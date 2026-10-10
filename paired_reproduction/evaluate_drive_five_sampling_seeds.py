"""New subject-disjoint experiment; never a recovery of manuscript 0.835."""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from official_model import LNet_RA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_refiner.models import MaskAutoencoder, ImageEncoder, DiffusionUNet

OUT = Path(__file__).resolve().parent / 'drive_five_sampling_seeds_2026-10-07_19-08-36'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = torch.device('cuda')

def dice(p, g):
    return (2*(p*g).sum()+1e-6)/(p.sum()+g.sum()+1e-6)

def score(p, g, fov):
    p=p[fov]; g=g[fov]
    p = p >= .5
    g = g >= .5
    tp = (p & g).sum().item(); fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item(); tn = (~p & ~g).sum().item()
    return dict(dice=2*tp/max(2*tp+fp+fn,1), iou=tp/max(tp+fp+fn,1),
                sensitivity=tp/max(tp+fn,1), specificity=tn/max(tn+fp,1),
                accuracy=(tp+tn)/(tp+fp+fn+tn), tp=tp, fp=fp, fn=fn, tn=tn)

base=ROOT/'diffusion_refiner'/'DRIVE'
gt_test=Path('F:/PHD/AI in Med/augmentaionforITinMED/data/drive_test_ground_truth')
training=sorted((base/'training'/'images').glob('*.tif'))
validation_ids=set(np.random.default_rng(42).choice([p.stem[:2] for p in training],4,replace=False).tolist())
data={}
for folder in ['training','test']:
    for path in sorted((base/folder/'images').glob('*.tif')):
        name=path.stem; num=name[:2]
        gt_path=base/'training'/'1st_manual'/f'{num}_manual1.gif' if folder=='training' else gt_test/f'{num}_manual1.png'
        fov_path=base/folder/'mask'/f'{num}_{folder}_mask.gif'
        assert gt_path.exists() and fov_path.exists(),(gt_path,fov_path)
        rgb=np.asarray(Image.open(path).convert('RGB'))
        green=cv2.createCLAHE(clipLimit=5.,tileGridSize=(32,32)).apply(rgb[:,:,1])
        green=np.uint8(np.clip((green/255.)**(1/1.2)*255,0,255))
        green=cv2.resize(green,(512,512),interpolation=cv2.INTER_LINEAR)
        color=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
        mask=np.asarray(Image.open(gt_path).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
        fov=np.asarray(Image.open(fov_path).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
        assert .01<mask.mean()<.3, 'Not a plausible vessel annotation'
        split='test' if folder=='test' else 'validation' if num in validation_ids else 'train'
        data[name]=dict(subject=int(num),split=split,x=torch.tensor(green/255.,dtype=torch.float32)[None],rgb=torch.tensor(color/255.,dtype=torch.float32).permute(2,0,1),g=torch.tensor(mask,dtype=torch.float32)[None],fov=torch.tensor(fov)[None],image_path=str(path),gt_path=str(gt_path),fov_path=str(fov_path))
manifest={k:[n for n,d in data.items() if d['split']==k] for k in ['train','validation','test']}
assert [len(manifest[k]) for k in manifest]==[16,4,20]
groups={'test':[data[n]['subject'] for n in manifest['test']]}
SOURCE = Path(__file__).resolve().parent / 'drive_results'
config=json.loads((SOURCE/'protocol.json').read_text())
config['sampling_seeds']=[42,123,2026,3000,3001]
config['claim']='Five sampling seeds of existing DRIVE checkpoint; separate from author-supplied corrected table; one training seed'
config['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
config['checkpoint_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.pt')}
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
model=LNet_RA(in_channels=1).to(DEVICE)
model.load_state_dict(torch.load(SOURCE/'baseline.pt',weights_only=False)['state']); model.eval()
with torch.no_grad():
    for d in data.values(): d['coarse']=model(d['x'][None].to(DEVICE)).cpu()[0]
ae=MaskAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(SOURCE/'autoencoder.pt',weights_only=True)); ae.eval(); ae.requires_grad_(False)
ie=ImageEncoder().to(DEVICE); un=DiffusionUNet().to(DEVICE)
time_mlp=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
ref=torch.load(SOURCE/'refiner.pt',weights_only=True)
ie.load_state_dict(ref['ie']); un.load_state_dict(ref['un']); time_mlp.load_state_dict(ref['time'])
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2; a=a/a[0]
betas=(1-a[1:]/a[:-1]).clamp(max=.999); alpha=torch.cumprod(1-betas,0)
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c): return un(z,emb(t),c+time_mlp(emb(t)))
ie.eval(); un.eval(); time_mlp.eval()
rows=[]
with torch.no_grad():
    for name in manifest['test']:
        d=data[name]; coarse=d['coarse'][None].to(DEVICE); z0=ae.encode(coarse); c=ie(d['rgb'][None].to(DEVICE))
        base=score(d['coarse'],d['g'],d['fov']); np.save(OUT/f'{name}_coarse.npy',d['coarse'].numpy())
        for seed in config['sampling_seeds']:
            gen=torch.Generator(device=DEVICE).manual_seed(seed)
            start=99
            z=alpha[start].sqrt()*z0+(1-alpha[start]).sqrt()*torch.randn(z0.shape,device=DEVICE,generator=gen)
            ts=torch.linspace(start,0,50,device=DEVICE).long()
            for j,ti in enumerate(ts):
                tv=ti[None]; eps0=denoise(z,tv,torch.zeros_like(c)); eps1=denoise(z,tv,c)
                eps=eps0+1.5*(eps1-eps0)
                estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
                prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
                z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
            pred=ae.decode(z).cpu()[0]; refined=score(pred,d['g'],d['fov'])
            np.save(OUT/f'{name}_refined_seed{seed}.npy',pred.numpy())
            rows.append(dict(image=name,subject=d['subject'],seed=seed,baseline_dice=base['dice'],refined_dice=refined['dice'],delta=refined['dice']-base['dice'],**{f'baseline_{k}':v for k,v in base.items() if k!='dice'},**{f'refined_{k}':v for k,v in refined.items() if k!='dice'}))
            print(name,seed,base['dice'],refined['dice'],flush=True)
with (OUT/'paired_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# Average stochastic runs per image, then bootstrap subjects (both eyes together).
subject_values=np.array([np.mean([r['delta'] for r in rows if r['subject']==s]) for s in groups['test']])
rng=np.random.default_rng(2026); boot=subject_values[rng.integers(0,len(subject_values),(10000,len(subject_values)))].mean(1)
summary=dict(baseline_dice=float(np.mean([r['baseline_dice'] for r in rows])),refined_dice=float(np.mean([r['refined_dice'] for r in rows])),delta=float(subject_values.mean()),image_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),n_images=20,n_subjects=None,sampling_seeds=config['sampling_seeds'],limitations=['Single training seed','Official DRIVE test set; does not reproduce manuscript 0.835','Local test annotation acquisition provenance not established','In-sample coarse training maps','Refiner time conditioning and sampler corrected; new implementation'],checkpoint_sha256=config['checkpoint_sha256'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

from collections import defaultdict
seed_scores=defaultdict(list)
for row in rows: seed_scores[row['seed']].append(row['refined_dice'])
means={k:float(np.mean(v)) for k,v in seed_scores.items()}
summary['seed_means']=means
summary['sd_between_sampling_seed_means']=float(np.std(list(means.values()),ddof=1))
summary['limitations'].append('Five noise samples of one trained model, not five training runs; does not extend author-supplied corrected scores')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'paired_metrics.txt').write_bytes((OUT/'paired_metrics.csv').read_bytes())
print('FINAL_SUMMARY',json.dumps(summary),flush=True)
