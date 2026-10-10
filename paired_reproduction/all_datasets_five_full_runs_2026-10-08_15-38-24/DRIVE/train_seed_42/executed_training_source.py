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

OUT = experiment_output
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
config=dict(split=manifest,seed=42,threshold=.5,region='official FOV resized nearest',size=512,precision='FP32',baseline='official RVS LNet_RA in_channels=1',baseline_max_epochs=1000,baseline_patience=20,baseline_lr=.001,weight_decay=.001,AE_epochs=200,AE_lr=.001,AE_selection='validation coarse reconstruction MSE',refiner_epochs=100,noise_strength=.1,ddim_transitions=50,sampling_seeds=[42,123,2026,3000,3001],annotation_source=str(gt_test),annotation_provenance='local manual1 PNGs; acquisition provenance not established',claim='new DRIVE experiment; not original manuscript 0.835')
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
(OUT/'file_provenance.json').write_text(json.dumps({n:{k:d[k] for k in ['image_path','gt_path','fov_path']} for n,d in data.items()},indent=2))

def batches(split, augment=False):
    names = manifest[split].copy()
    if augment: random.shuffle(names)
    for start in range(0,len(names),4):
        samples=[]
        for name in names[start:start+4]:
            d={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in data[name].items()}
            if augment:
                angle=random.choice([0,15,30,45,90,100,120])
                from torchvision.transforms.functional import rotate
                from torchvision.transforms import InterpolationMode
                for k in ('x','rgb','g','fov'):
                    d[k]=rotate(d[k].float(),angle,interpolation=InterpolationMode.NEAREST if k in ('g','fov') else InterpolationMode.BILINEAR)
                for dim in (-1,-2):
                    if random.random()<.5:
                        for k in ('x','rgb','g','fov'): d[k]=d[k].flip(dim)
            samples.append(d)
        yield {k:torch.stack([d[k] for d in samples]).to(DEVICE) for k in ('x','rgb','g','fov')}, names[start:start+4]


model=LNet_RA(in_channels=1).to(DEVICE)
stage_initial_rng=torch.get_rng_state()
job_stage('baseline_training'); print('Baseline parameters',sum(p.numel() for p in model.parameters()),flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
best=-1.; stale=0
log=[]
for epoch in range(0 if '--resume-baseline' in sys.argv else 1000):
    model.train(); losses=[]
    for b,_ in batches('train',True):
        opt.zero_grad(); p=model(b['x']); loss=1-dice(p,b['g']); loss.backward(); opt.step(); losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val=np.mean([score(model(b['x'])[i],b['g'][i],b['fov'][i]>.5)['dice'] for b,_ in batches('validation') for i in range(len(b['g']))])
    log.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_dice=float(val)))
    (OUT/'baseline_training.json').write_text(json.dumps(log))
    if val>best:
        best=val; stale=0
        torch.save(dict(state=model.state_dict(),epoch=epoch+1,validation_dice=best,protocol=config),OUT/'baseline.pt')
    else: stale+=1
    print(f'baseline epoch {epoch+1} val={val:.4f} best={best:.4f} patience={stale}',flush=True)
    if stale>=20: break
# This checkpoint was created by this script; older local saves include a NumPy scalar.
saved=torch.load(OUT/'baseline.pt',weights_only=False)
saved['validation_dice']=float(saved['validation_dice'])
torch.save(saved,OUT/'baseline.pt')
model.load_state_dict(saved['state']); model.eval()
if '--baseline-only' in sys.argv:
    sys.exit(0)
with torch.no_grad():
    for d in data.values(): d['coarse']=model(d['x'][None].to(DEVICE)).cpu()[0]
# Stage seeds are independent of how many baseline epochs ran; also reproduce resume.
torch.set_rng_state(stage_initial_rng)
torch.cuda.manual_seed_all(42)
job_stage('autoencoder_training'); ae=MaskAutoencoder().to(DEVICE)
opt=torch.optim.AdamW(ae.parameters(),lr=.001,weight_decay=.001)
AE_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
best=float('inf')
AE_logs=[]
for epoch in range(200):
    ae.train()
    for b,names in batches('train'):
        coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
        inputs=torch.cat([coarse,b['g']],0)
        opt.zero_grad(); p=ae(inputs); loss=F.mse_loss(p,inputs)+1-dice(p,inputs); loss.backward(); opt.step()
    AE_schedule.step()
    ae.eval()
    with torch.no_grad():
        val=float(np.mean([F.mse_loss(ae(data[n]['coarse'][None].to(DEVICE)),data[n]['coarse'][None].to(DEVICE)).item() for n in manifest['validation']]))
    AE_logs.append(val)
    (OUT/'AE_training.json').write_text(json.dumps(AE_logs))
    if val<best: best=val; torch.save(ae.state_dict(),OUT/'autoencoder.pt')
    print(f'AE epoch {epoch+1} validation_loss={val:.4f}',flush=True)
ae.load_state_dict(torch.load(OUT/'autoencoder.pt',weights_only=True)); ae.eval(); ae.requires_grad_(False)
job_stage('refiner_training'); ie=ImageEncoder().to(DEVICE); un=DiffusionUNet().to(DEVICE)
time_mlp=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
params=list(ie.parameters())+list(un.parameters())+list(time_mlp.parameters())
opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2; a=a/a[0]
betas=(1-a[1:]/a[:-1]).clamp(max=.999); alpha=torch.cumprod(1-betas,0)
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c): return un(z,emb(t),c+time_mlp(emb(t)))
ref_log=[]
for epoch in range(100):
    ie.train(); un.train(); time_mlp.train(); losses=[]
    for b,names in batches('train'):
        coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
        with torch.no_grad(): z=ae.encode(coarse)
        ti=torch.randint(0,1000,(len(names),),device=DEVICE); at=alpha[ti,None,None,None]
        noise=torch.randn_like(z); zt=at.sqrt()*z+(1-at).sqrt()*noise
        c=ie(b['rgb']); c=c*(torch.rand(len(names),1,device=DEVICE)>.1)
        pred=denoise(zt,ti,c); zhat=(zt-(1-at).sqrt()*pred)/at.sqrt()
        decoded=ae.decode(zhat)
        loss=F.mse_loss(pred,noise)+.5*(1-dice(decoded,b['g']))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step(); losses.append(loss.item())
    schedule.step(); ref_log.append(float(np.mean(losses)))
    torch.save(dict(ie=ie.state_dict(),un=un.state_dict(),time=time_mlp.state_dict(),epoch=epoch+1),OUT/'refiner.pt')
    (OUT/'refiner_training.json').write_text(json.dumps(ref_log))
    print(f'refiner epoch {epoch+1} loss={ref_log[-1]:.4f}',flush=True)
