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
torch.manual_seed(3001)
np.random.seed(3001)
random.seed(3001)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = torch.device('cuda')

def dice(p, g):
    return (2*(p*g).sum()+1e-6)/(p.sum()+g.sum()+1e-6)

def score(p, g):
    p = p >= .5
    g = g >= .5
    tp = (p & g).sum().item(); fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item(); tn = (~p & ~g).sum().item()
    return dict(dice=2*tp/max(2*tp+fp+fn,1), iou=tp/max(tp+fp+fn,1),
                sensitivity=tp/max(tp+fn,1), specificity=tn/max(tn+fp,1),
                accuracy=(tp+tn)/(tp+fp+fn+tn), tp=tp, fp=fp, fn=fn, tn=tn)

subjects = np.random.default_rng(42).permutation(np.arange(1,15)).tolist()
groups = dict(train=subjects[:8], validation=subjects[8:10], test=subjects[10:])
data = {}
for path in sorted((ROOT/'data').glob('*.jpg')):
    name = path.stem
    subject = int(name.split('_')[1][:2])
    rgb = np.asarray(Image.open(path).convert('RGB'))
    # Explicit paper order: green, CLAHE, inverse gamma, bilinear resize.
    green = cv2.createCLAHE(clipLimit=5., tileGridSize=(32,32)).apply(rgb[:,:,1])
    green = np.uint8(np.clip((green/255.)**(1/1.2)*255,0,255))
    green = cv2.resize(green,(512,512),interpolation=cv2.INTER_LINEAR)
    color = cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
    mask = np.asarray(Image.open(ROOT/'data'/f'{name}_1stHO.png').convert('L').resize((512,512),Image.Resampling.NEAREST)) > 127
    split = next(k for k,v in groups.items() if subject in v)
    data[name] = dict(subject=subject,split=split,x=torch.tensor(green/255.,dtype=torch.float32)[None],
                      rgb=torch.tensor(color/255.,dtype=torch.float32).permute(2,0,1),
                      g=torch.tensor(mask,dtype=torch.float32)[None])
manifest = {k:[n for n,d in data.items() if d['split']==k] for k in groups}
assert [len(manifest[k]) for k in groups] == [16,4,8]
config = dict(split=manifest,subjects=groups,seed=42,threshold=.5,region='whole image; no FOV mask',
              observer='1stHO',size=512,precision='FP32',baseline='official LNet_RA, input channels=1',
              baseline_max_epochs=1000,baseline_patience=20,baseline_lr=.001,weight_decay=.001,
              autoencoder_epochs=50,refiner_epochs=100,noise_strength=.1,ddim_transitions=50,
              sampling_seeds=[42,123,2026],claim='new experiment, not original manuscript partition')
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))

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
                for k in ('x','rgb','g'):
                    d[k]=rotate(d[k],angle,interpolation=InterpolationMode.NEAREST if k=='g' else InterpolationMode.BILINEAR)
                for dim in (-1,-2):
                    if random.random()<.5:
                        for k in ('x','rgb','g'): d[k]=d[k].flip(dim)
            samples.append(d)
        yield {k:torch.stack([d[k] for d in samples]).to(DEVICE) for k in ('x','rgb','g')}, names[start:start+4]


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
        val=np.mean([score(model(b['x'])[i],b['g'][i])['dice'] for b,_ in batches('validation') for i in range(len(b['g']))])
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
torch.cuda.manual_seed_all(3001)
job_stage('autoencoder_training'); ae=MaskAutoencoder().to(DEVICE)
opt=torch.optim.AdamW(ae.parameters(),lr=1e-4,weight_decay=.001)
best=float('inf')
for epoch in range(50):
    ae.train()
    for b,_ in batches('train'):
        opt.zero_grad(); p=ae(b['g']); loss=F.mse_loss(p,b['g'])+1-dice(p,b['g']); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): val=np.mean([(1-dice(ae(b['g']),b['g'])).item() for b,_ in batches('validation')])
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
