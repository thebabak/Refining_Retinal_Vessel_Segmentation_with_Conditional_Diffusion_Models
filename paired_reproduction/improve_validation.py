"""Repair reconstruction, retrain denoiser, choose only on validation."""
from pathlib import Path
import csv,json,math
import torch,numpy as np
from torch import nn
import torch.nn.functional as F

source=Path(__file__).with_name('run.py').read_text()
prefix=source.split('model=LNet_RA(in_channels=1)')[0]
prefix=prefix.replace("(OUT/'protocol.json').write_text(json.dumps(config,indent=2))",'')
exec(compile(prefix,str(Path(__file__).with_name('run.py')),'exec'))
OLD=OUT; OUT=Path(__file__).resolve().parent/'validation_improvement'; OUT.mkdir(exist_ok=True)
baseline=LNet_RA(in_channels=1).to(DEVICE)
baseline.load_state_dict(torch.load(OLD/'baseline.pt',weights_only=True)['state']); baseline.eval()
with torch.no_grad():
    for split in ['train','validation']:
        for name in manifest[split]:
            d=data[name]; d['coarse']=baseline(d['x'][None].to(DEVICE)).cpu()[0]
ae=MaskAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(OLD/'autoencoder.pt',weights_only=True))
opt=torch.optim.AdamW(ae.parameters(),lr=.001,weight_decay=.001)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
best=float('inf'); logs=[]
for epoch in range(200):
    ae.train(); losses=[]
    for b,names in batches('train'):
        coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
        inputs=torch.cat([coarse,b['g']],0)
        p=ae(inputs)
        # Identity targets: never substitute GT for a model prediction at inference.
        loss=F.mse_loss(p,inputs)+(1-dice(p,inputs))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    sch.step(); ae.eval()
    with torch.no_grad():
        errors=[]; scores=[]
        for name in manifest['validation']:
            d=data[name]; c=d['coarse'][None].to(DEVICE); p=ae(c)
            errors.append(F.mse_loss(p,c).item()); scores.append(score(p[0],d['g'].to(DEVICE))['dice'])
    err=float(np.mean(errors)); val=float(np.mean(scores))
    if err<best: best=err; torch.save(ae.state_dict(),OUT/'autoencoder.pt')
    logs.append(dict(epoch=epoch+1,identity_mse=err,validation_dice=val,train_loss=float(np.mean(losses))))
    if epoch%10==0: print('AE',epoch+1,'identity MSE',err,'validation Dice',val,flush=True)
(OUT/'AE_training.json').write_text(json.dumps(logs,indent=2))
ae.load_state_dict(torch.load(OUT/'autoencoder.pt',weights_only=True)); ae.eval(); ae.requires_grad_(False)
# Fresh denoiser required because the mask encoder changed.
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
ie=ImageEncoder().to(DEVICE); un=DiffusionUNet().to(DEVICE)
tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
params=list(ie.parameters())+list(un.parameters())+list(tm.parameters())
opt=torch.optim.AdamW(params,lr=.0001,weight_decay=.001)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2; a=a/a[0]
alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c): return un(z,emb(t),c+tm(emb(t)))
logs=[]
for epoch in range(100):
    ie.train(); un.train(); tm.train(); losses=[]
    for b,names in batches('train'):
        coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
        with torch.no_grad(): z=ae.encode(coarse)
        ti=torch.randint(0,1000,(len(names),),device=DEVICE); at=alpha[ti,None,None,None]
        noise=torch.randn_like(z); zt=at.sqrt()*z+(1-at).sqrt()*noise
        c=ie(b['rgb'])*(torch.rand(len(names),1,device=DEVICE)>.1)
        pred=denoise(zt,ti,c); decoded=ae.decode((zt-(1-at).sqrt()*pred)/at.sqrt())
        loss=F.mse_loss(pred,noise)+.5*(1-dice(decoded,b['g']))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step(); losses.append(loss.item())
    sch.step(); logs.append(float(np.mean(losses)))
    if epoch%10==0: print('diffusion',epoch+1,'loss',logs[-1],flush=True)
torch.save(dict(ie=ie.state_dict(),un=un.state_dict(),time=tm.state_dict()),OUT/'refiner.pt')
(OUT/'diffusion_training.json').write_text(json.dumps(logs))
ie.eval(); un.eval(); tm.eval(); rows=[]
with torch.no_grad():
    for name in manifest['validation']:
        d=data[name]; gt=d['g'].to(DEVICE); coarse=d['coarse'][None].to(DEVICE)
        def add(method,p,rho=0,seed=0):
            rows.append(dict(image=name,method=method,rho=rho,seed=seed,**score(p[0],gt)))
        add('baseline',coarse); add('AE_reconstruction',ae(coarse))
        c=ie(d['rgb'][None].to(DEVICE)); z0=ae.encode(coarse)
        for rho in [.01,.03,.05,.1]:
            start=int(rho*1000)-1; ts=torch.linspace(start,0,50,device=DEVICE).long().unique_consecutive()
            for seed in [42,123,2026]:
                gen=torch.Generator(device=DEVICE).manual_seed(seed)
                z=alpha[start].sqrt()*z0+(1-alpha[start]).sqrt()*torch.randn(z0.shape,device=DEVICE,generator=gen)
                for j,ti in enumerate(ts):
                    tv=ti[None]; e0=denoise(z,tv,torch.zeros_like(c)); e1=denoise(z,tv,c)
                    eps=e0+1.5*(e1-e0); estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
                    prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
                    z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
                add('diffusion',ae.decode(z),rho,seed)
        print('validation evaluated',name,flush=True)
with (OUT/'per_image.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
means={}
for method,rho in sorted({(r['method'],r['rho']) for r in rows}):
    means[f'{method}_rho{rho}']=float(np.mean([r['dice'] for r in rows if r['method']==method and r['rho']==rho]))
summary=dict(validation_dice=means,selected=max(means,key=means.get),selection='validation only; includes baseline fallback',test_evaluated=False,limitations=['Only 4 validation images from 2 subjects','Exploratory selection, not evidence of generalisation','No target-latent or spatial-conditioning architectural change yet'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
