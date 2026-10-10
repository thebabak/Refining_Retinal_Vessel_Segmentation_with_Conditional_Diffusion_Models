"""Validation-only ablations. Never score or select against the test set."""
from pathlib import Path
import csv
import json
import torch
import numpy as np
from torch import nn

# Reuse the frozen preprocessing and metrics without invoking training/testing.
source=(Path(__file__).with_name('run.py')).read_text()
prefix=source.split('model=LNet_RA(in_channels=1)')[0]
prefix=prefix.replace("(OUT/'protocol.json').write_text(json.dumps(config,indent=2))",'')
exec(compile(prefix,str(Path(__file__).with_name('run.py')),'exec'))
OLD=OUT
OUT=Path(__file__).resolve().parent/'validation_diagnostics'
OUT.mkdir(exist_ok=True)
baseline=LNet_RA(in_channels=1).to(DEVICE)
baseline.load_state_dict(torch.load(OLD/'baseline.pt',weights_only=True)['state'])
baseline.eval()
ae=MaskAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(OLD/'autoencoder.pt',weights_only=True)); ae.eval()
ie=ImageEncoder().to(DEVICE); un=DiffusionUNet().to(DEVICE)
tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
cpt=torch.load(OLD/'refiner.pt',weights_only=True)
ie.load_state_dict(cpt['ie']); un.load_state_dict(cpt['un']); tm.load_state_dict(cpt['time'])
ie.eval(); un.eval(); tm.eval()
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*np.pi/2)**2; a=a/a[0]
alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
def emb(t):
    v=t[:,None]*torch.exp(-np.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c): return un(z,emb(t),c+tm(emb(t)))
rows=[]
with torch.no_grad():
    for name in manifest['validation']:
        d=data[name]; g=d['g'].to(DEVICE); coarse=baseline(d['x'][None].to(DEVICE))
        z0=ae.encode(coarse); reconstruction=ae.decode(z0)
        def add(method,p,seed=0,rho=0,effective_steps=0):
            rows.append(dict(image=name,method=method,seed=seed,rho=rho,effective_steps=effective_steps,**score(p[0],g)))
        add('baseline',coarse); add('AE_coarse_reconstruction',reconstruction)
        add('AE_GT_reconstruction_sanity_only',ae(g[None]))
        c=ie(d['rgb'][None].to(DEVICE))
        for rho in [.01,.03,.05,.1]:
            start=max(0,int(rho*1000)-1)
            ts=torch.linspace(start,0,50,device=DEVICE).long().unique_consecutive()
            for seed in [42,123,2026]:
                gen=torch.Generator(device=DEVICE).manual_seed(seed)
                z=alpha[start].sqrt()*z0+(1-alpha[start]).sqrt()*torch.randn(z0.shape,device=DEVICE,generator=gen)
                for j,ti in enumerate(ts):
                    tv=ti[None]; eps0=denoise(z,tv,torch.zeros_like(c)); eps1=denoise(z,tv,c)
                    eps=eps0+1.5*(eps1-eps0)
                    estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
                    prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
                    z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
                add('diffusion',ae.decode(z),seed,rho,len(ts))
        print(name,'baseline',score(coarse[0],g)['dice'],'AE',score(reconstruction[0],g)['dice'],flush=True)
with (OUT/'per_image.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={}
for method,rho in sorted({(r['method'],r['rho']) for r in rows}):
    selected=[r for r in rows if r['method']==method and r['rho']==rho]
    summary[f'{method}_rho{rho}']={k:float(np.mean([r[k] for r in selected])) for k in ['dice','sensitivity','specificity']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
