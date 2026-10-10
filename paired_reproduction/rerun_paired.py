"""New paired CHASE/HRF evaluation; never recovery of deleted original runs."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['CHASE', 'HRF'], required=True)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
source = HERE / ('run.py' if args.dataset == 'CHASE' else 'run_hrf_baseline.py')
out = HERE / ('paired_rerun_' + args.dataset + '_' + datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S'))
out.mkdir()
# Reuse inspected data loading and metric definitions, without running training
# or touching prior results. Prefix ends before the baseline is instantiated.
prefix = source.read_text().split('model=LNet_RA(in_channels=1).to(DEVICE)')[0]
old_output = "OUT = Path(__file__).resolve().parent / '" + ('results' if args.dataset=='CHASE' else 'hrf_results') + "'"
assert prefix.count(old_output)==1
prefix = prefix.replace(old_output, 'OUT = out')
exec(compile(prefix, str(source), 'exec'), globals())
old = HERE / ('results' if args.dataset=='CHASE' else 'hrf_results')
seeds = [42, 123, 2026] + list(range(3000, 3017))
config.update(claim='New paired rerun, not deleted manuscript results', sampling_seeds=seeds,
              transition_counts=[20,50], initialization_timestep=99, guidance=1.5,
              baseline_source=str(old/'baseline.pt'), training_runs=1,
              checkpoint_policy='Previously validation-selected baseline; CHASE existing documented refiner; HRF AE validation selection and fixed 100 refiner epochs; no test-based tuning',
              AE_epochs=50, AE_lr=1e-4, refiner_epochs=100)
if args.dataset=='CHASE':
    config['refiner_source']=str(old/'refiner.pt')
else:
    config['refiner_source']='new training on HRF train partition only'
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
shutil.copy2(old/'baseline.pt',OUT/'baseline.pt')
model=LNet_RA(in_channels=1).to(DEVICE)
saved=torch.load(OUT/'baseline.pt',weights_only=False)
model.load_state_dict(saved['state']);model.eval();model.requires_grad_(False)
with torch.inference_mode():
    for name,d in data.items():d['coarse']=model(d['x'][None].to(DEVICE)).cpu()[0]
print('OUTPUT',OUT,flush=True)
torch.manual_seed(42);torch.cuda.manual_seed_all(42)
ae=MaskAutoencoder().to(DEVICE)
if args.dataset=='CHASE':
    shutil.copy2(old/'autoencoder.pt',OUT/'autoencoder.pt')
else:
    opt=torch.optim.AdamW(ae.parameters(),lr=1e-4,weight_decay=.001)
    best=float('inf');logs=[]
    for epoch in range(50):
        ae.train()
        for b,_ in batches('train'):
            opt.zero_grad();p=ae(b['g']);loss=F.mse_loss(p,b['g'])+1-dice(p,b['g']);loss.backward();opt.step()
        ae.eval()
        with torch.no_grad():val=float(np.mean([(1-dice(ae(b['g']),b['g'])).item() for b,_ in batches('validation')]))
        logs.append(dict(epoch=epoch+1,validation_loss=val))
        if val<best:best=val;torch.save(ae.state_dict(),OUT/'autoencoder.pt')
        (OUT/'autoencoder_training.json').write_text(json.dumps(logs,indent=2))
        if (epoch+1)%5==0:print('HRF AE',epoch+1,'validation loss',val,flush=True)
ae.load_state_dict(torch.load(OUT/'autoencoder.pt',weights_only=True));ae.eval();ae.requires_grad_(False)
ie=ImageEncoder().to(DEVICE);un=DiffusionUNet().to(DEVICE)
tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2;a=a/a[0]
alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c):return un(z,emb(t),c+tm(emb(t)))
if args.dataset=='CHASE':
    shutil.copy2(old/'refiner.pt',OUT/'refiner.pt')
else:
    params=list(ie.parameters())+list(un.parameters())+list(tm.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100);logs=[]
    for epoch in range(100):
        ie.train();un.train();tm.train();losses=[]
        for b,names in batches('train'):
            coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
            with torch.no_grad():z=ae.encode(coarse)
            ti=torch.randint(0,1000,(len(names),),device=DEVICE);at=alpha[ti,None,None,None]
            noise=torch.randn_like(z);zt=at.sqrt()*z+(1-at).sqrt()*noise
            c=ie(b['rgb']);c=c*(torch.rand(len(names),1,device=DEVICE)>.1)
            pred=denoise(zt,ti,c);zhat=(zt-(1-at).sqrt()*pred)/at.sqrt()
            loss=F.mse_loss(pred,noise)+.5*(1-dice(ae.decode(zhat),b['g']))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(loss.item())
        schedule.step();logs.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        torch.save(dict(ie=ie.state_dict(),un=un.state_dict(),time=tm.state_dict(),epoch=epoch+1),OUT/'refiner.pt')
        (OUT/'refiner_training.json').write_text(json.dumps(logs,indent=2))
        if (epoch+1)%5==0:print('HRF refiner',epoch+1,'loss',logs[-1]['loss'],flush=True)
ck=torch.load(OUT/'refiner.pt',weights_only=True)
for m,k in ((ie,'ie'),(un,'un'),(tm,'time')):m.load_state_dict(ck[k]);m.eval()
def metrics(p,d):
    return score(p,d['g']) if args.dataset=='CHASE' else score(p,d['g'],d['fov'])
rows=[]
with torch.inference_mode():
    for name in manifest['test']:
        d=data[name];baseline=metrics(d['coarse'],d)
        coarse=d['coarse'][None].to(DEVICE);z0=ae.encode(coarse);c=ie(d['rgb'][None].to(DEVICE))
        np.save(OUT/f'{name}_coarse.npy',d['coarse'].numpy())
        for seed in seeds:
            noise=torch.randn(z0.shape,device=DEVICE,generator=torch.Generator(device=DEVICE).manual_seed(seed))
            for steps in (20,50):
                z=alpha[99].sqrt()*z0+(1-alpha[99]).sqrt()*noise
                ts=torch.linspace(99,0,steps,device=DEVICE).long()
                for j,ti in enumerate(ts):
                    eps0=denoise(z,ti[None],torch.zeros_like(c));eps1=denoise(z,ti[None],c)
                    eps=eps0+1.5*(eps1-eps0)
                    estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
                    prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
                    z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
                pred=ae.decode(z).cpu()[0];refined=metrics(pred,d)
                assert torch.isfinite(pred).all()
                np.savez_compressed(OUT/f'{name}_steps{steps}_seed{seed}.npz',probability=pred.numpy())
                rows.append(dict(image=name,unit=str(d['subject']) if args.dataset=='CHASE' else name,
                                 pathology='' if args.dataset=='CHASE' else name.split('_')[1],seed=seed,steps=steps,
                                 **{f'baseline_{k}':v for k,v in baseline.items()},**{f'refined_{k}':v for k,v in refined.items()},delta=refined['dice']-baseline['dice']))
        with (OUT/'paired_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('evaluated',args.dataset,name,baseline['dice'],flush=True)
summary={};per=[]
rng=np.random.default_rng(20261006)
for steps in (20,50):
    for name in manifest['test']:
        rr=[r for r in rows if r['image']==name and r['steps']==steps]
        assert len(rr)==20
        per.append(dict(image=name,unit=rr[0]['unit'],pathology=rr[0]['pathology'],steps=steps,
                        baseline_dice=rr[0]['baseline_dice'],refined_dice=float(np.mean([r['refined_dice'] for r in rr])),delta=float(np.mean([r['delta'] for r in rr]))))
    pp=[r for r in per if r['steps']==steps]
    if args.dataset=='CHASE':
        units=sorted({r['unit'] for r in pp})
        values=np.array([[np.mean([r[k] for r in pp if r['unit']==u]) for k in ('baseline_dice','refined_dice','delta')] for u in units])
        boots=values[rng.integers(0,len(values),(50000,len(values)))].mean(1)
    else:
        strata=[np.array([[r[k] for k in ('baseline_dice','refined_dice','delta')] for r in pp if r['pathology']==g]) for g in ('h','dr','g')]
        boots=np.mean([v[rng.integers(0,len(v),(50000,len(v)))].mean(1) for v in strata],axis=0)
        values=np.array([[r[k] for k in ('baseline_dice','refined_dice','delta')] for r in pp])
    summary[str(steps)]=dict(mean=values.mean(0).tolist(),columns=['baseline_dice','refined_dice','paired_difference'],CI95=np.quantile(boots,[.025,.975],axis=0).tolist())
with (OUT/'per_image_means.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
summary.update(dataset=args.dataset,n_images=len(manifest['test']),sampling_seeds=20,training_runs=1,
               bootstrap='50000 subject-clustered resamples for CHASE; pathology-stratified image resamples for HRF; seed-averaged scores; conditional on fixed training run',
               limitations=['New configuration, not original manuscript results','Four independent CHASE subjects; HRF patient IDs unavailable','No training-run variability estimate','Test partitions previously exposed; no tuning performed on this rerun'],
               checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.pt')})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
