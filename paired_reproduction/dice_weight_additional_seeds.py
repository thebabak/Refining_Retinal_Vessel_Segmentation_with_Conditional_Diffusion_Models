"""Controlled Dice-weight ablation on NEW CHASE data."""
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
OUT=HERE/('dice_weight_additional_seeds_'+datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S'))
OUT.mkdir()
source=HERE/'run.py'
prefix=source.read_text().split('model=LNet_RA(in_channels=1).to(DEVICE)')[0]
prefix=prefix.replace("OUT = Path(__file__).resolve().parent / 'results'",'OUT = experiment_output')
experiment_output=OUT
exec(compile(prefix,str(source),'exec'),globals())
source_results=HERE/'results'
TRAIN_SEEDS=[3000,3001]
NOISE_SEEDS=[42,123,2026,3000,3001]
config.update(claim='New Dice-weight ablation, not original manuscript runs',
              training_seeds=TRAIN_SEEDS,sampling_seeds=NOISE_SEEDS,epochs=100,
              lambda_Dice_values=[0.,.25,.5,1.],image_conditioning='global 128-vector',
              fixed_baseline=str(source_results/'baseline.pt'),fixed_autoencoder=str(source_results/'autoencoder.pt'),
              selection='fixed epoch100; no test-based tuning',transitions=50,start_timestep=99,
              guidance=1.5,thin_metric='Recall on GT skeleton pixels with Euclidean vessel-mask radius <=2 resized pixels',
              boundary_metric='F1 of binary-mask boundaries with Euclidean tolerance <=2 resized pixels',
              limitations=['One fixed backbone/autoencoder; two additional refiner training seeds, to be combined with saved three-seed reproduction','Four CHASE subjects','Previously exposed test split; no tuning performed','Thin metric is a resized pixel-width proxy, not an anatomical capillary label'])
config.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,source_code_sha256=hashlib.sha256((HERE/'dice_weight_ablation.py').read_bytes()).hexdigest(),prior_three_seed_run=str(HERE/'dice_weight_ablation_2026-10-06_18-14-33'))
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
for filename in ('baseline.pt','autoencoder.pt'):shutil.copy2(source_results/filename,OUT/filename)
baseline=LNet_RA(in_channels=1).to(DEVICE)
baseline.load_state_dict(torch.load(OUT/'baseline.pt',weights_only=False)['state'])
baseline.eval();baseline.requires_grad_(False)
with torch.inference_mode():
    for d in data.values():d['coarse']=baseline(d['x'][None].to(DEVICE)).cpu()[0]
ae=MaskAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(OUT/'autoencoder.pt',weights_only=True));ae.eval();ae.requires_grad_(False)

class SpatialImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1,stride=2)
        self.conv2=nn.Conv2d(32,64,3,padding=1,stride=2)
        self.projection=nn.Conv2d(64,128,1)
    def forward(self,x):return self.projection(F.relu(self.conv2(F.relu(self.conv1(x)))))
    def copy_global(self,g):
        self.conv1.load_state_dict(g.net[0].state_dict());self.conv2.load_state_dict(g.net[2].state_dict())
        with torch.no_grad():
            self.projection.weight.copy_(g.net[6].weight[:,:,None,None]);self.projection.bias.copy_(g.net[6].bias)

class ConditioningUNet(DiffusionUNet):
    @staticmethod
    def block(block,x,c):
        if c.ndim==2:return block(x,c)
        h=block.conv1(block.act(block.norm1(x)))
        condition=F.interpolate(c,size=h.shape[-2:],mode='bilinear',align_corners=False)
        g,b=F.conv2d(condition,block.film.weight[:,:,None,None],block.film.bias).chunk(2,1)
        h=h*(1+g)+b
        return x+block.act(block.conv2(h))
    def forward(self,z,t,c):
        h=self.in_conv(z);h=self.block(self.res1,h,c)
        h=self.down(h);h=self.block(self.res2,h,c)
        return self.out_conv(self.up(h))

def skeletonize(mask):
    # Zhang-Suen two-subiteration thinning; zero padding outside the mask.
    x=np.pad(mask.astype(np.uint8),1)
    while True:
        changed=False
        for phase in (0,1):
            p=x[1:-1,1:-1]
            n=[x[:-2,1:-1],x[:-2,2:],x[1:-1,2:],x[2:,2:],x[2:,1:-1],x[2:,:-2],x[1:-1,:-2],x[:-2,:-2]]
            neighbours=sum(n)
            transitions=sum((n[i]==0)&(n[(i+1)%8]==1) for i in range(8))
            if phase==0:condition=(n[0]*n[2]*n[4]==0)&(n[2]*n[4]*n[6]==0)
            else:condition=(n[0]*n[2]*n[6]==0)&(n[0]*n[4]*n[6]==0)
            remove=(p==1)&(neighbours>=2)&(neighbours<=6)&(transitions==1)&condition
            if remove.any():p[remove]=0;changed=True
        if not changed:break
    return x[1:-1,1:-1].astype(bool)

def boundary(mask):return mask&~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
for name in manifest['test']:
    d=data[name];gt=d['g'].numpy()[0]>.5
    sk=skeletonize(gt);radius=cv2.distanceTransform(gt.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    d['thin']=sk&(radius<=2);assert d['thin'].any()
    d['boundary']=boundary(gt)
    d['boundary_distance']=cv2.distanceTransform((~d['boundary']).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    np.savez_compressed(OUT/f'{name}_reference_metrics.npz',gt=gt,skeleton=sk,thin_centerline=d['thin'],boundary=d['boundary'])

def extended_score(p,d):
    out=score(p,d['g']);mask=p.numpy()[0]>=.5;bp=boundary(mask)
    precision=float(np.mean(d['boundary_distance'][bp]<=2)) if bp.any() else 0.
    distance=cv2.distanceTransform((~bp).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    recall=float(np.mean(distance[d['boundary']]<=2)) if bp.any() else 0.
    out.update(thin_centerline_recall=float(mask[d['thin']].mean()),thin_reference_pixels=int(d['thin'].sum()),
               boundary_precision=precision,boundary_recall=recall,boundary_f1=2*precision*recall/max(precision+recall,1e-12))
    return out

t=torch.arange(1001,device=DEVICE)/1000
a=torch.cos((t+.008)/1.008*math.pi/2)**2;a=a/a[0]
alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
def emb(t):
    v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=DEVICE)/32)
    return torch.cat([v.sin(),v.cos()],1)
def denoise(z,t,c,un,tm):
    temporal=tm(emb(t))
    return un(z,emb(t),c+(temporal if c.ndim==2 else temporal[:,:,None,None]))
def sample(coarse,rgb,noise,ie,un,tm):
    z0=ae.encode(coarse);c=ie(rgb)
    z=alpha[99].sqrt()*z0+(1-alpha[99]).sqrt()*noise
    ts=torch.linspace(99,0,50,device=DEVICE).long()
    for j,ti in enumerate(ts):
        eps0=denoise(z,ti[None],torch.zeros_like(c),un,tm);eps1=denoise(z,ti[None],c,un,tm)
        eps=eps0+1.5*(eps1-eps0)
        estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt()
        prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=DEVICE)
        z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
    return ae.decode(z)

rows=[];counts={}
print('OUTPUT',OUT,flush=True)
for training_seed in TRAIN_SEEDS:
    torch.manual_seed(training_seed);torch.cuda.manual_seed_all(training_seed)
    initial_ie=ImageEncoder().to(DEVICE);initial_un=ConditioningUNet().to(DEVICE)
    initial_tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE)
    initial={k:{n:v.clone() for n,v in m.state_dict().items()} for k,m in [('ie',initial_ie),('un',initial_un),('time',initial_tm)]}
    for weight in (0.,.25,.5,1.):
        variant='dice'+str(weight).replace('.','p')
        ie=ImageEncoder().to(DEVICE)
        ie.load_state_dict(initial['ie'])
        un=ConditioningUNet().to(DEVICE);un.load_state_dict(initial['un'])
        tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(DEVICE);tm.load_state_dict(initial['time'])
        counts[variant]=dict(image_encoder=sum(p.numel() for p in ie.parameters()),diffusion_unet=sum(p.numel() for p in un.parameters()),time_MLP=sum(p.numel() for p in tm.parameters()),frozen_autoencoder=sum(p.numel() for p in ae.parameters()))
        assert sum(counts[variant].values())==527233
        if variant=='spatial':
            with torch.inference_mode():
                rgb=data[manifest['validation'][0]]['rgb'][None].to(DEVICE)
                assert torch.allclose(ie(rgb).mean((2,3)),initial_ie(rgb),atol=1e-6)
        params=list(ie.parameters())+list(un.parameters())+list(tm.parameters())
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
        # Reset training randomness, independent of initialization and evaluation.
        torch.manual_seed(100000+training_seed);torch.cuda.manual_seed_all(100000+training_seed)
        logs=[]
        for epoch in range(100):
            ie.train();un.train();tm.train();losses=[]
            for b,names in batches('train'):
                coarse=torch.stack([data[n]['coarse'] for n in names]).to(DEVICE)
                with torch.no_grad():z=ae.encode(coarse)
                ti=torch.randint(0,1000,(len(names),),device=DEVICE);at=alpha[ti,None,None,None]
                noise=torch.randn_like(z);zt=at.sqrt()*z+(1-at).sqrt()*noise
                c=ie(b['rgb']);drop=(torch.rand(len(names),device=DEVICE)>.1)
                c=c*drop.reshape((-1,)+(1,)*(c.ndim-1))
                pred=denoise(zt,ti,c,un,tm);estimate=(zt-(1-at).sqrt()*pred)/at.sqrt()
                noise_loss=F.mse_loss(pred,noise)
                segmentation_loss=1-dice(ae.decode(estimate),b['g'])
                loss=noise_loss+weight*segmentation_loss
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(loss.item())
            scheduler.step();logs.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
            if (epoch+1)%10==0:print('train',training_seed,variant,epoch+1,logs[-1]['loss'],flush=True)
        torch.save(dict(ie=ie.state_dict(),un=un.state_dict(),time=tm.state_dict(),variant=variant,lambda_Dice=weight,training_seed=training_seed,epoch=100),OUT/f'{variant}_trainseed{training_seed}.pt')
        (OUT/f'{variant}_trainseed{training_seed}_log.json').write_text(json.dumps(logs,indent=2))
        ie.eval();un.eval();tm.eval()
        with torch.inference_mode():
            for name in manifest['test']:
                d=data[name];coarse=d['coarse'][None].to(DEVICE);rgb=d['rgb'][None].to(DEVICE)
                base=extended_score(d['coarse'],d);shape=ae.encode(coarse).shape
                for _ in range(2):
                    warm=torch.randn(shape,device=DEVICE,generator=torch.Generator(device=DEVICE).manual_seed(999));sample(coarse,rgb,warm,ie,un,tm)
                for noise_seed in NOISE_SEEDS:
                    noise=torch.randn(shape,device=DEVICE,generator=torch.Generator(device=DEVICE).manual_seed(noise_seed))
                    torch.cuda.synchronize();start=time.perf_counter()
                    p=sample(coarse,rgb,noise,ie,un,tm)
                    torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
                    p=p.cpu()[0];assert torch.isfinite(p).all();metrics=extended_score(p,d)
                    np.savez_compressed(OUT/f'{name}_{variant}_train{training_seed}_noise{noise_seed}.npz',probability=p.numpy())
                    rows.append(dict(image=name,subject=d['subject'],variant=variant,lambda_Dice=weight,training_seed=training_seed,sampling_seed=noise_seed,ms=elapsed,**metrics,**{f'baseline_{k}':v for k,v in base.items()}))
                print('eval',training_seed,variant,name,flush=True)
        with (OUT/'per_image_seed_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    del initial_ie,initial_un,initial_tm,initial;torch.cuda.empty_cache()

metric_names=['dice','thin_centerline_recall','boundary_f1','sensitivity','specificity','accuracy','iou','ms']
subjects=sorted({r['subject'] for r in rows})
weights=[0.,.25,.5,1.]
summary=dict(parameter_counts=counts,refiner_total=527233,refiner_trainable=405824,baseline_total=1946961,complete_total=2474194,
             n_images=8,n_subjects=4,n_refiner_training_seeds=2,n_sampling_seeds=5,lambda_Dice_values=weights,metrics={},limitations=config['limitations'])
rng=np.random.default_rng(20261006)
for metric in metric_names:
    arrays={str(w):np.array([[np.mean([r[metric] for r in rows if r['lambda_Dice']==w and r['subject']==s and r['training_seed']==ts]) for ts in TRAIN_SEEDS] for s in subjects]) for w in weights}
    summary['metrics'][metric]={}
    for w in weights:
        arr=arrays[str(w)];delta=arr-arrays['0.0']
        boots=np.array([delta[rng.integers(0,4,4)][:,rng.integers(0,2,2)].mean() for _ in range(20000)])
        summary['metrics'][metric][str(w)]=dict(mean=float(arr.mean()),training_seed_means=arr.mean(0).tolist(),difference_vs_lambda0=float(delta.mean()),crossed_subject_training_seed_95ci=np.quantile(boots,[.025,.975]).tolist())
summary['checkpoint_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.pt')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
