"""New spatial RGB+coarse U-Net experiment; never alters supplied score tables."""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SEEDS=[42,123,2026,3000,3001]
REF=HERE/'all_datasets_five_full_runs_2026-10-08_15-38-24'/'CHASE'

def stamp():return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
def save(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def worker(out,seed):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_all_datasets_five_full_runs import compose
    tag=out.name[-19:];job=out/f'seed_{seed}';job.mkdir(exist_ok=True)
    done=job/f'completed_{tag}.json'
    if done.exists():return
    source,prefix,_=compose('CHASE',seed,job)
    prefix='\n'.join(l for l in prefix.splitlines() if not l.startswith("(OUT/'protocol.json')"))
    ns=dict(__file__=str(source),__name__='spatial_control',experiment_output=job)
    exec(compile(prefix,str(source),'exec'),ns)
    data=ns['data'];manifest=ns['manifest'];device=ns['DEVICE'];score=ns['score']
    reference=REF/f'train_seed_{seed}';baseline=ns['LNet_RA'](in_channels=1).to(device)
    baseline.load_state_dict(torch.load(reference/'baseline.pt',weights_only=False)['state'])
    baseline.eval();baseline.requires_grad_(False)
    with torch.inference_mode():
        for d in data.values():d['coarse']=baseline(d['x'][None].to(device)).cpu()[0]
    class Block(nn.Module):
        def __init__(self,a,b):
            super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
        def forward(self,x):return self.net(x)
    class Refiner(nn.Module):
        def __init__(self):
            super().__init__();self.enc=nn.ModuleList([Block(4,16),Block(16,32),Block(32,64),Block(64,128)])
            self.dec=nn.ModuleList([Block(192,64),Block(96,32),Block(48,16)])
            self.head=nn.Conv2d(16,1,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        def forward(self,rgb,coarse):
            x=torch.cat([rgb,coarse],1);features=[]
            for i,b in enumerate(self.enc):
                x=b(x);features.append(x)
                if i<3:x=F.max_pool2d(x,2)
            for b,skip in zip(self.dec,reversed(features[:-1])):
                x=b(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
            return torch.logit(coarse.clamp(1e-4,1-1e-4))+self.head(x)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Refiner().to(device)
    protocol=dict(dataset='CHASE-DB1',seed=seed,split=manifest,baseline_checkpoint=str(reference/'baseline.pt'),baseline_sha256=hashlib.sha256((reference/'baseline.pt').read_bytes()).hexdigest(),model='Spatial residual U-Net:4-channel RGB/coarse;16/32/64/128 channels;3 skip-connected decoder blocks;GroupNorm/SiLU;bilinear upsampling;zero-initialised logit residual',parameters=sum(p.numel() for p in model.parameters()),loss='0.5 BCEWithLogits +0.5 soft Dice loss;whole-image reduction',optimizer='AdamW',lr=3e-4,weight_decay=1e-4,max_epochs=120,patience=25,batch=4,selection='Highest validation mean image Dice, initial baseline checkpoint included',augmentation='Aligned random flips and right-angle rotations on RGB/coarse/GT',threshold=.5,evaluation='512x512;1stHO;whole image;subject-disjoint;test8images4subjects',original_tables_modified=False,scope='New experiment using retained executable coarse predictors. Not a paired comparison to original author-supplied diffusion scores. No AE, no diffusion. Architecture and objective differ from diffusion; not a parameter-matched ablation.')
    save(job/f'protocol_{tag}.json',protocol)
    def evaluate(names):
        model.eval()
        with torch.inference_mode():
            return float(np.mean([score(model(data[n]['rgb'][None].to(device),data[n]['coarse'][None].to(device)).sigmoid().cpu()[0],data[n]['g'])['dice'] for n in names]))
    def checkpoint(epoch,val):torch.save(dict(state=model.state_dict(),epoch=epoch,validation_dice=val,protocol=protocol),job/f'refiner_{tag}.pt')
    best=evaluate(manifest['validation']);checkpoint(0,best);wait=0
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,120)
    rng=np.random.default_rng(seed)
    with (job/f'training_log_{tag}.txt').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','training_loss','validation_dice','best_validation_dice'])
        for epoch in range(1,121):
            model.train();names=rng.permutation(manifest['train']).tolist();losses=[]
            for i in range(0,len(names),4):
                batch=[]
                for n in names[i:i+4]:
                    d=data[n];rgb=d['rgb'];coarse=d['coarse'];gt=d['g'];k=int(rng.integers(4))
                    rgb,coarse,gt=[torch.rot90(t,k,[-2,-1]) for t in (rgb,coarse,gt)]
                    for dim in [-1,-2]:
                        if rng.random()<.5:rgb,coarse,gt=[t.flip(dim) for t in (rgb,coarse,gt)]
                    batch.append((rgb,coarse,gt))
                rgb,coarse,gt=[torch.stack([d[j] for d in batch]).to(device) for j in range(3)]
                logits=model(rgb,coarse);p=logits.sigmoid()
                loss=.5*F.binary_cross_entropy_with_logits(logits,gt)+.5*(1-ns['dice'](p,gt))
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
            scheduler.step();val=evaluate(manifest['validation'])
            if val>best:best=val;wait=0;checkpoint(epoch,val)
            else:wait+=1
            w.writerow([epoch,float(np.mean(losses)),val,best]);f.flush()
            if epoch%5==0:print(seed,epoch,'validation',val,'best',best,flush=True)
            if wait>=25:break
    retained=torch.load(job/f'refiner_{tag}.pt',weights_only=False);model.load_state_dict(retained['state']);model.eval()
    rows=[]
    with torch.inference_mode():
        for name in manifest['test']:
            d=data[name];p=model(d['rgb'][None].to(device),d['coarse'][None].to(device)).sigmoid().cpu()[0]
            b=score(d['coarse'],d['g']);m=score(p,d['g'])
            np.savez_compressed(job/f'{name}_prediction_{tag}.npz',baseline=d['coarse'].numpy(),refined=p.numpy(),ground_truth=d['g'].numpy())
            rows.append(dict(image=name,subject=d['subject'],seed=seed,**{'baseline_'+k:v for k,v in b.items()},**{'refined_'+k:v for k,v in m.items()},delta=m['dice']-b['dice']))
    with (job/f'per_image_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(done,dict(seed=seed,selected_epoch=retained['epoch'],epochs_completed=epoch,validation_dice=best,baseline_mean=float(np.mean([r['baseline_dice'] for r in rows])),refined_mean=float(np.mean([r['refined_dice'] for r in rows])),parameters=protocol['parameters'],completed_at=stamp()))

def report(out):
    import numpy as np
    from scipy.stats import ttest_rel
    from report_all_datasets_full_runs import exact_signed_rank
    tag=out.name[-19:];rows=[];means=[]
    for seed in SEEDS:
        job=out/f'seed_{seed}';assert (job/f'completed_{tag}.json').exists()
        rr=list(csv.DictReader((job/f'per_image_{tag}.txt').open()))
        for r in rr:
            with np.load(job/f"{r['image']}_prediction_{tag}.npz") as p:
                gt=p['ground_truth']>=.5
                for method in ['baseline','refined']:
                    pred=p[method]>=.5;tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum());tn=int((~pred&~gt).sum())
                    for key,v in [('tp',tp),('fp',fp),('fn',fn),('tn',tn)]:assert int(float(r[method+'_'+key]))==v
                    assert abs(float(r[method+'_dice'])-2*tp/max(2*tp+fp+fn,1))<1e-12
        rows.extend(rr);means.append([np.mean([float(r['baseline_dice']) for r in rr]),np.mean([float(r['refined_dice']) for r in rr])])
    assert len(rows)==40
    means=np.asarray(means);delta=means[:,1]-means[:,0]
    subjects=sorted({int(r['subject']) for r in rows})
    matrix=np.array([[np.mean([float(r['delta']) for r in rows if int(r['seed'])==s and int(r['subject'])==u]) for u in subjects] for s in SEEDS])
    rng=np.random.default_rng(20261009);ri=rng.integers(5,size=(50000,5));ci=rng.integers(4,size=(50000,4));draws=matrix[ri[:,:,None],ci[:,None,:]].mean((1,2))
    _,wp,_=exact_signed_rank(delta)
    result=dict(training_seeds=SEEDS,verified_per_image_records=40,baseline_mean=float(means[:,0].mean()),baseline_SD=float(means[:,0].std(ddof=1)),refined_mean=float(means[:,1].mean()),refined_SD=float(means[:,1].std(ddof=1)),mean_delta=float(delta.mean()),crossed_bootstrap_95CI=np.quantile(draws,[.025,.975]).tolist(),paired_t_p=float(ttest_rel(means[:,1],means[:,0]).pvalue),exact_signed_rank_p=wp,bootstrap_draws=50000,original_user_data_modified=False,scope='New CHASE-only experiment;not paired to author-supplied diffusion outputs;4 independent test subjects;exploratory uncertainty')
    with (out/f'all_per_image_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/f'verified_results_{tag}.json',result)
    (out/f'experiment_report_{tag}.txt').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save(out/'status.json',dict(stage='completed',completed_jobs=5,verified_records=40,updated_at=stamp()))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--seed',type=int);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True);tag=out.name[-19:]
    if a.seed:worker(out,a.seed)
    else:
        (out/f'source_spatial_refiner_{tag}.py').write_bytes(Path(__file__).read_bytes())
        for i,s in enumerate(SEEDS):
            save(out/'status.json',dict(stage='training',job=i+1,total=5,seed=s,updated_at=stamp()))
            with (out/f'seed{s}_{tag}.log').open('w') as f:
                p=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(out),'--seed',str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode:save(out/'status.json',dict(stage='failed',seed=s));raise RuntimeError('Worker failed')
        report(out)
