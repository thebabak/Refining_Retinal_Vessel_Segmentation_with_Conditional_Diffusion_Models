"""New spatial RGB+coarse U-Net experiment; never alters supplied score tables."""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SEEDS=[42,123,2026,3000,3001]
REF=HERE/'all_datasets_five_full_runs_2026-10-08_15-38-24'

def stamp():return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
def save(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def worker(out,seed,dataset):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_all_datasets_five_full_runs import compose
    tag=out.name[-19:];job=out/dataset/f'seed_{seed}';job.mkdir(exist_ok=True,parents=True)
    done=job/f'completed_{tag}.json'
    if done.exists():return
    source,prefix,_=compose(dataset,seed,job)
    prefix='\n'.join(l for l in prefix.splitlines() if not l.startswith("(OUT/'protocol.json')") and not l.startswith("(OUT/'file_provenance.json')"))
    ns=dict(__file__=str(source),__name__='spatial_control',experiment_output=job)
    exec(compile(prefix,str(source),'exec'),ns)
    data=ns['data'];manifest=ns['manifest'];device=ns['DEVICE'];score=lambda p,g,fov: ns['score'](p,g,fov)
    reference=REF/dataset/f'train_seed_{seed}';baseline=ns['LNet_RA'](in_channels=1).to(device)
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
    protocol=dict(dataset=dataset,seed=seed,split=manifest,baseline_checkpoint=str(reference/'baseline.pt'),baseline_sha256=hashlib.sha256((reference/'baseline.pt').read_bytes()).hexdigest(),model='Spatial residual U-Net:4-channel RGB/coarse;16/32/64/128 channels;3 skip-connected decoder blocks;GroupNorm/SiLU;bilinear upsampling;zero-initialised logit residual',parameters=sum(p.numel() for p in model.parameters()),loss='0.5 BCEWithLogits +0.5 soft Dice loss;whole-image reduction',optimizer='AdamW',lr=3e-4,weight_decay=1e-4,max_epochs=120,patience=25,batch=4,selection='Highest validation mean image Dice, initial baseline checkpoint included',augmentation='Aligned random flips and right-angle rotations on RGB/coarse/GT',threshold=.5,evaluation=json.loads((reference/'protocol.json').read_text()),original_tables_modified=False,scope='New experiment using retained executable coarse predictors. Not a paired comparison to original author-supplied diffusion scores. No AE, no diffusion. Architecture and objective differ from diffusion; not a parameter-matched ablation.')
    save(job/f'protocol_{tag}.json',protocol)
    def evaluate(names):
        model.eval()
        with torch.inference_mode():
            return float(np.mean([score(model(data[n]['rgb'][None].to(device),data[n]['coarse'][None].to(device)).sigmoid().cpu()[0],data[n]['g'],data[n]['fov'])['dice'] for n in names]))
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
            b=score(d['coarse'],d['g'],d['fov']);m=score(p,d['g'],d['fov'])
            np.savez_compressed(job/f'{name}_prediction_{tag}.npz',baseline=d['coarse'].numpy(),refined=p.numpy(),ground_truth=d['g'].numpy(),fov=d['fov'].numpy())
            rows.append(dict(dataset=dataset,image=name,subject=d.get('subject',''),pathology=name.split('_')[-1] if dataset=='HRF' else '',seed=seed,**{'baseline_'+k:v for k,v in b.items()},**{'refined_'+k:v for k,v in m.items()},delta=m['dice']-b['dice']))
    with (job/f'per_image_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(done,dict(dataset=dataset,seed=seed,selected_epoch=retained['epoch'],epochs_completed=epoch,validation_dice=best,baseline_mean=float(np.mean([r['baseline_dice'] for r in rows])),refined_mean=float(np.mean([r['refined_dice'] for r in rows])),parameters=protocol['parameters'],completed_at=stamp()))

def report(out):
    import numpy as np
    from scipy.stats import ttest_rel
    from report_all_datasets_full_runs import exact_signed_rank
    import zipfile
    tag=out.name[-19:];allrows=[];results=[]
    for dataset,nimages in [('DRIVE',20),('HRF',12)]:
        matrix=[];means=[];history=[];image_names=None
        for seed in SEEDS:
            job=out/dataset/f'seed_{seed}';completion=json.loads((job/f'completed_{tag}.json').read_text());history.append(completion)
            rr=sorted(csv.DictReader((job/f'per_image_{tag}.txt').open()),key=lambda r:r['image'])
            assert len(rr)==nimages
            if image_names is None:image_names=[r['image'] for r in rr]
            assert image_names==[r['image'] for r in rr]
            for r in rr:
                with np.load(job/f"{r['image']}_prediction_{tag}.npz") as p:
                    gt=p['ground_truth']>=.5;fov=p['fov'].astype(bool)
                    for method in ['baseline','refined']:
                        pred=p[method]>=.5;tp=int((pred & gt & fov).sum());fp=int((pred & ~gt & fov).sum());fn=int((~pred & gt & fov).sum());tn=int((~pred & ~gt & fov).sum())
                        expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/(tp+fp+fn+tn))
                        for k,v in expected.items():assert abs(float(r[method+'_'+k])-v)<1e-12
                assert abs(float(r['delta'])-float(r['refined_dice'])+float(r['baseline_dice']))<1e-12
            allrows.extend(rr);matrix.append([float(r['delta']) for r in rr]);means.append([np.mean([float(r['baseline_dice']) for r in rr]),np.mean([float(r['refined_dice']) for r in rr])])
        matrix=np.asarray(matrix);means=np.asarray(means);delta=means[:,1]-means[:,0]
        rng=np.random.default_rng(20261009);ri=rng.integers(5,size=(50000,5))
        if dataset=='HRF':
            strata=[[i for i,n in enumerate(image_names) if n.split('_')[-1]==p] for p in ['h','dr','g']]
            assert all(len(g)==4 for g in strata)
            ci=np.concatenate([rng.choice(g,size=(50000,len(g)),replace=True) for g in strata],axis=1)
        else:ci=rng.integers(nimages,size=(50000,nimages))
        draws=matrix[ri[:,:,None],ci[:,None,:]].mean((1,2));_,wp,_=exact_signed_rank(delta)
        results.append(dict(dataset=dataset,training_seeds=SEEDS,verified_records=5*nimages,baseline_mean=float(means[:,0].mean()),baseline_SD=float(means[:,0].std(ddof=1)),refined_mean=float(means[:,1].mean()),refined_SD=float(means[:,1].std(ddof=1)),mean_delta=float(delta.mean()),crossed_bootstrap_95CI=np.quantile(draws,[.025,.975]).tolist(),paired_t_p=float(ttest_rel(means[:,1],means[:,0]).pvalue),exact_signed_rank_p=wp,bootstrap_draws=50000,training_history=history,identity_selections=sum(c['selected_epoch']==0 for c in history),resampling='Training seeds and images;HRF pathology-stratified;patient independence unverified',scope='New retained-checkpoint experiment;not paired to supplied original diffusion outputs'))
    for key in ['paired_t_p','exact_signed_rank_p']:
        highest=0
        for i,r in enumerate(sorted(results,key=lambda r:r[key])):highest=max(highest,(2-i)*r[key]);r['holm_'+key]=min(1,highest)
    assert len(allrows)==160
    with (out/f'all_per_image_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    save(out/f'verified_results_{tag}.json',results)
    (out/f'experiment_report_{tag}.txt').write_text(json.dumps(results,indent=2)+'\nOriginal user tables and manuscript unchanged. New paired coarse-to-refiner comparison only.\n')
    with zipfile.ZipFile(out/f'spatial_refiner_complete_data_{tag}.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.txt','.json','.log','.py','.pt','.npz'] and tag in p.name:z.write(p,arcname=str(p.relative_to(out)))
    with zipfile.ZipFile(out/f'spatial_refiner_complete_data_{tag}.zip') as z:assert z.testzip() is None
    save(out/'status.json',dict(stage='completed',completed_jobs=10,verified_records=160,updated_at=stamp()))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--seed',type=int);ap.add_argument('--dataset',choices=['DRIVE','HRF']);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True);tag=out.name[-19:]
    if a.seed:worker(out,a.seed,a.dataset)
    else:
        (out/f'source_spatial_refiner_{tag}.py').write_bytes(Path(__file__).read_bytes())
        for i,(d,s) in enumerate((d,s) for s in SEEDS for d in ['DRIVE','HRF']):
            save(out/'status.json',dict(stage='training',job=i+1,total=10,seed=s,dataset=d,updated_at=stamp()))
            with (out/f'{d}_seed{s}_{tag}.log').open('w') as f:
                p=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(out),'--dataset',d,'--seed',str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode:save(out/'status.json',dict(stage='failed',dataset=d,seed=s));raise RuntimeError('Worker failed')
        report(out)
