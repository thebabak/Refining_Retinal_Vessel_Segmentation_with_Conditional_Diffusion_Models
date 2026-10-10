"""Controlled NEW component ablations, separate from original supplied scores."""
import argparse,csv,hashlib,json,math,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REFERENCE=HERE/'all_datasets_five_full_runs_2026-10-08_15-38-24'
SEEDS=[42,123,2026,3000,3001]
VARIANTS=['global_frozen','deterministic','target_diffusion','no_image','spatial_film','global_additive','ae_finetuned']
def stamp():return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for i in range(30):
        try:tmp.replace(path);return
        except PermissionError:
            if i==29:raise
            time.sleep(.1)

def worker_impl(out,dataset,seed):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_all_datasets_five_full_runs import compose
    tag=out.name[-19:];job=out/dataset/f'seed_{seed}';job.mkdir(parents=True,exist_ok=True)
    done=job/f'completed_{tag}.json'
    if done.exists():return
    source,prefix,_=compose(dataset,seed,job)
    prefix='\n'.join(l for l in prefix.splitlines() if not l.startswith("(OUT/'protocol.json')") and not l.startswith("(OUT/'file_provenance.json')"))
    ns=dict(__file__=str(source),__name__='component_worker',experiment_output=job)
    exec(compile(prefix,str(source),'exec'),ns)
    data=ns['data'];manifest=ns['manifest'];device=ns['DEVICE'];dice=ns['dice'];batches=ns['batches']
    reference=REFERENCE/dataset/f'train_seed_{seed}'
    assert json.loads((reference/'protocol.json').read_text())['split']==manifest
    baseline=ns['LNet_RA'](in_channels=1).to(device)
    baseline.load_state_dict(torch.load(reference/'baseline.pt',weights_only=False)['state']);baseline.eval();baseline.requires_grad_(False)
    with torch.no_grad():
        for d in data.values():d['coarse']=baseline(d['x'][None].to(device)).cpu()[0]
    ae0=ns['MaskAutoencoder']().to(device);ae0.load_state_dict(torch.load(reference/'autoencoder.pt',weights_only=True))
    ie0=ns['ImageEncoder']().to(device);un0=ns['DiffusionUNet']().to(device)
    tm0=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(device)
    initial={key:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for key,m in [('ae',ae0),('ie',ie0),('un',un0),('time',tm0)]}
    del ae0,ie0,un0,tm0,baseline
    class SpatialEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.conv1=nn.Conv2d(3,32,3,stride=2,padding=1);self.conv2=nn.Conv2d(32,64,3,stride=2,padding=1);self.projection=nn.Conv2d(64,128,1)
        def forward(self,x):return self.projection(F.relu(self.conv2(F.relu(self.conv1(x)))))
        def initialise(self):
            self.conv1.load_state_dict({k:initial['ie']['net.0.'+k] for k in ['weight','bias']});self.conv2.load_state_dict({k:initial['ie']['net.2.'+k] for k in ['weight','bias']})
            with torch.no_grad():self.projection.weight.copy_(initial['ie']['net.6.weight'][:,:,None,None]);self.projection.bias.copy_(initial['ie']['net.6.bias'])
    class ConditionUNet(ns['DiffusionUNet']):
        def __init__(self,additive=False):super().__init__();self.additive=additive
        def block(self,block,x,c):
            h=block.conv1(block.act(block.norm1(x)))
            if c.ndim==4:
                c=F.interpolate(c,size=h.shape[-2:],mode='bilinear',align_corners=False)
                g,b=F.conv2d(c,block.film.weight[:,:,None,None],block.film.bias).chunk(2,1)
            else:g,b=[v[:,:,None,None] for v in block.film(c).chunk(2,1)]
            h=h+(g+b)/math.sqrt(2) if self.additive else h*(1+g)+b
            return x+block.act(block.conv2(h))
        def forward(self,z,t,c):
            h=self.block(self.res1,self.in_conv(z),c);h=self.block(self.res2,self.down(h),c)
            return self.out_conv(self.up(h))
    t=torch.arange(1001,device=device)/1000;a=torch.cos((t+.008)/1.008*math.pi/2)**2;a=a/a[0]
    alpha=torch.cumprod(1-(1-a[1:]/a[:-1]).clamp(max=.999),0)
    def emb(t):
        v=t[:,None]*torch.exp(-math.log(10000)*torch.arange(32,device=device)/32)
        return torch.cat([v.sin(),v.cos()],1)
    def metrics(p,d):return ns['score'](p,d['g']) if dataset=='CHASE' else ns['score'](p,d['g'],d['fov'])
    for name in manifest['test']:
        d=data[name]
        with np.load(reference/f'{name}_reference.npz') as src:
            assert np.array_equal(src['coarse'],d['coarse'].numpy());assert np.array_equal(src['ground_truth'],d['g'].numpy())
        np.savez_compressed(job/f'{name}_reference_{tag}.npz',baseline=d['coarse'].numpy(),ground_truth=d['g'].numpy(),fov=d['fov'].numpy() if 'fov' in d else np.ones_like(d['g'].numpy(),dtype=bool))
    for variant in VARIANTS:
        dest=job/variant;dest.mkdir(exist_ok=True)
        if (dest/f'completed_{tag}.json').exists():continue
        save(job/'status.json',dict(stage='training',variant=variant,updated_at=stamp()))
        ae=ns['MaskAutoencoder']().to(device);ae.load_state_dict(initial['ae']);ae.requires_grad_(variant=='ae_finetuned')
        ie=SpatialEncoder().to(device) if variant=='spatial_film' else ns['ImageEncoder']().to(device)
        if variant=='spatial_film':ie.initialise()
        else:ie.load_state_dict(initial['ie'])
        un=ConditionUNet(variant=='global_additive').to(device);un.load_state_dict(initial['un'])
        tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(device);tm.load_state_dict(initial['time'])
        params=list(un.parameters())+list(tm.parameters())
        if variant!='no_image':params+=list(ie.parameters())
        else:ie.requires_grad_(False)
        if variant=='ae_finetuned':params+=list(ae.parameters())
        count=sum(p.numel() for p in params)
        protocol=dict(dataset=dataset,training_seed=seed,variant=variant,split=manifest,baseline_checkpoint=str(reference/'baseline.pt'),baseline_sha256=hashlib.sha256((reference/'baseline.pt').read_bytes()).hexdigest(),initial_autoencoder_sha256=hashlib.sha256((reference/'autoencoder.pt').read_bytes()).hexdigest(),updated_parameters=count,epochs=100,selection='fixed epoch100;no test tuning',optimizer='AdamW1e-4;weight_decay.001;cosine100;batch4;gradient_clip1',augmentation=False,loss='Noise MSE +0.5 decoded-mask Dice;deterministic uses only0.5decoded-mask Dice',condition_dropout=.1,conditioning='128global vector' if variant!='spatial_film' else '128-channel128x128spatial grid',sampling_seeds=SEEDS,steps=50,inference='DDIM eta0;CFG1.5;coarse warm-start t99;no-image CFG branches identical;deterministic one pass',target_definition='target_diffusion trains on encoded GT;other diffusion variants encoded coarse',target_inference='Both coarse-warm start99 and pureGaussian start999 reported;same learned checkpoint',fine_tuning='All AE encoder/decoder parameters updated jointly only forae_finetuned',scope='New controlled retained-checkpoint experiment,not original supplied results',limitations=['In-sample coarse training maps','Small fixed splits;previously examined test cohorts','Joint AE tuning can change latent geometry','Additive injection uses both existing FiLM projections as(g+b)/sqrt2:matched parameter count but redundant additive parameterisation','No-image branch has fewer updated parameters','Target-warm and target-full inference differ ininitialisation;full comparison is notsinglefactor','Fixed checkpoint selection differs from separately reported spatial RGB+coarse U-Net experiment'])
        save(dest/f'protocol_{tag}.json',protocol)
        def denoise(z,ti,c):
            temporal=tm(emb(ti));return un(z,emb(ti),c+(temporal if c.ndim==2 else temporal[:,:,None,None]))
        def condition(rgb):return torch.zeros(len(rgb),128,device=device) if variant=='no_image' else ie(rgb)
        torch.manual_seed(seed+98765);torch.cuda.manual_seed_all(seed+98765)
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
        with (dest/f'training_log_{tag}.txt').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['epoch','loss','noise_loss','dice_loss'])
            for epoch in range(100):
                ie.train();un.train();tm.train();ae.train(variant=='ae_finetuned');values=[]
                for b,names in batches('train'):
                    coarse=torch.stack([data[n]['coarse'] for n in names]).to(device)
                    source_mask=b['g'] if variant=='target_diffusion' else coarse
                    if variant=='ae_finetuned':z=ae.encode(source_mask)
                    else:
                        with torch.no_grad():z=ae.encode(source_mask)
                    ti=torch.randint(0,1000,(len(names),),device=device);at=alpha[ti,None,None,None];noise=torch.randn_like(z)
                    drop=(torch.rand(len(names),1,device=device)>.1);c=condition(b['rgb']);c=c*(drop if c.ndim==2 else drop[:,:,None,None])
                    if variant=='deterministic':
                        prediction=ae.decode(z+denoise(z,torch.zeros_like(ti),c));nl=torch.zeros((),device=device)
                    else:
                        zt=at.sqrt()*z+(1-at).sqrt()*noise;eps=denoise(zt,ti,c);nl=F.mse_loss(eps,noise);prediction=ae.decode((zt-(1-at).sqrt()*eps)/at.sqrt())
                    dl=1-dice(prediction,b['g']);loss=nl+.5*dl
                    assert torch.isfinite(loss).item(),'Non-finite training loss'
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step();values.append([loss.item(),nl.item(),dl.item()])
                scheduler.step();w.writerow([epoch+1,*np.mean(values,axis=0)]);f.flush()
                if (epoch+1)%20==0:print(dataset,seed,variant,epoch+1,np.mean(values,axis=0).tolist(),flush=True)
        checkpoint=dest/f'model_{tag}.pt';torch.save(dict(ae=ae.state_dict(),ie=ie.state_dict(),un=un.state_dict(),time=tm.state_dict(),epoch=100,protocol=protocol),checkpoint)
        ae.eval();ie.eval();un.eval();tm.eval()
        save(job/'status.json',dict(stage='evaluation',variant=variant,updated_at=stamp()))
        records=[]
        with torch.inference_mode():
            for name in manifest['test']:
                d=data[name];coarse=d['coarse'][None].to(device);z0=ae.encode(coarse);c=condition(d['rgb'][None].to(device));base=metrics(d['coarse'],d)
                modes=['target_warm','target_full'] if variant=='target_diffusion' else [variant]
                for mode in modes:
                    outputs=[]
                    sampling_seeds=[None] if variant=='deterministic' else SEEDS
                    for sampling_seed in sampling_seeds:
                        if variant=='deterministic':prediction=ae.decode(z0+denoise(z0,torch.zeros(1,device=device),c))
                        else:
                            gen=torch.Generator(device=device).manual_seed(sampling_seed);noise=torch.randn(z0.shape,device=device,generator=gen)
                            start=999 if mode=='target_full' else 99
                            z=noise if mode=='target_full' else alpha[start].sqrt()*z0+(1-alpha[start]).sqrt()*noise
                            ts=torch.linspace(start,0,50,device=device).long()
                            for j,ti in enumerate(ts):
                                e0=denoise(z,ti[None],torch.zeros_like(c));e1=denoise(z,ti[None],c);eps=e0+1.5*(e1-e0)
                                estimate=(z-(1-alpha[ti]).sqrt()*eps)/alpha[ti].sqrt();prev=alpha[ts[j+1]] if j+1<len(ts) else torch.tensor(1.,device=device)
                                z=prev.sqrt()*estimate+(1-prev).sqrt()*eps
                            prediction=ae.decode(z)
                        array=prediction.cpu()[0].numpy();assert np.isfinite(array).all();outputs.append(array)
                        mm=metrics(torch.from_numpy(array),d)
                        records.append(dict(dataset=dataset,variant=mode,image=name,subject=d.get('subject',''),pathology=name.split('_')[-1] if dataset=='HRF' else'',training_seed=seed,sampling_seed=sampling_seed if sampling_seed is not None else'',steps=1 if variant=='deterministic' else 50,updated_parameters=count,**{'baseline_'+k:v for k,v in base.items()},**{'refined_'+k:v for k,v in mm.items()},delta=mm['dice']-base['dice']))
                    np.savez_compressed(dest/f'{name}_{mode}_predictions_{tag}.npz',probabilities=np.stack(outputs),sampling_seeds=np.asarray(sampling_seeds if variant!='deterministic' else[-1]))
        with (dest/f'per_image_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        expected=len(manifest['test'])*(1 if variant=='deterministic' else 10 if variant=='target_diffusion' else 5);assert len(records)==expected
        save(dest/f'completed_{tag}.json',dict(variant=variant,records=len(records),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),completed_at=stamp(),updated_parameters=count))
        print('COMPLETED',dataset,seed,variant,len(records),flush=True)
        del ae,ie,un,tm,opt,scheduler,params;torch.cuda.empty_cache()
    save(done,dict(dataset=dataset,seed=seed,trained_variants=7,evaluation_variants=8,completed_at=stamp()))

def worker(out,dataset,seed):
    job=out/dataset/f'seed_{seed}';job.mkdir(parents=True,exist_ok=True)
    done=job/f'completed_{out.name[-19:]}.json';lock=job/'worker.lock'
    if done.exists():return
    while True:
        try:
            with lock.open('x') as f:f.write(str(__import__('os').getpid()))
            break
        except FileExistsError:
            if done.exists():return
            time.sleep(1)
    try:
        if not done.exists():worker_impl(out,dataset,seed)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--dataset',choices=['CHASE','DRIVE','HRF']);ap.add_argument('--seed',type=int);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);tag=out.name[-19:]
    if a.dataset:worker(out,a.dataset,a.seed)
    else:
        (out/f'source_component_ablations_{tag}.py').write_bytes(Path(__file__).read_bytes())
        save(out/f'experiment_plan_{tag}.json',dict(datasets=['CHASE','DRIVE','HRF'],training_seeds=SEEDS,sampling_seeds=SEEDS,variants=VARIANTS,new_training_jobs=105,evaluation_variants=8,expected_records=7200,baseline_ae_reference=str(REFERENCE),epochs=100,steps=50,original_tables_unchanged=True,manuscript_unchanged=True))
        for i,(d,s) in enumerate((d,s) for s in SEEDS for d in ['CHASE','DRIVE','HRF']):
            save(out/'status.json',dict(stage='running',batch=i+1,total_batches=15,dataset=d,seed=s,updated_at=stamp()))
            with (out/f'{d}_seed{s}_{tag}.log').open('a') as f:p=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(out),'--dataset',d,'--seed',str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode:save(out/'status.json',dict(stage='failed',dataset=d,seed=s));raise RuntimeError('Ablation worker failed')
        save(out/'status.json',dict(stage='training_complete',batches=15,trained_models=105,updated_at=stamp()))
        subprocess.run([sys.executable,str(HERE/'report_component_ablations.py'),'--output',str(out)],cwd=ROOT,check=True)

