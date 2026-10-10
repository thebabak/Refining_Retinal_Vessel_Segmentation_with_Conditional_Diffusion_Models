"""Matched deterministic control of the actual local diffusion reproduction."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = HERE/'all_datasets_five_full_runs_2026-10-08_15-38-24'
SEEDS = [42,123,2026,3000,3001]


def stamp():
    return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')


def save_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    for attempt in range(30):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==29: raise
            time.sleep(.1)


def worker(dataset, seed, out):
    import numpy as np
    import torch
    from torch import nn
    from run_all_datasets_five_full_runs import compose

    tag=out.name[-19:]
    job=out/dataset/f'train_seed_{seed}'
    job.mkdir(parents=True,exist_ok=True)
    completion=job/f'completed_{tag}.json'
    if completion.exists():return
    ref=REFERENCE/dataset/f'train_seed_{seed}'
    assert (ref/'completed.json').exists()
    source,prefix,_=compose(dataset,seed,job)
    # Avoid writing the prefix's preliminary protocol before the matched protocol.
    prefix='\n'.join(line for line in prefix.splitlines() if not line.startswith("(OUT/'protocol.json')") and not line.startswith("(OUT/'file_provenance.json')"))
    ns=dict(__file__=str(source),__name__='deterministic_control',experiment_output=job)
    exec(compile(prefix,str(source),'exec'),ns)
    device=ns['DEVICE'];data=ns['data'];manifest=ns['manifest'];batches=ns['batches'];dice=ns['dice'];score=ns['score']
    source_protocol=json.loads((ref/'protocol.json').read_text())
    assert source_protocol['split']==manifest
    baseline=ns['LNet_RA'](in_channels=1).to(device)
    baseline.load_state_dict(torch.load(ref/'baseline.pt',weights_only=False)['state'])
    baseline.eval();baseline.requires_grad_(False)
    with torch.no_grad():
        for sample in data.values():sample['coarse']=baseline(sample['x'][None].to(device)).cpu()[0]
    # CPU initialisation order recreates the diffusion model's initialisation:
    # baseline -> autoencoder -> image encoder -> latent U-Net -> time MLP.
    ae=ns['MaskAutoencoder']().to(device)
    ae.load_state_dict(torch.load(ref/'autoencoder.pt',weights_only=True));ae.eval();ae.requires_grad_(False)
    ie=ns['ImageEncoder']().to(device);un=ns['DiffusionUNet']().to(device)
    tm=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128)).to(device)
    torch.cuda.manual_seed_all(seed)
    params=list(ie.parameters())+list(un.parameters())+list(tm.parameters())
    count=sum(p.numel() for p in params)
    diffuser=torch.load(ref/'refiner.pt',weights_only=True)
    reference_count=sum(t.numel() for key in ['ie','un','time'] for t in diffuser[key].values())
    assert count==reference_count
    del diffuser
    protocol=dict(dataset=dataset,training_seed=seed,split=manifest,reference=str(ref),
                  frozen_baseline_sha256=hashlib.sha256((ref/'baseline.pt').read_bytes()).hexdigest(),
                  frozen_autoencoder_sha256=hashlib.sha256((ref/'autoencoder.pt').read_bytes()).hexdigest(),
                  model='Same image encoder, latent U-Net and time MLP as diffusion; direct residual latent correction',
                  forward='z_ref=encode(coarse)+UNet(encode(coarse),emb(0),ImageEncoder(RGB)+timeMLP(emb(0)))',
                  objective='0.5 * (1 - decoded-mask Dice); same Dice reduction as reference',
                  noise_loss=False,DDIM=False,classifier_free_guidance=False,
                  training_condition_dropout=.1,epochs=100,optimizer='AdamW',lr=.0001,weight_decay=.001,
                  scheduler='CosineAnnealingLR T_max100',batch_size=4,grad_clip=1.,augmentation=False,
                  selection='Fixed epoch100, not test-selected',learned_parameters=count,
                  diffusion_learned_parameters=reference_count,precision='FP32',torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                  evaluation_protocol=source_protocol,
                  limitations=['Controlled direct-vs-diffusion formulation comparison; both objective and inference necessarily differ',
                               'Total parameter counts matched; fixed-time conditioning has lower effective time-dependent capacity',
                               'Conditional on available local checkpoints; not the author-supplied positive-results table'])
    save_json(job/f'protocol_{tag}.json',protocol)

    def embedding(t):
        frequencies=torch.exp(-math.log(10000)*torch.arange(32,device=device)/32)
        v=t[:,None]*frequencies
        return torch.cat([v.sin(),v.cos()],1)

    def forward(coarse, rgb, dropout=False):
        with torch.no_grad():z=ae.encode(coarse)
        e=embedding(torch.zeros(len(coarse),device=device))
        c=ie(rgb)
        if dropout:c=c*(torch.rand(len(coarse),1,device=device)>.1)
        return ae.decode(z+un(z,e,c+tm(e)))

    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
    with (job/f'training_log_{tag}.txt').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','mean_training_loss']);writer.writeheader()
        for epoch in range(100):
            ie.train();un.train();tm.train();losses=[]
            for b,names in batches('train'):
                coarse=torch.stack([data[name]['coarse'] for name in names]).to(device)
                prediction=forward(coarse,b['rgb'],True)
                loss=.5*(1-dice(prediction,b['g']))
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(float(loss.item()))
            schedule.step()
            writer.writerow(dict(epoch=epoch+1,mean_training_loss=float(np.mean(losses))));stream.flush()
            if (epoch+1)%10==0:print(dataset,seed,'epoch',epoch+1,'loss',float(np.mean(losses)),flush=True)
    checkpoint=job/f'deterministic_refiner_{tag}.pt'
    torch.save(dict(ie=ie.state_dict(),un=un.state_dict(),time=tm.state_dict(),epoch=100,protocol=protocol),checkpoint)
    ie.eval();un.eval();tm.eval()
    diffusion_rows=list(csv.DictReader((ref/'paired_per_image_metrics.txt').open(newline='')))
    diffusion_rows=[r for r in diffusion_rows if r['steps']=='50']
    records=[]

    def metrics(pred,sample):
        return score(pred,sample['g']) if dataset=='CHASE' else score(pred,sample['g'],sample['fov'])

    with torch.inference_mode():
        for image in manifest['test']:
            sample=data[image];coarse=sample['coarse'][None].to(device);rgb=sample['rgb'][None].to(device)
            prediction=forward(coarse,rgb).cpu()[0]
            repeated=forward(coarse,rgb).cpu()[0]
            assert torch.equal(prediction,repeated),'Inference is not deterministic'
            reconstruction=ae.decode(ae.encode(coarse)).cpu()[0]
            base=metrics(sample['coarse'],sample);det=metrics(prediction,sample);aem=metrics(reconstruction,sample)
            diff=[r for r in diffusion_rows if r['image']==image];assert len(diff)==5
            assert all(abs(float(r['baseline_dice'])-base['dice'])<1e-12 for r in diff)
            with np.load(ref/f'{image}_reference.npz') as original:
                assert np.array_equal(original['ground_truth'],sample['g'].numpy())
                assert np.array_equal(original['coarse'],sample['coarse'].numpy())
            np.savez_compressed(job/f'{image}_predictions_{tag}.npz',deterministic=prediction.numpy(),
                                autoencoder_reconstruction=reconstruction.numpy())
            records.append(dict(dataset=dataset,image=image,subject=sample.get('subject',''),
                                pathology=image.split('_')[-1] if dataset=='HRF' else '',seed=seed,
                                **{'baseline_'+k:v for k,v in base.items()},
                                **{'autoencoder_'+k:v for k,v in aem.items()},
                                **{'deterministic_'+k:v for k,v in det.items()},
                                diffusion_mean_dice=float(np.mean([float(r['refined_dice']) for r in diff])),
                                diffusion_sampling_realisations=5,reference_job=str(ref)))
    with (job/f'paired_per_image_{tag}.txt').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    save_json(completion,dict(dataset=dataset,seed=seed,images=len(records),parameters=count,
                             means={k:float(np.mean([r[k] for r in records])) for k in ['baseline_dice','autoencoder_dice','deterministic_dice','diffusion_mean_dice']},
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),completed_at=stamp()))
    print('COMPLETED',dataset,seed,flush=True)


def controller(out):
    out.mkdir(parents=True,exist_ok=True);tag=out.name[-19:]
    (out/f'source_run_deterministic_refiner_control_{tag}.py').write_bytes(Path(__file__).read_bytes())
    save_json(out/f'experiment_plan_{tag}.json',dict(reference=str(REFERENCE),datasets=['CHASE','DRIVE','HRF'],
              training_seeds=SEEDS,epochs=100,backbone_autoencoder='Frozen matched per-training-seed checkpoints',
              controls=['Coarse baseline','AE reconstruction','Direct deterministic learned refiner','Existing diffusion at50steps averaged over5noise seeds'],
              timestamp=tag,manuscript_modified=False,author_supplied_tables_modified=False))
    for i,(dataset,seed) in enumerate((d,s) for s in SEEDS for d in ['CHASE','DRIVE','HRF']):
        save_json(out/'status.json',dict(stage='running',job=i+1,total=15,dataset=dataset,seed=seed,updated_at=stamp()))
        with (out/f'{dataset}_seed{seed}_{tag}.log').open('a') as log:
            p=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(out),'--dataset',dataset,'--seed',str(seed)],
                             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:
            save_json(out/'status.json',dict(stage='failed',dataset=dataset,seed=seed,returncode=p.returncode));raise RuntimeError('Worker failed')
    save_json(out/'status.json',dict(stage='completed',jobs=15,updated_at=stamp()))
    subprocess.run([sys.executable,str(HERE/'report_deterministic_refiner_control.py'),'--output',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['CHASE','DRIVE','HRF']);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.dataset:worker(args.dataset,args.seed,args.output.resolve())
    else:controller(args.output.resolve())
