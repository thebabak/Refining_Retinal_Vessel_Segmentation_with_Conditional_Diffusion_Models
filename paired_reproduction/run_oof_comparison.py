"""New matched OOF experiment; never substitutes for original author tables."""
import argparse, csv, hashlib, json, subprocess, sys, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 123, 2026, 3000, 3001]
CONTROL = ROOT/'revise/component_ablations_2026-10-09_12-40-32'

def stamp():
    return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def worker(out, dataset, seed):
    import numpy as np
    import torch
    import run_component_ablations as component
    from run_all_datasets_five_full_runs import compose
    tag = out.name[-19:]
    job = out/dataset/f'seed_{seed}'
    job.mkdir(parents=True, exist_ok=True)
    if (job/f'oof_completed_{tag}.json').exists():
        return
    source, prefix, _ = compose(dataset, seed, job)
    prefix = '\n'.join(l for l in prefix.splitlines() if not l.startswith("(OUT/'protocol.json')") and not l.startswith("(OUT/'file_provenance.json')"))
    ns = dict(__file__=str(source), __name__='oof_worker', experiment_output=job)
    exec(compile(prefix, str(source), 'exec'), ns)
    data, manifest = ns['data'], ns['manifest']
    original_manifest = {k:list(v) for k,v in manifest.items()}
    rng = np.random.default_rng(20261009)
    if dataset == 'CHASE':
        units = sorted({data[n]['subject'] for n in manifest['train']})
        groups = np.array_split(rng.permutation(units), 4)
        folds = [[n for n in manifest['train'] if data[n]['subject'] in g] for g in groups]
    elif dataset == 'HRF':
        folds = [[] for _ in range(3)]
        for category in ['h','dr','g']:
            names = sorted(n for n in manifest['train'] if n.split('_')[-1] == category)
            assert len(names) == 9
            for i,g in enumerate(np.array_split(rng.permutation(names), 3)):
                folds[i].extend(g.tolist())
    else:
        folds = [g.tolist() for g in np.array_split(rng.permutation(sorted(manifest['train'])), 4)]
    assert sorted(n for f in folds for n in f) == sorted(manifest['train'])
    protocol = dict(dataset=dataset, training_seed=seed, split=original_manifest, folds=folds,
        fold_seed=20261009, scope='New matched in-sample versus out-of-fold comparison; original author tables unchanged',
        held_out_policy='Held-out fold is excluded from baseline fitting and checkpoint selection. Original validation partition is used for fold checkpoint selection; original test is excluded.',
        selection='Fold baselines: maximum validation mean image Dice, max1000epochs patience20. Refiner: fixed epoch100.',
        comparison='Only refiner training coarse maps differ. Full-training baseline test maps, initial AE, refiner initial weights, optimizer, loss, training RNG, image split and sampling noise are held fixed.',
        autoencoder='Same frozen retained AE in both arms; DRIVE AE was pretrained on in-sample coarse maps, so this comparison does not remove that additional possible shift.',
        validation_maps='Refiner fixed-epoch fitting uses training partition only; validation/test maps remain the retained full-training baseline outputs.',
        limitations=['Fold models have fewer training examples; OOF effect combines reduced training size and prediction generalisation.', 'DRIVE/HRF patient independence unverified.', 'Fixed, previously examined test cohorts.', 'Not recovery of original supplied diffusion scores.'])
    save(job/f'oof_protocol_{tag}.json', protocol)
    raw = source.read_text(encoding='utf-8-sig')
    start = raw.index('model=LNet_RA(in_channels=1).to(DEVICE)')
    end = raw.index('model.load_state_dict(saved[\'state\']); model.eval()', start) + len("model.load_state_dict(saved['state']); model.eval()")
    baseline_training = raw[start:end]
    predictions = {}
    for i, held in enumerate(folds):
        fold = job/f'fold_{i+1}'
        fold.mkdir(exist_ok=True)
        held_file = fold/f'held_out_predictions_{tag}.npz'
        if held_file.exists():
            with np.load(held_file) as a:
                predictions.update({n:a[n].copy() for n in a.files})
            continue
        manifest['train'] = [n for n in original_manifest['train'] if n not in held]
        assert not set(held)&set(manifest['train']) and not set(held)&set(manifest['validation'])
        if dataset == 'CHASE':
            assert not {data[n]['subject'] for n in held}&{data[n]['subject'] for n in manifest['train']+manifest['validation']+manifest['test']}
        ns['OUT'] = fold
        ns['config'] = dict(protocol, fold=i+1, fit_images=manifest['train'].copy(), held_out_images=held, validation_images=manifest['validation'].copy())
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ns['random'].seed(seed); np.random.seed(seed)
        save(job/'status.json',dict(stage='fold_baseline_training',fold=i+1,total_folds=len(folds),updated_at=stamp()))
        exec(compile(baseline_training, str(source), 'exec'), ns)
        with torch.inference_mode():
            maps = {n:ns['model'](data[n]['x'][None].to(ns['DEVICE'])).cpu()[0].numpy() for n in held}
        assert all(np.isfinite(a).all() for a in maps.values())
        np.savez_compressed(held_file, **maps)
        predictions.update(maps)
        save(fold/f'fold_inventory_{tag}.json',dict(held_out=held,fit=manifest['train'],validation=manifest['validation'],test_used_for_selection=False,checkpoint_sha256=hashlib.sha256((fold/'baseline.pt').read_bytes()).hexdigest()))
        print('FOLD COMPLETE',dataset,seed,i+1,flush=True)
    manifest['train'] = original_manifest['train']
    assert set(predictions)==set(original_manifest['train'])
    np.savez_compressed(job/f'oof_training_maps_{tag}.npz', **predictions)
    # Reuse exact component-reference implementation and initialisation. Only
    # training coarse maps are replaced, immediately after baseline inference.
    text = (HERE/'run_component_ablations.py').read_text()
    old = "    ae0=ns['MaskAutoencoder']().to(device);ae0.load_state_dict(torch.load(reference/'autoencoder.pt',weights_only=True))"
    insertion = "    with np.load(job/f'oof_training_maps_{tag}.npz') as oof:\n        assert set(oof.files)==set(manifest['train'])\n        for name in manifest['train']:data[name]['coarse']=torch.from_numpy(oof[name].copy())\n"
    assert text.count(old)==1
    text = text.replace(old, insertion+old)
    text = text.replace("trained_variants=7,evaluation_variants=8", "trained_variants=1,evaluation_variants=1")
    exec(compile(text, str(HERE/'run_component_ablations.py'), 'exec'), component.__dict__)
    component.VARIANTS = ['global_frozen']
    component.worker_impl(out, dataset, seed)
    cp = job/'global_frozen'/f'protocol_{tag}.json'
    c = json.loads(cp.read_text()); c['scope']=protocol['scope']; c['coarse_training_source']='out-of-fold; see oof_protocol'; save(cp,c)
    # Preserve the existing in-sample comparator, rather than refitting it.
    control = CONTROL/dataset/f'seed_{seed}'/'global_frozen'
    control_rows = list(csv.DictReader((control/'per_image_2026-10-09_12-40-32.txt').open()))
    oof_rows = list(csv.DictReader((job/'global_frozen'/f'per_image_{tag}.txt').open()))
    assert len(control_rows)==len(oof_rows)==len(manifest['test'])*5
    for a,b in zip(control_rows,oof_rows):
        for k in ['image','sampling_seed','training_seed'] + [k for k in a if k.startswith('baseline_')]:
            assert a[k]==b[k],(k,a[k],b[k])
    save(job/f'oof_completed_{tag}.json',dict(completed_at=stamp(),fold_models=len(folds),records=len(oof_rows),control=str(control),protocol=protocol))

def report(out):
    import numpy as np
    from scipy.stats import ttest_rel
    from report_all_datasets_full_runs import exact_signed_rank
    tag=out.name[-19:]; allrows=[]; summary=[]; tests=[]
    for dataset,n in [('CHASE',8),('DRIVE',20),('HRF',12)]:
        matrix={arm:[] for arm in ['in_sample','out_of_fold']}; names=None
        for seed in SEEDS:
            job=out/dataset/f'seed_{seed}'; assert (job/f'oof_completed_{tag}.json').exists()
            reference={}
            for arm,dest,t in [('in_sample',CONTROL/dataset/f'seed_{seed}','2026-10-09_12-40-32'),('out_of_fold',job,tag)]:
                rr=list(csv.DictReader((dest/'global_frozen'/f'per_image_{t}.txt').open()))
                images=sorted({r['image'] for r in rr}); assert len(images)==n
                if names is None:names=images
                assert names==images
                means=[]
                for image in images:
                    with np.load(dest/f'{image}_reference_{t}.npz') as a:
                        gt=a['ground_truth']>=.5; fov=a['fov'].astype(bool); coarse=a['baseline']
                    if image in reference:
                        assert np.array_equal(reference[image],coarse)
                    reference[image]=coarse
                    with np.load(dest/'global_frozen'/f'{image}_global_frozen_predictions_{t}.npz') as a: pp=a['probabilities']; seeds=a['sampling_seeds']
                    sub=[r for r in rr if r['image']==image]; assert len(sub)==len(pp)==5
                    for r,p,noise in zip(sub,pp,seeds):
                        assert int(r['sampling_seed'])==int(noise)
                        for prefix,prob in [('baseline',coarse),('refined',p)]:
                            binary=prob>=.5
                            counts={'tp':int((binary&gt&fov).sum()),'fp':int((binary&~gt&fov).sum()),'fn':int((~binary&gt&fov).sum()),'tn':int((~binary&~gt&fov).sum())}
                            for k,v in counts.items():assert int(r[prefix+'_'+k])==v
                            tp,fp,fn=counts['tp'],counts['fp'],counts['fn'];assert abs(float(r[prefix+'_dice'])-2*tp/max(2*tp+fp+fn,1))<1e-12
                        allrows.append(dict(arm=arm,**r))
                    means.append(np.mean([float(r['refined_dice']) for r in sub]))
                matrix[arm].append(means)
        for arm in matrix:
            matrix[arm]=np.array(matrix[arm]); runmeans=matrix[arm].mean(1)
            summary.append(dict(dataset=dataset,arm=arm,mean_dice=float(runmeans.mean()),sd_between_training_seeds=float(runmeans.std(ddof=1))))
        diff=matrix['out_of_fold']-matrix['in_sample']; delta=diff.mean(1)
        if dataset=='CHASE':
            units=sorted({int(name.split('_')[1][:2]) for name in names})
            diff=np.stack([diff[:,[i for i,name in enumerate(names) if int(name.split('_')[1][:2])==u]].mean(1) for u in units],axis=1)
        rng=np.random.default_rng(20261009)
        if dataset=='HRF':
            ui=np.concatenate([rng.choice([i for i,name in enumerate(names) if name.split('_')[-1]==p],(50000,4),replace=True) for p in ['h','dr','g']],axis=1)
        else:ui=rng.integers(diff.shape[1],size=(50000,diff.shape[1]))
        si=rng.integers(5,size=(50000,5));draws=diff[si[:,:,None],ui[:,None,:]].mean((1,2))
        stat,wp,nz=exact_signed_rank(delta)
        tests.append(dict(dataset=dataset,mean_oof_minus_in_sample=float(delta.mean()),paired_t_p=float(ttest_rel(matrix['out_of_fold'].mean(1),matrix['in_sample'].mean(1)).pvalue),exact_signed_rank_p=wp,crossed_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),bootstrap_resamples=50000))
    for key in ['paired_t_p','exact_signed_rank_p']:
        order=sorted(range(3),key=lambda i:tests[i][key]);prev=0
        for rank,i in enumerate(order):prev=max(prev,min(1,tests[i][key]*(3-rank)));tests[i]['holm_'+key]=prev
    assert len(allrows)==2000
    for name,rows in [('all_verified_per_image',allrows),('mean_and_SD',summary)]:
        with (out/f'{name}_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/f'paired_tests_{tag}.json',tests)
    save(out/f'completed_{tag}.json',dict(verified_records=2000,new_oof_baselines=55,new_refiners=15,completed_at=stamp()))
    save(out/'status.json',dict(stage='completed',verified_records=2000,updated_at=stamp()))
    print(json.dumps(dict(summary=summary,tests=tests),indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dataset');ap.add_argument('--seed',type=int);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);tag=out.name[-19:]
    if args.dataset:worker(out,args.dataset,args.seed)
    else:
        (out/f'source_run_oof_comparison_{tag}.py').write_bytes(Path(__file__).read_bytes())
        save(out/f'plan_{tag}.json',dict(datasets=['CHASE','DRIVE','HRF'],seeds=SEEDS,new_fold_baselines=55,new_refiners=15,paired_control=str(CONTROL),original_author_tables_unchanged=True,manuscript_unchanged=True))
        jobs=[(d,s) for s in SEEDS for d in ['CHASE','DRIVE','HRF']]
        def launch(pair):
            d,s=pair
            with (out/f'{d}_seed_{s}_{tag}.log').open('a') as f:
                result=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(out),'--dataset',d,'--seed',str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{d} seed{s} failed; inspect log')
        try:
            save(out/'status.json',dict(stage='running',started_at=stamp(),jobs=15))
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(launch,jobs))
            report(out)
        except Exception:
            save(out/'status.json',dict(stage='failed',traceback=traceback.format_exc(),updated_at=stamp()));raise
