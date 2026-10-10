"""Verify predictions and report controlled component ablations."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel
from run_component_ablations import SEEDS,VARIANTS,save,stamp
from report_all_datasets_full_runs import exact_signed_rank
MODES=['global_frozen','deterministic','target_warm','target_full','no_image','spatial_film','global_additive','ae_finetuned']

def main(out):
    tag=out.name[-19:];allrows=[];summaries=[];seedmeans=[];tests=[];inventory=[]
    for dataset,nimages in [('CHASE',8),('DRIVE',20),('HRF',12)]:
        matrices={mode:[] for mode in MODES};baseline_matrix=[];names=None;parameter_counts={}
        for seed in SEEDS:
            job=out/dataset/f'seed_{seed}';assert (job/f'completed_{tag}.json').exists()
            rr=[]
            references={}
            for variant in VARIANTS:
                dest=job/variant;completion=json.loads((dest/f'completed_{tag}.json').read_text());protocol=json.loads((dest/f'protocol_{tag}.json').read_text())
                assert hashlib.sha256((dest/f'model_{tag}.pt').read_bytes()).hexdigest()==completion['checkpoint_sha256']
                inventory.append(dict(dataset=dataset,seed=seed,variant=variant,checkpoint=str(dest/f'model_{tag}.pt'),records=completion['records'],parameters=completion['updated_parameters']))
                rows=list(csv.DictReader((dest/f'per_image_{tag}.txt').open()));assert len(rows)==completion['records']
                mode_names=['target_warm','target_full'] if variant=='target_diffusion' else[variant]
                for mode in mode_names:
                    images=sorted({r['image'] for r in rows if r['variant']==mode});assert len(images)==nimages
                    for image in images:
                        if image not in references:
                            with np.load(job/f'{image}_reference_{tag}.npz') as a:references[image]={k:a[k].copy() for k in a.files}
                        ref=references[image];gt=ref['ground_truth']>=.5;fov=ref['fov'].astype(bool)
                        with np.load(dest/f'{image}_{mode}_predictions_{tag}.npz') as a:predictions=a['probabilities'];noises=a['sampling_seeds']
                        sub=[r for r in rows if r['image']==image and r['variant']==mode]
                        assert len(sub)==len(predictions)==(1 if mode=='deterministic' else 5)
                        for r,probability,noise in zip(sub,predictions,noises):
                            assert (int(r['sampling_seed']) if r['sampling_seed'] else-1)==int(noise)
                            assert np.isfinite(probability).all()
                            for prefix,array in [('baseline',ref['baseline']),('refined',probability)]:
                                p=array>=.5;tp=int((p & gt & fov).sum());fp=int((p & ~gt & fov).sum());fn=int((~p & gt & fov).sum());tn=int((~p & ~gt & fov).sum())
                                metrics=dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/(tp+fp+fn+tn))
                                for k,v in metrics.items():assert abs(float(r[prefix+'_'+k])-v)<1e-12
                            assert abs(float(r['delta'])-float(r['refined_dice'])+float(r['baseline_dice']))<1e-12
                rr+=rows
            allrows+=rr
            image_names=sorted(references)
            if names is None:names=image_names
            assert names==image_names
            bm=np.array([float(next(r for r in rr if r['image']==n)['baseline_dice']) for n in names]);baseline_matrix.append(bm)
            seedmeans.append(dict(dataset=dataset,training_seed=seed,variant='coarse_baseline',mean_dice=float(bm.mean()),sampling_realisations=0))
            for mode in MODES:
                parameter_counts[mode]=int(next(r for r in rr if r['variant']==mode)['updated_parameters'])
                matrixrow=np.array([np.mean([float(r['refined_dice']) for r in rr if r['image']==n and r['variant']==mode]) for n in names])
                matrices[mode].append(matrixrow);seedmeans.append(dict(dataset=dataset,training_seed=seed,variant=mode,mean_dice=float(matrixrow.mean()),sampling_realisations=1 if mode=='deterministic' else 5))
            print('Verified',dataset,seed,len(rr),flush=True)
        matrices={k:np.asarray(v) for k,v in matrices.items()};baseline_matrix=np.asarray(baseline_matrix)
        for mode,matrix in [('coarse_baseline',baseline_matrix),*matrices.items()]:
            means=matrix.mean(1);summaries.append(dict(dataset=dataset,variant=mode,mean_dice=float(means.mean()),sd_between_training_seeds=float(means.std(ddof=1)),training_runs=5,updated_parameters=parameter_counts.get(mode,0)))
        comparisons=[('reference_vs_'+mode,'global_frozen',mode) for mode in MODES if mode!='global_frozen']
        comparisons+=[('target_initialisation','target_warm','target_full')]
        for comparison,first,second in comparisons:
            differences=matrices[second]-matrices[first];deltas=differences.mean(1)
            if dataset=='CHASE':
                subjects=sorted({int(n.split('_')[1][:2]) for n in names})
                units=np.stack([differences[:,[i for i,n in enumerate(names) if int(n.split('_')[1][:2])==s]].mean(1) for s in subjects],axis=1)
            else:units=differences
            rng=np.random.default_rng(20261009)
            if dataset=='HRF':
                strata=[[i for i,n in enumerate(names) if n.split('_')[-1]==p] for p in ['h','dr','g']];assert all(len(s)==4 for s in strata)
                ui=np.concatenate([rng.choice(s,(50000,len(s)),replace=True) for s in strata],axis=1)
            else:ui=rng.integers(units.shape[1],size=(50000,units.shape[1]))
            si=rng.integers(5,size=(50000,5));draws=units[si[:,:,None],ui[:,None,:]].mean((1,2));stat,wp,n=exact_signed_rank(deltas)
            tp=1. if np.all(deltas==0) else float(ttest_rel(matrices[second].mean(1),matrices[first].mean(1)).pvalue)
            tests.append(dict(dataset=dataset,comparison=comparison,first=first,second=second,mean_delta=float(deltas.mean()),sd_delta=float(deltas.std(ddof=1)),crossed_bootstrap_95CI=np.quantile(draws,[.025,.975]).tolist(),paired_t_p=tp,exact_signed_rank_p=wp,wilcoxon_statistic=stat,nonzero_pairs=n,bootstrap_draws=50000,bootstrap_seed=20261009,resampling='Training seeds and test subjects forCHASE;images forDRIVE;pathology-stratified images forHRF;patient independence unverified forDRIVE/HRF'))
    assert len(inventory)==105 and len(allrows)==7200
    for key in ['paired_t_p','exact_signed_rank_p']:
        maximum=0
        for i,r in enumerate(sorted(tests,key=lambda r:r[key])):maximum=max(maximum,(len(tests)-i)*r[key]);r['holm_'+key]=min(1.,maximum)
    for stem,rows in [('all_verified_per_image',allrows),('mean_and_SD',summaries),('training_seed_means',seedmeans)]:
        with (out/f'{stem}_{tag}.txt').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/f'paired_tests_{tag}.json',tests);save(out/f'completion_inventory_{tag}.json',inventory)
    lines=['NEW controlled component-ablation experiment. Original supplied tables/manuscript unchanged.',
           '105 independently trained refiners:7variants x5training seeds x3datasets;fixed retained per-seed baseline/initialAE.',
           'Eight evaluated modes:target diffusion has bothcoarse-warm andfullGaussian initialisation;same target-trained checkpoint.',
           '100epochs,fixed finalcheckpoint,matched training/noise seeds,input data,split,objective exceptdeterministic;five sampling seeds at50DDIMsteps.',
           'Conditioning global/spatial matched parameters and initialweights. Additive usesexisting(g+b)/sqrt2 projection:parameter budget matched but additive parameterisation redundant.',
           'No-image removesRGBbranch fromoptimizer and haslower updated count. Fine-tuning updatesbothAEencoder anddecoder,addingtheir parameters.',
           'Ablation deterministic shareslatentAE andconditioning architecture;differs from separatelyreported spatialRGB/coarse U-Net experiment.',
           'All7200records verified againstsaved FP32 predictions,GT,FOV andallconfusioncounts/metrics.',
           'Means/SD across5trainedmodels,afteraveraging samplingrealisationsperimage. Pairedtraining-seed t andexactsign-rank;Holmseparatelyacross24comparisons.',
           'Crossedbootstrap50000resamples oftraining seeds andtestunits;4CHASEsubjects,20DRIVEimages,12pathology-stratifiedHRFimages.',
           'Smallfixed,previouslyexaminedtestcohorts;in-samplecoarsetrainingmaps;unknownpatientindependenceforDRIVE/HRF;five-pairtests lowresolution.',
           'This is not recovery/validation oforiginalpositive author-supplied score table;do not mix its scores into thesepairedcomparisons.','']
    for r in summaries:lines.append(f"{r['dataset']} {r['variant']}: Dice={r['mean_dice']:.9f} +/- {r['sd_between_training_seeds']:.9f};updatedparams={r['updated_parameters']}")
    lines.append('')
    for r in tests:lines.append(f"{r['dataset']} {r['comparison']}: change={r['mean_delta']:+.9f};95CI={r['crossed_bootstrap_95CI']};Holm t p={r['holm_paired_t_p']:.9g};Holm W p={r['holm_exact_signed_rank_p']:.9g}")
    (out/f'experiment_report_{tag}.txt').write_text('\n'.join(lines)+'\n')
    (out/f'source_report_component_ablations_{tag}.py').write_bytes(Path(__file__).read_bytes())
    with zipfile.ZipFile(out/f'component_ablation_tables_{tag}.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.txt','.json','.py','.log'] and tag in p.name:z.write(p,arcname=str(p.relative_to(out)))
    with zipfile.ZipFile(out/f'component_ablation_tables_{tag}.zip') as z:assert z.testzip() is None
    save(out/f'verified_completion_{tag}.json',dict(trained_models=105,verified_records=7200,evaluation_modes=8,completed_at=stamp(),weights_and_FP32_predictions='Retained in experiment subfolders;not included in tables-only ZIP'))
    save(out/'status.json',dict(stage='completed',trained_models=105,verified_records=7200,updated_at=stamp()))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();main(a.output.resolve())

