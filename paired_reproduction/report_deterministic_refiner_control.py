"""Verify and analyse paired deterministic-control predictions."""
import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp
from report_all_datasets_full_runs import exact_signed_rank


def main(out):
    tag=out.name[-19:];seeds=[42,123,2026,3000,3001]
    allrows=[];seedrows=[];summaries=[];tests=[];verified=0
    for dataset in ['CHASE','DRIVE','HRF']:
        matrices={key:[] for key in ['baseline_dice','autoencoder_dice','deterministic_dice','diffusion_mean_dice']}
        image_names=None
        for seed in seeds:
            job=out/dataset/f'train_seed_{seed}'
            assert (job/f'completed_{tag}.json').exists()
            protocol=json.loads((job/f'protocol_{tag}.json').read_text())
            assert protocol['learned_parameters']==protocol['diffusion_learned_parameters']
            reference=Path(protocol['reference'])
            for filename,key in [('baseline.pt','frozen_baseline_sha256'),('autoencoder.pt','frozen_autoencoder_sha256')]:
                assert hashlib.sha256((reference/filename).read_bytes()).hexdigest()==protocol[key]
            rows=sorted(csv.DictReader((job/f'paired_per_image_{tag}.txt').open(newline='')),key=lambda r:r['image'])
            assert len({r['image'] for r in rows})==len(rows)
            if image_names is None:image_names=[r['image'] for r in rows]
            assert image_names==[r['image'] for r in rows]
            for r in rows:
                with np.load(reference/f"{r['image']}_reference.npz") as f:
                    gt=f['ground_truth']>=.5;fov=f['fov'].astype(bool);coarse=f['coarse']>=.5
                with np.load(job/f"{r['image']}_predictions_{tag}.npz") as f:
                    predictions=dict(baseline=coarse,autoencoder=f['autoencoder_reconstruction']>=.5,
                                     deterministic=f['deterministic']>=.5)
                for prefix,prediction in predictions.items():
                    tp=int((prediction & gt & fov).sum());fp=int((prediction & ~gt & fov).sum())
                    fn=int((~prediction & gt & fov).sum());tn=int((~prediction & ~gt & fov).sum())
                    for k,value in [('tp',tp),('fp',fp),('fn',fn),('tn',tn)]:assert int(r[prefix+'_'+k])==value
                    expected=dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),
                                  specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/(tp+fp+fn+tn))
                    for k,value in expected.items():assert abs(float(r[prefix+'_'+k])-value)<1e-12
                verified+=1
            allrows.extend(rows)
            meanrow=dict(dataset=dataset,seed=seed)
            for key in matrices:
                values=[float(r[key]) for r in rows];matrices[key].append(values);meanrow[key]=float(np.mean(values))
            seedrows.append(meanrow)
        matrices={k:np.asarray(v) for k,v in matrices.items()}
        for key,matrix in matrices.items():
            runmeans=matrix.mean(1)
            summaries.append(dict(dataset=dataset,method=key,mean_dice=float(runmeans.mean()),
                                  sd_between_training_seeds=float(runmeans.std(ddof=1)),training_runs=5))
        comparisons=[('baseline_to_deterministic','baseline_dice','deterministic_dice'),
                     ('autoencoder_to_deterministic','autoencoder_dice','deterministic_dice'),
                     ('deterministic_to_diffusion','deterministic_dice','diffusion_mean_dice')]
        for name,first,second in comparisons:
            differences=matrices[second]-matrices[first];run_deltas=differences.mean(1)
            ttest=ttest_1samp(run_deltas,0);w,wp,n_nonzero=exact_signed_rank(run_deltas)
            if dataset=='CHASE':
                subjects=sorted({int(image.split('_')[1][:2]) for image in image_names})
                unit_matrix=np.stack([differences[:,[i for i,image in enumerate(image_names) if int(image.split('_')[1][:2])==subject]].mean(1) for subject in subjects],axis=1)
                unit='subject; both eyes averaged'
            else:unit_matrix=differences;unit='image; patient independence unverified'
            rng=np.random.default_rng(20261009)
            if dataset=='HRF':
                strata=[[i for i,image in enumerate(image_names) if image.split('_')[-1]==category] for category in ['h','dr','g']]
                indices=np.concatenate([rng.choice(group,(50000,len(group)),replace=True) for group in strata],axis=1)
            else:indices=rng.integers(unit_matrix.shape[1],size=(50000,unit_matrix.shape[1]))
            training_indices=rng.integers(5,size=(50000,5))
            draws=unit_matrix[training_indices[:,:,None],indices[:,None,:]].mean((1,2))
            tests.append(dict(dataset=dataset,comparison=name,mean_delta=float(run_deltas.mean()),sd_delta=float(run_deltas.std(ddof=1)),
                              paired_t_p=float(ttest.pvalue),exact_wilcoxon_p=wp,wilcoxon_statistic=w,nonzero_training_pairs=n_nonzero,
                              crossed_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),bootstrap_draws=50000,bootstrap_seed=20261009,
                              resampling_unit=unit,scope='Paired training-seed inference on fixed test set; crossed bootstrap includes test-unit resampling'))
    assert verified==len(allrows)==200
    for key in ['paired_t_p','exact_wilcoxon_p']:
        maximum=0.
        for i,r in enumerate(sorted(tests,key=lambda r:r[key])):
            maximum=max(maximum,(len(tests)-i)*r[key]);r['holm_'+key]=min(maximum,1.)
    for stem,rows in [('all_paired_per_image',allrows),('seed_means',seedrows),('mean_and_SD',summaries)]:
        with (out/f'{stem}_{tag}.txt').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/f'paired_tests_{tag}.json').write_text(json.dumps(tests,indent=2))
    report=['Matched deterministic direct-refiner control of the ACTUAL LOCAL reproduction, not supplied positive-score tables.',
            '15 direct-refiner training jobs:5 per dataset. Frozen per-seed LU-Net+RA and AE checkpoints match diffusion reference exactly.',
            'Same image encoder, latent U-Net and time MLP; fixed t=0. Residual latent prediction, one conditional forward pass.',
            '100epochs; AdamW1e-4,weight_decay.001;CosineAnnealing;batch4;gradient clipping1;no augmentation;conditioning dropout.1.',
            'Loss:0.5*(1-Dice) using identical decoded-mask Dice reduction. No epsilon-MSE, Gaussian corruption, DDIM or CFG.',
            'Total learned parameter budget matched; fixed-time conditioning is less expressive than variable-time conditioning.',
            'This compares direct supervised refinement with the diffusion training/inference formulation; it is not a single-objective-factor ablation.',
            'Diffusion numbers refer to the same frozen per-seed checkpoints at50steps, averaged over5 sampling realisations.',
            'Two-sided paired training-seed t and exact signed-rank tests, separately Holm-adjusted across9 comparisons.',
            'Crossed bootstrap resamples5 training seeds and test units;50,000 draws;4subjects forCHASE,images forDRIVE,HRF pathology-stratified.',
            '5-run signed-rank tests have low resolution; distributional assumptions and small test cohorts limit interpretation.',
            f'PASS: all {verified} paired records checked against saved FP32 masks/GT/FOV; repeated inference bitwise deterministic.','']
    for r in summaries:report.append(f"{r['dataset']} {r['method']}: {r['mean_dice']:.9f} +/- {r['sd_between_training_seeds']:.9f}")
    report.append('')
    for r in tests:report.append(f"{r['dataset']} {r['comparison']}: delta={r['mean_delta']:+.9f}, Holm t p={r['holm_paired_t_p']:.9g}, Holm Wilcoxon p={r['holm_exact_wilcoxon_p']:.9g}, crossed95CI={r['crossed_bootstrap_95ci']}")
    (out/f'experiment_report_{tag}.txt').write_text('\n'.join(report)+'\n')
    (out/f'source_report_deterministic_refiner_control_{tag}.py').write_bytes(Path(__file__).read_bytes())
    paths=[]
    with zipfile.ZipFile(out/f'deterministic_control_data_{tag}.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.txt','.json','.py','.log'] and tag in p.name:
                z.write(p,arcname=str(p.relative_to(out)));paths.append(str(p))
    savepath=out/f'verified_completion_{tag}.json'
    savepath.write_text(json.dumps(dict(completed_training_jobs=15,verified_records=200,report_files=paths),indent=2))
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();main(args.output.resolve())
