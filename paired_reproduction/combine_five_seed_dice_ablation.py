"""Combine the five actual refiner-training seeds; never rescale to paper scores."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import t,ttest_rel,wilcoxon
HERE=Path(__file__).resolve().parent
OLD=HERE/'dice_weight_ablation_2026-10-06_18-14-33'
NEW=sorted(HERE.glob('dice_weight_additional_seeds_*/summary.json'))[-1].parent
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=HERE/f'dice_weight_five_seed_summary_{stamp}';OUT.mkdir()
records=[]
for folder in (OLD,NEW):
 records+=list(csv.DictReader((folder/'per_image_seed_metrics.csv').open()))
assert len(records)==800
seeds=[42,123,2026,3000,3001];weights=[0.,.25,.5,1.]
subjects=sorted({r['subject'] for r in records});assert len(subjects)==4
for r in records:
 tp,fp,fn=[int(r[k]) for k in ('tp','fp','fn')]
 assert abs(2*tp/max(2*tp+fp+fn,1)-float(r['dice']))<1e-12
for filename in ('baseline.pt','autoencoder.pt'):
 assert hashlib.sha256((OLD/filename).read_bytes()).hexdigest()==hashlib.sha256((NEW/filename).read_bytes()).hexdigest()
oldp=json.loads((OLD/'protocol.json').read_text());newp=json.loads((NEW/'protocol.json').read_text())
assert oldp['split']==newp['split'] and oldp['sampling_seeds']==newp['sampling_seeds']
assert newp['training_seeds']==[3000,3001]
arrays={};statistics={};rng=np.random.default_rng(20261007)
for w in weights:
 arr=np.array([[np.mean([float(r['dice']) for r in records if float(r['lambda_Dice'])==w and r['subject']==u and int(r['training_seed'])==s]) for s in seeds] for u in subjects])
 assert np.isfinite(arr).all();arrays[w]=arr
 for s in seeds:assert len([r for r in records if float(r['lambda_Dice'])==w and int(r['training_seed'])==s])==40
for w in weights:
 a=arrays[w].mean(0);delta=a-arrays[0].mean(0);d=arrays[w]-arrays[0]
 indices_subject=rng.integers(0,4,(50000,4));indices_seed=rng.integers(0,5,(50000,5))
 boots=d[indices_subject[:,:,None],indices_seed[:,None,:]].mean((1,2))
 seed_ci=(delta.mean()+np.array([-1,1])*t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)).tolist()
 statistics[str(w)]=dict(training_seed_means=dict(zip(map(str,seeds),a.tolist())),mean_dice=float(a.mean()),SD_between_training_seed_means=float(a.std(ddof=1)),paired_difference_vs_zero=float(delta.mean()),paired_difference_SD=float(delta.std(ddof=1)),seed_level_t95ci=seed_ci,crossed_subject_training_seed_bootstrap95ci=np.quantile(boots,[.025,.975]).tolist())
 if w!=0:
  statistic,p=ttest_rel(a,arrays[0].mean(0));wr=wilcoxon(delta,alternative='two-sided',method='exact')
  statistics[str(w)].update(paired_t_statistic=float(statistic),paired_t_pvalue=float(p),exact_Wilcoxon_statistic=float(wr.statistic),exact_Wilcoxon_pvalue=float(wr.pvalue))
# Holm correction for the three exploratory nonzero-vs-zero t comparisons.
ordered=sorted((statistics[str(w)]['paired_t_pvalue'],w) for w in weights if w!=0);running=0
for rank,(p,w) in enumerate(ordered):
 running=max(running,min(1.,p*(3-rank)));statistics[str(w)]['paired_t_Holm_adjusted_pvalue']=running
summary=dict(training_seeds=seeds,sampling_seeds=oldp['sampling_seeds'],images=oldp['split']['test'],n_images=8,n_subjects=4,n_refiner_training_seeds=5,transitions=50,weights=statistics,source_runs=[str(OLD),str(NEW)],new_runtime=dict(torch=newp.get('torch_version'),cuda=newp.get('cuda_version')),limitations=['Actual executable reproduction; not the supplied corrected table near0.835','Only the refiner is trained with five seeds; backbone and autoencoder remain one shared training run','The old three-seed protocol does not record runtime versions; new runtime is recorded, and training sessions may differ in software environment','Only four subjects; bootstrap intervals exploratory','t tests use five training-seed means on fixed test images, assuming approximately normal paired differences','Wilcoxon two-sided exact tests with n5 have limited power; smallest possible p0.0625 when all differences have the same sign','This fixed50step ablation does not evaluate the10-to100transition trend'],checkpoint_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for f in (OLD,NEW) for p in f.glob('*.pt')})
(OUT/'five_seed_statistics.json').write_text(json.dumps(summary,indent=2))
(OUT/'all_800_prediction_records.json').write_text(json.dumps(records,indent=2))
lines=['FIVE TRAINING SEEDS: ACTUAL DICE-WEIGHT ABLATION','CHASE,8images/four subjects,50DDIM transitions,5sampling seeds per trained refiner.','Training seeds42,123,2026,3000,3001; baseline and autoencoder fixed.','These results are not rescaled to the supplied author table near0.835.','', 'weight | seed42 | seed123 | seed2026 | seed3000 | seed3001 | mean | SD']
for w in weights:
 s=statistics[str(w)];lines.append(f'{w:g} | '+' | '.join(f"{s['training_seed_means'][str(seed)]:.9f}" for seed in seeds)+f" | {s['mean_dice']:.9f} | {s['SD_between_training_seed_means']:.9f}")
lines+=['','Paired comparisons versusweight0:']
for w in weights[1:]:lines.append(str(w)+': '+json.dumps(statistics[str(w)],indent=2))
lines+=['','Limitations:']+summary['limitations']
(OUT/'five_seed_results_report.txt').write_text('\n'.join(lines))
print(json.dumps(dict(report=str(OUT/'five_seed_results_report.txt'),statistics=str(OUT/'five_seed_statistics.json'),weights=statistics),indent=2))
