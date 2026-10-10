"""Recalculate CHASE from the supplied200-row five-training-seed table only."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import ttest_rel,wilcoxon
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'C:\Users\FS-GAMING PRO\.codex\attachments\50782590-1c0c-425c-ab2f-2a516935612c\Pasted text.txt')
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=ROOT/'reviewer_comments'/f'CHASE_supplied_five_seed_recalculation_{stamp}';OUT.mkdir()
manuscript_hash=hashlib.sha256((ROOT/'template261006.tex').read_bytes()).hexdigest()
rows=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
seeds=[42,123,2026,3000,3001];noise=[42,123,2026,3000,3001]
names=sorted({r['image'] for r in rows});subjects=sorted({r['subject'] for r in rows})
assert len(rows)==200 and len(names)==8 and len(subjects)==4
assert len({(r['image'],r['seed'],r['sampling_seed']) for r in rows})==200
for r in rows:
 for prefix in ('baseline','refined'):
  tp,fp,fn,tn=[int(r[f'{prefix}_{k}']) for k in ('tp','fp','fn','tn')]
  assert min(tp,fp,fn,tn)>=0
  vals=dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/max(tp+fp+fn+tn,1))
  assert all(abs(v-float(r[f'{prefix}_{m}']))<1e-8 for m,v in vals.items())
 assert abs(float(r['delta'])-(float(r['refined_dice'])-float(r['baseline_dice'])))<1e-8
 assert int(r['baseline_tp'])+int(r['baseline_fn'])==int(r['refined_tp'])+int(r['refined_fn'])
 assert int(r['baseline_tn'])+int(r['baseline_fp'])==int(r['refined_tn'])+int(r['refined_fp'])
per=[]
for name in names:
 record=dict(image=name,subject=next(r['subject'] for r in rows if r['image']==name),training_seed_means={})
 values=[]
 for seed in seeds:
  rr=[r for r in rows if r['image']==name and int(r['seed'])==seed];assert len(rr)==5 and {int(r['sampling_seed']) for r in rr}==set(noise)
  v=float(np.mean([float(r['refined_dice']) for r in rr]));record['training_seed_means'][str(seed)]=v;values.append(v)
 base=[float(r['baseline_dice']) for r in rows if r['image']==name];assert np.ptp(base)<1e-10
 record.update(baseline_dice=float(base[0]),refined_dice=float(np.mean(values)),training_seed_SD=float(np.std(values,ddof=1)),delta=float(np.mean(values)-base[0]))
 per.append(record)
array=np.array([[np.mean([r['training_seed_means'][str(seed)] for r in per if r['subject']==u]) for seed in seeds] for u in subjects])
base=np.array([np.mean([r['baseline_dice'] for r in per if r['subject']==u]) for u in subjects]);delta=array-base[:,None]
rng=np.random.default_rng(20261007)
si=rng.integers(0,4,(50000,4));ti=rng.integers(0,5,(50000,5))
boot_delta=delta[si[:,:,None],ti[:,None,:]].mean((1,2));boot_ref=array[si[:,:,None],ti[:,None,:]].mean((1,2));boot_base=base[si].mean(1)
subject_delta=delta.mean(1);subject_t=ttest_rel(array.mean(1),base);subject_w=wilcoxon(subject_delta,method='exact',alternative='two-sided')
seed_ref=array.mean(0);seed_delta=delta.mean(0);seed_t=ttest_rel(seed_ref,np.repeat(base.mean(),5));seed_w=wilcoxon(seed_delta,method='exact',alternative='two-sided')
metrics={k:float(np.mean([float(r[k]) for r in rows])) for k in ('baseline_dice','refined_dice','baseline_iou','refined_iou','baseline_sensitivity','refined_sensitivity','baseline_specificity','refined_specificity','baseline_accuracy','refined_accuracy')}
result=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),n_images=8,n_subjects=4,n_training_seeds=5,n_sampling_seeds_per_training_seed=5,n_predictions=200,training_seed_means=dict(zip(map(str,seeds),seed_ref.tolist())),mean_baseline=float(base.mean()),mean_refined=float(seed_ref.mean()),SD_between_training_seed_means=float(seed_ref.std(ddof=1)),mean_paired_difference=float(delta.mean()),paired_difference_SD_between_training_seed_means=float(seed_delta.std(ddof=1)),crossed_subject_training_seed_bootstrap95CI=dict(baseline=np.quantile(boot_base,[.025,.975]).tolist(),refined=np.quantile(boot_ref,[.025,.975]).tolist(),paired_difference=np.quantile(boot_delta,[.025,.975]).tolist()),subject_level_tests=dict(n=4,paired_t_statistic=float(subject_t.statistic),paired_t_pvalue=float(subject_t.pvalue),exact_Wilcoxon_statistic=float(subject_w.statistic),exact_Wilcoxon_pvalue=float(subject_w.pvalue)),seed_level_tests_conditional_on_fixed_test_set=dict(n=5,paired_t_statistic=float(seed_t.statistic),paired_t_pvalue=float(seed_t.pvalue),exact_Wilcoxon_statistic=float(seed_w.statistic),exact_Wilcoxon_pvalue=float(seed_w.pvalue)),per_image=per,all_metric_means=metrics,improved_images=sum(r['delta']>0 for r in per),worsened_images=sum(r['delta']<0 for r in per),bootstrap_replicates=50000,bootstrap_seed=20261007,limitations=['Only the supplied table used; no rescaling or substitution from other executable runs','Listed counts and metrics are algebraically consistent; no independent prediction/checkpoint provenance verification','Crossed bootstrap resamples four subject clusters and five training-seed labels; sampling realisations averaged within each image/training-seed first','Four subjects too few for reliable population-level coverage or strong test claims','Training-seed SD describes fixed-test-set variability; not test-image SD','t assumptions cannot be checked reliably with four subjects or five training seeds','Counts and sampling records do not themselves prove independent complete-model training','No10-to100DDIM comparison in this file'],manuscript_unchanged=manuscript_hash==hashlib.sha256((ROOT/'template261006.tex').read_bytes()).hexdigest())
assert result['manuscript_unchanged']
(OUT/'CHASE_five_seed_statistics.json').write_text(json.dumps(result,indent=2))
lines=['CHASE: RECALCULATION FROM SUPPLIED FIVE-SEED TABLE',f'Analysis {stamp} Asia/Tehran','200 predictions;8images/four subjects,5training seeds and5sampling seeds per trained model.','Each seed mean averages5sampling realisations per image, then8images. No rounding applied before statistics.','Bootstrap:50,000 crossed subject/training-seed draws; both eyes retained together.','', 'training_seed | mean Dice']+[f'{seed} | {seed_ref[i]:.9f}' for i,seed in enumerate(seeds)]
lines+=['','image | baseline Dice | mean refined Dice | training-seed SD | paired difference']+[f"{r['image']} | {r['baseline_dice']:.9f} | {r['refined_dice']:.9f} | {r['training_seed_SD']:.9f} | {r['delta']:+.9f}" for r in per]
lines+=['','Summary and methods:',json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2)]
(OUT/'CHASE_five_seed_results.txt').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in result.items() if k in ('mean_baseline','mean_refined','training_seed_means','SD_between_training_seed_means','mean_paired_difference','crossed_subject_training_seed_bootstrap95CI','subject_level_tests','seed_level_tests_conditional_on_fixed_test_set','improved_images','worsened_images')},indent=2));print('REPORT',OUT/'CHASE_five_seed_results.txt')
