"""Extract paired descriptive statistics from the accepted supplied tables."""
import csv,json,hashlib,math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=ROOT/'reviewer_comments'/f'reviewer_2_comment_1_results_{stamp}';OUT.mkdir()
sources={'CHASE-DB1':Path(r'C:\Users\FS-GAMING PRO\.codex\attachments\8ef7711f-d75c-4638-8520-8c5b89744f9a\Pasted text.txt'),'DRIVE':Path(r'C:\Users\FS-GAMING PRO\.codex\attachments\3e9ea676-ad6f-4aed-b5ef-3805fe335eaf\Pasted text.txt')}
manuscript=ROOT/'template261006.tex';before=hashlib.sha256(manuscript.read_bytes()).hexdigest()
metrics=['dice','iou','sensitivity','specificity','accuracy'];summary={};per={}
lines=['REVIEWER 2, COMMENT 1: PAIRED RESULTS',f'Extracted {stamp} Asia/Tehran','Only the latest supplied CHASE and DRIVE tables are used. No substitution from separate executable runs.','Raw source files preserved. Full precision used for every calculation.','Dice differences = refined minus baseline; sample scores averaged per image before paired resampling.','Displayed 3-decimal means use upward rounding as requested; inference and confidence intervals use unrounded values.','Confidence interval endpoints are provided at9decimals with ordinary display rounding, not upward-rounded; full precision retained in JSON.','50,000 paired percentile bootstrap draws, RNG seed20261007.','CHASE resampling unit: four subjects, retaining both eyes. DRIVE resampling unit:20images, conditional on image independence; actual patient identity not independently verified.','Intervals condition on supplied fitted-model results and sampled-seed averages; no training-run variability included.','Algebraic consistency can be verified, but checkpoint/protocol and prediction-mask provenance cannot be established from scalar tables alone.','No numerical results for original HRF Dice0.793 are inferred from these tables.','']
for dataset,source in sources.items():
 raw=list(csv.DictReader(source.open(encoding='utf-8-sig')))
 rows=[r for r in raw if r.get('steps','50')=='50']
 seed_count=20 if dataset=='CHASE-DB1' else 3
 image_count=8 if dataset=='CHASE-DB1' else 20
 names=sorted({r['image'] for r in rows});assert len(names)==image_count
 for r in rows:
  for prefix in ('baseline','refined'):
   tp,fp,fn,tn=[int(r[f'{prefix}_{k}']) for k in ('tp','fp','fn','tn')]
   assert min(tp,fp,fn,tn)>=0
   calculated=dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/max(tp+fp+fn+tn,1))
   assert all(abs(calculated[m]-float(r[f'{prefix}_{m}']))<1e-10 for m in metrics)
  # Matching annotation-positive and negative counts support pairing within listed image domain.
  assert int(r['baseline_tp'])+int(r['baseline_fn'])==int(r['refined_tp'])+int(r['refined_fn'])
  assert int(r['baseline_tn'])+int(r['baseline_fp'])==int(r['refined_tn'])+int(r['refined_fp'])
  assert abs(float(r['delta'])-(float(r['refined_dice'])-float(r['baseline_dice'])))<1e-10
 per[dataset]=[]
 for name in names:
  rr=[r for r in rows if r['image']==name];assert len(rr)==seed_count and len({r['seed'] for r in rr})==seed_count
  entry=dict(image=name,unit=rr[0].get('unit',rr[0].get('subject')),sampling_realisations=seed_count)
  for prefix in ('baseline','refined'):
   for metric in metrics:entry[f'{prefix}_{metric}']=float(np.mean([float(r[f'{prefix}_{metric}']) for r in rr]))
  entry['delta']=entry['refined_dice']-entry['baseline_dice'];per[dataset].append(entry)
 keys=['baseline_dice','refined_dice','delta'];rng=np.random.default_rng(20261007)
 if dataset=='CHASE-DB1':
  units=sorted({r['unit'] for r in per[dataset]});assert len(units)==4
  v=np.array([[np.mean([r[k] for r in per[dataset] if r['unit']==u]) for k in keys] for u in units]);boots=v[rng.integers(0,4,(50000,4))].mean(1)
 else:
  v=np.array([[r[k] for k in keys] for r in per[dataset]]);boots=v[rng.integers(0,20,(50000,20))].mean(1)
 ci=np.quantile(boots,[.025,.975],axis=0);means=v.mean(0)
 summary[dataset]=dict(n_images=image_count,sampling_realisations_per_image=seed_count,transitions=50,CI95={k:ci[:,i].tolist() for i,k in enumerate(keys)},means={k:float(means[i]) for i,k in enumerate(keys)},upward_display_3dec={k:math.ceil(means[i]*1000)/1000 for i,k in enumerate(keys)},other_metric_means={f'{prefix}_{m}':float(np.mean([r[f'{prefix}_{m}'] for r in per[dataset]])) for prefix in ('baseline','refined') for m in metrics},improved_images=sum(r['delta']>0 for r in per[dataset]),worsened_images=sum(r['delta']<0 for r in per[dataset]),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 lines += [dataset,'image | unit | sampling realisations | baseline Dice | refined Dice | paired difference']
 for r in per[dataset]:lines.append(f"{r['image']} | {r['unit']} | {seed_count} | {r['baseline_dice']:.9f} | {r['refined_dice']:.9f} | {r['delta']:+.9f}")
 lines+=['','Unrounded-calculation means and exploratory95% intervals:']
 for i,k in enumerate(keys):lines.append(f'{k}: {means[i]:.9f}; interval [{ci[0,i]:.9f}, {ci[1,i]:.9f}]')
 lines+=['','All metric means:']
 for k,vv in summary[dataset]['other_metric_means'].items():lines.append(f'{k}: {vv:.9f}')
 lines+=['',f"Improved images: {summary[dataset]['improved_images']}; worsened images: {summary[dataset]['worsened_images']}",'Upward-rounded means for display only: '+json.dumps(summary[dataset]['upward_display_3dec']),'']
after=hashlib.sha256(manuscript.read_bytes()).hexdigest();assert before==after
audit=dict(summary=summary,per_image_results=per,manuscript_unchanged=True,manuscript_sha256=after,bootstrap_draws=50000,random_seed=20261007,limitations=lines[4:12])
(OUT/'paired_results_and_intervals.json').write_text(json.dumps(audit,indent=2))
(OUT/'paired_per_image_results.txt').write_text('\n'.join(lines),encoding='utf8')
c=summary['CHASE-DB1'];d=summary['DRIVE']
reply=f'''Thank you for this comment. We have prepared paired per-image analyses of the supplied CHASE-DB1 and DRIVE evaluation tables, comparing baseline and refinement scores within each image. Sampling realisations were averaged per image before calculating mean paired changes and confidence intervals.

On CHASE-DB1 (eight images), baseline and refined mean Dice were {c['means']['baseline_dice']:.6f} and {c['means']['refined_dice']:.6f}. The mean paired difference was {c['means']['delta']:.6f}, with an exploratory95% bootstrap interval [{c['CI95']['delta'][0]:.6f}, {c['CI95']['delta'][1]:.6f}]. We averaged20 sampling realisations per image and retained both eyes of each of four subjects together during resampling.

On DRIVE (20 images), the corresponding means were {d['means']['baseline_dice']:.6f} and {d['means']['refined_dice']:.6f}. The paired difference was {d['means']['delta']:.6f}, with an exploratory95% interval [{d['CI95']['delta'][0]:.6f}, {d['CI95']['delta'][1]:.6f}]. We averaged three sampling realisations per image and resampled image pairs, conditional on independence between test images.

Each analysis used50,000 bootstrap resamples. The supplied counts and metrics were checked for algebraic consistency and matching positive/negative reference totals within each image. These checks do not independently establish checkpoint, threshold, annotation, or prediction-mask provenance. Intervals condition on the supplied fitted models and sampling averages and do not include training-run variability. Four CHASE subject clusters are too few for strong population-level claims. A corresponding supplied per-image table for the original HRF result remains necessary to address that dataset. This extraction has not changed the manuscript or established that every part of the reviewer's request is resolved.
'''
(OUT/'reviewer_2_comment_1_response_draft.txt').write_text(reply,encoding='utf8')
print(json.dumps(dict(report=str(OUT/'paired_per_image_results.txt'),response=str(OUT/'reviewer_2_comment_1_response_draft.txt'),summary=summary,manuscript_unchanged=True),indent=2))
