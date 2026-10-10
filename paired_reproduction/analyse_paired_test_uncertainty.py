"""Recompute paired per-image Dice and bootstrap CIs from saved rerun records."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=HERE/f'paired_test_uncertainty_{stamp}';OUT.mkdir()
manuscript=ROOT/'template261006.tex';before=hashlib.sha256(manuscript.read_bytes()).hexdigest()
sources={'CHASE-DB1':HERE/'paired_rerun_CHASE_2026-10-06_17-08-15','HRF':HERE/'paired_rerun_HRF_2026-10-06_17-09-02'}
summary={};all_per=[];lines=['PAIRED TEST-SET UNCERTAINTY ANALYSIS',f'Analysis date: {stamp} Asia/Tehran','Source: saved NEW reproduction runs. This is an analysis of existing outputs, not new training or recovery of the original manuscript scores.','Primary setting: 50 transitions; 20 transitions reported descriptively.','One training run per dataset; 20 sampling seeds averaged per image. CI conditions on these fitted checkpoints and seed averages.','50,000 paired percentile-bootstrap draws, RNG seed 20261006, retaining baseline/refiner pairing.','CHASE: four subject clusters, both eyes retained together. With only four clusters, bootstrap coverage is uncertain and intervals are exploratory.','HRF: 12 images, four each healthy/diabetic-retinopathy/glaucoma; resample images within pathology strata. Patient IDs unavailable: image-level independence cannot be verified, and these are not validated patient-level CIs.','No inference about the original manuscript results, no multiplicity-adjusted secondary inference.','']
verified=0
for dataset,source in sources.items():
 raw=list(csv.DictReader((source/'paired_metrics.csv').open()))
 n_images=8 if dataset=='CHASE-DB1' else 12
 assert len(raw)==n_images*20*2
 for r in raw:
  for prefix in ('baseline','refined'):
   tp,fp,fn,tn=[int(r[f'{prefix}_{k}']) for k in ('tp','fp','fn','tn')]
   assert min(tp,fp,fn,tn)>=0
   reconstructed=2*tp/max(2*tp+fp+fn,1)
   assert abs(reconstructed-float(r[f'{prefix}_dice']))<1e-12
  assert abs(float(r['delta'])-(float(r['refined_dice'])-float(r['baseline_dice'])))<1e-12
  verified+=1
 per=[];rng=np.random.default_rng(20261006);summary[dataset]={}
 lines += [dataset,'image | unit | pathology | steps | baseline Dice | mean refined Dice | paired difference']
 for steps in (20,50):
  rr=[r for r in raw if int(r['steps'])==steps]
  names=sorted({r['image'] for r in rr});assert len(names)==n_images
  for name in names:
   records=[r for r in rr if r['image']==name]
   assert len(records)==20 and len({r['seed'] for r in records})==20
   base=np.array([float(r['baseline_dice']) for r in records]);assert np.ptp(base)<1e-12
   refined=float(np.mean([float(r['refined_dice']) for r in records]))
   item=dict(dataset=dataset,image=name,unit=records[0]['unit'],pathology=records[0]['pathology'],steps=steps,baseline_dice=float(base[0]),refined_dice=refined,delta=refined-base[0])
   per.append(item);all_per.append(item)
   lines.append(f"{name} | {item['unit']} | {item['pathology']} | {steps} | {base[0]:.9f} | {refined:.9f} | {item['delta']:+.9f}")
  pp=[r for r in per if r['steps']==steps];keys=('baseline_dice','refined_dice','delta')
  if dataset=='CHASE-DB1':
   units=sorted({r['unit'] for r in pp});assert len(units)==4
   assert all(sum(r['unit']==u for r in pp)==2 for u in units)
   values=np.array([[np.mean([r[k] for r in pp if r['unit']==u]) for k in keys] for u in units])
   boots=values[rng.integers(0,4,(50000,4))].mean(1)
  else:
   strata=[np.array([[r[k] for k in keys] for r in pp if r['pathology']==g]) for g in ('h','dr','g')]
   assert all(len(v)==4 for v in strata)
   boots=np.mean([v[rng.integers(0,4,(50000,4))].mean(1) for v in strata],axis=0)
   values=np.array([[r[k] for k in keys] for r in pp])
  means=values.mean(0);ci=np.quantile(boots,[.025,.975],axis=0)
  previous=json.loads((source/'summary.json').read_text())[str(steps)]
  assert np.allclose(means,previous['mean'],atol=1e-12,rtol=0)
  assert np.allclose(ci,previous['CI95'],atol=1e-12,rtol=0)
  summary[dataset][str(steps)]={k:dict(mean=float(means[j]),bootstrap95ci=ci[:,j].tolist()) for j,k in enumerate(keys)}
  summary[dataset][str(steps)]['n_images']=n_images
  lines.append(f'SUMMARY {steps} transitions:')
  for j,k in enumerate(keys):lines.append(f'  {k}: {means[j]:.9f}; exploratory95CI [{ci[0,j]:.9f}, {ci[1,j]:.9f}]')
 lines+=['','Interpretation: mean paired differences and their bootstrap intervals are negative in this saved configuration. Refinement underperformed the coarse baseline; this analysis does not support a positive refinement gain.','']
after=hashlib.sha256(manuscript.read_bytes()).hexdigest();assert after==before
audit=dict(analysis_date=stamp,summary=summary,per_image_results=all_per,raw_rows_verified=verified,manuscript_unchanged=True,manuscript_sha256=after,source_sha256={str(p/'paired_metrics.csv'):hashlib.sha256((p/'paired_metrics.csv').read_bytes()).hexdigest() for p in sources.values()},resamples=50000,rng_seed=20261006)
(OUT/'results_and_protocol.json').write_text(json.dumps(audit,indent=2))
lines += [f'Verified {verified} raw prediction metric records against stored TP/FP/FN counts.',f'Manuscript unchanged, SHA256: {after}']
(OUT/'per_image_and_confidence_intervals.txt').write_text('\n'.join(lines),encoding='utf8')
c=summary['CHASE-DB1']['50'];h=summary['HRF']['50']
reply=f'''Thank you for highlighting the uncertainty associated with the small test sets. We performed a separate paired analysis of the saved new reproduction experiments, reporting per-image baseline and refinement Dice together with exploratory 95% bootstrap intervals. This analysis is separate from the unavailable original manuscript outputs and cannot validate their reported gains.

At 50 DDIM transitions, mean Dice on CHASE-DB1 was {c['baseline_dice']['mean']:.6f} for the baseline and {c['refined_dice']['mean']:.6f} for refinement. The paired mean difference was {c['delta']['mean']:.6f}, with an exploratory 95% interval [{c['delta']['bootstrap95ci'][0]:.6f}, {c['delta']['bootstrap95ci'][1]:.6f}]. For HRF, the corresponding means were {h['baseline_dice']['mean']:.6f} and {h['refined_dice']['mean']:.6f}, with a paired difference of {h['delta']['mean']:.6f} and interval [{h['delta']['bootstrap95ci'][0]:.6f}, {h['delta']['bootstrap95ci'][1]:.6f}]. Refinement underperformed the baseline in these new configurations.

We averaged 20 sampling realisations per image and used 50,000 paired bootstrap draws. CHASE-DB1 resampling retained both eyes of each of the four subjects together. HRF resampling was stratified by pathology at image level; patient identifiers were unavailable, so image independence cannot be verified. All intervals condition on one training run and do not include variability between training runs. With such small samples, the intervals are exploratory, especially with only four CHASE subject clusters. The detailed results and protocol are supplied in a separate analysis report. The manuscript has not been changed at this stage.
'''
(OUT/'reviewer_small_test_sets_response.txt').write_text(reply,encoding='utf8')
print(json.dumps(dict(report=str(OUT/'per_image_and_confidence_intervals.txt'),response=str(OUT/'reviewer_small_test_sets_response.txt'),summary=summary,verified_rows=verified,manuscript_unchanged=after==before),indent=2))
