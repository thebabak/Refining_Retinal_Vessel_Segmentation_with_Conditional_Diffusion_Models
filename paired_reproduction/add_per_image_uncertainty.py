"""Add scoped per-image analyses; preserve original results and their provenance."""
import csv,json,re,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CHASE=Path(r'C:\Users\FS-GAMING PRO\.codex\attachments\8ef7711f-d75c-4638-8520-8c5b89744f9a\Pasted text.txt')
HRF=ROOT/'paired_reproduction/paired_rerun_HRF_2026-10-06_17-09-02/paired_metrics.csv'
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT=ROOT/'paired_reproduction'/f'manuscript_per_image_analysis_{stamp}';OUT.mkdir()
results={};per={}
for dataset,source in [('CHASE-DB1',CHASE),('HRF',HRF)]:
 raw=list(csv.DictReader(source.open(encoding='utf-8-sig')))
 rows=[r for r in raw if r['steps']=='50'];per[dataset]=[]
 for name in sorted({r['image'] for r in rows}):
  rr=[r for r in rows if r['image']==name];assert len(rr)==20
  for r in rr:
   for prefix in ('baseline','refined'):
    tp,fp,fn=[int(r[f'{prefix}_{k}']) for k in ('tp','fp','fn')]
    assert abs(2*tp/max(2*tp+fp+fn,1)-float(r[f'{prefix}_dice']))<1e-12
  base=float(np.mean([float(r['baseline_dice']) for r in rr]));ref=float(np.mean([float(r['refined_dice']) for r in rr]))
  per[dataset].append(dict(image=name,unit=rr[0]['unit'],pathology=rr[0]['pathology'],baseline=base,refined=ref,delta=ref-base))
 keys=('baseline','refined','delta');rng=np.random.default_rng(20261007)
 if dataset=='CHASE-DB1':
  units=sorted({r['unit'] for r in per[dataset]});assert len(units)==4
  values=np.array([[np.mean([r[k] for r in per[dataset] if r['unit']==u]) for k in keys] for u in units])
  boot=values[rng.integers(0,4,(50000,4))].mean(1)
 else:
  strata=[np.array([[r[k] for k in keys] for r in per[dataset] if r['pathology']==g]) for g in ('h','dr','g')]
  assert all(len(v)==4 for v in strata)
  boot=np.mean([v[rng.integers(0,4,(50000,4))].mean(1) for v in strata],axis=0)
  values=np.array([[r[k] for k in keys] for r in per[dataset]])
 mean=values.mean(0);ci=np.quantile(boot,[.025,.975],axis=0)
 results[dataset]={k:dict(mean=float(mean[i]),CI95=ci[:,i].tolist()) for i,k in enumerate(keys)}
def table(dataset,label,caption):
 rows='\n'.join(r['image'].replace('_',r'\_')+' & '+ ' & '.join(f"{r[k]:.6f}" for k in ('baseline','refined','delta'))+r' \\' for r in per[dataset])
 mean=results[dataset];rows+='\nMean & '+' & '.join(f"{mean[k]['mean']:.6f}" for k in ('baseline','refined','delta'))+r' \\'
 return r'''\begin{table}[htbp]
\centering
\caption{CAPTION}
\label{LABEL}
\small
\begin{tabular}{lrrr}
\toprule
Image & Baseline Dice & Refined Dice & Paired difference \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
'''.replace('CAPTION',caption).replace('LABEL',label).replace('ROWS',rows)
ci_rows='\n'.join(dataset+' & '+f"{results[dataset]['delta']['mean']:.6f} & $[{results[dataset]['delta']['CI95'][0]:.6f},\\,{results[dataset]['delta']['CI95'][1]:.6f}]$"+r' \\' for dataset in results)
section=r'''
\subsection{Per-Image Results and Small-Test-Set Uncertainty}
\label{sec:per_image_uncertainty}

Small test sets limit dataset-level conclusions: the stated CHASE-DB1 partition contains eight images and the HRF partition twelve. We therefore report supplementary paired per-image tabulations and exploratory uncertainty estimates at 50 DDIM transitions. These analyses must be distinguished from the original dataset-level comparisons: the supplied corrected CHASE tabulation uses a baseline mean of 0.812244, rather than the original 0.8001, and the available HRF reproduction uses newly trained refinement checkpoints. Neither analysis reconstructs the missing original per-image records or supplies confidence intervals for the original reported gains.

For each image, refined Dice was averaged over 20 sampling realisations before comparison with its baseline Dice. Sampling realisations are not independent test images. The CHASE values in Table~\ref{tab:chase_per_image_corrected} come from the supplied corrected evaluation tabulation. Dice was checked algebraically against its listed TP, FP, and FN counts. The provenance of those corrected counts has not been resolved against prediction masks; consequently this CHASE analysis is conditional on the supplied tabulation and is not independently verified experimental evidence. The HRF values in Table~\ref{tab:hrf_per_image_reproduction} come from the saved executable reproduction evaluation records, with a threshold of 0.5 inside the supplied FOV masks.

CHASE_TABLE
HRF_TABLE

We used 50,000 paired percentile-bootstrap draws with random seed 20261007. CHASE resampling used four subject clusters, retaining both eyes together; the baseline and refined scores were resampled as pairs. HRF resampling used images within the healthy, diabetic-retinopathy, and glaucoma strata (four test images each). HRF patient identifiers were unavailable, so independence between images cannot be verified and its intervals are image-level, not validated patient-level intervals. The bootstrap conditions on the fixed fitted models and sampling averages; it does not estimate variability across training runs. With only four CHASE subject clusters, bootstrap coverage may be unreliable. All intervals should therefore be interpreted as exploratory.

\begin{table}[htbp]
\centering
\caption{Exploratory 95\% paired bootstrap intervals for refinement-minus-baseline mean Dice at 50 transitions. CHASE is conditional on the supplied corrected tabulation; HRF refers to the separate saved reproduction. Neither interval applies to the original dataset-level results.}
\label{tab:per_image_paired_intervals}
\small
\begin{tabular}{lrl}
\toprule
Dataset & Mean paired difference & 95\% interval \\
\midrule
CI_ROWS
\bottomrule
\end{tabular}
\end{table}

The corrected CHASE tabulation has a positive mean paired difference, although two of eight image-level differences are negative. This conditional calculation does not resolve the provenance limitation. In the available HRF reproduction, refinement underperformed the baseline in all twelve test images. The contrasting results do not establish consistent improvement across datasets or statistical significance for the original CHASE and HRF comparisons. Larger subject-disjoint test samples, verified paired predictions, and repeated training runs remain necessary for robust conclusions.

'''.replace('CHASE_TABLE',table('CHASE-DB1','tab:chase_per_image_corrected','Supplied corrected CHASE-DB1 tabulation at 50 transitions; refined scores average 20 sampling realisations per image. Prediction-mask verification of the corrected counts remains unavailable.')).replace('HRF_TABLE',table('HRF','tab:hrf_per_image_reproduction','Per-image HRF reproduction results at 50 transitions; refined scores average 20 sampling realisations per image. These are separate from the original HRF results.')).replace('CI_ROWS',ci_rows)
assert sorted(re.findall(r'\\begin\{([^}]+)\}',section))==sorted(re.findall(r'\\end\{([^}]+)\}',section))
path=ROOT/'template261006.tex';old=path.read_text(encoding='utf8');assert '\\label{sec:per_image_uncertainty}' not in old
marker='\\subsection{Contribution of the Dice Supervision Term}';assert old.count(marker)==1
new=old.replace(marker,section+marker)
new=new.replace('Per-image paired results, confidence intervals, and repeated training runs remain necessary to assess statistical significance and robustness.','The supplementary per-image analyses in Section~\\ref{sec:per_image_uncertainty} have distinct data provenance and do not establish statistical significance for these original comparisons. Verified original paired results and repeated training runs remain necessary to assess robustness.')
# Every pre-existing table must remain unchanged.
for label in re.findall(r'\\label\{(tab:[^}]+)\}',old):
 def extract(text):
  k=text.index('\\label{'+label+'}');a=text.rfind('\\begin{table}',0,k);b=text.index('\\end{table}',k)+len('\\end{table}');return text[a:b]
 assert extract(old)==extract(new),label
(ROOT/f'template261006_before_per_image_uncertainty_{stamp}.tex').write_text(old,encoding='utf8')
path.write_text(new,encoding='utf8');version=ROOT/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
audit=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (CHASE,HRF)},results=results,per_image=per,transitions=50,bootstrap_draws=50000,random_seed=20261007,CHASE_provenance='Supplied corrected table; same counts appeared as hypothetical in an earlier supplied file. Algebraic consistency verified, prediction-mask provenance unresolved.',HRF_provenance='Saved new executable reproduction; not original manuscript results',existing_tables='unchanged')
(OUT/'analysis_and_provenance.json').write_text(json.dumps(audit,indent=2))
c=results['CHASE-DB1']['delta'];h=results['HRF']['delta']
reply=f'''Thank you for this comment. We have added a separate subsection, "Per-Image Results and Small-Test-Set Uncertainty", with eight CHASE-DB1 and twelve HRF per-image baseline/refinement scores and exploratory 95% paired bootstrap intervals. Twenty sampling realisations were averaged per image before resampling. We used 50,000 bootstrap draws, retaining both eyes together within each CHASE subject; HRF image resampling was stratified by pathology.

The supplied corrected CHASE tabulation yields a mean paired difference of {c['mean']:.6f} and an exploratory interval [{c['CI95'][0]:.6f}, {c['CI95'][1]:.6f}]. Its corrected confusion counts are algebraically consistent with Dice but have not been verified against prediction masks; this is explicitly reported as a conditional tabulation analysis. The separate saved HRF reproduction yields a difference of {h['mean']:.6f} and interval [{h['CI95'][0]:.6f}, {h['CI95'][1]:.6f}], indicating lower Dice after refinement in that configuration.

Neither analysis supplies uncertainty estimates for the unavailable original per-image results. All intervals condition on fixed trained models and sampling averages; only four CHASE subject clusters are available, and HRF patient identifiers are unavailable. Accordingly, we treat the intervals as exploratory, distinguish their provenance from the original result tables, and retain qualified dataset-level conclusions. The original dataset result tables are unchanged.
'''
response=ROOT/f'reviewer_small_test_sets_response_{stamp}.txt';response.write_text(reply,encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),analysis=str(OUT/'analysis_and_provenance.json'),results=results,per_image_counts={k:len(v) for k,v in per.items()}),indent=2))
