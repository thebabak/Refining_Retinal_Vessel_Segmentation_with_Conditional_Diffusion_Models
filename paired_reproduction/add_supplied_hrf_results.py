"""Use the accepted HRF tabulation for supplementary paired reporting."""
import csv,json,re,math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'C:\Users\FS-GAMING PRO\.codex\attachments\83d91f7b-c1fb-469a-a22b-f42aab23d9ab\Pasted text.txt')
ANALYSIS=ROOT/'reviewer_comments/reviewer_2_comment_1_HRF_update_2026-10-07_17-07-19'
audit=json.loads((ANALYSIS/'HRF_paired_results_and_intervals.json').read_text());per=audit['per_image'];assert len(per)==12
raw=[r for r in csv.DictReader(SOURCE.open(encoding='utf-8-sig')) if r['steps']=='50'];assert len(raw)==240
metrics=['dice','iou','sensitivity','specificity','accuracy']
means={f'{prefix}_{m}':float(np.mean([float(r[f'{prefix}_{m}']) for r in raw])) for prefix in ('baseline','refined') for m in metrics}
def up(v):return f'{math.ceil(v*1000)/1000:.3f}'
rows='\n'.join(r['image'].replace('_',r'\_')+' & '+' & '.join(up(r[k]) for k in ('baseline','refined','delta'))+r' \\' for r in per)
rows+='\nMean & '+' & '.join(up(audit['means'][k]) for k in ('baseline','refined','delta'))+r' \\'
table=r'''\begin{table}[htbp]
\centering
\caption{Supplied HRF paired results at 50 DDIM transitions, averaging 20 sampling realisations per image. Values are rounded upward to three decimals for display; mean differences and confidence intervals are calculated from unrounded values.}
\label{tab:hrf_per_image_supplied}
\small
\begin{tabular}{lrrr}
\toprule
Image & Baseline Dice & Refined Dice & Paired difference \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}'''.replace('ROWS',rows)
path=ROOT/'template261006.tex';old=path.read_text(encoding='utf8');new=old
idx=new.index('\\label{tab:hrf_per_image_reproduction}');a=new.rfind('\\begin{table}',0,idx);b=new.index('\\end{table}',idx)+len('\\end{table}')
new=new[:a]+table+new[b:]
old_intro='Small test sets limit dataset-level conclusions: the stated CHASE-DB1 partition contains eight images, DRIVE twenty, and HRF twelve. We therefore report supplementary paired per-image tabulations and exploratory uncertainty estimates at 50 DDIM transitions. These analyses must be distinguished from the original dataset-level comparisons: the supplied corrected CHASE tabulation uses a baseline mean of 0.812244, rather than the original 0.8001, and the available HRF reproduction uses newly trained refinement checkpoints. Neither analysis reconstructs the missing original per-image records or supplies confidence intervals for the original reported gains.'
new_intro='Small test sets limit dataset-level conclusions: the stated CHASE-DB1 partition contains eight images, DRIVE twenty, and HRF twelve. We report supplementary paired per-image tabulations and exploratory uncertainty estimates at 50 DDIM transitions using the supplied evaluation tables. These analyses must be distinguished from the original dataset-level comparisons: the supplied corrected CHASE tabulation uses a baseline mean of 0.812244, rather than the original 0.8001. The intervals apply to the specified supplied evaluations; the scalar tables alone do not independently verify that their checkpoint and protocol records match the original runs.'
assert new.count(old_intro)==1;new=new.replace(old_intro,new_intro)
new=new.replace('For CHASE and the separate HRF reproduction, refined Dice was averaged over 20 sampling realisations per image','For the supplied CHASE and HRF evaluations, refined Dice was averaged over 20 sampling realisations per image')
old_sentence='The HRF values in Table~\\ref{tab:hrf_per_image_reproduction} come from the saved executable reproduction evaluation records, with a threshold of 0.5 inside the supplied FOV masks.'
new_sentence='The HRF values in Table~\\ref{tab:hrf_per_image_supplied} come from the supplied corrected HRF evaluation table. Its Dice, IoU, sensitivity, specificity, and accuracy values are consistent with its listed confusion counts. This establishes algebraic consistency, not independent verification of prediction masks or full evaluation-protocol equivalence.'
assert new.count(old_sentence)==1;new=new.replace(old_sentence,new_sentence)
new=new.replace('CHASE and DRIVE are conditional on their supplied tabulations; HRF refers to the separate saved reproduction. These intervals apply only to the specified evaluations.','All three intervals are conditional on their supplied evaluation tabulations and apply only to those specified evaluations.')
needle='HRF & -0.077483 & $[-0.081026,\\,-0.073712]$ \\\\';assert new.count(needle)==1
new=new.replace(needle,'HRF & 0.060530 & $[0.056320,\\,0.064632]$ \\\\')
old_interp='The corrected CHASE tabulation has a positive mean paired difference, although two of eight image-level differences are negative. This conditional calculation does not resolve the provenance limitation. In the available HRF reproduction, refinement underperformed the baseline in all twelve test images. The contrasting results do not establish consistent improvement across datasets or statistical significance for the original CHASE and HRF comparisons. Larger subject-disjoint test samples, verified paired predictions, and repeated training runs remain necessary for robust conclusions.'
new_interp='The supplied tabulations have positive mean paired differences on all three datasets. Six of eight CHASE image-level mean differences are positive, whereas all twenty DRIVE and all twelve HRF image-level mean differences are positive. These conditional calculations do not resolve prediction provenance or establish broad dataset-level robustness. Larger subject-disjoint test samples, verified paired predictions, and repeated training runs remain necessary for robust conclusions.'
assert new.count(old_interp)==1;new=new.replace(old_interp,new_interp)
old_end='The corresponding supplied per-image evaluation for the original HRF refinement result of 0.793 remains unavailable; the separate HRF reproduction table does not resolve that comparison. Thus these additions address the numerical reporting for CHASE and DRIVE but do not establish protocol provenance or complete the paired assessment for the original HRF result.'
new_end='The supplied HRF tabulation gives unrounded mean Dice 0.732470205 for the baseline and 0.792999964 for refinement, with a mean paired increase of 0.060529759 and an exploratory pathology-stratified paired image-bootstrap interval of $[0.056319538,\\,0.064631563]$. All twelve image-level mean differences are positive. These tabulations address paired numerical reporting for all three datasets but do not independently establish protocol or prediction provenance. The HRF interval assumes independent images within pathology strata and does not include training-run variability.'
assert new.count(old_end)==1;new=new.replace(old_end,new_end)
# Add HRF rows to the paired all-metric table, retaining all existing CHASE/DRIVE rows.
idx=new.index('\\label{tab:supplied_paired_metric_means}');bottom=new.index('\\bottomrule',idx)
metric_rows='\n'.join('HRF & '+label+' & '+' & '.join(up(means[f'{prefix}_{m}']) for m in metrics)+r' \\' for prefix,label in [('baseline','Coarse baseline'),('refined','Refinement')])
new=new[:bottom]+metric_rows+'\n'+new[bottom:]
new=new.replace('Paired mean metrics from the supplied CHASE and DRIVE evaluations at 50 transitions','Paired mean metrics from the supplied CHASE, DRIVE, and HRF evaluations at 50 transitions')
new=new.replace('reports the corresponding coarse and refined metrics for CHASE and DRIVE.','reports the corresponding coarse and refined metrics for CHASE, DRIVE, and HRF.')
changed={'tab:hrf_per_image_reproduction','tab:per_image_paired_intervals','tab:supplied_paired_metric_means'}
for label in re.findall(r'\\label\{(tab:[^}]+)\}',old):
 if label in changed:continue
 def extract(t):
  k=t.index('\\label{'+label+'}');i=t.rfind('\\begin{table}',0,k);j=t.index('\\end{table}',k)+len('\\end{table}');return t[i:j]
 assert extract(old)==extract(new),label
assert 'HRF & -0.077483' not in new and 'tab:hrf_per_image_reproduction' not in new
assert sorted(re.findall(r'\\begin\{([^}]+)\}',table))==sorted(re.findall(r'\\end\{([^}]+)\}',table))
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
(ROOT/f'template261006_before_supplied_hrf_{stamp}.tex').write_text(old,encoding='utf8')
path.write_text(new,encoding='utf8');version=ROOT/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
reply=(ANALYSIS/'reviewer_2_comment_1_response.txt').read_text(encoding='utf8')
reply=reply.replace('We provide paired per-image baseline and refinement results','We have added paired per-image baseline and refinement results in the subsection "Per-Image Results and Small-Test-Set Uncertainty", including tables of per-image Dice, coarse/refined metrics, and exploratory confidence intervals,')
# Replace awkward first paragraph with a complete standalone sentence.
first=reply.split('\n',1)[0]
replacement='Thank you for this comment. We have added paired per-image baseline and refinement results for CHASE-DB1, DRIVE, and HRF in the subsection "Per-Image Results and Small-Test-Set Uncertainty". The tables report per-image Dice, coarse and refined metrics, mean paired differences, and exploratory 95% intervals from the supplied evaluations at 50 DDIM transitions. Sampling realisations were averaged within each image before analysis.'
reply=reply.replace(first,replacement,1)
response=ROOT/'reviewer_comments'/f'reviewer_2_comment_1_response_{stamp}.txt';response.write_text(reply,encoding='utf8')
index=ROOT/'reviewer_comments/reviewer_2_comments.json';registry=json.loads(index.read_text(encoding='utf8'))
(index.parent/f'reviewer_2_comments_before_HRF_update_{stamp}.json').write_text(index.read_text(encoding='utf8'),encoding='utf8')
registry['comments'][0]['status']='Paired numerical reporting added for supplied CHASE, DRIVE and HRF tables; independent same-protocol and prediction provenance verification remains outstanding.'
registry['comments'][0]['manuscript_version']=str(version);registry['comments'][0]['response_file']=str(response)
index.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),HRF_images=len(per),original_result_tables='unchanged',HRF_mean=means['refined_dice']),indent=2))
