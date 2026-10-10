"""Insert supplied DRIVE paired results without changing original result tables."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,math,re
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reviewer_comments/reviewer_2_comment_1_results_2026-10-07_16-36-00'
audit=json.loads((SOURCE/'paired_results_and_intervals.json').read_text())
summary=audit['summary'];rows=audit['per_image_results']['DRIVE']
assert len(rows)==20
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
path=ROOT/'template261006.tex';old=path.read_text(encoding='utf8')
assert '\\label{tab:drive_per_image_supplied}' not in old
def upward(x):return f'{math.ceil(x*1000)/1000:.3f}'
table_rows='\n'.join(r['image'].replace('_',r'\_')+' & '+' & '.join(upward(r[k]) for k in ('baseline_dice','refined_dice','delta'))+r' \\' for r in rows)
table_rows+='\nMean & '+' & '.join(upward(summary['DRIVE']['means'][k]) for k in ('baseline_dice','refined_dice','delta'))+r' \\'
coarse_rows=[]
for dataset in ('CHASE-DB1','DRIVE'):
 for prefix,label in [('baseline','Coarse baseline'),('refined','Refinement')]:
  coarse_rows.append(dataset+' & '+label+' & '+' & '.join(upward(summary[dataset]['other_metric_means'][f'{prefix}_{m}']) for m in ('dice','iou','sensitivity','specificity','accuracy'))+r' \\')
add=r'''
The supplied DRIVE evaluation table additionally provides paired baseline and refinement records for the same twenty image identifiers, with three sampling realisations per image. The five listed segmentation metrics were recomputed from TP, FP, FN, and TN and found algebraically consistent. Within each image, baseline and refinement counts have matching totals for reference-positive and reference-negative pixels. These scalar checks support pairing within the tabulation but do not independently verify common checkpoints, annotation sources, FOV masks, thresholds, or the provenance of the underlying predictions. Table~\ref{tab:drive_per_image_supplied} reports the image-level means, and Table~\ref{tab:supplied_paired_metric_means} reports the corresponding coarse and refined metrics for CHASE and DRIVE.

\begin{table}[htbp]
\centering
\caption{Supplied DRIVE paired results at 50 DDIM transitions. Each image-level value averages three sampling realisations. All values in this table are rounded upward to three decimals for display; means and paired differences are calculated before rounding. Full-precision values are retained for statistical analysis.}
\label{tab:drive_per_image_supplied}
\small
\begin{tabular}{lrrr}
\toprule
Image & Baseline Dice & Refined Dice & Paired difference \\
\midrule
DRIVE_ROWS
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Paired mean metrics from the supplied CHASE and DRIVE evaluations at 50 transitions, with upward rounding to three decimals for display. These tabulations are separate from the original dataset-level result tables; interval calculations use unrounded values.}
\label{tab:supplied_paired_metric_means}
\scriptsize
\begin{tabular}{llrrrrr}
\toprule
Dataset & Method & Dice & IoU & Sensitivity & Specificity & Accuracy \\
\midrule
METRIC_ROWS
\bottomrule
\end{tabular}
\end{table}

For DRIVE, unrounded mean Dice was 0.787935379 for the coarse baseline and 0.834482557 for refinement. The mean paired increase was 0.046547178, with an exploratory 95\% paired image-bootstrap interval of $[0.045769681,\,0.047272189]$ using 50,000 draws and random seed 20261007. All twenty image-level mean Dice differences were positive. This interval averages the three sampling realisations within each image before resampling image pairs. It assumes independence between test images; patient identity cannot be verified from the listed image identifiers alone. As for CHASE, the interval conditions on the supplied fitted-model outputs and sampling averages and does not include variability across training runs. The raw DRIVE mean 0.834482557 is displayed as 0.835 only under the explicitly stated upward-rounding convention.

The supplied CHASE tabulation has mean Dice 0.812243575 for the baseline and 0.834999296 for refinement, with mean paired increase 0.022755721 and exploratory subject-bootstrap interval $[0.001685607,\,0.043825834]$. Six of eight image-level mean differences were positive. The corresponding supplied per-image evaluation for the original HRF refinement result of 0.793 remains unavailable; the separate HRF reproduction table does not resolve that comparison. Thus these additions address the numerical reporting for CHASE and DRIVE but do not establish protocol provenance or complete the paired assessment for the original HRF result.

'''.replace('DRIVE_ROWS',table_rows).replace('METRIC_ROWS','\n'.join(coarse_rows))
assert sorted(re.findall(r'\\begin\{([^}]+)\}',add))==sorted(re.findall(r'\\end\{([^}]+)\}',add))
marker='\\subsection{Contribution of the Dice Supervision Term}';assert old.count(marker)==1
new=old.replace(marker,add+marker)
new=new.replace('the stated CHASE-DB1 partition contains eight images and the HRF partition twelve.','the stated CHASE-DB1 partition contains eight images, DRIVE twenty, and HRF twelve.')
new=new.replace('For each image, refined Dice was averaged over 20 sampling realisations before comparison with its baseline Dice.','For CHASE and the separate HRF reproduction, refined Dice was averaged over 20 sampling realisations per image before comparison with its baseline Dice; the supplied DRIVE tabulation averages three realisations per image.')
new=new.replace('CHASE is conditional on the supplied corrected tabulation; HRF refers to the separate saved reproduction. Neither interval applies to the original dataset-level results.','CHASE and DRIVE are conditional on their supplied tabulations; HRF refers to the separate saved reproduction. These intervals apply only to the specified evaluations.')
needle='CHASE-DB1 & 0.022756 & $[0.001686,\\,0.043826]$ \\\\'
assert new.count(needle)==1
new=new.replace(needle,needle+'\nDRIVE & 0.046547 & $[0.045770,\\,0.047272]$ \\\\')
for label in re.findall(r'\\label\{(tab:[^}]+)\}',old):
 if label=='tab:per_image_paired_intervals':continue
 def extract(text):
  k=text.index('\\label{'+label+'}');a=text.rfind('\\begin{table}',0,k);b=text.index('\\end{table}',k)+len('\\end{table}');return text[a:b]
 assert extract(old)==extract(new),label
(ROOT/f'template261006_before_drive_paired_results_{stamp}.tex').write_text(old,encoding='utf8')
path.write_text(new,encoding='utf8');version=ROOT/f'template261006_{stamp}.tex';version.write_text(new,encoding='utf8')
reply=(SOURCE/'reviewer_2_comment_1_response_draft.txt').read_text(encoding='utf8')
reply=reply.replace('We have prepared paired per-image analyses','We have added paired per-image analyses in the subsection "Per-Image Results and Small-Test-Set Uncertainty", with tables of per-image Dice, baseline/refinement metrics, and paired confidence intervals,')
reply=reply.replace('This extraction has not changed the manuscript or established that every part of the reviewer\'s request is resolved.','These analyses have been added to the manuscript. The original dataset-level result tables are unchanged, and the HRF requirement remains outstanding. Upward rounding is explicitly labelled for displayed three-decimal tables; all statistics use full-precision values.')
response=ROOT/'reviewer_comments'/f'reviewer_2_comment_1_response_{stamp}.txt';response.write_text(reply,encoding='utf8')
index=ROOT/'reviewer_comments/reviewer_2_comments.json';registry=json.loads(index.read_text(encoding='utf8'))
(index.parent/f'reviewer_2_comments_before_update_{stamp}.json').write_text(index.read_text(encoding='utf8'),encoding='utf8')
registry['comments'][0]['status']='Partial: CHASE and DRIVE supplied per-image results, paired metrics and exploratory confidence intervals added to manuscript; original HRF paired table and independent protocol verification outstanding.'
registry['comments'][0]['manuscript_version']=str(version);registry['comments'][0]['response_file']=str(response)
index.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(version=str(version),response=str(response),new_drive_images=len(rows),original_dataset_tables='unchanged',rounding='upward3display only; inference full precision'),indent=2))
