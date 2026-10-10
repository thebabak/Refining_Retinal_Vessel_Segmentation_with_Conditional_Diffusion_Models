"""Update only the supplementary paired-results section from the supplied table."""
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'reviewer_comments/uploaded_full_paired_tests_2026-10-08_21-12-51'
data = json.loads((ANALYSIS/'paired_tests.json').read_text())
means = list(csv.DictReader((ANALYSIS/'mean_and_SD_by_steps.txt').open()))
raw = list(csv.DictReader((ANALYSIS/'source_table.txt').open()))
per_image = list(csv.DictReader((ANALYSIS/'per_image_50steps.txt').open()))
names = {'CHASE':'CHASE-DB1', 'DRIVE':'DRIVE', 'HRF':'HRF'}


def latex_row(cells):
    return ' & '.join(cells)+r' \\'


def table(caption,label,columns,header,body):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',
                      r'\label{'+label+'}',r'\scriptsize',r'\begin{tabular}{'+columns+'}',
                      r'\toprule',latex_row(header),r'\midrule',*body,r'\bottomrule',
                      r'\end{tabular}',r'\end{table}',''])


def number(value):
    return f'{float(value):.9f}'


def p_value(value):
    if value>=.0001:
        return f'{value:.6f}'
    coefficient,exponent=f'{value:.3e}'.split('e')
    return '$'+coefficient+r'\times10^{'+str(int(exponent))+'}$'


section = r'''\subsection{Paired Results, Repeated Seeds, and Small-Test-Set Uncertainty}
\label{sec:per_image_uncertainty}

We analyse the current author-supplied evaluation table containing 5000 records: 1000 for CHASE-DB1, 2500 for DRIVE, and 1500 for HRF. Each test image has five labels in the training-seed column, five sampling-seed labels, and five transition settings (10, 20, 30, 50, and 100). Both seed columns use 42, 123, 2026, 3000, and 3001. The table explicitly separates these two levels; sampling realisations are not treated as independent training runs or independent test images. The statistical analysis treats the training-seed labels as paired run units, conditional on their representing independent training executions; the associated execution records were not independently inspected.

The authors confirm that the coarse and refined predictions used the same test images, ground-truth masks, preprocessing, image resolution, field-of-view masks, binarisation thresholds, and Dice calculation. The refiner received the same coarse predictions used to calculate the paired baseline metrics. All five segmentation metrics and paired differences in the current table are numerically consistent with its TP, FP, FN, and TN counts. This is a numerical check of the supplied tabulation, rather than independent verification against prediction masks.

This analysis supersedes the earlier supplementary paired tabulations in this section and must be distinguished from the original single-run dataset and transition-count tables elsewhere in the manuscript. It also differs from the separately executed local reproduction. In particular, its baseline means are 0.810514 on CHASE-DB1, 0.791814 on DRIVE, and 0.732121 on HRF; these should not be substituted into comparisons with unrelated predictions.

We first average the five sampling realisations within each image and training label. Each training-label mean then averages the fixed test images. Table~\ref{tab:paired_five_seed_means} reports refined Dice at 50 transitions by training label, and Table~\ref{tab:uploaded_step_mean_sd} reports means and sample standard deviations across the five training-label means at every transition setting. These SDs describe the supplied run-level variation on a fixed test set, not uncertainty across patients.

'''
body=[]
for dataset in names:
    scores=[]
    for seed in [42,123,2026,3000,3001]:
        scores.append(sum(float(r['refined_dice']) for r in raw if r['dataset']==dataset and r['seed']==str(seed) and r['steps']=='50')/
                      sum(r['dataset']==dataset and r['seed']==str(seed) and r['steps']=='50' for r in raw))
    body.append(latex_row([names[dataset]]+[number(v) for v in scores]))
section+=table('Refined mean Dice at 50 transitions by training-seed label, after averaging sampling realisations and test images.','tab:paired_five_seed_means','lrrrrr',['Dataset','42','123','2026','3000','3001'],body)
body=[]
for r in means:
    body.append(latex_row([names[r['dataset']],r['steps']]+[number(r[k]) for k in ['baseline_mean','baseline_sd_across_training_labels','refined_mean','refined_sd_across_training_labels']]))
section+=table('Current supplied-table mean Dice and SD across five training-label means. The five sampling realisations are averaged within each run. Calculations retain full precision.','tab:uploaded_step_mean_sd','lrrrrr',['Dataset','Steps','Baseline mean','Baseline SD','Refined mean','Refined SD'],body)
section+=r'''
For the three planned comparisons on each dataset---baseline versus 50 transitions, 10 versus 50, and 50 versus 100---we use two-sided paired $t$ tests and exact Wilcoxon signed-rank tests on the five training-label means. The signed-rank probabilities are obtained by exhaustive sign enumeration, excluding zero differences and averaging tied absolute ranks. Holm adjustment is applied separately to each test type across the nine comparisons. The $t$ test assumes approximately normal independent run-level differences; the signed-rank interpretation assumes independent units and symmetric differences. Neither assumption can be assessed reliably with only five runs. We report both analyses rather than choosing the test giving a preferred conclusion.

'''
body=[]
comparison_names={'baseline_vs_50':'Baseline to 50','10_vs_50':'10 to 50','50_vs_100':'50 to 100'}
for r in data['training_label_tests']:
    body.append(latex_row([names[r['dataset']],comparison_names[r['comparison']],number(r['mean_delta']),
                           p_value(r['holm_paired_t_p']),p_value(r['holm_exact_wilcoxon_p'])]))
section+=table('Paired comparisons across five training-label means. Differences are second setting minus first. The two $p$ columns are separately Holm-adjusted over nine comparisons per test type.','tab:uploaded_training_paired_tests','llrrr',['Dataset','Comparison','Mean difference','Holm $p_t$','Holm $p_W$'],body)
section+=r'''
Baseline--refinement differences are positive on all three datasets and have Holm-adjusted paired $t$ probabilities below 0.05, conditional on the stated run-level assumptions. However, all three corresponding exact signed-rank probabilities are 0.0625 before adjustment and 0.5625 after adjustment. With five nonzero untied pairs, 0.0625 is the smallest attainable two-sided exact signed-rank probability. Thus the parametric and nonparametric analyses do not support identical significance claims.

None of the 10-versus-50 or 50-versus-100 comparisons reaches 0.05 in either adjusted test. The 50-to-100 mean changes are only 0.000000480 for CHASE-DB1, 0.000000053 for DRIVE, and 0.000000902 for HRF. These current-table results do not establish a reliable benefit of additional transitions or equivalence between settings. They do not explain the original single-run increase from 0.835 to 0.836, which remains a distinct result and may reflect sampling variation.

To assess test-unit variation, we additionally average all training and sampling realisations within each image. CHASE testing averages both eyes within each of four subjects; DRIVE and HRF use twenty and twelve image units. Patient independence cannot be verified for DRIVE or HRF from the supplied identifiers. Secondary two-sided tests use separate Holm correction across the three datasets for each test type; these tests answer a different question from the run-level comparisons.

'''
body=[]
for r in data['test_unit_tests']:
    body.append(latex_row([names[r['dataset']],str(r['n_pairs']),p_value(r['holm_paired_t_p']),p_value(r['holm_exact_wilcoxon_p'])]))
section+=table('Secondary subject/image-level paired tests at 50 transitions. Both test families are separately Holm-adjusted across three datasets. CHASE units are subjects; DRIVE and HRF units are images with unverified patient independence.','tab:paired_statistical_tests','lrrr',['Dataset','Units','Holm $p_t$','Holm $p_W$'],body)
section+=r'''
The CHASE subject-level comparison does not reach 0.05 in either test ($p_t=0.098927$ and $p_W=0.125$ after adjustment). DRIVE and HRF reach this threshold under the stated image-level assumptions. These conditional findings should not be described as broad patient-level robustness.

Exploratory 95\% percentile-bootstrap intervals use 50,000 draws with random seed 20261008, jointly resampling training labels and test units after averaging sampling realisations. CHASE resampling retains both eyes together within subjects. DRIVE resamples images; HRF resamples images within pathology strata. Table~\ref{tab:per_image_paired_intervals} reports baseline--refinement differences at 50 transitions. With five training labels and only four CHASE subjects, bootstrap coverage may be unreliable. A positive bootstrap interval should not override the nonsignificant subject-level tests.

'''
body=[]
for r in data['training_label_tests']:
    if r['comparison']!='baseline_vs_50':continue
    m=next(m for m in means if m['dataset']==r['dataset'] and m['steps']=='50')
    lo,hi=r['crossed_bootstrap_95ci']
    body.append(latex_row([names[r['dataset']],f"{float(m['baseline_mean']):.6f}",f"{float(m['refined_mean']):.6f}",f"{r['mean_delta']:+.6f}",f'$[{lo:.6f},\\,{hi:.6f}]$']))
section+=table('Current supplied-table paired mean Dice at 50 transitions and exploratory crossed training-label/test-unit bootstrap intervals for the difference.','tab:per_image_paired_intervals','lrrrl',['Dataset','Baseline','Refined','Difference','95\\% interval'],body)
labels={'CHASE':'tab:chase_per_image_corrected','DRIVE':'tab:drive_per_image_supplied','HRF':'tab:hrf_per_image_supplied'}
for dataset in names:
    body=[]
    for r in per_image:
        if r['dataset']==dataset:
            body.append(latex_row([r['image'].replace('_',r'\_')]+[f'{math.ceil(float(r[k])*1000)/1000:.3f}' for k in ['baseline_dice','refined_dice','delta']]))
    section+=table(names[dataset]+' current supplied-table per-image means at 50 transitions, averaging 25 training/sampling realisations per image. Values are rounded upward to three decimals for display; statistical calculations use unrounded values.',labels[dataset],'lrrr',['Image','Baseline Dice','Refined Dice','Difference'],body)
body=[]
for dataset in names:
    selected=[r for r in raw if r['dataset']==dataset and r['steps']=='50']
    for prefix,method in [('baseline','Coarse baseline'),('refined','Refinement')]:
        values=[sum(float(r[prefix+'_'+k]) for r in selected)/len(selected) for k in ['dice','iou','sensitivity','specificity','accuracy']]
        body.append(latex_row([names[dataset],method]+[f'{math.ceil(v*1000)/1000:.3f}' for v in values]))
section+=table('Mean segmentation metrics at 50 transitions from the current supplied table, rounded upward to three decimals for display.','tab:supplied_paired_metric_means','llrrrrr',['Dataset','Method','Dice','IoU','Sensitivity','Specificity','Accuracy'],body)
section+='\nComplete per-image, training-label, sampling-seed and transition records, including confusion counts, are preserved in the accompanying machine-readable source table. Repeated sampling does not increase the independent test-subject count. Larger independent test cohorts and execution-level confirmation of the supplied training labels remain necessary for strong generalisation claims.\n\n'

manuscript=ROOT/'template261006.tex'
original=manuscript.read_bytes()
text=original.decode('utf-8-sig').replace('\r\n','\n')
start=text.index(r'\subsection{Per-Image Results and Small-Test-Set Uncertainty}')
end=text.index(r'\subsection{Contribution of the Dice Supervision Term}',start)
updated=text[:start]+section+text[end:]
assert section.count(r'\begin{table}')==section.count(r'\end{table}')==9
for label in ['tab:paired_five_seed_means','tab:uploaded_step_mean_sd','tab:uploaded_training_paired_tests','tab:paired_statistical_tests','tab:per_image_paired_intervals',*labels.values(),'tab:supplied_paired_metric_means']:
    assert updated.count(r'\label{'+label+'}')==1
assert updated[:start]==text[:start] and updated.endswith(text[end:])
stamp=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
backup=ROOT/('template261006_before_uploaded_paired_analysis_'+stamp+'.tex')
version=ROOT/('template261006_'+stamp+'.tex')
assert not backup.exists() and not version.exists()
backup.write_bytes(original)
version.write_text(updated,encoding='utf-8')
manuscript.write_bytes(version.read_bytes())
assert manuscript.read_bytes()==version.read_bytes()
(ANALYSIS/'manuscript_incorporation.json').write_text(json.dumps(dict(version=str(version),prior_version_archive=str(backup),scope='Replaced supplementary paired-results section only; original single-run tables unchanged; source labels not independently execution-verified',compiled=False),indent=2))
print(version)
print('Nine balanced tables, unique labels; all other sections unchanged. PDF not compiled.')
