"""Paired tests on the user-designated tables; no model training is implied."""
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
SOURCE = ROOT / 'reviewer_comments/paired_current_tables_2026-10-08_15-11-25'
OUT = ROOT / 'reviewer_comments' / ('paired_statistical_tests_' + STAMP)
OUT.mkdir()
results = {}
for dataset in ['CHASE-DB1', 'DRIVE', 'HRF']:
    with (SOURCE / (dataset + '_per_image_means.txt')).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if dataset == 'CHASE-DB1':
        subjects = sorted({r['subject'] for r in rows})
        values = np.array([statistics.mean(float(r['delta']) for r in rows
                                          if r['subject'] == s) for s in subjects])
        unit = 'subject (both eyes averaged)'
    else:
        values = np.array([float(r['delta']) for r in rows])
        unit = 'image (patient independence unverified)'
    t = ttest_1samp(values, 0, alternative='two-sided')
    w = wilcoxon(values, alternative='two-sided', method='exact')
    results[dataset] = dict(n_units=len(values), unit=unit,
                            mean_delta=float(values.mean()),
                            paired_t_statistic=float(t.statistic),
                            paired_t_p=float(t.pvalue),
                            wilcoxon_statistic=float(w.statistic),
                            exact_wilcoxon_p=float(w.pvalue),
                            differences=values.tolist())

# One prespecified family: three baseline-versus-refinement Wilcoxon tests.
ordered = sorted(results, key=lambda d: results[d]['exact_wilcoxon_p'])
running = 0.
for i, dataset in enumerate(ordered):
    running = max(running, (len(ordered)-i)*results[dataset]['exact_wilcoxon_p'])
    results[dataset]['holm_wilcoxon_p'] = min(running, 1.)

(OUT / 'paired_tests.json').write_text(json.dumps(results, indent=2))
lines = ['Two-sided paired tests after averaging realisations within images.',
         'CHASE: average both eyes within subject; n=4 subjects.',
         'DRIVE and HRF: image-level tests; patient independence unverified.',
         'Primary reported test: exact signed-rank Wilcoxon, Holm correction across datasets.',
         'Secondary paired t tests are exploratory, unadjusted.',
         'Signed-rank interpretation assumes independent units and symmetric differences.',
         'No five-run DDIM transition experiment was conducted by this analysis.', '']
for dataset, r in results.items():
    lines.append(f"{dataset}: n={r['n_units']}, t={r['paired_t_statistic']:.9g}, "
                 f"t p={r['paired_t_p']:.9g}, W={r['wilcoxon_statistic']:.9g}, "
                 f"exact Wilcoxon p={r['exact_wilcoxon_p']:.9g}, "
                 f"Holm p={r['holm_wilcoxon_p']:.9g}")
(OUT / 'paired_tests_report.txt').write_text('\n'.join(lines)+'\n')

body = []
for dataset, r in results.items():
    body.append(f"{dataset} & {r['n_units']} & {r['paired_t_statistic']:.3f} & "
                f"{r['paired_t_p']:.6g} & {r['wilcoxon_statistic']:.0f} & "
                f"{r['exact_wilcoxon_p']:.6g} & {r['holm_wilcoxon_p']:.6g}" + r' \\')
addition = r'''
\paragraph{Exploratory paired statistical tests.}
After averaging the available realisations within each image, we compared baseline and refined Dice using two-sided paired tests. For CHASE-DB1, both eyes were averaged within each subject, giving four independent subject units rather than eight image units. DRIVE and HRF use twenty and twelve image units, respectively; patient independence remains unverified. Table~\ref{tab:paired_statistical_tests} reports exact Wilcoxon signed-rank tests, with Holm correction across the three dataset comparisons. Secondary paired $t$ tests are reported as exploratory unadjusted analyses. The signed-rank interpretation assumes independent units and approximately symmetric paired differences; the $t$ test assumes approximately normal unit-level differences. These assumptions cannot be reliably assessed with four CHASE subjects.

\begin{table}[htbp]
\centering
\caption{Exploratory two-sided paired baseline--refinement tests at 50 transitions, using the same current tables as the paired mean and bootstrap analyses. Holm adjustment applies to the three Wilcoxon tests. CHASE units are subjects; DRIVE and HRF units are images.}
\label{tab:paired_statistical_tests}
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
Dataset & Units & $t$ & $p_t$ & $W$ & $p_W$ & Holm $p_W$ \\
\midrule
''' + '\n'.join(body) + r'''
\bottomrule
\end{tabular}
\end{table}

The CHASE-DB1 comparison does not reach the 0.05 threshold in either paired test, despite its positive exploratory bootstrap interval. With only four subject units, the smallest attainable two-sided exact signed-rank $p$ value without ties is 0.125. The bootstrap interval should therefore not be treated as decisive evidence of a population-level benefit. DRIVE and HRF have positive differences with Holm-adjusted signed-rank $p$ values below 0.05 under the stated image-level assumptions; these conditional results do not establish broad patient-level robustness or variability across independent training runs.

The five-seed summaries and paired tests address variability reporting and the baseline--refinement comparison only. They do not provide five independent full-pipeline training runs for each dataset or a repeated-training analysis of the original 10--100-transition series. That original series remains descriptive; the 0.001 increase from 50 to 100 transitions may reflect sampling variation, but its cause has not been established. No standard deviations or paired test results for the original transition series are inferred from its point estimates.

'''
manuscript = ROOT / 'template261006.tex'
original = manuscript.read_bytes()
text = original.decode('utf-8-sig').replace('\r\n', '\n')
anchor = r'\subsection{Contribution of the Dice Supervision Term}'
assert text.count(anchor) == 1
assert r'\label{tab:paired_statistical_tests}' not in text
updated = text.replace(anchor, addition+anchor, 1)
assert updated.replace(addition, '', 1) == text
backup = ROOT / ('template261006_before_paired_tests_' + STAMP + '.tex')
version = ROOT / ('template261006_' + STAMP + '.tex')
assert not backup.exists() and not version.exists()
backup.write_bytes(original)
version.write_text(updated, encoding='utf-8')
manuscript.write_bytes(version.read_bytes())

response = '''Thank you for this comment. We have added five-seed mean Dice results, standard deviations between seed-level means, and paired baseline--refinement tests. Available realisations were averaged within images; CHASE-DB1 testing used four subject-level averages retaining both eyes together. Exact two-sided Wilcoxon tests were Holm-adjusted across the three dataset comparisons. The adjusted p values were 0.125 for CHASE-DB1, 0.00000572205 for DRIVE, and 0.0009765625 for HRF. Secondary exploratory paired t tests are also reported. DRIVE and HRF inference is conditional on image independence, which could not be established from the supplied identifiers.

We explicitly distinguish training-seed and sampling-seed variability. These tables do not establish five independent full-pipeline training runs across all datasets. The original 10--100-transition comparison also remains a single-run descriptive series. Its 0.001 Dice difference between 50 and 100 transitions may reflect sampling variation, but this has not been demonstrated. A repeated-training transition-count experiment remains necessary to fully address that part of the comment. We therefore do not claim that the reviewer request has been completely resolved.
'''
response_file = OUT / 'reviewer_2_comment_2_response.txt'
response_file.write_text(response)
registry = ROOT / 'reviewer_comments/reviewer_2_comments.json'
registry_bytes = registry.read_bytes()
tracking = json.loads(registry_bytes.decode('utf-8-sig'))
for item in tracking['comments']:
    if item['number'] == 2:
        item['status'] = 'Partially addressed: current five-seed summaries, SDs, paired tests and multiplicity correction added; five independent full-pipeline runs and original DDIM transition trend remain incomplete.'
        item['manuscript_version'] = str(version)
        item['response_file'] = str(response_file)
        item['current_paired_test_report'] = str(OUT / 'paired_tests_report.txt')
(registry.parent / ('reviewer_2_comments_before_paired_tests_' + STAMP + '.json')).write_bytes(registry_bytes)
registry.write_text(json.dumps(tracking, indent=2, ensure_ascii=False), encoding='utf-8')
(registry.parent / ('reviewer_2_comments_' + STAMP + '.json')).write_bytes(registry.read_bytes())
verbatim = registry.parent / 'reviewer_2_comments.txt'
(registry.parent / ('reviewer_2_comments_' + STAMP + '.txt')).write_bytes(verbatim.read_bytes())
print('\n'.join(lines))
print('MANUSCRIPT', version)
print('RESPONSE', response_file)
