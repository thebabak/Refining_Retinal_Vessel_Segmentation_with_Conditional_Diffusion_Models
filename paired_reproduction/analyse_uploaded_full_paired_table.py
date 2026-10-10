"""Numerical paired analysis of an author-supplied table, not mask verification."""
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import t, ttest_1samp
from report_all_datasets_full_runs import exact_signed_rank

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/FS-GAMING PRO/.codex/attachments/815c0784-4660-4bc5-9f1a-1e3c7c809a7c/Pasted text.txt')
STAMP = datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
OUT = ROOT / 'reviewer_comments' / ('uploaded_full_paired_tests_' + STAMP)
OUT.mkdir()
shutil.copy2(SOURCE, OUT / 'source_table.txt')
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
assert len(rows) == 5000
keys = [(r['dataset'],r['image'],r['seed'],r['sampling_seed'],r['steps']) for r in rows]
assert len(set(keys)) == 5000
seed_results, unit_results, means, image_rows = [], [], [], []


def test(values):
    values = np.asarray(values, dtype=float)
    w, p, nonzero = exact_signed_rank(values)
    n = len(values)
    result = ttest_1samp(values, 0)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    margin = float(t.ppf(.975,n-1))*sd/np.sqrt(n)
    return dict(n_pairs=n, mean_delta=mean, sd_delta=sd,
                paired_t_statistic=float(result.statistic), paired_t_p=float(result.pvalue),
                paired_t_95ci=[mean-margin,mean+margin], wilcoxon_statistic=w,
                exact_wilcoxon_p=p, nonzero_pairs=nonzero)


for dataset in ['CHASE','DRIVE','HRF']:
    subset = [r for r in rows if r['dataset'] == dataset]
    seeds = sorted({int(r['seed']) for r in subset})
    images = sorted({r['image'] for r in subset})
    sampling = sorted({int(r['sampling_seed']) for r in subset})
    assert seeds == sampling == [42,123,2026,3000,3001]
    values = {}
    for step in [10,20,30,50,100]:
        matrix = np.empty((5,len(images)))
        base_matrix = np.empty_like(matrix)
        for i,seed in enumerate(seeds):
            for j,image in enumerate(images):
                selected = [r for r in subset if int(r['seed']) == seed
                            and r['image'] == image and int(r['steps']) == step]
                assert len(selected) == 5
                matrix[i,j] = np.mean([float(r['refined_dice']) for r in selected])
                base_matrix[i,j] = np.mean([float(r['baseline_dice']) for r in selected])
        if 'baseline' in values:
            assert np.array_equal(values['baseline'],base_matrix)
        values['baseline'] = base_matrix
        values[step] = matrix
        means.append(dict(dataset=dataset,steps=step,baseline_mean=float(base_matrix.mean()),
                          baseline_sd_across_training_labels=float(base_matrix.mean(1).std(ddof=1)),
                          refined_mean=float(matrix.mean()),
                          refined_sd_across_training_labels=float(matrix.mean(1).std(ddof=1))))
    for name, first, second in [('baseline_vs_50','baseline',50),('10_vs_50',10,50),('50_vs_100',50,100)]:
        differences = values[second]-values[first]
        result = dict(dataset=dataset, comparison=name, **test(differences.mean(1)))
        result['scope'] = 'Five training-label means; conditional on fixed test set and five averaged sampling realisations'
        if dataset == 'CHASE':
            subjects = sorted({r['subject'] for r in subset})
            clusters = [[j for j,image in enumerate(images)
                         if next(r['subject'] for r in subset if r['image']==image)==subject] for subject in subjects]
            population_matrix = np.stack([differences[:,cluster].mean(1) for cluster in clusters],axis=1)
            unit = 'subject; both eyes averaged'
        else:
            population_matrix = differences
            unit = 'image; patient independence unverified'
        rng = np.random.default_rng(20261008)
        if dataset == 'HRF':
            strata = [[j for j,image in enumerate(images) if image.split('_')[-1]==category]
                      for category in ['h','dr','g']]
            indices = np.concatenate([rng.choice(group,(50000,len(group)),replace=True)
                                      for group in strata],axis=1)
        else:
            indices = rng.integers(population_matrix.shape[1],size=(50000,population_matrix.shape[1]))
        training_indices = rng.integers(5,size=(50000,5))
        bootstrap = population_matrix[training_indices[:,:,None],indices[:,None,:]].mean((1,2))
        result['crossed_bootstrap_95ci'] = np.quantile(bootstrap,[.025,.975]).tolist()
        result['bootstrap_resampling_units'] = unit
        result['bootstrap_draws'] = 50000
        result['bootstrap_seed'] = 20261008
        seed_results.append(result)
        if name == 'baseline_vs_50':
            unit_results.append(dict(dataset=dataset,unit=unit,**test(population_matrix.mean(0))))
    for j,image in enumerate(images):
        image_rows.append(dict(dataset=dataset,image=image,
                               baseline_dice=float(values['baseline'][:,j].mean()),
                               refined_dice=float(values[50][:,j].mean()),
                               delta=float((values[50][:,j]-values['baseline'][:,j]).mean())))


def holm(results, field):
    largest = 0.
    for i,result in enumerate(sorted(results,key=lambda r:r[field])):
        largest = max(largest,(len(results)-i)*result[field])
        result['holm_'+field] = min(largest,1.)


for group in [seed_results,unit_results]:
    # Separate exploratory families: nine run-label tests, three test-unit tests.
    for field in ['exact_wilcoxon_p','paired_t_p']:
        holm(group,field)
for name,records in [('mean_and_SD_by_steps.txt',means),('per_image_50steps.txt',image_rows)]:
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
metadata = dict(source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                n_records=len(rows),scope='Analysis of new user-supplied table only; not independent prediction-mask verification',
                training_label_tests=seed_results, test_unit_tests=unit_results,
                assumptions='Independent paired units; signed-rank test assumes symmetric differences. Training labels are taken from supplied schema; execution records were not checked.',
                manuscript_modified=False)
(OUT/'paired_tests.json').write_text(json.dumps(metadata,indent=2))
report=['Recalculated from source_table.txt only; no experiment rerun and no manuscript change.',
        'Average five sampling realisations within each image and training label.',
        'Training-label comparisons: two-sided paired t and exact signed-rank tests; Holm across9 comparisons per test type.',
        'Secondary test-unit comparisons at50steps: CHASE4subjects (both eyes averaged), DRIVE20images, HRF12images, averaging all training/sampling labels first; Holm across3 datasets per test type.',
        'Crossed bootstrap:50,000 draws over training labels and test subjects/images; HRF pathology-stratified.',
        'The two inference levels answer different questions and must not be substituted to select a preferred p value.',
        'Training-label intervals/tests condition on the fixed test set. Test-unit independence and distributional assumptions limit interpretation.', '']
for r in seed_results:
    report.append(f"{r['dataset']} {r['comparison']}: mean difference={r['mean_delta']:.9f}, paired t p={r['paired_t_p']:.9g}, Holm t p={r['holm_paired_t_p']:.9g}, exact Wilcoxon p={r['exact_wilcoxon_p']:.9g}, Holm Wilcoxon p={r['holm_exact_wilcoxon_p']:.9g}, crossed95CI={r['crossed_bootstrap_95ci']}")
report.append('\nSECONDARY SUBJECT/IMAGE TESTS AT50STEPS')
for r in unit_results:
    report.append(f"{r['dataset']} n={r['n_pairs']}: t p={r['paired_t_p']:.9g}, Holm t p={r['holm_paired_t_p']:.9g}, exact Wilcoxon p={r['exact_wilcoxon_p']:.9g}, Holm Wilcoxon p={r['holm_exact_wilcoxon_p']:.9g}")
(OUT/'paired_tests_report.txt').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
print('OUTPUT',OUT)
