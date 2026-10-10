"""Verify saved maps and export full-run data without changing manuscript tables."""
import argparse
import csv
import json
import itertools
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp, rankdata

SEEDS = [42, 123, 2026, 3000, 3001]
STEPS = [10, 20, 30, 50, 100]
DATASETS = ['CHASE', 'DRIVE', 'HRF']
COMPARISONS = [('baseline_vs_50', 'baseline', 50), ('10_vs_50', 10, 50), ('50_vs_100', 50, 100)]


def exact_signed_rank(differences):
    """Enumerate signs of nonzero pairs; handles zero differences and tied ranks."""
    nonzero = np.asarray(differences)[np.asarray(differences) != 0]
    if not len(nonzero):
        return 0., 1., 0
    ranks = rankdata(np.abs(nonzero), method='average')
    observed = abs(float(np.sum(ranks*np.sign(nonzero))))
    signs = np.array(list(itertools.product([-1,1], repeat=len(nonzero))))
    distribution = np.abs(signs @ ranks)
    p = float(np.mean(distribution >= observed-1e-12))
    statistic = float(min(ranks[nonzero>0].sum(), ranks[nonzero<0].sum()))
    return statistic, p, len(nonzero)


def run(folder):
    records, seed_rows, summary_rows, test_rows = [], [], [], []
    verified_rows = 0
    for dataset in DATASETS:
        image_order = None
        by_seed = {}
        split = None
        for seed in SEEDS:
            job = folder / dataset / f'train_seed_{seed}'
            assert (job / 'completed.json').exists(), f'Incomplete job: {job}'
            protocol = json.loads((job / 'protocol.json').read_text())
            if split is None:
                split = protocol['split']
            assert split == protocol['split'], 'Split changed across training runs'
            assert protocol['training_seed'] == seed
            rows = list(csv.DictReader((job / 'paired_per_image_metrics.txt').open(newline='')))
            records.extend(rows)
            images = sorted({r['image'] for r in rows})
            if image_order is None:
                image_order = images
            assert image_order == images
            by_seed[seed] = {}
            for image in images:
                with np.load(job / f'{image}_reference.npz') as f:
                    gt = f['ground_truth'] >= .5
                    fov = f['fov'].astype(bool)
                    baseline = f['coarse'] >= .5
                for steps in STEPS:
                    with np.load(job / f'{image}_steps{steps}_probabilities.npz') as f:
                        maps = f['probabilities']
                        assert f['sampling_seeds'].tolist() == SEEDS
                    for index, sample_seed in enumerate(SEEDS):
                        selected = [r for r in rows if r['image'] == image
                                    and int(r['steps']) == steps and int(r['sampling_seed']) == sample_seed]
                        assert len(selected) == 1
                        row = selected[0]
                        for prefix, pred in [('baseline', baseline), ('refined', maps[index] >= .5)]:
                            tp = int((pred & gt & fov).sum())
                            fp = int((pred & ~gt & fov).sum())
                            fn = int((~pred & gt & fov).sum())
                            tn = int((~pred & ~gt & fov).sum())
                            for key, value in [('tp',tp), ('fp',fp), ('fn',fn), ('tn',tn)]:
                                assert int(row[prefix+'_'+key]) == value
                            expected = dict(dice=2*tp/max(2*tp+fp+fn,1), iou=tp/max(tp+fp+fn,1),
                                            sensitivity=tp/max(tp+fn,1), specificity=tn/max(tn+fp,1),
                                            accuracy=(tp+tn)/(tp+fp+fn+tn))
                            for key, value in expected.items():
                                assert abs(float(row[prefix+'_'+key])-value) < 1e-12
                        assert abs(float(row['delta'])-(float(row['refined_dice'])-float(row['baseline_dice']))) < 1e-12
                        verified_rows += 1
            for steps in STEPS:
                subset = [r for r in rows if int(r['steps']) == steps]
                values = {key: statistics.mean(float(r[key]) for r in subset)
                          for key in ['baseline_dice', 'refined_dice', 'delta']}
                seed_rows.append(dict(dataset=dataset, training_seed=seed, steps=steps, **values))
                by_seed[seed][steps] = [statistics.mean(float(r['refined_dice']) for r in subset if r['image'] == image)
                                       for image in images]
            by_seed[seed]['baseline'] = [statistics.mean(float(r['baseline_dice']) for r in rows if r['image'] == image)
                                         for image in images]
            print('VERIFIED', dataset, seed, flush=True)
        for steps in STEPS:
            means = np.array([np.mean(by_seed[s][steps]) for s in SEEDS])
            baselines = np.array([np.mean(by_seed[s]['baseline']) for s in SEEDS])
            summary_rows.append(dict(dataset=dataset, steps=steps, n_training_runs=5,
                                     baseline_mean=float(baselines.mean()), baseline_sd=float(baselines.std(ddof=1)),
                                     refined_mean=float(means.mean()), refined_sd=float(means.std(ddof=1))))
        for name, first, second in COMPARISONS:
            differences = np.array([np.array(by_seed[s][second])-np.array(by_seed[s][first]) for s in SEEDS])
            training_deltas = differences.mean(axis=1)
            paired_t = ttest_1samp(training_deltas, 0)
            w_statistic, w_p, nonzero_pairs = exact_signed_rank(training_deltas)
            seed_se = float(training_deltas.std(ddof=1)/np.sqrt(5))
            mean = float(training_deltas.mean())
            training_ci = [mean-float(t.ppf(.975, 4))*seed_se, mean+float(t.ppf(.975, 4))*seed_se]
            rng = np.random.default_rng(20261008)
            if dataset == 'CHASE':
                units = sorted({int(image.split('_')[1][:2]) for image in image_order})
                matrix = np.stack([differences[:, [i for i, image in enumerate(image_order)
                                                  if int(image.split('_')[1][:2]) == subject]].mean(axis=1)
                                   for subject in units], axis=1)
                indices = rng.integers(len(units), size=(50000,len(units)))
                unit_note = 'subject clusters, both eyes averaged'
            else:
                matrix = differences
                if dataset == 'HRF':
                    strata = [[i for i,image in enumerate(image_order) if image.split('_')[-1] == category]
                              for category in ['h','dr','g']]
                    indices = np.concatenate([rng.choice(stratum, size=(50000,len(stratum)), replace=True)
                                              for stratum in strata], axis=1)
                    unit_note = 'pathology-stratified images; patient independence unverified'
                else:
                    indices = rng.integers(len(image_order), size=(50000,len(image_order)))
                    unit_note = 'images; patient independence unverified'
            training_indices = rng.integers(5, size=(50000,5))
            boot = matrix[training_indices[:,:,None], indices[:,None,:]].mean(axis=(1,2))
            test_rows.append(dict(dataset=dataset, comparison=name, mean_delta=mean,
                                  sd_training_deltas=float(training_deltas.std(ddof=1)),
                                  paired_t_p=float(paired_t.pvalue),
                                  wilcoxon_statistic=w_statistic,
                                  exact_wilcoxon_p=w_p, nonzero_training_pairs=nonzero_pairs,
                                  wilcoxon_method='Exhaustive sign enumeration of nonzero signed ranks, average ranks for ties',
                                  training_seed_t_95ci=training_ci,
                                  crossed_training_test_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                                  test_resampling_unit=unit_note, bootstrap_draws=50000,
                                  bootstrap_seed=20261008))
    assert len(records) == verified_rows == 5000
    largest = 0.
    for index, row in enumerate(sorted(test_rows, key=lambda r:r['exact_wilcoxon_p'])):
        largest = max(largest, (len(test_rows)-index)*row['exact_wilcoxon_p'])
        row['holm_wilcoxon_p'] = min(largest, 1.)
    for filename, data in [('ALL_verified_per_image_records.txt', records),
                           ('training_seed_means.txt', seed_rows), ('mean_and_SD_by_steps.txt', summary_rows)]:
        with (folder / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (folder / 'paired_tests_and_intervals.json').write_text(json.dumps(test_rows, indent=2))
    (folder / 'paired_training_seed_tests.json').write_text(json.dumps(test_rows, indent=2))
    (folder / 'verification.txt').write_text('PASS: all 5000 rows verified against saved FP32 probability maps, ground-truth masks and FOVs. All derived metrics and paired differences agree. Test splits identical across training seeds within each dataset.\n')
    report = ['NEW independent full-pipeline reproduction; separate from author-supplied original tables.',
              '5 training seeds x 5 noise seeds x 5 step counts, each dataset.',
              'Every training run independently trains baseline, mask autoencoder, and refiner.',
              'Means/SD refer to 5 training-run means after averaging sampling realisations and test images.',
              'Seed-level tests condition on the fixed test set. Crossed bootstrap additionally resamples test units.',
              'Only 4 CHASE subjects; DRIVE/HRF patient independence unverified.',
              'Exact two-sided signed-rank tests with 5 nonzero untied pairs cannot yield p below 0.0625.',
              'Paired t tests and bootstrap intervals are exploratory with 5 training runs.',
              'Holm adjustment is across the 9 planned signed-rank comparisons.',
              'Preprocessing, training schedules and augmentation are described in per-job protocol.json; no test-based tuning.', '']
    for row in summary_rows:
        report.append(f"{row['dataset']} {row['steps']} steps: baseline {row['baseline_mean']:.9f} +/- {row['baseline_sd']:.9f}; refined {row['refined_mean']:.9f} +/- {row['refined_sd']:.9f}")
    (folder / 'experiment_report.txt').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(folder / 'experiment_data_tables.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['ALL_verified_per_image_records.txt','training_seed_means.txt','mean_and_SD_by_steps.txt',
                     'paired_tests_and_intervals.json','verification.txt','experiment_report.txt','experiment_plan.json']:
            archive.write(folder / name, arcname=name)
        for job in folder.glob('*/train_seed_*'):
            for name in ['protocol.json','completed.json']:
                archive.write(job / name, arcname=str((job / name).relative_to(folder)))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        while True:
            status_file = args.output / 'status.json'
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            if status.get('stage') == 'failed':
                raise RuntimeError('Training controller failed; inspect status.json and job logs')
            if status.get('stage') == 'completed':
                break
            time.sleep(10)
    (args.output / 'source_report_all_datasets_full_runs.py').write_bytes(Path(__file__).read_bytes())
    run(args.output.resolve())
