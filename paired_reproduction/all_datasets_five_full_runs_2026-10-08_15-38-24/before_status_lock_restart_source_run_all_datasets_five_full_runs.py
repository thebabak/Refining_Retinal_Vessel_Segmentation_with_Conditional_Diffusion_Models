"""Independent full-pipeline reruns, separate from author-supplied tables.

Controller: python this_file.py --output <new-folder>
Worker: controller invokes the same script with --dataset and --seed.
Original experiment sources/checkpoints and the manuscript are not modified.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 123, 2026, 3000, 3001]
STEPS = [10, 20, 30, 50, 100]


def now():
    return datetime.now(ZoneInfo('Asia/Tehran')).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def compose(dataset, seed, destination):
    filename = {'CHASE': 'run.py', 'DRIVE': 'run_drive.py',
                'HRF': 'run_hrf_baseline.py'}[dataset]
    source = HERE / filename
    text = source.read_text(encoding='utf-8-sig')
    marker = 'model=LNet_RA(in_channels=1).to(DEVICE)'
    prefix = text.split(marker)[0]
    output_statement = next(line for line in prefix.splitlines()
                            if line.startswith('OUT = Path('))
    prefix = prefix.replace(output_statement, 'OUT = experiment_output')
    prefix = prefix.replace('torch.manual_seed(42)', f'torch.manual_seed({seed})')
    prefix = prefix.replace('np.random.seed(42)', f'np.random.seed({seed})')
    prefix = prefix.replace('random.seed(42)', f'random.seed({seed})')
    if dataset != 'HRF':
        training = marker + text.split(marker, 1)[1].split(
            'ie.eval(); un.eval(); time_mlp.eval()')[0]
    else:
        baseline = marker + text.split(marker, 1)[1].split('\nrows=[]')[0]
        chase = (HERE / 'run.py').read_text()
        refinement = chase[chase.index('# Stage seeds are independent'):chase.index(
            'ie.eval(); un.eval(); time_mlp.eval()')]
        training = baseline + '''
with torch.no_grad():
    for d in data.values(): d['coarse']=model(d['x'][None].to(DEVICE)).cpu()[0]
''' + refinement
    training = training.replace('torch.cuda.manual_seed_all(42)',
                                f'torch.cuda.manual_seed_all({seed})')
    # Local job history additionally records training stages without changing loss.
    training = training.replace("print('Baseline parameters'",
                                "job_stage('baseline_training'); print('Baseline parameters'")
    training = training.replace('ae=MaskAutoencoder().to(DEVICE)',
                                "job_stage('autoencoder_training'); ae=MaskAutoencoder().to(DEVICE)")
    training = training.replace('ie=ImageEncoder().to(DEVICE)',
                                "job_stage('refiner_training'); ie=ImageEncoder().to(DEVICE)")
    compile(prefix, str(source), 'exec')
    compile(training, str(source), 'exec')
    return source, prefix, training


def worker(dataset, seed, output, preflight=False):
    import numpy as np
    import torch

    job = output / dataset / f'train_seed_{seed}'
    job.mkdir(parents=True, exist_ok=True)
    if (job / 'completed.json').exists():
        print('Already completed', job, flush=True)
        return
    status = dict(dataset=dataset, training_seed=seed, started_at=now())

    def job_stage(stage):
        status.update(stage=stage, updated_at=now())
        write_json(job / 'status.json', status)

    job_stage('loading_data')
    source, prefix, training = compose(dataset, seed, job)
    namespace = dict(__file__=str(source), __name__='independent_rerun',
                     experiment_output=job, job_stage=job_stage)
    exec(compile(prefix, str(source), 'exec'), namespace)
    config = namespace['config']
    config.update(training_seed=seed, seed=seed, split_seed=42,
                  sampling_seeds=SEEDS, transition_counts=STEPS,
                  start_timestep=99, guidance=1.5,
                  independent_stages=['baseline', 'mask_autoencoder', 'refiner'],
                  claim='New independently trained full-pipeline runs; not the author-supplied original results',
                  precision='FP32', torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),
                  refiner_epochs=100,
                  autoencoder_epochs=200 if dataset == 'DRIVE' else 50,
                  baseline_selection='validation Dice; maximum 1000 epochs, patience 20',
                  refiner_selection='fixed epoch 100; no test-based selection',
                  loss='noise MSE plus 0.5 decoded-mask Dice loss',
                  augmentation='baseline geometric augmentation inherited from executable reproduction; refiner none',
                  baseline_training_maps='in-sample coarse maps',
                  limitations=['Uses existing executable reproduction, not independently verified original-paper code',
                               'Small fixed test split',
                               'DRIVE/HRF annotation conversion provenance and patient independence unresolved'])
    # The split is fixed across training seeds; only model/training RNGs differ.
    write_json(job / 'protocol.json', config)
    (job / 'executed_training_source.py').write_text(prefix + '\n' + training, encoding='utf-8')
    if preflight:
        print(dataset, 'preflight', {k: len(v) for k, v in namespace['manifest'].items()}, flush=True)
        job_stage('preflight_passed')
        return
    exec(compile(training, str(source), 'exec'), namespace)
    job_stage('evaluation')
    model, ae = namespace['model'], namespace['ae']
    ie, un, tm = namespace['ie'], namespace['un'], namespace['time_mlp']
    for module in (model, ae, ie, un, tm):
        module.eval()
        module.requires_grad_(False)
    data, manifest = namespace['data'], namespace['manifest']
    alpha, device = namespace['alpha'], namespace['DEVICE']
    denoise, score = namespace['denoise'], namespace['score']

    def metrics(prediction, sample):
        if dataset == 'CHASE':
            return score(prediction, sample['g'])
        return score(prediction, sample['g'], sample['fov'])

    records = []
    with (job / 'paired_per_image_metrics.txt').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        with torch.inference_mode():
            for image in manifest['test']:
                sample = data[image]
                coarse = sample['coarse'][None].to(device)
                z0 = ae.encode(coarse)
                conditioning = ie(sample['rgb'][None].to(device))
                baseline = metrics(sample['coarse'], sample)
                np.savez_compressed(job / f'{image}_reference.npz',
                                    coarse=sample['coarse'].numpy(),
                                    ground_truth=sample['g'].numpy(),
                                    fov=(sample['fov'].numpy() if 'fov' in sample
                                         else np.ones_like(sample['g'].numpy(), dtype=bool)))
                maps = {steps: [] for steps in STEPS}
                for sampling_seed in SEEDS:
                    generator = torch.Generator(device=device).manual_seed(sampling_seed)
                    # The identical Gaussian tensor is reused across all step counts.
                    noise = torch.randn(z0.shape, device=device, generator=generator)
                    for steps in STEPS:
                        z = alpha[99].sqrt()*z0 + (1-alpha[99]).sqrt()*noise
                        timesteps = torch.linspace(99, 0, steps, device=device).long()
                        for j, ti in enumerate(timesteps):
                            unconditional = denoise(z, ti[None], torch.zeros_like(conditioning))
                            conditional = denoise(z, ti[None], conditioning)
                            epsilon = unconditional + 1.5*(conditional-unconditional)
                            estimate = (z-(1-alpha[ti]).sqrt()*epsilon)/alpha[ti].sqrt()
                            previous = alpha[timesteps[j+1]] if j+1 < steps else torch.tensor(1., device=device)
                            z = previous.sqrt()*estimate + (1-previous).sqrt()*epsilon
                        prediction = ae.decode(z).cpu()[0]
                        refined = metrics(prediction, sample)
                        array = prediction.numpy()
                        assert np.isfinite(array).all()
                        maps[steps].append(array)
                        record = dict(dataset=dataset, image=image,
                                      subject=sample.get('subject', ''),
                                      pathology=image.split('_')[-1] if dataset == 'HRF' else '',
                                      seed=seed, sampling_seed=sampling_seed, steps=steps,
                                      baseline_dice=baseline['dice'], refined_dice=refined['dice'],
                                      delta=refined['dice']-baseline['dice'],
                                      **{'baseline_'+k: v for k, v in baseline.items() if k != 'dice'},
                                      **{'refined_'+k: v for k, v in refined.items() if k != 'dice'})
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        stream.flush()
                        records.append(record)
                for steps, arrays in maps.items():
                    np.savez_compressed(job / f'{image}_steps{steps}_probabilities.npz',
                                        probabilities=np.stack(arrays), sampling_seeds=np.array(SEEDS))
                status.update(completed_images=sum(r['steps'] == 50 and r['sampling_seed'] == 42
                                                    for r in records), rows=len(records))
                job_stage('evaluation')
                print(dataset, seed, image, 'evaluated all 5 sampling seeds and 5 step counts', flush=True)
    expected = len(manifest['test'])*len(SEEDS)*len(STEPS)
    assert len(records) == expected
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in job.glob('*.pt')}
    summary = dict(dataset=dataset, training_seed=seed, rows=len(records), checkpoint_sha256=hashes,
                   completed_at=now(), metrics_by_steps={})
    for steps in STEPS:
        subset = [r for r in records if r['steps'] == steps]
        summary['metrics_by_steps'][steps] = {k: float(np.mean([r[k] for r in subset]))
                                             for k in ['baseline_dice', 'refined_dice', 'delta']}
    write_json(job / 'completed.json', summary)
    job_stage('completed')


def aggregate(output):
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon

    all_records = []
    complete = []
    for dataset in ['CHASE', 'DRIVE', 'HRF']:
        for seed in SEEDS:
            job = output / dataset / f'train_seed_{seed}'
            if (job / 'completed.json').exists():
                complete.append((dataset, seed))
                with (job / 'paired_per_image_metrics.txt').open(newline='') as f:
                    all_records.extend(csv.DictReader(f))
    if not all_records:
        return
    with (output / 'ALL_completed_per_image_records.txt').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    summaries, tests = [], []
    for dataset in ['CHASE', 'DRIVE', 'HRF']:
        seed_scores = {}
        for seed in SEEDS:
            subset = [r for r in all_records if r['dataset'] == dataset and int(r['seed']) == seed]
            if not subset:
                continue
            baseline = float(np.mean([float(r['baseline_dice']) for r in subset if int(r['steps']) == 50]))
            seed_scores[seed] = {'baseline': baseline}
            for steps in STEPS:
                selected = [r for r in subset if int(r['steps']) == steps]
                seed_scores[seed][steps] = float(np.mean([float(r['refined_dice']) for r in selected]))
        if not seed_scores:
            continue
        for steps in STEPS:
            values = [v[steps] for v in seed_scores.values()]
            baselines = [v['baseline'] for v in seed_scores.values()]
            summaries.append(dict(dataset=dataset, steps=steps, n_training_runs=len(values),
                                  baseline_mean=float(np.mean(baselines)),
                                  baseline_sd=float(np.std(baselines, ddof=1)) if len(values)>1 else None,
                                  refined_mean=float(np.mean(values)),
                                  refined_sd=float(np.std(values, ddof=1)) if len(values)>1 else None))
        if len(seed_scores) == 5:
            for comparison, first, second in [('baseline_vs_50','baseline',50), ('10_vs_50',10,50), ('50_vs_100',50,100)]:
                a = np.array([v[first] for v in seed_scores.values()])
                b = np.array([v[second] for v in seed_scores.values()])
                delta = b-a
                t = ttest_rel(b,a)
                w = wilcoxon(delta, alternative='two-sided', method='exact') if np.any(delta) else None
                tests.append(dict(dataset=dataset, comparison=comparison, n_training_runs=5,
                                  mean_delta=float(delta.mean()), sd_delta=float(delta.std(ddof=1)),
                                  paired_t_p=float(t.pvalue), exact_wilcoxon_p=float(w.pvalue) if w else 1.,
                                  scope='Conditional on fixed test set; training-run inference, not patient-level inference'))
    write_json(output / 'training_seed_means_and_SD.json', summaries)
    # One family of nine planned Wilcoxon comparisons once all jobs finish.
    if tests:
        ordered = sorted(tests, key=lambda r: r['exact_wilcoxon_p'])
        maximum = 0.
        for i, record in enumerate(ordered):
            maximum = max(maximum, (len(ordered)-i)*record['exact_wilcoxon_p'])
            record['holm_wilcoxon_p'] = min(maximum, 1.)
        write_json(output / 'paired_training_seed_tests.json', tests)
    write_json(output / 'completion_inventory.json', dict(completed_jobs=complete, expected_jobs=15,
                                                         rows=len(all_records), expected_rows=5000))


def controller(output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'experiment_plan.json', dict(started_at=now(), datasets=['CHASE','DRIVE','HRF'],
               training_seeds=SEEDS, sampling_seeds=SEEDS, transition_counts=STEPS,
               full_pipeline_runs=15, expected_per_image_records=5000,
               split_seed=42, paired_noise_across_steps=True,
               source='Existing executable reproduction; newly trained baseline, AE and refiner per job',
               manuscript_changes=False, original_author_tables_changed=False,
               source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__),HERE/'run.py',HERE/'run_drive.py',HERE/'run_hrf_baseline.py']}))
    for p in [Path(__file__), HERE/'run.py',HERE/'run_drive.py',HERE/'run_hrf_baseline.py']:
        (output / ('source_'+p.name)).write_bytes(p.read_bytes())
    jobs = [(dataset, seed) for seed in SEEDS for dataset in ['CHASE','DRIVE','HRF']]
    for index, (dataset, seed) in enumerate(jobs):
        status = dict(stage='running', current_dataset=dataset, current_training_seed=seed,
                      job_number=index+1, total_jobs=15, updated_at=now())
        write_json(output / 'status.json', status)
        log_path = output / f'{dataset}_train_seed_{seed}.log'
        with log_path.open('a', encoding='utf-8') as log:
            process = subprocess.run([sys.executable, str(Path(__file__)), '--output', str(output),
                                      '--dataset', dataset, '--seed', str(seed)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            status.update(stage='failed', returncode=process.returncode, log_file=str(log_path), updated_at=now())
            write_json(output / 'status.json', status)
            raise RuntimeError(f'Job failed: {dataset} {seed}; see {log_path}')
        aggregate(output)
    write_json(output / 'status.json', dict(stage='completed', completed_jobs=15, updated_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['CHASE','DRIVE','HRF'])
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.dataset:
        worker(args.dataset, args.seed, args.output.resolve(), args.preflight)
    else:
        controller(args.output.resolve())
