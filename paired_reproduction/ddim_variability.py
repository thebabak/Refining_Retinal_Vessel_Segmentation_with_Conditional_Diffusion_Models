"""Matched sampling experiment on the documented NEW CHASE partition.

This does not recover or validate the original manuscript 0.835/0.836 results.
"""
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_refiner.models import MaskAutoencoder, ImageEncoder, DiffusionUNet

SOURCE = ROOT / 'paired_reproduction/results'
OUT = ROOT / 'paired_reproduction' / ('ddim_variability_' + datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S'))
OUT.mkdir()
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device('cuda')
protocol = json.loads((SOURCE / 'protocol.json').read_text())
seeds = [42, 123, 2026] + list(range(3000, 3017))
ae = MaskAutoencoder().to(device).eval()
ae.load_state_dict(torch.load(SOURCE / 'autoencoder.pt', weights_only=True))
ie = ImageEncoder().to(device).eval()
un = DiffusionUNet().to(device).eval()
tm = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)).to(device).eval()
saved = torch.load(SOURCE / 'refiner.pt', weights_only=True)
for model, key in ((ie, 'ie'), (un, 'un'), (tm, 'time')):
    model.load_state_dict(saved[key])
t = torch.arange(1001, device=device) / 1000
a = torch.cos((t + .008) / 1.008 * math.pi / 2) ** 2
a = a / a[0]
alpha = torch.cumprod(1 - (1 - a[1:] / a[:-1]).clamp(max=.999), 0)

def emb(t):
    v = t[:, None] * torch.exp(-math.log(10000) * torch.arange(32, device=device) / 32)
    return torch.cat([v.sin(), v.cos()], 1)

def denoise(z, t, c):
    return un(z, emb(t), c + tm(emb(t)))

def sample(coarse, rgb, noise, steps):
    z0 = ae.encode(coarse)
    c = ie(rgb)
    z = alpha[99].sqrt() * z0 + (1 - alpha[99]).sqrt() * noise
    ts = torch.linspace(99, 0, steps, device=device).long()
    assert len(torch.unique(ts)) == steps
    for j, ti in enumerate(ts):
        eps0 = denoise(z, ti[None], torch.zeros_like(c))
        eps1 = denoise(z, ti[None], c)
        eps = eps0 + 1.5 * (eps1 - eps0)
        estimate = (z - (1 - alpha[ti]).sqrt() * eps) / alpha[ti].sqrt()
        prev = alpha[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1., device=device)
        z = prev.sqrt() * estimate + (1 - prev).sqrt() * eps
    return ae.decode(z)

record = dict(source_protocol=protocol, sampling_seeds=seeds, transitions=[50, 100],
              initialization='same Gaussian tensor per image and seed; start timestep 99',
              guidance=1.5, precision='FP32', timing='batch 1; CUDA synchronized; encoding, image conditioning, DDIM, decoding; excludes backbone, transfers and scoring; two warmups per setting',
              claim='New experiment on documented reproduction checkpoints, NOT original manuscript 0.835/0.836',
              checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.pt')},
              gpu=torch.cuda.get_device_name(), torch=torch.__version__)
(OUT / 'protocol.json').write_text(json.dumps(record, indent=2))
rows = []
print('OUTPUT', OUT, flush=True)
with torch.inference_mode():
    for name in protocol['split']['test']:
        rgb = cv2.resize(np.asarray(Image.open(ROOT / 'data' / f'{name}.jpg').convert('RGB')), (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = torch.tensor(rgb / 255., dtype=torch.float32).permute(2, 0, 1)[None].to(device)
        coarse = torch.tensor(np.load(SOURCE / f'{name}_coarse.npy'))[None].to(device)
        gt = np.asarray(Image.open(ROOT / 'data' / f'{name}_1stHO.png').convert('L').resize((512, 512), Image.Resampling.NEAREST)) > 127
        shape = ae.encode(coarse).shape
        warm = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(999))
        for steps in (50, 100):
            for _ in range(2):
                sample(coarse, rgb, warm, steps)
        for index, seed in enumerate(seeds):
            noise = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
            pair = {}
            # Alternate order to reduce systematic timing-order effects.
            for steps in ((50, 100) if index % 2 == 0 else (100, 50)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred = sample(coarse, rgb, noise, steps)
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - start)
                mask = pred.cpu().numpy()[0, 0] >= .5
                tp = np.count_nonzero(mask & gt)
                value = 2 * tp / max(np.count_nonzero(mask) + np.count_nonzero(gt), 1)
                pair[steps] = (float(value), elapsed)
                if steps == 50 and seed in protocol['sampling_seeds']:
                    old = np.load(SOURCE / f'{name}_refined_seed{seed}.npy')
                    assert np.max(np.abs(old - pred.cpu().numpy()[0])) < 1e-5, '50-step source reproduction failed'
            rows.append(dict(image=name, subject=int(name.split('_')[1][:2]), seed=seed,
                             dice50=pair[50][0], dice100=pair[100][0], delta=pair[100][0]-pair[50][0],
                             ms50=pair[50][1], ms100=pair[100][1]))
            with (OUT / 'paired_metrics.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print('completed', name, 'mean delta', np.mean([r['delta'] for r in rows if r['image']==name]), flush=True)

# Crossed bootstrap: resample subjects (retaining both eyes) and sampling seeds.
subjects = sorted({r['subject'] for r in rows})
matrix = np.array([[np.mean([r['delta'] for r in rows if r['subject']==subject and r['seed']==seed]) for seed in seeds] for subject in subjects])
rng = np.random.default_rng(20261006)
boots = np.array([matrix[rng.integers(0, len(subjects), len(subjects))][:, rng.integers(0, len(seeds), len(seeds))].mean() for _ in range(10000)])
seed_means = np.array([np.mean([r['delta'] for r in rows if r['seed']==seed]) for seed in seeds])
summary = dict(n_images=8, n_subjects=len(subjects), n_sampling_seeds=len(seeds), n_training_runs=1,
               dice50=float(np.mean([r['dice50'] for r in rows])), dice100=float(np.mean([r['dice100'] for r in rows])),
               mean_delta=float(matrix.mean()), crossed_subject_seed_bootstrap_95ci=np.quantile(boots, [.025, .975]).tolist(),
               seed_mean_delta_sd=float(seed_means.std(ddof=1)), seed_mean_delta_range=[float(seed_means.min()), float(seed_means.max())],
               mean_ms50=float(np.mean([r['ms50'] for r in rows])), mean_ms100=float(np.mean([r['ms100'] for r in rows])),
               source_50step_predictions_reproduced=True,
               limitations=['Different checkpoints and partition from original manuscript results', 'Only four independent subjects', 'One training run; sampling seeds are not training runs', 'Exploratory bootstrap interval'])
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
