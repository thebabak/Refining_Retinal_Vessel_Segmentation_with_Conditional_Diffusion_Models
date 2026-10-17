# Conditional diffusion refiner implementation notes

This directory contains shared model definitions and legacy training, inference and evaluation utilities. The current manuscript reference is trained/evaluated by `../paired_reproduction/run_component_ablations.py` (`global_frozen`), not by the legacy `eval.py` CLI. It includes multiple historical variants rather than a single validated manuscript reproduction entry point. See the [repository README](../README.md) for source separation and release scope.

## Code locations

| Files | Role |
| --- | --- |
| `models.py`, `models_corrected.py`, `models_corrected2.py` | Different model implementations. Select the version required by the experiment protocol; do not assume they are interchangeable. |
| `dataset.py`, `dataset2.py` | Real-data loaders. Development-only dummy dataset classes have been removed. |
| `train.py` and other `train*.py` files | Training variants with different settings and assumptions. Inspect the chosen driver and its inputs before execution. |
| `utils.py`, `inference.py`, `refine_low_noise.py` | Scheduling, sampling and inference utilities. |
| `eval.py`, `eval_baseline_same_protocol.py`, `uncertainty_metrics.py` | Evaluation-related utilities. `eval.py` requires actual coarse predictions and does not synthesize them from target masks. Check the remaining evaluation assumptions for the selected script. |
| `plot_results.py` and figure-generation scripts | Plotting utilities. A generated figure is not an independent evaluation. |
| `requirements.txt` | Dependency list. The unused demonstration `config.yaml` has been removed; manuscript settings are specified by the mapped experiment drivers. |

## Setup

From the repository root, install the existing dependency list in a separate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r diffusion_refiner/requirements.txt
```

Configure dataset directories, annotations, coarse predictions, splits and required checkpoints for the chosen script. Some scripts retain local paths or prototype assumptions. Dependency installation does not confirm manuscript reproducibility.

## Experiment drivers

The `../paired_reproduction/` directory contains separate experiment drivers and reporting scripts for repeated training, deterministic refiners, component studies, out-of-fold comparisons and timing. Match each driver to its dated records and source manifest before using its outputs in an article.

Dummy dataset classes and test helpers have been removed. Missing CHASE data raises `FileNotFoundError`. The `inference.py` random-input demo does not run from the command line. The synthetic-only `train_new.py` and `testUnet.py` utilities have been removed; use a documented real-data driver instead.

The `eval.py` CLI requires `--baseline-dir`. Its evaluator requires `batch['coarse']` and raises an error when that input is missing; it does not accept a human-observer reference as an implicit coarse input. The legacy DRIVE/HRF loaders add `coarse` and `baseline` only when a prediction directory is supplied. With no such directory, they return image/mask pairs only. The CHASE loader's second-observer comparison is still a human annotation, not an LU-Net+RA prediction.

Obsolete development documents and utilities are kept only in the ignored local recovery archive described in the repository README. They are not manuscript evaluation inputs.

## Interpretation and availability

This documentation makes no estimated Dice, throughput or parameter-budget claims. Those quantities depend on the architecture, checkpoint, partition, evaluator and timing scope and belong in the matching experimental records.

Model weights and raw prediction arrays are excluded by the publication rules, even when local copies exist. No training, evaluation, statistical calculation or timing measurement was performed for this documentation update. Publishing the source cleanup does not run or reproduce an experiment.

Legacy diagnosis/figure utilities that used reference annotations as coarse inputs or displayed autoencoder reconstructions as predictions have also been removed from the active tree. DRIVE/HRF sampling figure CLIs require an explicit saved-coarse directory. Their earlier code and already generated figures remain preserved for provenance; no figure or metric was regenerated.

For the current article and exact table-to-code map, follow the repository README. Older model variants are retained for source history and are not interchangeable with the current global/frozen reference.
