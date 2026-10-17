# Refining Retinal Vessel Segmentation with Conditional Diffusion Models

**A Controlled Evaluation on LU-Net+RA Probability Maps**

Implementation code, experiment drivers and versioned records from the authors' retinal vessel segmentation study.

## Current revision

The [current manuscript, complete reviewer responses, figures and numerical records](revise/current_revision_2026-10-17_10-29-17/README_2026-10-17_10-29-17.md) correspond to the **23-page MDPI revision**. The main title is retained; five contributions are reported. All results are in the article, including the per-image comparison and matched OOF experiment.

Table 6 compares the same coarse and global/frozen refined predictions at 50 DDIM updates. Its mean Dice ± SD across five training-seed means is copied below from the manuscript:

| Dataset | Coarse LU-Net+RA | Global/frozen diffusion refiner |
| --- | --- | --- |
| CHASE-DB1 | 0.810514 ± 0.001806 | 0.746146 ± 0.002622 |
| DRIVE | 0.791814 ± 0.007557 | 0.789757 ± 0.008286 |
| HRF | 0.732121 ± 0.003490 | 0.665726 ± 0.006320 |

The tested reference does not improve mean Dice. Table 6 includes exploratory paired 95% intervals; Tables 7–9 provide all per-image means; Table 10 reports the matched OOF comparison. OOF nonsignificance does not establish equivalence or remove the training-input-shift concern. Historical positive single-run scores are not current efficacy results.

## Which code matches the manuscript?

| Manuscript content | Implementation |
| --- | --- |
| Reference and component study, Tables 6 and 11 | [run_component_ablations.py](paired_reproduction/run_component_ablations.py), `global_frozen` reference |
| Shared model and data/training definitions | [models.py](diffusion_refiner/models.py), [run.py](paired_reproduction/run.py), [run_drive.py](paired_reproduction/run_drive.py), [run_hrf_baseline.py](paired_reproduction/run_hrf_baseline.py), [official_model.py](paired_reproduction/official_model.py) |
| Matched OOF, Table 10 | [run_oof_comparison.py](paired_reproduction/run_oof_comparison.py) |
| Spatial control, Table 12 | [CHASE driver](paired_reproduction/run_spatial_residual_refiner.py), [DRIVE/HRF driver](paired_reproduction/run_spatial_refiner_drive_hrf.py) |
| Fixed-checkpoint 50/100-update study, Section 4.8 | [ddim_variability.py](paired_reproduction/ddim_variability.py) |
| Complete-pipeline / CUDA-event timing, Section 4.9 | [benchmark_pipeline.py](paired_reproduction/benchmark_pipeline.py), [benchmark_cuda_events.py](paired_reproduction/benchmark_cuda_events.py) |

The manuscript reference starts coarse-initialised diffusion at stored index 99; the Gaussian-start target ablation starts at 999, and the deterministic control uses no DDIM. Independent training runs and repeated initial-noise sampling are separate sources of variation. The reference, spatial control and fixed-checkpoint timing experiments have different training/measurement protocols; use the corresponding manuscript section.

`diffusion_refiner/` also contains legacy implementations and utilities. Read its [module notes](diffusion_refiner/README.md); the legacy evaluation CLI is not the complete Table 6 evaluator. The separate full-pipeline rerun must not be confused with the matched component-study reference. Historical records in `revise/` and frozen executed-source snapshots are retained for provenance, not as interchangeable current results.

## Setup and prerequisites

Install the existing [dependencies](diffusion_refiner/requirements.txt) in a separate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r diffusion_refiner/requirements.txt
```

The manuscript experiment drivers require CUDA, matching data layouts, split/protocol files and the retained baseline/autoencoder checkpoints. Obtain CHASE-DB1, DRIVE and HRF from their original providers and follow their terms. Installing dependencies does not download datasets or checkpoints. Inspect a driver and its artifact prerequisites before running it; there is no claim that a fresh clone alone reproduces every historical result.

## Availability and historical material

The current revision folder contains byte-preserved sources, class/figure assets, PDFs and existing numerical CSVs. Model weights, dataset images and raw probability arrays are not included in this source release; the manuscript describes how to request retained artifacts. Complete arrays for every repeat of the fixed-checkpoint sampling experiment were not retained. Code or a numerical table alone does not substitute for those artifacts.

Dummy datasets/helpers, synthetic-only training utilities, hypothetical plotting scripts, obsolete January documents, unused demo configuration and duplicate `ALL/` code have been removed from the active tree. Retained figure CLIs require real saved coarse predictions. Missing coarse inputs raise an error; reference annotations are not silently substituted. Historical Git versions and frozen experiment snapshots remain provenance records. Local recovery archives and caches are ignored by Git.

This update copies existing manuscript values and records and cleans code/documentation. It performs no training, prediction evaluation, statistical recalculation or timing measurement. Overleaf is not updated automatically by publishing this repository.
