# Retinal vessel diffusion refinement

This repository contains source code and the `revise/` manuscript, numerical tables, reviewer responses, and experiment reports.

## Code

- `diffusion_refiner/`: segmentation and diffusion implementation, training and evaluation scripts.
- `paired_reproduction/`: paired evaluation, statistical analyses, timing, deterministic controls, component ablations, and out-of-fold training scripts.
- `ALL/` and root Python scripts: plotting and earlier implementation utilities.
- `reference_RVS/`: reference baseline notebooks; retain their original attribution.

## Revision records

`revise/` preserves multiple dated sources. Author-supplied score tables and separately executed reproduction studies have different provenance and must not be mixed. Source manifests and reviewer responses document these distinctions. Some supplied adjusted tables have unresolved execution provenance; their values must not be described as independently verified experiments.

The current manuscript is the newest `revise/last_for_git_*.tex`. Manuscript snapshots do not include every required figure or the MDPI class. Existing scripts may require local dataset paths and archived checkpoints.

Model checkpoints, raw prediction tensors, logs, and the two large raw-output archives are omitted from this publication. This repository does not contain the deleted original score-producing checkpoint. Dataset images, local environments, temporary files, and unrelated manuscript drafts are excluded. Historical Git commits remain intact.
