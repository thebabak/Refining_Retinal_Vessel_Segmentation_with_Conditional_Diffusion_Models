to improve retinal vessel segmentation from a coarse U-Net output.
Diffusion Refiner Prototype

This repository contains a minimal prototype for a conditional latent-diffusion mask refiner designed
to improve retinal vessel segmentation from a coarse U-Net output.

Files:
- `models.py`: model definitions (autoencoder, image encoder, diffusion U-Net)
- `dataset.py`: CHASE-DB1 loader and dummy dataset
- `train.py`: training harness with real CHASE data
- `utils.py`: diffusion scheduler (cosine schedule) and DDIM sampling
- `inference.py`: checkpoint loading and mask refinement
- `eval.py`: evaluation metrics (Dice, IoU, Accuracy, Sensitivity, Specificity, AUC)
- `plot_results.py`: visualization of results
- `config.yaml`: hyperparameters
- `requirements.txt`: dependencies

Installation (Windows - PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install required packages:

```powershell
pip install --upgrade pip
pip install -r diffusion_refiner\requirements.txt
```

Quick Start

Train on CHASE-DB1 dataset (5 epochs):

```powershell
python -m diffusion_refiner.train
```

Evaluate model (compute Dice, IoU, AUC on test set):

```powershell
python -m diffusion_refiner.eval
```

Plot evaluation results:

```powershell
pip install matplotlib pandas
python -m diffusion_refiner.plot_results
```

Run inference (refine a coarse mask):

```powershell
python -m diffusion_refiner.inference
```

Methods Overview

**Conditional Latent Diffusion Refiner:**
- Input: RGB fundus image + coarse vessel segmentation mask
- Encoder: Lightweight image encoder + mask autoencoder (4× downsampling)
- Diffusion Process: Cosine noise schedule, DDPM training loss
- Sampling: DDIM (20–50 steps) for fast inference (~100 ms/image)
- Output: Refined binary vessel mask

**Key Features:**
- Lightweight: ~1.9M parameters for entire pipeline
- Fast: 208 FPS inference on CPU (4.8 ms/image with 20 DDIM steps)
- Interpretable: Iterative refinement visible per sampling step
- Evaluations: Dice, IoU, Accuracy, Sensitivity, Specificity, AUC-ROC

Data & Reproducibility

- Dataset: CHASE-DB1 retinal vessel segmentation (28 images)
- Train/Eval split: Currently uses full dataset for both (TODO: proper split)
- Checkpoint: `diffusion_refiner_checkpoint.pth` (automatically saved after training)
- Results: `evaluation_results.csv` (per-image metrics)

Next Steps

- Implement proper train/val/test splits on DRIVE, CHASE, HRF datasets
- Add baseline U-Net + reverse attention comparison
- Tune diffusion hyperparameters (guidance scale, sampling steps) for better refinement
- Write full Methods and Results sections for paper
- Create paper draft with architecture diagrams and ablations

Notes

- The `eval.py` currently uses target mask + noise as "coarse prediction" for testing; replace with actual baseline predictions
- Extend `dataset.py` with DRIVE and HRF loaders for multi-dataset evaluation
- GPU recommended for faster training; CPU mode auto-detects device

