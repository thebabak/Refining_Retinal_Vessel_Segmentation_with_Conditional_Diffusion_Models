"""
Generate real-data-based versions of schematic/diagram images for the paper.
Saves all outputs to plots/realdata/.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusion_refiner.dataset import CHASEDataset
from diffusion_refiner.inference import load_checkpoint, refine_mask_ensemble
import torch

# Output directory
OUTPUT_DIR = Path("plots/realdata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CHASE data directory
CHASE_DATA_DIR = Path("data")
CKPT_PATH = Path("diffusion_refiner_checkpoint.pth")
IMG_SIZE = (512, 512)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print(f"Loading model from {CKPT_PATH}...")
model = load_checkpoint(CKPT_PATH, DEVICE)

# Load CHASE data
print(f"Loading CHASE-DB1 data from {CHASE_DATA_DIR}...")
dataset = CHASEDataset(CHASE_DATA_DIR, img_size=IMG_SIZE)

# Use the first image for demonstration
item = dataset[0]
image = item['image'].to(DEVICE)
coarse_mask = item['mask'].to(DEVICE)

# Run ensemble refinement
NUM_SAMPLES = 5
NUM_STEPS = 50
GUIDANCE_SCALE = 1.5
mean_mask, uncertainty, all_samples = refine_mask_ensemble(
    model, image, coarse_mask, num_samples=NUM_SAMPLES, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, device=DEVICE
)
mean_mask_np = mean_mask.squeeze().cpu().numpy()
uncertainty_np = uncertainty.squeeze().cpu().numpy()
image_np = image.permute(1, 2, 0).cpu().numpy()
coarse_mask_np = coarse_mask.squeeze().cpu().numpy()

# 1. Real-data-based Uncertainty Quantification via Ensemble Sampling
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Uncertainty Quantification via Ensemble Sampling (Real Data)', fontsize=16, fontweight='bold')
axes[0, 0].imshow(image_np)
axes[0, 0].set_title('RGB Image')
axes[0, 0].axis('off')
axes[1, 0].imshow(coarse_mask_np, cmap='gray')
axes[1, 0].set_title('Coarse Mask')
axes[1, 0].axis('off')
for i in range(NUM_SAMPLES):
    ax = axes[i//3, 1 + i%3]
    ax.imshow(all_samples[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Sample {i+1}')
    ax.axis('off')
axes[1, 3].imshow(mean_mask_np, cmap='gray', vmin=0, vmax=1)
axes[1, 3].set_title('Mean Prediction')
axes[1, 3].axis('off')
axes[0, 3].imshow(uncertainty_np, cmap='hot', vmin=0, vmax=uncertainty_np.max())
axes[0, 3].set_title('Uncertainty (Variance)')
axes[0, 3].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_uncertainty_ensemble.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. Real-data-based Single vs Ensemble Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Single Sample vs. Ensemble Sampling (Real Data)', fontsize=16, fontweight='bold')
axes[0, 0].imshow(all_samples[0].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Single Sample')
axes[0, 0].axis('off')
axes[1, 0].imshow(mean_mask_np, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('Ensemble Mean')
axes[1, 0].axis('off')
axes[0, 1].imshow(uncertainty_np, cmap='hot', vmin=0, vmax=uncertainty_np.max())
axes[0, 1].set_title('Uncertainty (Variance)')
axes[0, 1].axis('off')
axes[1, 1].imshow(image_np)
axes[1, 1].set_title('Fundus Image')
axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_single_vs_ensemble.png', dpi=200, bbox_inches='tight')
plt.close()

print("All schematic/diagram images have been regenerated using real data and saved to:", OUTPUT_DIR)
