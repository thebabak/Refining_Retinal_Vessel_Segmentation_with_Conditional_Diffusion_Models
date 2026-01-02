"""
Generate uncertainty quantification image from real CHASE-DB1 data using ensemble sampling.
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from diffusion_refiner.dataset import CHASEDataset
from diffusion_refiner.inference import load_checkpoint, refine_mask_ensemble

# Paths
DATA_DIR = Path("data")
CKPT_PATH = Path("diffusion_refiner_checkpoint.pth")
OUTPUT_PATH = Path("plots/chase/chase_uncertainty_real.png")

# Parameters
IMG_SIZE = (512, 512)
NUM_SAMPLES = 5
NUM_STEPS = 50
GUIDANCE_SCALE = 1.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print(f"Loading model from {CKPT_PATH}...")
model = load_checkpoint(CKPT_PATH, DEVICE)

# Load CHASE data
print(f"Loading CHASE-DB1 data from {DATA_DIR}...")
dataset = CHASEDataset(DATA_DIR, img_size=IMG_SIZE)

# Use the first image for demonstration
sample = dataset[0]
image = sample['image'].to(DEVICE)
coarse_mask = sample['mask'].to(DEVICE)

# Run ensemble refinement
print("Running ensemble refinement for uncertainty quantification...")
mean_mask, uncertainty, all_samples = refine_mask_ensemble(
    model, image, coarse_mask, num_samples=NUM_SAMPLES, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, device=DEVICE
)

# Convert tensors to numpy
mean_mask_np = mean_mask.squeeze().cpu().numpy()
uncertainty_np = uncertainty.squeeze().cpu().numpy()
image_np = image.permute(1, 2, 0).cpu().numpy()

# Plot and save
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(image_np)
axes[0].set_title('CHASE Fundus Image')
axes[0].axis('off')
axes[1].imshow(mean_mask_np, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Refined Mask (Mean)')
axes[1].axis('off')
im = axes[2].imshow(uncertainty_np, cmap='hot', vmin=0, vmax=uncertainty_np.max())
axes[2].set_title('Uncertainty (Variance)')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.suptitle('Uncertainty Quantification via Ensemble Sampling (Real Data)', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_PATH}")
