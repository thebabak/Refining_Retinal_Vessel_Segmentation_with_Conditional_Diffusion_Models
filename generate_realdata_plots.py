"""
Generate all real-data-based plots for the paper using CHASE-DB1 images and masks.
Saves all plots to plots/realdata/.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from PIL import Image
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

# 1. Dataset overview (all images)
fig, axes = plt.subplots(4, 7, figsize=(16, 10))
fig.suptitle('CHASE-DB1 Dataset Overview (Real Data)', fontsize=16, fontweight='bold')
for idx in range(len(dataset)):
    item = dataset[idx]
    row = idx // 7
    col = idx % 7
    ax = axes[row, col]
    img = item['image'].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    ax.set_title(f"{item['path'].split('/')[-1]}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Example images with mask and overlay
n_examples = 6
fig, axes = plt.subplots(n_examples, 3, figsize=(12, 14))
fig.suptitle('CHASE-DB1: Real Image, Mask, and Overlay Examples', fontsize=14, fontweight='bold')
for i in range(n_examples):
    item = dataset[i]
    img = item['image'].permute(1, 2, 0).cpu().numpy()
    mask = item['mask'].squeeze().cpu().numpy()
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"{item['path'].split('/')[-1]} (Image)" if i == 0 else item['path'].split('/')[-1], fontsize=9)
    axes[i, 0].axis('off')
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title("Ground Truth" if i == 0 else "", fontsize=9)
    axes[i, 1].axis('off')
    overlay = img.copy()
    overlay[mask > 0.5] = [0, 1, 0]
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay" if i == 0 else "", fontsize=9)
    axes[i, 2].axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_02_mask_examples.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Vessel statistics
vessel_percentages = []
for i in range(len(dataset)):
    mask = dataset[i]['mask'].squeeze().cpu().numpy()
    vessel_percentages.append((mask > 0.5).sum() / mask.size * 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vessel_percentages, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(np.mean(vessel_percentages), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vessel_percentages):.1f}%')
ax.set_xlabel('Vessel Coverage (%)')
ax.set_ylabel('Number of Images')
ax.set_title('Vessel Percentage Distribution (Real Data)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_03_vessel_statistics.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Vessel thickness (distance transform)
from scipy import ndimage
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Vessel Thickness Maps (Real Data)', fontsize=14, fontweight='bold')
for idx in range(6):
    mask = dataset[idx]['mask'].squeeze().cpu().numpy().astype(np.uint8)
    dist = ndimage.distance_transform_edt(mask)
    ax = axes.flat[idx]
    im = ax.imshow(dist, cmap='hot')
    ax.set_title(f"{dataset[idx]['path'].split('/')[-1]}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Distance (px)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'realdata_04_vessel_thickness.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Uncertainty quantification (ensemble sampling)
NUM_SAMPLES = 5
NUM_STEPS = 50
GUIDANCE_SCALE = 1.5
item = dataset[0]
image = item['image'].to(DEVICE)
coarse_mask = item['mask'].to(DEVICE)
mean_mask, uncertainty, all_samples = refine_mask_ensemble(
    model, image, coarse_mask, num_samples=NUM_SAMPLES, num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE, device=DEVICE
)
mean_mask_np = mean_mask.squeeze().cpu().numpy()
uncertainty_np = uncertainty.squeeze().cpu().numpy()
image_np = image.permute(1, 2, 0).cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(image_np)
axes[0].set_title('Fundus Image (Real)')
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
fig.savefig(OUTPUT_DIR / 'realdata_05_uncertainty.png', dpi=200, bbox_inches='tight')
plt.close()

print("All real-data-based plots saved to:", OUTPUT_DIR)
