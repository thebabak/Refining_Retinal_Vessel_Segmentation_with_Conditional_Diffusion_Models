"""Generate a CHASE-style DRIVE visualization with image, GT mask, synthetic coarse baseline, refined mean, and uncertainty."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from diffusion_refiner.inference import load_checkpoint, refine_mask_ensemble


def parse_args():
    parser = ArgumentParser(description="Generate a DRIVE CHASE-style sample visualization.")
    parser.add_argument("--images", type=str, default="DRIVE/training/images", help="Path to DRIVE images directory")
    parser.add_argument("--masks", type=str, default="DRIVE/training/mask", help="Path to DRIVE mask directory")
    parser.add_argument("--checkpoint", type=str, default="diffusion_refiner_checkpoint.pth", help="Path to diffusion refiner checkpoint")
    parser.add_argument("--output", type=str, default="plots/drive/drive_chase_style.png", help="Output figure path")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of DRIVE sample to visualize")
    parser.add_argument("--img-size", type=int, nargs=2, default=(512, 512), help="Resize image and mask to this size")
    parser.add_argument("--num-samples", type=int, default=5, help="Ensemble sample count")
    parser.add_argument("--num-steps", type=int, default=30, help="DDIM sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="Guidance scale")
    parser.add_argument("--first-manual-dir", type=str, default="DRIVE/training/1st_manual", help="Path to 1st_manual masks")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Optional directory with baseline coarse predictions")
    return parser.parse_args()


def load_image(path: Path, img_size):
    image = Image.open(path).convert("RGB")
    image = image.resize(tuple(img_size), Image.Resampling.BILINEAR)
    return np.array(image).astype(np.float32) / 255.0


def load_mask(path: Path, img_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize(tuple(img_size), Image.Resampling.NEAREST)
    mask = np.array(mask) > 127
    return mask.astype(np.float32)


def find_matching_file_by_number(directory: Path, number_prefix: str):
    if not directory.exists():
        return None
    for p in sorted(directory.iterdir()):
        name = p.stem
        if name.startswith(number_prefix):
            return p
    return None


def make_synthetic_coarse(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
    eroded = np.ones_like(mask_bool, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    eroded = eroded.astype(np.uint8)
    noise = np.random.RandomState(42).rand(*mask.shape) < 0.02
    eroded[noise] = 0
    return eroded.astype(np.float32)


def main():
    args = parse_args()
    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif'}])
    mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif'}])

    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")
    if not mask_files:
        raise RuntimeError(f"No masks found in {mask_dir}")
    if args.sample_index < 0 or args.sample_index >= len(image_files):
        raise ValueError("sample-index out of range")

    image_path = image_files[args.sample_index]
    mask_path = mask_files[args.sample_index]

    print(f"Loading image: {image_path}")
    print(f"Loading mask: {mask_path}")

    image_np = load_image(image_path, args.img_size)
    mask_np = load_mask(mask_path, args.img_size)
    # try to load 1st_manual (first observer) if available
    first_manual_np = None
    first_manual_dir = Path(args.first_manual_dir)
    number_prefix = image_path.stem.split('_')[0]
    fm_file = find_matching_file_by_number(first_manual_dir, number_prefix)
    if fm_file is not None:
        try:
            first_manual_np = load_mask(fm_file, args.img_size)
            print(f"Loaded 1st manual: {fm_file}")
        except Exception:
            first_manual_np = None

    # baseline: load real baseline if provided, otherwise synthesize
    baseline_np = None
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        b_file = find_matching_file_by_number(baseline_dir, number_prefix)
        if b_file is not None:
            try:
                baseline_np = load_mask(b_file, args.img_size)
                print(f"Loaded baseline prediction: {b_file}")
            except Exception:
                baseline_np = None

    coarse_np = baseline_np if baseline_np is not None else make_synthetic_coarse(mask_np)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, device)

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    coarse_tensor = torch.from_numpy(coarse_np).unsqueeze(0).to(device)

    print("Running ensemble refinement...")
    mean_mask, uncertainty, _ = refine_mask_ensemble(
        model,
        image_tensor,
        coarse_tensor,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    mean_np = mean_mask.squeeze().cpu().numpy()
    uncertainty_np = uncertainty.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(image_np)
    axes[0].set_title('DRIVE Image')
    axes[0].axis('off')

    # show 1st manual if available, otherwise show available training mask
    if first_manual_np is not None:
        axes[1].imshow(first_manual_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('1st Manual (GT)')
    else:
        axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('GT Mask')
    axes[1].axis('off')

    axes[2].imshow(coarse_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Coarse Baseline')
    axes[2].axis('off')

    axes[3].imshow(mean_np, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Diffusion-Refined')
    axes[3].axis('off')

    # error map: absolute difference between GT (prefer 1st manual) and refined mean
    gt_for_error = first_manual_np if first_manual_np is not None else mask_np
    error_map = np.abs(mean_np - gt_for_error)
    im = axes[4].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
    axes[4].set_title('Error Map')
    axes[4].axis('off')
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.suptitle('DRIVE CHASE-style Refinement Sample', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
