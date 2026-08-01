"""Generate uncertainty quantification figures for HRF using ensemble refinement."""
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from diffusion_refiner.dataset2 import HRFDataset
from diffusion_refiner.inference import load_checkpoint, refine_mask_ensemble


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HRF uncertainty visualization.")
    parser.add_argument("--images", type=str, required=True, help="Path to HRF images directory")
    parser.add_argument("--masks", type=str, required=True, help="Path to HRF masks directory")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Optional path to HRF coarse/baseline masks directory")
    parser.add_argument("--checkpoint", type=str, default="diffusion_refiner_checkpoint.pth", help="Checkpoint path")
    parser.add_argument("--output", type=str, default="plots/hrf/hrf_uncertainty.png", help="Output figure path")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of ensemble samples")
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="Guidance scale")
    parser.add_argument("--img-size", type=int, nargs=2, default=(512, 512), help="Image size (H W)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = load_checkpoint(Path(args.checkpoint), device)

    print(f"Loading HRF data from {args.images} and {args.masks}...")
    dataset = HRFDataset(
        args.images,
        args.masks,
        img_size=tuple(args.img_size),
        baseline_dir=args.baseline_dir,
    )

    item = dataset[0]
    image = item['image'].to(device)
    coarse_mask = item['baseline'].to(device)

    print("Running ensemble refinement for uncertainty quantification...")
    mean_mask, uncertainty, _ = refine_mask_ensemble(
        model,
        image,
        coarse_mask,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    mean_mask_np = mean_mask.squeeze().cpu().numpy()
    uncertainty_np = uncertainty.squeeze().cpu().numpy()
    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('HRF Fundus Image')
    axes[0].axis('off')
    axes[1].imshow(mean_mask_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Refined Mask (Mean)')
    axes[1].axis('off')
    im = axes[2].imshow(uncertainty_np, cmap='hot', vmin=0, vmax=uncertainty_np.max())
    axes[2].set_title('Uncertainty (Variance)')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.suptitle('Uncertainty Quantification via Ensemble Sampling (HRF)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
