import torch
from pathlib import Path
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel
from .utils import ddim_sample


def load_checkpoint(ckpt_path, device):
    """Load trained diffusion model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128, timesteps=1000)
    
    model.ae.load_state_dict(ckpt['ae'])
    model.img_enc.load_state_dict(ckpt['img_enc'])
    model.unet.load_state_dict(ckpt['unet'])
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    return model


def refine_mask(
    model,
    image,
    coarse_mask,
    num_steps=50,
    guidance_scale=1.5,
    device='cuda'
):
    """Refine a coarse vessel mask using the diffusion refiner.
    
    Args:
        model: LatentDiffusionModel
        image: RGB fundus image (H, W, 3) or (3, H, W) in range [0, 1]
        coarse_mask: coarse vessel mask (H, W) or (1, H, W) in range [0, 1]
        num_steps: DDIM sampling steps
        guidance_scale: guidance scale for conditioning
        device: device to run on
    
    Returns:
        refined_mask: refined vessel mask (1, H, W) in range [0, 1]
    """
    refined = _refine_single(model, image, coarse_mask, num_steps, guidance_scale, device)
    return refined


def refine_mask_ensemble(
    model,
    image,
    coarse_mask,
    num_samples=5,
    num_steps=50,
    guidance_scale=1.5,
    device='cuda'
):
    """Refine a coarse vessel mask using ensemble sampling for uncertainty quantification.
    
    Args:
        model: LatentDiffusionModel
        image: RGB fundus image (H, W, 3) or (3, H, W) in range [0, 1]
        coarse_mask: coarse vessel mask (H, W) or (1, H, W) in range [0, 1]
        num_samples: number of independent samples for ensemble
        num_steps: DDIM sampling steps per sample
        guidance_scale: guidance scale for conditioning
        device: device to run on
    
    Returns:
        mean_mask: mean of ensemble predictions (1, H, W)
        uncertainty: pixel-wise variance across ensemble (1, H, W)
        all_samples: all ensemble samples (num_samples, 1, H, W)
    """
    all_samples = []
    
    print(f"Running ensemble sampling with {num_samples} samples...")
    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...", end=' ', flush=True)
        refined = _refine_single(model, image, coarse_mask, num_steps, guidance_scale, device)
        all_samples.append(refined)
        print("✓")
    
    # Stack all samples
    stack = torch.stack(all_samples, dim=0)  # (num_samples, 1, H, W)
    
    # Compute mean and variance
    mean_mask = stack.mean(dim=0)  # (1, H, W)
    uncertainty = stack.var(dim=0)  # (1, H, W) - pixel-wise variance
    
    return mean_mask, uncertainty, stack


def _refine_single(model, image, coarse_mask, num_steps, guidance_scale, device):
    """Helper: perform single refinement pass."""
def _refine_single(model, image, coarse_mask, num_steps, guidance_scale, device):
    """Helper: perform single refinement pass."""
    if image.dim() == 3 and image.shape[0] == 3:
        # Already (3, H, W)
        image_tensor = image.unsqueeze(0).to(device)
    else:
        # (H, W, 3)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    if coarse_mask.dim() == 2:
        mask_tensor = coarse_mask.unsqueeze(0).unsqueeze(0).to(device)
    else:
        # (1, H, W)
        mask_tensor = coarse_mask.unsqueeze(0).to(device)
    
    # Encode coarse mask to latent space
    with torch.no_grad():
        z_init = model.ae.encode(mask_tensor)
        cond = model.img_enc(image_tensor)
        cond = model.cond_mlp(cond)
        
        # DDIM sampling starting from coarse latent + noise
        z_refined = ddim_sample(
            model,
            model.scheduler,
            z_init.shape,
            cond,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        )
        
        # Decode refined latent to mask space
        refined_mask = model.ae.decode(z_refined)
    
    return refined_mask.squeeze(0)  # Remove batch dim, return (1, H, W)


if __name__ == '__main__':
    # Example: load checkpoint and refine a dummy mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path("diffusion_refiner_checkpoint.pth")
    
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        model = load_checkpoint(ckpt_path, device)
        
        # Create dummy image and coarse mask
        image = torch.rand(3, 512, 512)
        coarse_mask = torch.rand(1, 512, 512) > 0.8
        
        # Single refinement
        print("Refining mask with DDIM sampling (50 steps)...")
        refined = refine_mask(model, image, coarse_mask.float(), num_steps=50, device=device)
        print(f"Refined mask shape: {refined.shape}, min={refined.min():.3f}, max={refined.max():.3f}")
        
        # Ensemble refinement with uncertainty
        print("\n" + "="*60)
        print("Ensemble Uncertainty Quantification")
        print("="*60)
        mean_mask, uncertainty, samples = refine_mask_ensemble(
            model, image, coarse_mask.float(), 
            num_samples=3, num_steps=30, device=device
        )
        
        print(f"\nResults:")
        print(f"  Mean mask shape: {mean_mask.shape}")
        print(f"  Mean mask range: [{mean_mask.min():.3f}, {mean_mask.max():.3f}]")
        print(f"  Uncertainty (variance) shape: {uncertainty.shape}")
        print(f"  Uncertainty range: [{uncertainty.min():.6f}, {uncertainty.max():.6f}]")
        print(f"  All samples shape: {samples.shape}")
        
        # Find high-uncertainty regions (potential error regions)
        uncertainty_threshold = uncertainty.mean() + 1.0 * uncertainty.std()
        high_uncertainty_regions = (uncertainty > uncertainty_threshold).sum().item()
        total_pixels = uncertainty.numel()
        print(f"\n  High-uncertainty pixels: {high_uncertainty_regions}/{total_pixels} ({100*high_uncertainty_regions/total_pixels:.1f}%)")
        
    else:
        print(f"Checkpoint not found at {ckpt_path}")
