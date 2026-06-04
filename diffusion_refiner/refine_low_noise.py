"""
Corrected refinement using only low timesteps where the model works well.
Save as: diffusion_refiner/refine_low_noise.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from diffusion_refiner.models_corrected import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
)


class LowNoiseRefiner:
    """
    Diffusion-based refiner that only uses low timesteps (t < 250)
    where the model can actually reconstruct z0 accurately.
    """
    
    def __init__(self, model, max_timestep=250, device='cuda'):
        """
        Args:
            model: Trained LatentDiffusionModel
            max_timestep: Maximum timestep to use (only t < max_timestep are reliable)
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.max_timestep = max_timestep
        self.model.eval()
    
    @torch.no_grad()
    def refine(self, coarse_mask, image, num_steps=20, noise_strength=0.1, 
               guidance_scale=1.0, blend_ratio=0.3):
        """
        Refine a coarse mask using only low-timestep diffusion.
        
        Args:
            coarse_mask: [B, 1, H, W] initial vessel probability map
            image: [B, 3, H, W] RGB fundus image
            num_steps: Number of DDIM steps (within low-timestep range)
            noise_strength: Starting noise level (0-1)
                           Higher means more refinement but also more deviation
            guidance_scale: CFG scale (1.0 = no guidance)
            blend_ratio: How much to blend with original coarse mask (0-1)
                        0 = fully refined, 1 = original coarse mask
        
        Returns:
            refined_mask: [B, 1, H, W] refined vessel probability map
        """
        batch_size = coarse_mask.shape[0]
        device = self.device
        
        # Move inputs to device
        coarse_mask = coarse_mask.to(device).float()
        image = image.to(device).float()
        
        # Ensure 4D
        if coarse_mask.ndim == 3:
            coarse_mask = coarse_mask.unsqueeze(1)
        
        # Encode coarse mask and get conditioning
        z0 = self.model.ae.encode(coarse_mask)
        img_cond = self.model.img_enc(image)
        
        # Map noise_strength to a starting timestep
        # noise_strength=0.1 → start_t ≈ 100
        # noise_strength=0.2 → start_t ≈ 200
        start_t = int(noise_strength * self.max_timestep)
        start_t = min(start_t, self.max_timestep - 1)
        
        # Add noise at the starting timestep
        alpha_start = self.model.alphas_cumprod[start_t].view(1, 1, 1, 1)
        noise = torch.randn_like(z0)
        z_t = torch.sqrt(alpha_start) * z0 + torch.sqrt(1 - alpha_start) * noise
        
        # Create DDIM timesteps from start_t down to 0
        step_indices = torch.linspace(start_t, 0, num_steps + 1, device=device).long()
        
        for i in range(num_steps):
            t = step_indices[i].repeat(batch_size)
            t_next = step_indices[i + 1].repeat(batch_size)
            
            # Predict noise
            eps_pred = self.model.unet(z_t, t, img_cond)
            
            # Classifier-free guidance (mild)
            if guidance_scale != 1.0:
                null_cond = torch.zeros_like(img_cond)
                eps_uncond = self.model.unet(z_t, t, null_cond)
                eps_pred = eps_uncond + guidance_scale * (eps_pred - eps_uncond)
            
            # Get alpha values
            alpha_t = self.model.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_next = self.model.alphas_cumprod[t_next].view(-1, 1, 1, 1)
            
            # Predict x0
            x0_pred = (z_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            
            # Blend with original z0 for stability
            if blend_ratio > 0:
                x0_pred = (1 - blend_ratio) * x0_pred + blend_ratio * z0
            
            # DDIM step
            dir_xt = torch.sqrt(1 - alpha_next) * eps_pred
            z_t = torch.sqrt(alpha_next) * x0_pred + dir_xt
        
        # Decode refined latent
        refined = self.model.ae.decode(z_t)
        refined = torch.sigmoid(refined)
        
        # Resize if needed
        if refined.shape[-2:] != coarse_mask.shape[-2:]:
            refined = F.interpolate(
                refined, size=coarse_mask.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        return refined
    
    @torch.no_grad()
    def refine_with_uncertainty(self, coarse_mask, image, K=5, **kwargs):
        """
        Refine with ensemble for uncertainty estimation.
        """
        samples = []
        for _ in range(K):
            refined = self.refine(coarse_mask, image, **kwargs)
            samples.append(refined)
        
        samples = torch.stack(samples, dim=0)
        mean_mask = samples.mean(dim=0)
        uncertainty = samples.var(dim=0)
        
        return mean_mask, uncertainty


def grid_search_refinement(model, coarse_mask, image, gt_mask, device):
    """
    Grid search to find optimal refinement parameters.
    """
    refiner = LowNoiseRefiner(model, max_timestep=250, device=device)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH: Refinement Parameters")
    print("=" * 80)
    
    # Baseline
    baseline_dice = compute_dice(coarse_mask, gt_mask)
    print(f"Baseline Dice: {baseline_dice:.4f}")
    print()
    
    best_dice = baseline_dice
    best_params = None
    
    # Grid search
    noise_strengths = [0.05, 0.1, 0.15, 0.2, 0.25]
    num_steps_list = [10, 20, 30]
    blend_ratios = [0.0, 0.1, 0.2, 0.3]
    
    for ns in noise_strengths:
        for steps in num_steps_list:
            for br in blend_ratios:
                refined = refiner.refine(
                    coarse_mask, image,
                    num_steps=steps,
                    noise_strength=ns,
                    blend_ratio=br,
                    guidance_scale=1.0
                )
                
                dice = compute_dice(refined, gt_mask)
                improvement = dice - baseline_dice
                
                status = "✓ IMPROVED" if improvement > 0.005 else ""
                
                print(f"noise={ns:.2f}, steps={steps:2d}, blend={br:.1f}: "
                      f"Dice={dice:.4f} ({improvement:+.4f}) {status}")
                
                if dice > best_dice:
                    best_dice = dice
                    best_params = {
                        "noise_strength": ns,
                        "num_steps": steps,
                        "blend_ratio": br
                    }
    
    print(f"\nBest: Dice={best_dice:.4f} (improvement: {best_dice - baseline_dice:+.4f})")
    print(f"Parameters: {best_params}")
    
    return best_params, best_dice


def compute_dice(pred, target, threshold=0.5, eps=1e-6):
    """Compute Dice coefficient."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    
    pred_flat = pred_bin.reshape(pred_bin.size(0), -1)
    target_flat = target_bin.reshape(target_bin.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean().item()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from diffusion_refiner.dataset2 import CHASEDataset
    
    # Setup
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = Path("seed_outputs_properly_trained/CHASE-DB1/diffusion_refiner.pth")
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        exit(1)
    
    print(f"Loading model from: {checkpoint_path}")
    
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
    unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
    
    model = LatentDiffusionModel(
        ae=ae, img_enc=img_enc, unet=unet,
        cond_dim=128, timesteps=1000
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.ae.load_state_dict(checkpoint["ae"])
    model.img_enc.load_state_dict(checkpoint["img_enc"])
    model.unet.load_state_dict(checkpoint["unet"])
    model.eval()
    
    print("Model loaded.")
    
    # Load data
    chase_path = Path("data")
    if not chase_path.exists():
        print(f"Data not found: {chase_path}")
        exit(1)
    
    dataset = CHASEDataset(chase_path, img_size=(512, 512))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Get a batch
    batch = next(iter(dataloader))
    image = batch["image"].to(device).float()
    gt_mask = batch["mask"].to(device).float()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    
    # Get or create coarse mask
    coarse_mask = None
    for key in ["baseline", "coarse", "coarse_mask"]:
        if key in batch:
            coarse_mask = batch[key].to(device).float()
            if coarse_mask.ndim == 3:
                coarse_mask = coarse_mask.unsqueeze(1)
            break
    
    if coarse_mask is None:
        print("No coarse mask found. Using GT with noise as coarse mask.")
        coarse_mask = (gt_mask > 0.5).float()
        coarse_mask = coarse_mask + torch.randn_like(coarse_mask) * 0.15
        coarse_mask = torch.clamp(coarse_mask, 0, 1)
    
    # Run grid search
    best_params, best_dice = grid_search_refinement(
        model, coarse_mask, image, gt_mask, device
    )
    
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print(f"noise_strength: {best_params['noise_strength']}")
    print(f"num_steps: {best_params['num_steps']}")
    print(f"blend_ratio: {best_params['blend_ratio']}")
    print(f"Expected Dice: {best_dice:.4f}")