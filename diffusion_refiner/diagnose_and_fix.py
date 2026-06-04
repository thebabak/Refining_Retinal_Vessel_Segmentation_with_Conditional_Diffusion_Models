"""
Diagnostic and fix script for diffusion refinement failure.
Save as: diffusion_refiner/diagnose_and_fix.py
Run: python -m diffusion_refiner.diagnose_and_fix
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from diffusion_refiner.models_corrected import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
)


def diagnose_refinement(model, dataloader, device):
    """
    Step-by-step diagnosis to find where the refinement breaks.
    """
    model.eval()
    
    # Get a single batch
    batch = next(iter(dataloader))
    image = batch["image"].to(device).float()
    gt_mask = batch["mask"].to(device).float()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    
    # Get coarse mask
    coarse_mask = None
    for key in ["baseline", "coarse", "coarse_mask", "prediction"]:
        if key in batch:
            coarse_mask = batch[key].to(device).float()
            if coarse_mask.ndim == 3:
                coarse_mask = coarse_mask.unsqueeze(1)
            break
    
    if coarse_mask is None:
        print("No coarse mask found in batch. Using GT mask for diagnosis.")
        coarse_mask = gt_mask.clone()
        coarse_mask = (coarse_mask > 0.5).float() + torch.randn_like(coarse_mask) * 0.1
        coarse_mask = torch.clamp(coarse_mask, 0, 1)
    
    batch_size = image.shape[0]
    
    print("=" * 80)
    print("DIAGNOSIS: Step-by-step refinement analysis")
    print("=" * 80)
    
    # Step 1: Check autoencoder reconstruction of coarse mask
    print("\n1. Autoencoder reconstruction quality:")
    with torch.no_grad():
        z_coarse = model.ae.encode(coarse_mask)
        recon_coarse = model.ae.decode(z_coarse)
        recon_coarse = torch.sigmoid(recon_coarse)
    
    mse_ae = F.mse_loss(recon_coarse, coarse_mask).item()
    print(f"   MSE of AE reconstruction: {mse_ae:.6f}")
    print(f"   Latent shape: {z_coarse.shape}")
    print(f"   Latent mean: {z_coarse.mean():.4f}, std: {z_coarse.std():.4f}")
    print(f"   Latent min: {z_coarse.min():.4f}, max: {z_coarse.max():.4f}")
    
    # Step 2: Test different noise strengths
    print("\n2. Testing different noise strengths:")
    noise_strengths = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    
    for ns in noise_strengths:
        with torch.no_grad():
            z = model.ae.encode(coarse_mask)
            noise = torch.randn_like(z)
            z_noisy = z * np.sqrt(1 - ns) + noise * np.sqrt(ns)
            recon = model.ae.decode(z_noisy)
            recon = torch.sigmoid(recon)
        
        mse = F.mse_loss(recon, gt_mask).item()
        dice = compute_dice(recon, gt_mask)
        print(f"   noise_strength={ns:.3f}: MSE={mse:.4f}, Dice={dice:.4f}")
    
    # Step 3: Check diffusion model predictions
    print("\n3. Diffusion model noise prediction test:")
    
    with torch.no_grad():
        z0 = model.ae.encode(gt_mask)  # Use GT for this test
        img_cond = model.img_enc(image)
        
        for t_val in [0, 100, 250, 500, 750, 999]:
            t = torch.ones(batch_size, device=device).long() * t_val
            alpha_bar = model.alphas_cumprod[t_val].view(1, 1, 1, 1)
            
            # Forward diffusion
            noise_true = torch.randn_like(z0)
            z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise_true
            
            # Predict noise
            noise_pred = model.unet(z_t, t, img_cond)
            
            # Compute prediction error
            pred_error = F.mse_loss(noise_pred, noise_true).item()
            
            # Try to reconstruct z0
            z0_pred = (z_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            recon_error = F.mse_loss(z0_pred, z0).item()
            
            print(f"   t={t_val}: noise_pred_error={pred_error:.6f}, z0_recon_error={recon_error:.6f}")
    
    # Step 4: Test full DDIM with very conservative settings
    print("\n4. Conservative DDIM test:")
    
    conservative_configs = [
        {"num_steps": 5, "noise_strength": 0.001, "guidance_scale": 1.0},
        {"num_steps": 5, "noise_strength": 0.005, "guidance_scale": 1.0},
        {"num_steps": 5, "noise_strength": 0.01, "guidance_scale": 1.0},
        {"num_steps": 10, "noise_strength": 0.005, "guidance_scale": 1.0},
        {"num_steps": 10, "noise_strength": 0.01, "guidance_scale": 1.0},
        {"num_steps": 20, "noise_strength": 0.005, "guidance_scale": 1.0},
        {"num_steps": 20, "noise_strength": 0.01, "guidance_scale": 1.0},
    ]
    
    for config in conservative_configs:
        with torch.no_grad():
            refined = model.ddim_sample(
                coarse_mask, image,
                num_steps=config["num_steps"],
                noise_strength=config["noise_strength"],
                guidance_scale=config["guidance_scale"]
            )
        
        dice = compute_dice(refined, gt_mask)
        print(f"   steps={config['num_steps']}, noise={config['noise_strength']:.3f}, "
              f"guidance={config['guidance_scale']}: Dice={dice:.4f}")
    
    # Step 5: Check if the model is actually trained
    print("\n5. Model training check:")
    print(f"   AE parameters require_grad: {any(p.requires_grad for p in model.ae.parameters())}")
    print(f"   Image encoder requires_grad: {any(p.requires_grad for p in model.img_enc.parameters())}")
    print(f"   U-Net requires_grad: {any(p.requires_grad for p in model.unet.parameters())}")
    
    # Check if weights look trained (not random)
    for name, param in model.unet.named_parameters():
        if 'weight' in name and param.ndim >= 2:
            print(f"   {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
            break
    
    return {
        "ae_mse": mse_ae,
        "latent_stats": {
            "mean": z_coarse.mean().item(),
            "std": z_coarse.std().item(),
            "min": z_coarse.min().item(),
            "max": z_coarse.max().item(),
        }
    }


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


def fix_noise_schedule(model):
    """
    Fix the model to use a more conservative noise schedule
    and add a method for mild refinement.
    """
    
    @torch.no_grad()
    def mild_refinement(self, mask, image, num_steps=5, noise_strength=0.005):
        """
        Very conservative refinement that barely changes the mask.
        Use this as a baseline to verify the pipeline works at all.
        """
        batch_size = mask.shape[0]
        device = mask.device
        
        # Encode
        z0 = self.ae.encode(mask)
        img_cond = self.img_enc(image)
        
        # Add very small noise
        noise = torch.randn_like(z0)
        z_t = z0 * np.sqrt(1 - noise_strength) + noise * np.sqrt(noise_strength)
        
        # Very few DDIM steps
        step_indices = torch.linspace(self.timesteps - 1, 0, num_steps + 1, device=device).long()
        
        for i in range(num_steps):
            t = step_indices[i].repeat(batch_size)
            t_next = step_indices[i + 1].repeat(batch_size)
            
            eps_pred = self.unet(z_t, t, img_cond)
            
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1)
            
            x0_pred = (z_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            
            # Blend towards original z0 for stability
            x0_pred = 0.9 * x0_pred + 0.1 * z0
            
            dir_xt = torch.sqrt(1 - alpha_next) * eps_pred
            z_t = torch.sqrt(alpha_next) * x0_pred + dir_xt
        
        refined = self.ae.decode(z_t)
        refined = torch.sigmoid(refined)
        
        if refined.shape[-2:] != mask.shape[-2:]:
            refined = F.interpolate(refined, size=mask.shape[-2:],
                                   mode='bilinear', align_corners=False)
        
        return refined
    
    # Add the method to the model
    import types
    model.mild_refinement = types.MethodType(mild_refinement, model)
    
    return model


def create_simple_refiner(model):
    """
    If DDIM doesn't work, create a simple learned residual refiner
    that operates in pixel space. This is a fallback approach.
    """
    
    class SimpleRefiner(torch.nn.Module):
        """Simple U-Net that refines coarse masks in pixel space."""
        def __init__(self, in_ch=4):  # 1 channel mask + 3 channel image
            super().__init__()
            
            self.enc1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, 32, 3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
            )
            self.enc2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
            )
            self.enc3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
            )
            
            self.dec1 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
            )
            self.dec2 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
            )
            self.final = torch.nn.Conv2d(32, 1, 3, padding=1)
        
        def forward(self, coarse_mask, image):
            x = torch.cat([coarse_mask, image], dim=1)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            d1 = self.dec1(e3)
            d2 = self.dec2(d1 + e2)
            out = self.final(d2 + e1)
            return torch.sigmoid(coarse_mask + out * 0.1)  # Residual connection
    
    return SimpleRefiner()


# ============================================================
# Main diagnostic execution
# ============================================================

if __name__ == "__main__":
    import sys
    from diffusion_refiner.dataset2 import CHASEDataset
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    chase_path = Path("data")
    if not chase_path.exists():
        print("CHASE-DB1 data not found. Please check the path.")
        sys.exit(1)
    
    dataset = CHASEDataset(chase_path, img_size=(512, 512))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} images")
    
    # Load or create model
    # First, check if a trained model exists
    checkpoint_path = Path("seed_outputs_corrected/CHASE-DB1/seed_42/diffusion_refiner.pth")
    
    if checkpoint_path.exists():
        print(f"\nLoading trained model from: {checkpoint_path}")
        
        # Create model architecture
        ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
        img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
        unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
        
        model = LatentDiffusionModel(
            ae=ae, img_enc=img_enc, unet=unet,
            cond_dim=128, timesteps=1000
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.ae.load_state_dict(checkpoint["ae"])
        model.img_enc.load_state_dict(checkpoint["img_enc"])
        model.unet.load_state_dict(checkpoint["unet"])
        model.eval()
        
        print("Model loaded successfully.")
    else:
        print(f"\nNo trained model found at: {checkpoint_path}")
        print("Creating untrained model for architecture diagnosis...")
        
        ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
        img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
        unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
        
        model = LatentDiffusionModel(
            ae=ae, img_enc=img_enc, unet=unet,
            cond_dim=128, timesteps=1000
        ).to(device)
    
    # Run diagnosis
    print("\n" + "=" * 80)
    print("RUNNING FULL DIAGNOSIS")
    print("=" * 80)
    
    results = diagnose_refinement(model, dataloader, device)
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    print("\nKey findings:")
    print(f"1. Autoencoder reconstruction error: {results['ae_mse']:.6f}")
    print(f"2. Latent space statistics: mean={results['latent_stats']['mean']:.4f}, "
          f"std={results['latent_stats']['std']:.4f}")
    
    if results['ae_mse'] > 0.01:
        print("   ⚠ WARNING: Autoencoder reconstruction error is high!")
        print("   The autoencoder may need more training.")
    
    if abs(results['latent_stats']['mean']) > 2 or results['latent_stats']['std'] > 2:
        print("   ⚠ WARNING: Latent space statistics are extreme!")
        print("   This can cause diffusion to fail.")