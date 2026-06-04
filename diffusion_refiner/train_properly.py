"""
Corrected training script with monitoring and validation.
Save as: diffusion_refiner/train_properly.py
Run: python -m diffusion_refiner.train_properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
import time

from diffusion_refiner.dataset2 import CHASEDataset, DRIVEDataset
from diffusion_refiner.models_corrected import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Training with validation monitoring
# ============================================================

def train_autoencoder_with_validation(ae, train_loader, val_loader, device, 
                                      epochs=100, lr=1e-4, patience=20):
    """
    Train autoencoder with validation monitoring and early stopping.
    """
    ae = ae.to(device)
    optimizer = optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print("=" * 80)
    print("Stage 1: Training Mask Autoencoder (with validation)")
    print("=" * 80)
    
    for epoch in range(epochs):
        # Training
        ae.train()
        train_loss = 0.0
        
        for batch in train_loader:
            mask = batch["mask"].to(device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            recon_logits = ae(mask)
            
            if recon_logits.shape[-2:] != mask.shape[-2:]:
                recon_logits = F.interpolate(
                    recon_logits, size=mask.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            
            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(recon_logits, mask)
            
            # Dice loss
            probs = torch.sigmoid(recon_logits)
            probs_flat = probs.view(probs.size(0), -1)
            target_flat = mask.view(mask.size(0), -1)
            intersection = (probs_flat * target_flat).sum(1)
            denominator = probs_flat.sum(1) + target_flat.sum(1)
            dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
            dice_loss = (1.0 - dice).mean()
            
            loss = bce_loss + dice_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / max(len(train_loader), 1)
        
        # Validation
        ae.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mask = batch["mask"].to(device).float()
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                
                recon_logits = ae(mask)
                
                if recon_logits.shape[-2:] != mask.shape[-2:]:
                    recon_logits = F.interpolate(
                        recon_logits, size=mask.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                
                # Compute metrics
                probs = torch.sigmoid(recon_logits)
                probs_flat = probs.view(probs.size(0), -1)
                target_flat = mask.view(mask.size(0), -1)
                
                intersection = (probs_flat * target_flat).sum(1)
                denominator = probs_flat.sum(1) + target_flat.sum(1)
                dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
                
                val_loss += (1.0 - dice).mean().item()
                val_dice += dice.mean().item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_dice = val_dice / max(len(val_loader), 1)
        
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Dice: {avg_val_dice:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best state
    if best_state is not None:
        ae.load_state_dict(best_state)
        print(f"Loaded best model with val Dice: {1 - best_val_loss:.4f}")
    
    return ae


def train_diffusion_with_monitoring(model, train_loader, val_loader, device,
                                    epochs=200, lr=1e-4, patience=30):
    """
    Train diffusion model with comprehensive monitoring.
    """
    model = model.to(device)
    model.train()
    
    # Only train image encoder and U-Net
    optimizer = optim.AdamW(
        list(model.img_enc.parameters()) + list(model.unet.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("Stage 2: Training Diffusion Refiner (with monitoring)")
    print("=" * 80)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_noise_error = 0.0
        
        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device).float()
            mask = batch["mask"].to(device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            batch_size = image.shape[0]
            
            # Sample timesteps with importance sampling (more at higher t)
            # This helps the model learn the full noise range
            t_probs = torch.ones(model.timesteps) / model.timesteps
            t = torch.multinomial(t_probs, batch_size, replacement=True).to(device)
            
            # Forward pass
            eps_pred, eps_true, z0 = model(mask, image, t)
            
            # Simple MSE loss on noise prediction
            noise_loss = F.mse_loss(eps_pred, eps_true)
            
            # Also compute loss at a mid-range timestep for auxiliary supervision
            t_mid = torch.ones(batch_size, device=device).long() * 500
            eps_pred_mid, eps_true_mid, z0_mid = model(mask, image, t_mid)
            noise_loss_mid = F.mse_loss(eps_pred_mid, eps_true_mid)
            
            # Total loss
            loss = noise_loss + 0.3 * noise_loss_mid
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_noise_error += noise_loss.item()
        
        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_noise_error = train_noise_error / max(len(train_loader), 1)
        
        # Validation: check noise prediction at different timesteps
        model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device).float()
                mask = batch["mask"].to(device).float()
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                
                batch_size = image.shape[0]
                img_cond = model.img_enc(image)
                z0 = model.ae.encode(mask)
                
                # Test noise prediction at key timesteps
                for t_val in [10, 100, 250, 500, 750, 990]:
                    t = torch.ones(batch_size, device=device).long() * t_val
                    alpha_bar = model.alphas_cumprod[t_val].view(1, 1, 1, 1)
                    
                    noise = torch.randn_like(z0)
                    z_t = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
                    
                    eps_pred = model.unet(z_t, t, img_cond)
                    noise_error = F.mse_loss(eps_pred, noise).item()
                    
                    # Try to reconstruct z0
                    z0_pred = (z_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
                    z0_error = F.mse_loss(z0_pred, z0).item()
                    
                    if t_val not in val_metrics:
                        val_metrics[t_val] = {"noise_error": [], "z0_error": []}
                    
                    val_metrics[t_val]["noise_error"].append(noise_error)
                    val_metrics[t_val]["z0_error"].append(z0_error)
                
                break  # Only one batch for validation
        
        # Average validation metrics
        avg_val_noise_error = 0
        for t_val, metrics in val_metrics.items():
            metrics["noise_error"] = np.mean(metrics["noise_error"])
            metrics["z0_error"] = np.mean(metrics["z0_error"])
            avg_val_noise_error += metrics["noise_error"]
        avg_val_noise_error /= len(val_metrics)
        
        scheduler.step(avg_val_noise_error)
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Noise Error: {avg_noise_error:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            print("  Validation noise prediction errors:")
            for t_val in sorted(val_metrics.keys()):
                metrics = val_metrics[t_val]
                print(f"    t={t_val:4d}: noise_err={metrics['noise_error']:.4f}, "
                      f"z0_err={metrics['z0_error']:.4f}")
        
        # Early stopping
        if avg_val_noise_error < best_val_loss:
            best_val_loss = avg_val_noise_error
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    return model


# ============================================================
# Main training script
# ============================================================

if __name__ == "__main__":
    # Configuration
    SEED = 42
    AE_EPOCHS = 200
    DIFF_EPOCHS = 500
    BATCH_SIZE = 4
    LR = 1e-4
    
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    
    # ============================================================
    # CHASE-DB1
    # ============================================================
    CHASE_PATH = Path("data")
    
    if CHASE_PATH.exists():
        print("\n" + "=" * 80)
        print("TRAINING ON CHASE-DB1")
        print("=" * 80)
        
        # Load dataset
        dataset = CHASEDataset(CHASE_PATH, img_size=(512, 512))
        
        # Split into train/val (80/20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED)
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        
        print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")
        
        # Create models
        ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
        img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
        unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
        
        # Stage 1: Train autoencoder with validation
        print("\nStage 1: Training Autoencoder")
        ae = train_autoencoder_with_validation(
            ae, train_loader, val_loader, device,
            epochs=AE_EPOCHS, lr=LR, patience=30
        )
        
        # Save autoencoder
        output_dir = Path("seed_outputs_properly_trained/CHASE-DB1")
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ae.state_dict(), output_dir / "mask_autoencoder.pth")
        print(f"Saved autoencoder to: {output_dir / 'mask_autoencoder.pth'}")
        
        # Freeze autoencoder
        for param in ae.parameters():
            param.requires_grad = False
        
        # Stage 2: Train diffusion model with monitoring
        print("\nStage 2: Training Diffusion Model")
        model = LatentDiffusionModel(
            ae=ae, img_enc=img_enc, unet=unet,
            cond_dim=128, timesteps=1000
        ).to(device)
        
        model = train_diffusion_with_monitoring(
            model, train_loader, val_loader, device,
            epochs=DIFF_EPOCHS, lr=LR, patience=50
        )
        
        # Save diffusion model
        torch.save({
            "ae": model.ae.state_dict(),
            "img_enc": model.img_enc.state_dict(),
            "unet": model.unet.state_dict(),
        }, output_dir / "diffusion_refiner.pth")
        print(f"Saved diffusion model to: {output_dir / 'diffusion_refiner.pth'}")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
    else:
        print(f"CHASE-DB1 data not found at: {CHASE_PATH}")