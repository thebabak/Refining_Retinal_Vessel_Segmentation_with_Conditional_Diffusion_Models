"""
Train new model with current architecture (~443K params).
Works WITHOUT pre-computed coarse masks.
Save as: diffusion_refiner/train_new.py
Run: python -m diffusion_refiner.train_new
"""

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from diffusion_refiner.models_corrected import build_model


class SimpleChaseDB1(Dataset):
    """Simple CHASE-DB1 dataset - no coarse masks needed."""
    
    def __init__(self, data_dir, img_size=(512, 512)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples = []
        
        for f in sorted(self.data_dir.glob("*.jpg")):
            name = f.stem
            if "_1stHO" in name or "_2ndHO" in name:
                continue
            mask_path = self.data_dir / f"{name}_1stHO.png"
            if mask_path.exists():
                self.samples.append((f, mask_path))
        
        print(f"Found {len(self.samples)} image-mask pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.img_size, Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = (mask > 0.5).float().unsqueeze(0)
        
        # Create synthetic coarse mask (add noise + erode thin vessels)
        coarse = mask.clone()
        coarse = coarse + torch.randn_like(coarse) * 0.2
        coarse = torch.clamp(coarse, 0, 1)
        # Randomly remove some vessel pixels
        rand_mask = torch.rand_like(mask)
        thin = (mask > 0.5) & (rand_mask < 0.15)
        coarse[thin] = 0.0
        
        return {"image": img, "mask": mask, "coarse": coarse}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # ================================================================
    # Data
    # ================================================================
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found!")
        return
    
    dataset = SimpleChaseDB1(data_dir, img_size=(512, 512))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(f"Training images: {len(dataset)}")
    print()
    
    # ================================================================
    # Model
    # ================================================================
    model, counts = build_model(device)
    print(f"Model: {counts['total']:,} params")
    print(f"  AE: {counts['ae']:,}")
    print(f"  IE: {counts['ie']:,}")
    print(f"  UN: {counts['un']:,}")
    print()
    
    # ================================================================
    # Stage 1: Train Autoencoder
    # ================================================================
    print("=" * 60)
    print("Stage 1: Training Autoencoder (100 epochs)")
    print("=" * 60)
    
    opt_ae = optim.AdamW(model.ae.parameters(), lr=1e-4, weight_decay=1e-5)
    best_loss = float('inf')
    
    for epoch in range(100):
        model.ae.train()
        total_loss = 0.0
        
        for batch in loader:
            mask = batch["mask"].to(device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            recon = model.ae(mask)
            loss = F.mse_loss(recon, mask)
            
            # Add Dice loss for better reconstruction
            probs = torch.sigmoid(recon)
            probs_flat = probs.view(probs.size(0), -1)
            target_flat = mask.view(mask.size(0), -1)
            intersection = (probs_flat * target_flat).sum(1)
            dice = (2.0 * intersection + 1e-6) / (probs_flat.sum(1) + target_flat.sum(1) + 1e-6)
            dice_loss = (1.0 - dice).mean()
            
            total_loss_val = loss + 0.5 * dice_loss
            
            opt_ae.zero_grad()
            total_loss_val.backward()
            opt_ae.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/100 | MSE: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    print(f"  AE complete. Best MSE: {best_loss:.6f}")
    
    # Freeze autoencoder
    for p in model.ae.parameters():
        p.requires_grad = False
    model.ae.eval()
    print()
    
    # ================================================================
    # Stage 2: Train Diffusion
    # ================================================================
    print("=" * 60)
    print("Stage 2: Training Diffusion Refiner (200 epochs)")
    print("=" * 60)
    
    opt_diff = optim.AdamW(
        list(model.img_enc.parameters()) + list(model.unet.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_diff, T_max=200)
    best_loss = float('inf')
    
    for epoch in range(200):
        model.img_enc.train()
        model.unet.train()
        total_loss = 0.0
        
        for batch in loader:
            image = batch["image"].to(device).float()
            mask = batch["mask"].to(device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            
            bs = image.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, 1000, (bs,), device=device)
            
            # Forward diffusion
            eps_pred, eps_true, z0 = model(mask, image, t)
            
            # MSE loss on noise prediction
            loss = F.mse_loss(eps_pred, eps_true)
            
            opt_diff.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_diff.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 50 == 0:
            lr = opt_diff.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/200 | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {lr:.2e}")
    
    print(f"  Diffusion complete. Best loss: {best_loss:.6f}")
    print()
    
    # ================================================================
    # Save
    # ================================================================
    output_dir = Path("seed_outputs_new")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "diffusion_refiner.pth"
    
    torch.save({
        "ae": model.ae.state_dict(),
        "img_enc": model.img_enc.state_dict(),
        "unet": model.unet.state_dict(),
    }, save_path)
    
    print(f"✓ Model saved to: {save_path.resolve()}")
    print(f"  Parameters: {counts['total']:,}")
    print(f"  Best diffusion loss: {best_loss:.6f}")
    print()
    print("Done! Now run: python -m diffusion_refiner.evaluate_uncertainty")


if __name__ == "__main__":
    train()