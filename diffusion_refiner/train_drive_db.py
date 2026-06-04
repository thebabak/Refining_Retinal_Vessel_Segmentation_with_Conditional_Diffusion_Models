import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel, ddpm_loss

def drive_dataset(images_dir, masks_dir, img_size=(584, 565)):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    images = sorted(images_dir.glob("*.tif"))
    print(f"Found {len(images)} DRIVE images in {images_dir}")
    class DRIVEDataset(Dataset):
        def __len__(self):
            return len(images)
        def __getitem__(self, idx):
            img_path = images[idx]
            mask_name = img_path.stem + "_mask.gif"
            mask_path = masks_dir / mask_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_size)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return {'image': image, 'mask': mask, 'path': str(img_path)}
    return DRIVEDataset()

def train_drive_db(images_dir, masks_dir, epochs=5, batch_size=2, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = drive_dataset(images_dir, masks_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"Training for {epochs} epochs on {len(ds)} DRIVE images...")
    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} starting...")
        epoch_loss = 0.0
        for i, batch in enumerate(dl):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            B = image.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            loss = ddpm_loss(eps_pred, eps_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, step {i+1}/{len(dl)}, loss={loss:.4f}")
        avg_loss = epoch_loss / len(dl)
        scheduler.step()
        print(f"Epoch {epoch+1} complete, avg loss={avg_loss:.4f}\n")
    print("Training loop finished. Proceeding to save checkpoint...")
    ckpt_path = Path(__file__).parent / "diffusion_refiner_drive_checkpoint.pth"
    print("Saving checkpoint...")
    try:
        torch.save({
            'ae': ae.state_dict(),
            'img_enc': img_enc.state_dict(),
            'unet': unet.state_dict(),
            'model': model.state_dict(),
        }, ckpt_path)
        import os, time
        time.sleep(1)
        if ckpt_path.exists():
            print(f"Confirmed: DRIVE checkpoint saved to {ckpt_path}")
        else:
            print(f"ERROR: Checkpoint file not found after save: {ckpt_path}")
            raise RuntimeError("Checkpoint save step reached but file not found!")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise
    print("Training script completed.")

print("train_drive_db.py module loaded.")

if __name__ == "__main__":
    import sys
    print("__main__ block entered.")
    if len(sys.argv) < 3:
        print("Usage: python -m diffusion_refiner.train_drive_db <images_dir> <masks_dir>")
    else:
        images_dir = sys.argv[1]
        masks_dir = sys.argv[2]
        train_drive_db(images_dir, masks_dir)
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel, ddpm_loss

def drive_dataset(images_dir, masks_dir, img_size=(584, 565)):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    images = sorted(images_dir.glob("*.tif"))
    print(f"Found {len(images)} DRIVE images in {images_dir}")
    class DRIVEDataset(Dataset):
        def __len__(self):
            return len(images)
        def __getitem__(self, idx):
            img_path = images[idx]
            mask_name = img_path.stem + "_mask.gif"
            mask_path = masks_dir / mask_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_size)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return {'image': image, 'mask': mask, 'path': str(img_path)}
    return DRIVEDataset()

def train_drive_db(images_dir, masks_dir, epochs=5, batch_size=2, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = drive_dataset(images_dir, masks_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"Training for {epochs} epochs on {len(ds)} DRIVE images...")
    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} starting...")
        epoch_loss = 0.0
        for i, batch in enumerate(dl):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            B = image.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            loss = ddpm_loss(eps_pred, eps_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, step {i+1}/{len(dl)}, loss={loss:.4f}")
        avg_loss = epoch_loss / len(dl)
        scheduler.step()
        print(f"Epoch {epoch+1} complete, avg loss={avg_loss:.4f}\n")
    print("Training loop finished. Proceeding to save checkpoint...")
    ckpt_path = Path(__file__).parent / "diffusion_refiner_drive_checkpoint.pth"
    print(f"Attempting to save DRIVE checkpoint to {ckpt_path}")
    print("Saving checkpoint...")
    try:
        torch.save({
            'ae': ae.state_dict(),
            'img_enc': img_enc.state_dict(),
            'unet': unet.state_dict(),
            'model': model.state_dict(),
        }, ckpt_path)
        import os, time
        time.sleep(1)
        if ckpt_path.exists():
            print(f"Confirmed: DRIVE checkpoint saved to {ckpt_path}")
        else:
            print(f"ERROR: Checkpoint file not found after save: {ckpt_path}")
            raise RuntimeError("Checkpoint save step reached but file not found!")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise
    print("Training script completed.")
