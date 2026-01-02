import torch
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from .dataset import CHASEDataset, DummyDriveDataset
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel, ddpm_loss


def train_step(model, optimizer, batch, device):
    """Single training step."""
    model.train()
    image = batch['image'].to(device)
    mask = batch['mask'].to(device)
    B = image.shape[0]
    t = torch.randint(0, 1000, (B,), device=device)
    eps_pred, eps_true, z0 = model(mask, image, t)
    loss = ddpm_loss(eps_pred, eps_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_chase(data_dir, epochs=5, batch_size=2, lr=2e-4):
    """Train diffusion refiner on CHASE-DB1 dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CHASE dataset
    ds = CHASEDataset(data_dir, img_size=(512, 512))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Instantiate models
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training for {epochs} epochs on {len(ds)} images...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(dl):
            loss = train_step(model, optimizer, batch, device)
            epoch_loss += loss
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, step {i+1}/{len(dl)}, loss={loss:.4f}")
        
        avg_loss = epoch_loss / len(dl)
        scheduler.step()
        print(f"Epoch {epoch+1} complete, avg loss={avg_loss:.4f}\n")
    
    # Save checkpoint
    ckpt_path = Path("diffusion_refiner_checkpoint.pth")
    torch.save({
        'ae': ae.state_dict(),
        'img_enc': img_enc.state_dict(),
        'unet': unet.state_dict(),
        'model': model.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def test_dummy():
    """Quick test on dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ds = DummyDriveDataset(n=4, img_size=(3, 128, 128))
    dl = DataLoader(ds, batch_size=2)
    for batch in dl:
        loss = train_step(model, optimizer, batch, device)
        print('dummy loss', loss)


if __name__ == '__main__':
    # Check if CHASE data exists
    chase_path = Path("data")
    if chase_path.exists():
        train_chase(chase_path, epochs=5, batch_size=2)
    else:
        print("CHASE dataset not found; running dummy test instead...")
        test_dummy()
