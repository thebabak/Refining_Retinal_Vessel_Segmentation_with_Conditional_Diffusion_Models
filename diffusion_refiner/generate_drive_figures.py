import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel
from .train_drive_db import drive_dataset

def generate_drive_figures(images_dir, masks_dir, checkpoint_path, out_dir="plots/drive/figures"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = drive_dataset(images_dir, masks_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Load model
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    ckpt = torch.load(str(Path(__file__).parent / checkpoint_path), map_location=device)
    ae.load_state_dict(ckpt['ae'])
    img_enc.load_state_dict(ckpt['img_enc'])
    unet.load_state_dict(ckpt['unet'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    # 1. mask_examples.png
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        sample = ds[i]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            t = torch.tensor([999], device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            decoded_mask = ae.decode(z0)
            pred_mask = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        img_np = np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0))
        pred_mask_resized = cv2.resize(pred_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        axs[i//5, i%5].imshow(img_np)
        axs[i//5, i%5].imshow(pred_mask_resized, cmap='Reds', alpha=0.5)
        axs[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "mask_examples.png")
    plt.close(fig)
    print("Saved mask_examples.png")
    # 2. vessel_statistics.png
    vessel_pixels = []
    for i in range(len(ds)):
        sample = ds[i]
        mask = sample['mask'].unsqueeze(0).to(device)
        image = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            t = torch.tensor([999], device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            decoded_mask = ae.decode(z0)
            pred_mask = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        vessel_pixels.append(np.sum(pred_mask))
    plt.figure(figsize=(8, 5))
    plt.hist(vessel_pixels, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Vessel Pixel Count')
    plt.ylabel('Frequency')
    plt.title('Vessel Statistics (DRIVE)')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "vessel_statistics.png")
    plt.close()
    print("Saved vessel_statistics.png")
    # 3. vessel_thickness.png
    thicknesses = []
    for i in range(len(ds)):
        sample = ds[i]
        mask = sample['mask'].unsqueeze(0).to(device)
        image = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            t = torch.tensor([999], device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            decoded_mask = ae.decode(z0)
            pred_mask = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        # Estimate thickness by counting vessel pixels per row
        row_thickness = np.mean(np.sum(pred_mask, axis=1))
        thicknesses.append(row_thickness)
    plt.figure(figsize=(8, 5))
    plt.hist(thicknesses, bins=20, color='green', alpha=0.7)
    plt.xlabel('Average Vessel Thickness (pixels)')
    plt.ylabel('Frequency')
    plt.title('Vessel Thickness (DRIVE)')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "vessel_thickness.png")
    plt.close()
    print("Saved vessel_thickness.png")
    # 4. uncertainty.png
    # Use ensemble sampling for uncertainty
    uncertainties = []
    for i in range(5):
        sample = ds[i]
        mask = sample['mask'].unsqueeze(0).to(device)
        image = sample['image'].unsqueeze(0).to(device)
        ensemble = []
        for _ in range(10):
            t = torch.randint(900, 1000, (1,), device=device)
            with torch.no_grad():
                eps_pred, eps_true, z0 = model(mask, image, t)
                decoded_mask = ae.decode(z0)
                pred_mask = decoded_mask.squeeze().cpu().numpy()
                ensemble.append(pred_mask)
        ensemble = np.stack(ensemble)
        uncertainty = np.var(ensemble, axis=0)
        uncertainties.append(uncertainty)
        plt.figure(figsize=(6, 6))
        plt.imshow(np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0)))
        plt.imshow(uncertainty, cmap='hot', alpha=0.5)
        plt.axis('off')
        plt.title(f'Uncertainty Example {i+1}')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"uncertainty_{i+1}.png")
        plt.close()
    print("Saved uncertainty_*.png")
    # 5. single_vs_ensemble.png
    sample = ds[0]
    mask = sample['mask'].unsqueeze(0).to(device)
    image = sample['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        t = torch.tensor([999], device=device)
        eps_pred, eps_true, z0 = model(mask, image, t)
        decoded_mask = ae.decode(z0)
        single_pred = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    ensemble = []
    for _ in range(10):
        t = torch.randint(900, 1000, (1,), device=device)
        with torch.no_grad():
            eps_pred, eps_true, z0 = model(mask, image, t)
            decoded_mask = ae.decode(z0)
            pred_mask = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            ensemble.append(pred_mask)
    ensemble_mean = np.mean(ensemble, axis=0)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0)))
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(single_pred, cmap='gray')
    plt.title('Single Prediction')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(ensemble_mean, cmap='gray')
    plt.title('Ensemble Mean')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "single_vs_ensemble.png")
    plt.close()
    print("Saved single_vs_ensemble.png")
    # 6. uncertainty_ensemble.png
    plt.figure(figsize=(8, 6))
    plt.imshow(np.var(ensemble, axis=0), cmap='hot')
    plt.title('Uncertainty (Ensemble)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "uncertainty_ensemble.png")
    plt.close()
    print("Saved uncertainty_ensemble.png")

if __name__ == "__main__":
    generate_drive_figures(
        images_dir="F:/PHD/AI in Med/paper3/DRIVE/training/images",
        masks_dir="F:/PHD/AI in Med/paper3/DRIVE/training/mask",
        checkpoint_path="diffusion_refiner_drive_checkpoint.pth"
    )
