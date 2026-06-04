import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from .models import MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel
from .train_drive_db import drive_dataset

def plot_drive_results(images_dir, masks_dir, checkpoint_path, out_dir="plots/drive/results"):
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
    # Inference and plotting
    for i in range(len(ds)):
        sample = ds[i]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            t = torch.tensor([999], device=device)
            eps_pred, eps_true, z0 = model(mask, image, t)
            decoded_mask = ae.decode(z0)
            pred_mask = (decoded_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        # Overlay prediction on image
        img_np = np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0))
        # Resize pred_mask to match image shape
        pred_mask_resized = cv2.resize(pred_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = img_np.copy()
        overlay[pred_mask_resized == 1] = [1, 0, 0]  # Red overlay for vessels
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[1].imshow(overlay)
        axs[1].set_title("Prediction Overlay")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        out_path = Path(out_dir) / f"drive_result_{i+1}.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_drive_results(
        images_dir="F:/PHD/AI in Med/paper3/DRIVE/training/images",
        masks_dir="F:/PHD/AI in Med/paper3/DRIVE/training/mask",
        checkpoint_path="diffusion_refiner_drive_checkpoint.pth"
    )
