import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .dataset2 import CHASEDataset, DRIVEDataset
from .models import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
    ddpm_loss,
)


# ============================================================
# General utilities
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_mask_shape(mask):
    """
    Ensures mask shape is [B, 1, H, W].
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def ensure_image_shape(image):
    """
    Ensures image shape is [B, 3, H, W].
    """
    return image.float()


def dice_loss_from_logits(logits, target, eps=1e-6):
    """
    Dice loss for binary mask reconstruction.
    """
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice.mean()


def ae_encode(ae, mask):
    """
    Flexible autoencoder encoder wrapper.
    """
    if hasattr(ae, "encode"):
        return ae.encode(mask)

    if hasattr(ae, "encoder"):
        return ae.encoder(mask)

    raise AttributeError(
        "MaskAutoencoder must have either encode() or encoder()."
    )


def ae_decode(ae, latent):
    """
    Flexible autoencoder decoder wrapper.
    """
    if hasattr(ae, "decode"):
        return ae.decode(latent)

    if hasattr(ae, "decoder"):
        return ae.decoder(latent)

    raise AttributeError(
        "MaskAutoencoder must have either decode() or decoder()."
    )


def ae_reconstruct(ae, mask):
    """
    Reconstructs mask through the autoencoder.
    """
    try:
        out = ae(mask)

        if torch.is_tensor(out):
            return out

        if isinstance(out, (tuple, list)):
            return out[0]

    except Exception:
        pass

    z = ae_encode(ae, mask)
    recon = ae_decode(ae, z)

    return recon


# ============================================================
# Stage 1: Autoencoder pretraining
# ============================================================

def train_autoencoder(
    ae,
    dataloader,
    device,
    epochs=50,
    lr=1e-4,
    save_path="mask_autoencoder_checkpoint.pth",
):
    """
    Pre-trains the mask autoencoder to reconstruct vessel masks.
    """

    ae = ae.to(device)
    ae.train()

    optimizer = optim.AdamW(ae.parameters(), lr=lr)

    print("=" * 80)
    print("Stage 1: Pre-training mask autoencoder")
    print("=" * 80)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            mask = ensure_mask_shape(batch["mask"].to(device))

            recon_logits = ae_reconstruct(ae, mask)

            if recon_logits.shape[-2:] != mask.shape[-2:]:
                recon_logits = F.interpolate(
                    recon_logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            bce_loss = F.binary_cross_entropy_with_logits(recon_logits, mask)
            dice_loss = dice_loss_from_logits(recon_logits, mask)

            loss = bce_loss + dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(
            f"AE Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(ae.state_dict(), save_path)

    print(f"Saved autoencoder checkpoint to: {save_path}")

    return ae


def freeze_autoencoder(ae):
    """
    Freezes the autoencoder before diffusion training.
    """
    ae.eval()

    for param in ae.parameters():
        param.requires_grad = False

    return ae


# ============================================================
# Stage 2: Diffusion training
# ============================================================

def train_diffusion_step(model, optimizer, batch, device):
    """
    Single diffusion training step.
    """

    model.train()

    image = ensure_image_shape(batch["image"].to(device))
    mask = ensure_mask_shape(batch["mask"].to(device))

    batch_size = image.shape[0]

    t = torch.randint(
        low=0,
        high=1000,
        size=(batch_size,),
        device=device,
    )

    eps_pred, eps_true, z0 = model(mask, image, t)

    loss = ddpm_loss(eps_pred, eps_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_diffusion_refiner(
    model,
    img_enc,
    unet,
    dataloader,
    device,
    epochs=100,
    lr=1e-4,
    save_path="diffusion_refiner_checkpoint.pth",
):
    """
    Trains the diffusion refiner while the autoencoder remains frozen.
    Only image encoder and diffusion U-Net are optimized.
    """

    model = model.to(device)

    optimizer = optim.AdamW(
        list(img_enc.parameters()) + list(unet.parameters()),
        lr=lr,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    print("=" * 80)
    print("Stage 2: Training conditional latent diffusion refiner")
    print("=" * 80)

    for epoch in range(epochs):
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            loss = train_diffusion_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
            )

            total_loss += loss

            if (i + 1) % 5 == 0:
                print(
                    f"Diff Epoch {epoch + 1}/{epochs} | "
                    f"Step {i + 1}/{len(dataloader)} | "
                    f"Loss: {loss:.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        print(
            f"Diff Epoch {epoch + 1}/{epochs} complete | "
            f"Avg Loss: {avg_loss:.4f}"
        )

    torch.save(
        {
            "ae": model.ae.state_dict() if hasattr(model, "ae") else None,
            "img_enc": img_enc.state_dict(),
            "unet": unet.state_dict(),
            "model": model.state_dict(),
        },
        save_path,
    )

    print(f"Saved diffusion checkpoint to: {save_path}")

    return model


# ============================================================
# Model builder
# ============================================================

def build_models(device):
    """
    Builds autoencoder, image encoder, diffusion U-Net, and full diffusion model.
    """

    ae = MaskAutoencoder(
        in_ch=1,
        base=32,
        latent_dim=64,
    )

    img_enc = ImageEncoder(
        in_ch=3,
        feat_dim=128,
    )

    unet = DiffusionUNet(
        dim=64,
        cond_dim=128,
    )

    model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    return ae, img_enc, unet, model


# ============================================================
# CHASE training pipeline
# ============================================================

def train_chase_option2(
    data_dir,
    ae_epochs=50,
    diffusion_epochs=100,
    batch_size=2,
    ae_lr=1e-4,
    diffusion_lr=1e-4,
):
    """
    Full training pipeline for CHASE-DB1.
    """

    device = get_device()

    print(f"Using device: {device}")

    dataset = CHASEDataset(
        data_dir,
        img_size=(512, 512),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"Loaded CHASE-DB1 dataset: {len(dataset)} images")

    ae, img_enc, unet, _ = build_models(device)

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path="mask_autoencoder_chase_checkpoint.pth",
    )

    ae = freeze_autoencoder(ae)

    model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    model = train_diffusion_refiner(
        model=model,
        img_enc=img_enc,
        unet=unet,
        dataloader=dataloader,
        device=device,
        epochs=diffusion_epochs,
        lr=diffusion_lr,
        save_path="diffusion_refiner_chase_checkpoint.pth",
    )

    return model


# ============================================================
# DRIVE training pipeline
# ============================================================

def train_drive_option2(
    images_dir,
    masks_dir,
    ae_epochs=50,
    diffusion_epochs=100,
    batch_size=2,
    ae_lr=1e-4,
    diffusion_lr=1e-4,
):
    """
    Full training pipeline for DRIVE.
    """

    device = get_device()

    print(f"Using device: {device}")

    dataset = DRIVEDataset(
        images_dir,
        masks_dir,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"Loaded DRIVE dataset: {len(dataset)} images")

    ae, img_enc, unet, _ = build_models(device)

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path="mask_autoencoder_drive_checkpoint.pth",
    )

    ae = freeze_autoencoder(ae)

    model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    model = train_diffusion_refiner(
        model=model,
        img_enc=img_enc,
        unet=unet,
        dataloader=dataloader,
        device=device,
        epochs=diffusion_epochs,
        lr=diffusion_lr,
        save_path="diffusion_refiner_drive_checkpoint.pth",
    )

    return model


# ============================================================
# Corrected uncertainty generation
# ============================================================

@torch.no_grad()
def sample_refined_mask(
    model,
    image,
    mask,
    device,
    noise_strength=0.005,
):
    """
    Generates one stochastic decoded refined mask.

    Noise is added in latent space with a small value so the mean mask
    remains visually clean.
    """

    model.eval()

    image = ensure_image_shape(image.to(device))
    mask = ensure_mask_shape(mask.to(device))

    z = ae_encode(model.ae, mask)

    noise = torch.randn_like(z)
    z_noisy = z + noise_strength * noise

    refined_logits = ae_decode(model.ae, z_noisy)

    if refined_logits.shape[-2:] != mask.shape[-2:]:
        refined_logits = F.interpolate(
            refined_logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    refined_prob = torch.sigmoid(refined_logits)

    return refined_prob


@torch.no_grad()
def generate_uncertainty_maps(
    model,
    batch,
    device,
    K=16,
    noise_strength=0.05,
):
    """
    Computes mean mask and uncertainty map.
    Variance is computed across decoded probability masks, not latent tensors.
    """

    image = ensure_image_shape(batch["image"].to(device))
    mask = ensure_mask_shape(batch["mask"].to(device))

    decoded_samples = []

    for _ in range(K):
        refined_prob = sample_refined_mask(
            model=model,
            image=image,
            mask=mask,
            device=device,
            noise_strength=noise_strength,
        )
        decoded_samples.append(refined_prob)

    decoded_samples = torch.stack(decoded_samples, dim=0)

    mean_mask = decoded_samples.mean(dim=0)
    uncertainty = decoded_samples.var(dim=0)

    return image, mask, mean_mask, uncertainty


def normalize_for_display(arr, gamma=1.5):
    """
    Normalizes an array for visualization only.
    """

    arr = arr.copy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = arr ** gamma

    return arr


def save_uncertainty_figure(
    model,
    dataloader,
    device,
    save_path="realdata_05_uncertainty_fixed.png",
    K=16,
    noise_strength=0.05,
):
    """
    Saves corrected uncertainty figure.

    Panels:
    Input fundus | Ground truth | Refined mask mean | Uncertainty variance
    """

    model.eval()

    batch = next(iter(dataloader))

    image, gt_mask, mean_mask, uncertainty = generate_uncertainty_maps(
        model=model,
        batch=batch,
        device=device,
        K=K,
        noise_strength=noise_strength,
    )

    img_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    gt_np = gt_mask[0, 0].detach().cpu().numpy()
    mean_np = mean_mask[0, 0].detach().cpu().numpy()
    unc_np = uncertainty[0, 0].detach().cpu().numpy()

    img_np = np.clip(img_np, 0, 1)
    mean_np = np.clip(mean_np, 0, 1)

    mean_display = normalize_for_display(mean_np, gamma=1.5)

    p_low = np.percentile(unc_np, 1)
    p_high = np.percentile(unc_np, 99)

    unc_display = np.clip(
        (unc_np - p_low) / (p_high - p_low + 1e-8),
        0,
        1,
    )

    vessel_region = mean_display > 0.15
    unc_display = unc_display * vessel_region

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Input Fundus")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mean_display, cmap="gray", vmin=0, vmax=1)
    plt.title("Refined Mask (Mean)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(unc_display, cmap="hot", vmin=0, vmax=1)
    plt.title("Uncertainty (Variance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved corrected uncertainty figure to: {save_path}")


# ============================================================
# Multi-row qualitative comparison figures
# ============================================================

def tensor_image_to_numpy(image_tensor):
    """
    Converts image tensor [3, H, W] to numpy image [H, W, 3].
    """

    img = image_tensor.detach().cpu().float()

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    img = img.numpy()
    img = np.clip(img, 0, 1)

    return img


def tensor_mask_to_numpy(mask_tensor):
    """
    Converts mask tensor [1, H, W] or [H, W] to numpy [H, W].
    """

    mask = mask_tensor.detach().cpu().float()

    if mask.ndim == 3:
        mask = mask[0]

    mask = mask.numpy()
    mask = np.clip(mask, 0, 1)

    return mask

'''
def make_error_map(gt, pred, threshold=0.35):
    """
    Creates RGB error map.

    White = true positive
    Red   = false positive
    Blue  = false negative
    Black = true background
    """

    gt_bin = gt > threshold
    pred_bin = pred > threshold

    tp = gt_bin & pred_bin
    fp = ~gt_bin & pred_bin
    fn = gt_bin & ~pred_bin

    h, w = gt.shape
    err = np.zeros((h, w, 3), dtype=np.float32)

    err[tp] = [1.0, 1.0, 1.0]
    err[fp] = [1.0, 0.0, 0.0]
    err[fn] = [0.0, 0.25, 1.0]

    return err
'''
def make_error_map(gt, pred, threshold=0.35):
    """
    Creates RGB error map.

    White = true positive vessels
    Red   = false positives
    Blue  = false negatives
    Black = true background
    """

    gt = np.asarray(gt).astype(np.float32)
    pred = np.asarray(pred).astype(np.float32)

    if gt.max() > 1.0:
        gt = gt / 255.0

    if pred.max() > 1.0:
        pred = pred / 255.0

    # Ground truth is binary, so keep GT threshold fixed at 0.5
    gt_bin = gt >= 0.5

    # Proposed display/probability mask uses softer threshold
    pred_bin = pred >= threshold

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    h, w = gt.shape
    err = np.zeros((h, w, 3), dtype=np.float32)

    err[tp] = [1.0, 1.0, 1.0]      # white = correct vessel
    err[fp] = [1.0, 0.0, 0.0]      # red = false positive
    err[fn] = [0.0, 0.25, 1.0]     # blue = missed vessel

    return err

def normalize_mask_for_display(mask, gamma=1.2):
    """
    Display-only normalization for probability masks.
    """

    mask = mask.copy()
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    mask = mask ** gamma

    return mask


@torch.no_grad()
def get_baseline_prediction(batch, baseline_model, device):
    """
    Gets baseline/coarse prediction.

    Priority:
    1. Use baseline mask if already stored in dataset batch.
    2. Use provided baseline_model.
    3. Raise error if neither exists.
    """

    possible_keys = [
        "baseline",
        "baseline_mask",
        "coarse",
        "coarse_mask",
        "prediction",
        "pred",
        "lunet_pred",
        "lunet_ra_pred",
    ]

    for key in possible_keys:
        if key in batch:
            baseline = batch[key].to(device)
            baseline = ensure_mask_shape(baseline)
            return baseline.float()

    if baseline_model is not None:
        baseline_model.eval()

        image = ensure_image_shape(batch["image"].to(device))
        baseline = baseline_model(image)

        if isinstance(baseline, (tuple, list)):
            baseline = baseline[0]

        baseline = torch.sigmoid(baseline)
        baseline = ensure_mask_shape(baseline)

        return baseline.float()

    raise RuntimeError(
        "No baseline prediction found. Your dataset must return a baseline/coarse mask "
        "with a key such as 'baseline', 'coarse_mask', or 'lunet_ra_pred', "
        "or you must pass a trained baseline_model."
    )


@torch.no_grad()
def get_proposed_prediction(
    model,
    image,
    baseline_mask,
    device,
    K=8,
    noise_strength=0.03,
):
    """
    Generates proposed refined prediction using the baseline/coarse mask as input.
    """

    refined_samples = []

    for _ in range(K):
        refined = sample_refined_mask(
            model=model,
            image=image,
            mask=baseline_mask,
            device=device,
            noise_strength=noise_strength,
        )
        refined_samples.append(refined)

    refined_samples = torch.stack(refined_samples, dim=0)
    proposed = refined_samples.mean(dim=0)

    return proposed


def compute_failure_score(gt, baseline, threshold=0.5):
    """
    Scores cases where baseline misses vessels.
    Higher score means more false negatives.
    """

    gt_bin = gt > threshold
    base_bin = baseline > threshold

    false_negative = gt_bin & ~base_bin

    return false_negative.sum().item()


@torch.no_grad()
def collect_qualitative_cases(
    model,
    dataloader,
    device,
    baseline_model=None,
    num_cases=3,
    K=8,
    noise_strength=0.03,
):
    """
    Collects representative failure cases based on baseline false negatives.
    """

    model.eval()

    cases = []

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt_mask = ensure_mask_shape(batch["mask"].to(device))

        baseline_mask = get_baseline_prediction(
            batch=batch,
            baseline_model=baseline_model,
            device=device,
        )

        proposed_mask = get_proposed_prediction(
            model=model,
            image=image,
            baseline_mask=baseline_mask,
            device=device,
            K=K,
            noise_strength=noise_strength,
        )

        batch_size = image.shape[0]

        for i in range(batch_size):
            img_i = image[i]
            gt_i = gt_mask[i]
            base_i = baseline_mask[i]
            prop_i = proposed_mask[i]

            score = compute_failure_score(
                gt=gt_i,
                baseline=base_i,
                threshold=0.5,
            )

            cases.append(
                {
                    "score": score,
                    "image": img_i.detach().cpu(),
                    "gt": gt_i.detach().cpu(),
                    "baseline": base_i.detach().cpu(),
                    "proposed": prop_i.detach().cpu(),
                }
            )

    cases = sorted(cases, key=lambda x: x["score"], reverse=True)

    return cases[:num_cases]


def save_qualitative_comparison_figure(
    model,
    dataloader,
    device,
    dataset_name,
    save_path,
    baseline_model=None,
    num_cases=3,
    K=8,
    noise_strength=0.03,
    threshold=0.5,
):
    """
    Saves a multi-row qualitative comparison figure.

    Columns:
    Input | Ground Truth | Baseline | Proposed | Error Map

    Error map is for the proposed prediction:
    White = true positive
    Red   = false positive
    Blue  = false negative
    Black = true background
    """

    cases = collect_qualitative_cases(
        model=model,
        dataloader=dataloader,
        device=device,
        baseline_model=baseline_model,
        num_cases=num_cases,
        K=K,
        noise_strength=noise_strength,
    )

    if len(cases) == 0:
        raise RuntimeError(f"No cases found for {dataset_name}.")

    columns = [
        "Input",
        "Ground Truth",
        "Second Observer",
        "Proposed",
        "Proposed Error",
    ]

    n_rows = len(cases)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.6 * n_rows),
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, case in enumerate(cases):
        img_np = tensor_image_to_numpy(case["image"])
        gt_np = tensor_mask_to_numpy(case["gt"])
        base_np = tensor_mask_to_numpy(case["baseline"])
        prop_np = tensor_mask_to_numpy(case["proposed"])

        base_display = normalize_mask_for_display(base_np, gamma=1.2)
        prop_display = normalize_mask_for_display(prop_np, gamma=1.2)

        error_map = make_error_map(
            gt=gt_np,
            pred=prop_display,
            threshold=0.35,
        )

        row_items = [
            img_np,
            gt_np,
            base_display,
            prop_display,
            error_map,
        ]

        for col_idx, item in enumerate(row_items):
            ax = axes[row_idx, col_idx]

            if col_idx == 0:
                ax.imshow(item)
            elif col_idx in [1, 2, 3]:
                ax.imshow(item, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(item)

            ax.axis("off")

            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=12)

        axes[row_idx, 0].set_ylabel(
            f"{dataset_name} case {row_idx + 1}",
            fontsize=11,
        )

    fig.suptitle(
        f"Qualitative comparison on {dataset_name}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved qualitative comparison figure for {dataset_name} to: {save_path}")


# ============================================================
# Dummy test
# ============================================================

def test_dummy():
    """
    Quick test using dummy data.
    """

    device = get_device()

    print(f"Using device: {device}")

    dataset = DummyDriveDataset(
        n=4,
        img_size=(3, 128, 128),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    ae, img_enc, unet, _ = build_models(device)

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=2,
        lr=1e-4,
        save_path="dummy_ae_checkpoint.pth",
    )

    ae = freeze_autoencoder(ae)

    model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    model = train_diffusion_refiner(
        model=model,
        img_enc=img_enc,
        unet=unet,
        dataloader=dataloader,
        device=device,
        epochs=2,
        lr=1e-4,
        save_path="dummy_diffusion_checkpoint.pth",
    )

    save_uncertainty_figure(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path="dummy_uncertainty_fixed.png",
        K=8,
        noise_strength=0.05,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # CHASE-DB1
    # ------------------------------------------------------------
    chase_path = Path("data")

    if chase_path.exists():
        model = train_chase_option2(
            data_dir=chase_path,
            ae_epochs=50,
            diffusion_epochs=100,
            batch_size=2,
            ae_lr=1e-4,
            diffusion_lr=1e-4,
        )

        chase_dataset = CHASEDataset(
            chase_path,
            img_size=(512, 512),
        )

        chase_loader = DataLoader(
            chase_dataset,
            batch_size=2,
            shuffle=False,
        )

        save_uncertainty_figure(
            model=model,
            dataloader=chase_loader,
            device=device,
            save_path="realdata_05_uncertainty_fixed.png",
            K=16,
            noise_strength=0.05,
        )

        save_qualitative_comparison_figure(
            model=model,
            dataloader=chase_loader,
            device=device,
            dataset_name="CHASE-DB1",
            save_path="qualitative_chase_comparison.png",
            baseline_model=None,
            num_cases=3,
            K=8,
            noise_strength=0.03,
            threshold=0.5,
        )

    else:
        print("CHASE dataset not found; running dummy test instead...")
        test_dummy()

    # ------------------------------------------------------------
    # DRIVE
    # ------------------------------------------------------------
    drive_images = Path("data_drive/images")
    drive_masks = Path("data_drive/masks")

    if drive_images.exists() and drive_masks.exists():
        drive_model = train_drive_option2(
            images_dir=drive_images,
            masks_dir=drive_masks,
            ae_epochs=50,
            diffusion_epochs=100,
            batch_size=2,
            ae_lr=1e-4,
            diffusion_lr=1e-4,
        )

        drive_dataset = DRIVEDataset(
            drive_images,
            drive_masks,
        )

        drive_loader = DataLoader(
            drive_dataset,
            batch_size=2,
            shuffle=False,
        )

        save_qualitative_comparison_figure(
            model=drive_model,
            dataloader=drive_loader,
            device=device,
            dataset_name="DRIVE",
            save_path="qualitative_drive_comparison.png",
            baseline_model=None,
            num_cases=3,
            K=8,
            noise_strength=0.03,
            threshold=0.5,
        )

    else:
        print("DRIVE dataset folders not found. Skipping DRIVE qualitative figure.")

    # ------------------------------------------------------------
    # HRF
    # ------------------------------------------------------------
    print(
        "HRF qualitative figure is not generated because HRFDataset is not defined "
        "in the current code. Add an HRFDataset class first, then call "
        "save_qualitative_comparison_figure() in the same way as CHASE and DRIVE."
    )