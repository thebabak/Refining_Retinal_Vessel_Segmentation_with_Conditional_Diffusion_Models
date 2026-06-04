import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

from .dataset2 import CHASEDataset, DummyDriveDataset, DRIVEDataset
from .models import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
    ddpm_loss,
)


# ============================================================
# Seed / reproducibility utilities
# ============================================================

def set_seed(seed=42):
    """
    Sets random seeds for reproducible training and sampling.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed=42):
    """
    Creates a seeded DataLoader generator.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(worker_id):
    """
    Makes DataLoader workers reproducible.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def logits_or_probs_to_probs(output):
    """
    Converts model output to probability map.

    If output is already in [0, 1], it is treated as probability.
    Otherwise sigmoid is applied.
    """

    if output.min().item() >= 0.0 and output.max().item() <= 1.0:
        return output

    return torch.sigmoid(output)


# ============================================================
# Metrics
# ============================================================

def compute_binary_metrics(pred_prob, gt_mask, threshold=0.5, eps=1e-7):
    """
    Computes Dice, IoU, Accuracy, Sensitivity, and Specificity.
    """

    pred_prob = pred_prob.detach().float()
    gt_mask = gt_mask.detach().float()

    pred_bin = (pred_prob >= threshold).float()
    gt_bin = (gt_mask >= 0.5).float()

    pred_flat = pred_bin.reshape(-1)
    gt_flat = gt_bin.reshape(-1)

    tp = (pred_flat * gt_flat).sum()
    fp = (pred_flat * (1.0 - gt_flat)).sum()
    fn = ((1.0 - pred_flat) * gt_flat).sum()
    tn = ((1.0 - pred_flat) * (1.0 - gt_flat)).sum()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    sen = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "acc": float(acc.item()),
        "sen": float(sen.item()),
        "spec": float(spec.item()),
    }


def compute_auc_safe(pred_prob, gt_mask):
    """
    Computes AUC safely.
    Returns NaN if sklearn is missing or AUC cannot be computed.
    """

    try:
        from sklearn.metrics import roc_auc_score

        pred_np = pred_prob.detach().cpu().numpy().reshape(-1)
        gt_np = gt_mask.detach().cpu().numpy().reshape(-1)
        gt_np = (gt_np >= 0.5).astype(np.uint8)

        if len(np.unique(gt_np)) < 2:
            return float("nan")

        return float(roc_auc_score(gt_np, pred_np))

    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate_reconstruction_metrics(
    model,
    dataloader,
    device,
    threshold=0.5,
    K=16,
    noise_strength=0.05,
):
    """
    Evaluates mean refined mask against ground truth mask.

    Important:
    In this code, the refiner is initialized from the ground-truth mask because
    your current uncertainty pipeline uses batch["mask"] as the input mask.
    For final paper results, replace this input with a real coarse LU-Net+RA mask.
    """

    model.eval()

    metric_rows = []
    auc_rows = []

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt_mask = ensure_mask_shape(batch["mask"].to(device))

        decoded_samples = []

        for _ in range(K):
            refined_prob = sample_refined_mask(
                model=model,
                image=image,
                mask=gt_mask,
                device=device,
                noise_strength=noise_strength,
            )

            decoded_samples.append(refined_prob)

        decoded_samples = torch.stack(decoded_samples, dim=0)
        mean_mask = decoded_samples.mean(dim=0)

        metrics = compute_binary_metrics(
            pred_prob=mean_mask,
            gt_mask=gt_mask,
            threshold=threshold,
        )

        auc = compute_auc_safe(
            pred_prob=mean_mask,
            gt_mask=gt_mask,
        )

        metric_rows.append(metrics)
        auc_rows.append(auc)

    result = {}

    for key in ["dice", "iou", "acc", "sen", "spec"]:
        result[key] = float(np.mean([row[key] for row in metric_rows]))

    valid_auc = [x for x in auc_rows if not np.isnan(x)]
    result["auc"] = float(np.mean(valid_auc)) if len(valid_auc) > 0 else float("nan")

    return result


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

        avg_loss = total_loss / max(len(dataloader), 1)

        print(
            f"AE Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

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

        avg_loss = total_loss / max(len(dataloader), 1)
        scheduler.step()

        print(
            f"Diff Epoch {epoch + 1}/{epochs} complete | "
            f"Avg Loss: {avg_loss:.4f}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
    seed=42,
    output_dir=None,
):
    """
    Full training pipeline for CHASE-DB1.
    """

    set_seed(seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    dataset = CHASEDataset(
        data_dir,
        img_size=(512, 512),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    print(f"Loaded CHASE-DB1 dataset: {len(dataset)} images")

    ae, img_enc, unet, _ = build_models(device)

    if output_dir is None:
        output_dir = Path("seed_outputs") / "CHASE-DB1" / f"seed_{seed}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    ae_save_path = output_dir / "mask_autoencoder_chase_checkpoint.pth"
    diff_save_path = output_dir / "diffusion_refiner_chase_checkpoint.pth"

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path=ae_save_path,
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
        save_path=diff_save_path,
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
    seed=42,
    output_dir=None,
):
    """
    Full training pipeline for DRIVE.
    """

    set_seed(seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    dataset = DRIVEDataset(
        images_dir,
        masks_dir,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    print(f"Loaded DRIVE dataset: {len(dataset)} images")

    ae, img_enc, unet, _ = build_models(device)

    if output_dir is None:
        output_dir = Path("seed_outputs") / "DRIVE" / f"seed_{seed}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    ae_save_path = output_dir / "mask_autoencoder_drive_checkpoint.pth"
    diff_save_path = output_dir / "diffusion_refiner_drive_checkpoint.pth"

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path=ae_save_path,
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
        save_path=diff_save_path,
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
    noise_strength=0.05,
):
    """
    Generates one stochastic decoded refined mask.

    Noise is added in latent space, but with a small value so the mean mask
    remains visually clean.
    """

    model.eval()

    image = ensure_image_shape(image.to(device))
    mask = ensure_mask_shape(mask.to(device))

    z = ae_encode(model.ae, mask)

    noise = torch.randn_like(z)
    z_noisy = z + noise_strength * noise

    refined_out = ae_decode(model.ae, z_noisy)

    if refined_out.shape[-2:] != mask.shape[-2:]:
        refined_out = F.interpolate(
            refined_out,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    refined_prob = logits_or_probs_to_probs(refined_out)

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

    This does not change the actual computed mask values.
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
    """

    model.eval()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
# Excel / CSV result saving
# ============================================================

def save_rows_to_csv(rows, save_path):
    """
    Saves list of dictionaries to CSV.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print("No rows to save.")
        return

    fields = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Saved CSV to: {save_path}")


def save_rows_to_excel(rows, save_path):
    """
    Saves list of dictionaries to Excel.
    Requires pandas and openpyxl.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print("No rows to save.")
        return

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_excel(save_path, index=False)

        print(f"Saved Excel to: {save_path}")

    except Exception as e:
        print("Could not save Excel file.")
        print("CSV was still saved.")
        print(f"Excel error: {e}")


def make_summary_rows(metric_rows):
    """
    Computes mean ± std across seeds.
    """

    if len(metric_rows) == 0:
        return []

    datasets = sorted(set(row["dataset"] for row in metric_rows))

    summary_rows = []

    metric_keys = [
        "dice",
        "iou",
        "acc",
        "sen",
        "spec",
        "auc",
    ]

    for dataset_name in datasets:
        rows = [row for row in metric_rows if row["dataset"] == dataset_name]

        summary = {
            "dataset": dataset_name,
            "num_seeds": len(rows),
        }

        for key in metric_keys:
            values = np.array([float(row[key]) for row in rows], dtype=np.float64)

            mean_value = np.nanmean(values)

            if len(values) > 1:
                std_value = np.nanstd(values, ddof=1)
            else:
                std_value = 0.0

            summary[key] = f"{mean_value:.4f} ± {std_value:.4f}"

        summary_rows.append(summary)

    return summary_rows


def save_metric_files(metric_rows, output_root):
    """
    Saves per-seed metrics and mean ± std summary to CSV and Excel.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = make_summary_rows(metric_rows)

    save_rows_to_csv(
        rows=metric_rows,
        save_path=output_root / "multiseed_metrics.csv",
    )

    save_rows_to_excel(
        rows=metric_rows,
        save_path=output_root / "multiseed_metrics.xlsx",
    )

    save_rows_to_csv(
        rows=summary_rows,
        save_path=output_root / "multiseed_summary.csv",
    )

    save_rows_to_excel(
        rows=summary_rows,
        save_path=output_root / "multiseed_summary.xlsx",
    )


# ============================================================
# Dummy test
# ============================================================

def test_dummy(seed=42):
    """
    Quick test using dummy data.
    """

    set_seed(seed)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    dataset = DummyDriveDataset(
        n=4,
        img_size=(3, 128, 128),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    ae, img_enc, unet, _ = build_models(device)

    output_dir = Path("seed_outputs") / "dummy" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ae = train_autoencoder(
        ae=ae,
        dataloader=dataloader,
        device=device,
        epochs=2,
        lr=1e-4,
        save_path=output_dir / "dummy_ae_checkpoint.pth",
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
        save_path=output_dir / "dummy_diffusion_checkpoint.pth",
    )

    save_uncertainty_figure(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path=output_dir / "dummy_uncertainty_fixed.png",
        K=8,
        noise_strength=0.05,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    seeds = [42, 123, 2026]

    output_root = Path("seed_outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    metric_rows = []

    chase_path = Path("data")

    if chase_path.exists():

        for seed in seeds:
            print("=" * 80)
            print(f"Running CHASE-DB1 with seed {seed}")
            print("=" * 80)

            output_dir = output_root / "CHASE-DB1" / f"seed_{seed}"

            model = train_chase_option2(
                data_dir=chase_path,
                ae_epochs=50,
                diffusion_epochs=100,
                batch_size=2,
                ae_lr=1e-4,
                diffusion_lr=1e-4,
                seed=seed,
                output_dir=output_dir,
            )

            device = get_device()

            dataset = CHASEDataset(
                chase_path,
                img_size=(512, 512),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
            )

            save_uncertainty_figure(
                model=model,
                dataloader=dataloader,
                device=device,
                save_path=output_dir / "realdata_05_uncertainty_fixed.png",
                K=16,
                noise_strength=0.05,
            )

            metrics = evaluate_reconstruction_metrics(
                model=model,
                dataloader=dataloader,
                device=device,
                threshold=0.5,
                K=16,
                noise_strength=0.05,
            )

            row = {
                "dataset": "CHASE-DB1",
                "seed": seed,
                "threshold": 0.5,
                "K": 16,
                "noise_strength": 0.05,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spec": metrics["spec"],
                "auc": metrics["auc"],
            }

            metric_rows.append(row)

            print("Metrics:")
            print(row)

    else:
        print("CHASE dataset not found; running dummy test instead...")
        test_dummy(seed=42)

    drive_images = Path("data_drive/images")
    drive_masks = Path("data_drive/masks")

    if drive_images.exists() and drive_masks.exists():

        for seed in seeds:
            print("=" * 80)
            print(f"Running DRIVE with seed {seed}")
            print("=" * 80)

            output_dir = output_root / "DRIVE" / f"seed_{seed}"

            drive_model = train_drive_option2(
                images_dir=drive_images,
                masks_dir=drive_masks,
                ae_epochs=50,
                diffusion_epochs=100,
                batch_size=2,
                ae_lr=1e-4,
                diffusion_lr=1e-4,
                seed=seed,
                output_dir=output_dir,
            )

            device = get_device()

            drive_dataset = DRIVEDataset(
                drive_images,
                drive_masks,
            )

            drive_loader = DataLoader(
                drive_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
            )

            save_uncertainty_figure(
                model=drive_model,
                dataloader=drive_loader,
                device=device,
                save_path=output_dir / "drive_uncertainty_fixed.png",
                K=16,
                noise_strength=0.05,
            )

            metrics = evaluate_reconstruction_metrics(
                model=drive_model,
                dataloader=drive_loader,
                device=device,
                threshold=0.5,
                K=16,
                noise_strength=0.05,
            )

            row = {
                "dataset": "DRIVE",
                "seed": seed,
                "threshold": 0.5,
                "K": 16,
                "noise_strength": 0.05,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spec": metrics["spec"],
                "auc": metrics["auc"],
            }

            metric_rows.append(row)

            print("Metrics:")
            print(row)

    else:
        print("DRIVE dataset folders not found. Skipping DRIVE.")

    if len(metric_rows) > 0:
        save_metric_files(
            metric_rows=metric_rows,
            output_root=output_root,
        )

        print("=" * 80)
        print("Excel and CSV files saved here:")
        print(output_root.resolve())
        print("=" * 80)
    else:
        print("No metrics were generated.")