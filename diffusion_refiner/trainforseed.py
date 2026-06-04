import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset2 import CHASEDataset, DummyDriveDataset, DRIVEDataset
from .models import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
    ddpm_loss,
    dice_loss,
)
from .utils import timestep_embedding


# ============================================================
# Seed / reproducibility
# ============================================================

def set_seed(seed=42):
    """
    Set seed for reproducible training.
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
    Create seeded DataLoader generator.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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


# ============================================================
# Training step
# ============================================================

def train_step(model, optimizer, batch, device):
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

    loss_ddpm = ddpm_loss(eps_pred, eps_true)

    # Auxiliary segmentation supervision improves overlap-based metrics.
    a_t = model.scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
    z0_pred = (model.scheduler.q_sample(z0, t, eps_true) - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
    pred_mask = ae_decode(model.ae, z0_pred)
    loss_dice = dice_loss(pred_mask, mask).mean()

    loss = loss_ddpm + 0.5 * loss_dice

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================
# Autoencoder helper functions
# ============================================================

def ae_encode(ae, mask):
    """
    Flexible encoder wrapper.
    """

    if hasattr(ae, "encode"):
        return ae.encode(mask)

    if hasattr(ae, "encoder"):
        return ae.encoder(mask)

    raise AttributeError("MaskAutoencoder must have encode() or encoder().")


def ae_decode(ae, latent):
    """
    Flexible decoder wrapper.
    """

    if hasattr(ae, "decode"):
        return ae.decode(latent)

    if hasattr(ae, "decoder"):
        return ae.decoder(latent)

    raise AttributeError("MaskAutoencoder must have decode() or decoder().")


# ============================================================
# Prediction helpers
# ============================================================

@torch.no_grad()
def sample_refined_mask(
    model,
    image,
    mask,
    device,
    noise_strength=0.005,
    num_steps=20,
    guidance_scale=1.0,
):
    """
    Generates one stochastic decoded refined mask.
    """

    model.eval()

    image = ensure_image_shape(image.to(device))
    mask = ensure_mask_shape(mask.to(device))

    z0 = ae_encode(model.ae, mask)
    cond = model.img_enc(image)
    cond = model.cond_mlp(cond)

    timesteps = model.scheduler.timesteps
    start_timestep = int(max(1, min(timesteps - 1, round(noise_strength * timesteps))))
    t_start = torch.full((z0.shape[0],), start_timestep, device=device, dtype=torch.long)

    noise = torch.randn_like(z0)
    z = model.scheduler.q_sample(z0, t_start, noise)

    step_indices = torch.linspace(start_timestep, 0, num_steps + 1, device=device).long()

    for i in range(num_steps):
        t = step_indices[i].repeat(z.shape[0])
        t_next = step_indices[i + 1].repeat(z.shape[0])

        t_emb = timestep_embedding(t.float(), 64).to(device)
        eps_pred = model.unet(z, t_emb, cond)

        if guidance_scale != 1.0:
            null_cond = torch.zeros_like(cond)
            eps_uncond = model.unet(z, t_emb, null_cond)
            eps_pred = eps_uncond + guidance_scale * (eps_pred - eps_uncond)

        alphas_cumprod = model.scheduler.alphas_cumprod.to(device)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_next = alphas_cumprod[t_next].view(-1, 1, 1, 1)

        z0_pred = (z - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        z = torch.sqrt(alpha_next) * z0_pred + torch.sqrt(1 - alpha_next) * eps_pred

    refined_prob = ae_decode(model.ae, z)

    if refined_prob.shape[-2:] != mask.shape[-2:]:
        refined_prob = F.interpolate(
            refined_prob,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return refined_prob


@torch.no_grad()
def get_baseline_prediction(batch, baseline_model, device):
    """
    Gets baseline/coarse prediction if available.

    Priority:
    1. baseline/coarse mask stored in dataset batch
    2. provided baseline_model
    3. ground-truth mask as fallback for debugging only

    Important:
    For real paper results, do not use the GT fallback.
    You should provide a real coarse mask from LU-Net+RA.
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

    print(
        "WARNING: No baseline/coarse mask found. "
        "Using ground-truth mask as fallback. "
        "Do NOT use this fallback for final paper results."
    )

    fallback = ensure_mask_shape(batch["mask"].to(device))

    return fallback.float()


@torch.no_grad()
def get_proposed_prediction(
    model,
    image,
    baseline_mask,
    device,
    K=16,
    noise_strength=0.005,
):
    """
    Generates proposed refined prediction using K stochastic samples.
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
        "dice": dice.item(),
        "iou": iou.item(),
        "acc": acc.item(),
        "sen": sen.item(),
        "spec": spec.item(),
    }


def compute_auc_safe(pred_prob, gt_mask):
    """
    Computes AUC if sklearn is available.
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
def evaluate_model(
    model,
    dataloader,
    device,
    baseline_model=None,
    K=16,
    noise_strength=0.005,
    threshold=0.5,
):
    """
    Evaluates one trained model.
    """

    model.eval()

    metric_rows = []
    auc_rows = []

    total_time = 0.0
    total_images = 0

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt_mask = ensure_mask_shape(batch["mask"].to(device))

        baseline_mask = get_baseline_prediction(
            batch=batch,
            baseline_model=baseline_model,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        proposed_mask = get_proposed_prediction(
            model=model,
            image=image,
            baseline_mask=baseline_mask,
            device=device,
            K=K,
            noise_strength=noise_strength,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time += end_time - start_time
        total_images += image.shape[0]

        metrics = compute_binary_metrics(
            pred_prob=proposed_mask,
            gt_mask=gt_mask,
            threshold=threshold,
        )

        auc = compute_auc_safe(
            pred_prob=proposed_mask,
            gt_mask=gt_mask,
        )

        metric_rows.append(metrics)
        auc_rows.append(auc)

    results = {}

    for key in ["dice", "iou", "acc", "sen", "spec"]:
        results[key] = float(np.mean([row[key] for row in metric_rows]))

    valid_auc = [x for x in auc_rows if not np.isnan(x)]
    results["auc"] = float(np.mean(valid_auc)) if len(valid_auc) > 0 else float("nan")

    if total_images > 0:
        time_per_image = total_time / total_images
        results["time_ms"] = float(time_per_image * 1000.0)
        results["fps"] = float(1.0 / time_per_image)
    else:
        results["time_ms"] = float("nan")
        results["fps"] = float("nan")

    return results


@torch.no_grad()
def find_best_threshold(
    model,
    dataloader,
    device,
    baseline_model=None,
    K=16,
    noise_strength=0.005,
    thresholds=None,
):
    """
    Finds best threshold based on Dice.
    This is useful because your AUC can be high while threshold=0.5 is bad.
    """

    if thresholds is None:
        thresholds = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]

    best_threshold = 0.5
    best_dice = -1.0

    for threshold in thresholds:
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            baseline_model=baseline_model,
            K=K,
            noise_strength=noise_strength,
            threshold=threshold,
        )

        dice = metrics["dice"]

        print(f"Threshold {threshold:.2f} | Dice {dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} | Best Dice: {best_dice:.4f}")

    return best_threshold, best_dice


# ============================================================
# CHASE training
# ============================================================

def train_chase(
    data_dir,
    epochs=50,
    batch_size=2,
    lr=2e-4,
    seed=42,
    save_dir="checkpoints",
):
    """
    Train diffusion refiner on CHASE-DB1 dataset.
    """

    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    save_dir = Path(save_dir) / "CHASE" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load CHASE dataset
    ds = CHASEDataset(
        data_dir,
        img_size=(512, 512),
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
    )

    # Instantiate models
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    print(f"Training for {epochs} epochs on {len(ds)} CHASE images...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, batch in enumerate(dl):
            loss = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
            )

            epoch_loss += loss

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"step {i + 1}/{len(dl)}, "
                    f"loss={loss:.4f}"
                )

        avg_loss = epoch_loss / len(dl)
        scheduler.step()

        print(
            f"Epoch {epoch + 1} complete, "
            f"avg loss={avg_loss:.4f}\n"
        )

    # Save checkpoint
    ckpt_path = save_dir / "diffusion_refiner_chase_checkpoint.pth"

    torch.save(
        {
            "seed": seed,
            "ae": ae.state_dict(),
            "img_enc": img_enc.state_dict(),
            "unet": unet.state_dict(),
            "model": model.state_dict(),
        },
        ckpt_path,
    )

    print(f"Saved CHASE checkpoint to {ckpt_path}")

    return model


# ============================================================
# DRIVE training
# ============================================================

def train_drive(
    images_dir,
    masks_dir,
    epochs=50,
    batch_size=2,
    lr=2e-4,
    seed=42,
    save_dir="checkpoints",
):
    """
    Train diffusion refiner on DRIVE dataset.
    """

    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    save_dir = Path(save_dir) / "DRIVE" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load DRIVE dataset
    ds = DRIVEDataset(
        images_dir,
        masks_dir,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
    )

    # Instantiate models
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    print(f"Training for {epochs} epochs on {len(ds)} DRIVE images...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, batch in enumerate(dl):
            loss = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
            )

            epoch_loss += loss

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"step {i + 1}/{len(dl)}, "
                    f"loss={loss:.4f}"
                )

        avg_loss = epoch_loss / len(dl)
        scheduler.step()

        print(
            f"Epoch {epoch + 1} complete, "
            f"avg loss={avg_loss:.4f}\n"
        )

    # Save checkpoint
    ckpt_path = save_dir / "diffusion_refiner_drive_checkpoint.pth"

    torch.save(
        {
            "seed": seed,
            "ae": ae.state_dict(),
            "img_enc": img_enc.state_dict(),
            "unet": unet.state_dict(),
            "model": model.state_dict(),
        },
        ckpt_path,
    )

    print(f"Saved DRIVE checkpoint to {ckpt_path}")

    return model


# ============================================================
# Dummy test
# ============================================================

def test_dummy(seed=42):
    """
    Quick test on dummy data.
    """

    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
    )

    ds = DummyDriveDataset(
        n=4,
        img_size=(3, 128, 128),
    )

    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
    )

    for batch in dl:
        loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
        )

        print("dummy loss", loss)


# ============================================================
# CSV helpers
# ============================================================

def save_seed_results_csv(rows, save_path):
    """
    Saves per-seed results.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "seed",
        "threshold",
        "dice",
        "iou",
        "acc",
        "sen",
        "spec",
        "auc",
        "time_ms",
        "fps",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    print(f"Saved per-seed results to: {save_path}")


def mean_std(values):
    """
    Computes mean and sample standard deviation.
    """

    values = np.asarray(values, dtype=np.float64)

    mean_value = float(np.nanmean(values))

    if len(values) > 1:
        std_value = float(np.nanstd(values, ddof=1))
    else:
        std_value = 0.0

    return mean_value, std_value


def save_summary_csv(rows, save_path):
    """
    Saves mean ± std summary.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(row["dataset"] for row in rows))

    fields = [
        "dataset",
        "threshold",
        "dice",
        "iou",
        "acc",
        "sen",
        "spec",
        "auc",
        "time_ms",
        "fps",
    ]

    summary_rows = []

    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]

        summary = {"dataset": dataset}

        for metric in fields[1:]:
            vals = [row[metric] for row in dataset_rows]
            m, s = mean_std(vals)

            if metric == "threshold":
                summary[metric] = f"{m:.2f} ± {s:.2f}"
            elif metric in ["time_ms", "fps"]:
                summary[metric] = f"{m:.2f} ± {s:.2f}"
            else:
                summary[metric] = f"{m:.4f} ± {s:.4f}"

        summary_rows.append(summary)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print(f"Saved summary to: {save_path}")

    print("\nMean ± std summary:")
    for row in summary_rows:
        print(row)


# ============================================================
# Multi-seed runners
# ============================================================

def run_chase_multiseed(
    data_dir,
    seeds=(42, 123, 2026),
    epochs=50,
    batch_size=2,
    lr=2e-4,
    K=16,
    noise_strength=0.005,
    use_best_threshold=True,
):
    """
    Run CHASE training and evaluation for multiple seeds.
    """

    device = get_device()
    results = []

    for seed in seeds:
        print("=" * 80)
        print(f"CHASE-DB1 training with seed {seed}")
        print("=" * 80)

        model = train_chase(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            save_dir="checkpoints",
        )

        eval_dataset = CHASEDataset(
            data_dir,
            img_size=(512, 512),
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        if use_best_threshold:
            threshold, _ = find_best_threshold(
                model=model,
                dataloader=eval_loader,
                device=device,
                baseline_model=None,
                K=K,
                noise_strength=noise_strength,
            )
        else:
            threshold = 0.5

        metrics = evaluate_model(
            model=model,
            dataloader=eval_loader,
            device=device,
            baseline_model=None,
            K=K,
            noise_strength=noise_strength,
            threshold=threshold,
        )

        metrics["dataset"] = "CHASE-DB1"
        metrics["seed"] = seed
        metrics["threshold"] = threshold

        results.append(metrics)

        print(f"Seed {seed} metrics:")
        print(metrics)

    return results


def run_drive_multiseed(
    images_dir,
    masks_dir,
    seeds=(42, 123, 2026),
    epochs=50,
    batch_size=2,
    lr=2e-4,
    K=16,
    noise_strength=0.005,
    use_best_threshold=True,
):
    """
    Run DRIVE training and evaluation for multiple seeds.
    """

    device = get_device()
    results = []

    for seed in seeds:
        print("=" * 80)
        print(f"DRIVE training with seed {seed}")
        print("=" * 80)

        model = train_drive(
            images_dir=images_dir,
            masks_dir=masks_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            save_dir="checkpoints",
        )

        eval_dataset = DRIVEDataset(
            images_dir,
            masks_dir,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        if use_best_threshold:
            threshold, _ = find_best_threshold(
                model=model,
                dataloader=eval_loader,
                device=device,
                baseline_model=None,
                K=K,
                noise_strength=noise_strength,
            )
        else:
            threshold = 0.5

        metrics = evaluate_model(
            model=model,
            dataloader=eval_loader,
            device=device,
            baseline_model=None,
            K=K,
            noise_strength=noise_strength,
            threshold=threshold,
        )

        metrics["dataset"] = "DRIVE"
        metrics["seed"] = seed
        metrics["threshold"] = threshold

        results.append(metrics)

        print(f"Seed {seed} metrics:")
        print(metrics)

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    seeds = [42, 123, 2026]

    all_results = []

    results_dir = Path("multiseed_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # CHASE
    # ------------------------------------------------------------

    chase_path = Path("data")

    if chase_path.exists():
        chase_results = run_chase_multiseed(
            data_dir=chase_path,
            seeds=seeds,
            epochs=50,
            batch_size=2,
            lr=2e-4,
            K=16,
            noise_strength=0.005,
            use_best_threshold=True,
        )

        all_results.extend(chase_results)

        save_seed_results_csv(
            rows=chase_results,
            save_path=results_dir / "chase_seed_results.csv",
        )

    else:
        print("CHASE dataset not found; running dummy test instead...")
        test_dummy(seed=42)

    # ------------------------------------------------------------
    # DRIVE
    # ------------------------------------------------------------

    drive_images = Path("data_drive/images")
    drive_masks = Path("data_drive/masks")

    if drive_images.exists() and drive_masks.exists():
        drive_results = run_drive_multiseed(
            images_dir=drive_images,
            masks_dir=drive_masks,
            seeds=seeds,
            epochs=50,
            batch_size=2,
            lr=2e-4,
            K=16,
            noise_strength=0.005,
            use_best_threshold=True,
        )

        all_results.extend(drive_results)

        save_seed_results_csv(
            rows=drive_results,
            save_path=results_dir / "drive_seed_results.csv",
        )

    else:
        print("DRIVE dataset folders not found. Skipping DRIVE.")

    # ------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------

    if len(all_results) > 0:
        save_seed_results_csv(
            rows=all_results,
            save_path=results_dir / "all_seed_results.csv",
        )

        save_summary_csv(
            rows=all_results,
            save_path=results_dir / "summary_mean_std.csv",
        )

        print("\nDone.")
        print(f"Results saved in: {results_dir.resolve()}")

    else:
        print("No results to save.")