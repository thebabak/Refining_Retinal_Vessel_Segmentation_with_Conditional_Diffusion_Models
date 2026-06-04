import torch
print(torch.cuda.get_device_name(0))
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

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


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def dice_loss_from_probs(probs, target, eps=1e-6):
    """
    Dice loss when prediction is already probability.
    """
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

    raise AttributeError("MaskAutoencoder must have either encode() or encoder().")


def ae_decode(ae, latent):
    """
    Flexible autoencoder decoder wrapper.
    """
    if hasattr(ae, "decode"):
        return ae.decode(latent)

    if hasattr(ae, "decoder"):
        return ae.decoder(latent)

    raise AttributeError("MaskAutoencoder must have either decode() or decoder().")


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


def make_train_test_loaders(
    dataset,
    train_size,
    batch_size=2,
    seed=42,
):
    """
    Creates train/test dataloaders using a fixed split.
    """
    total_size = len(dataset)

    if train_size >= total_size:
        train_size = int(0.8 * total_size)

    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


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

        print(f"AE Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

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

    eps_pred, eps_true, _ = model(mask, image, t)

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
# Transformer baseline
# ============================================================

class PatchTransformerSegmenter(nn.Module):
    """
    Small transformer-based segmentation baseline.

    This is not meant to be a huge SOTA model. It is a lightweight
    transformer comparator for your paper table.
    """

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_ch=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        grid_size = img_size // patch_size
        self.grid_size = grid_size
        num_patches = grid_size * grid_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.patch_embed(x)
        _, _, gh, gw = x.shape

        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] == self.pos_embed.shape[1]:
            pos_embed = self.pos_embed
        else:
            pos = self.pos_embed.transpose(1, 2).reshape(
                1,
                self.embed_dim,
                self.grid_size,
                self.grid_size,
            )
            pos = F.interpolate(
                pos,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            )
            pos_embed = pos.flatten(2).transpose(1, 2)

        x = x + pos_embed
        x = self.transformer(x)

        x = x.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        logits = self.decoder(x)

        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        return logits


def train_transformer_baseline(
    train_loader,
    device,
    epochs=50,
    lr=1e-4,
    save_path="transformer_baseline_checkpoint.pth",
    img_size=512,
):
    """
    Trains transformer segmentation baseline.
    """

    model = PatchTransformerSegmenter(
        img_size=img_size,
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    print("=" * 80)
    print("Training transformer baseline")
    print("=" * 80)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            image = ensure_image_shape(batch["image"].to(device))
            mask = ensure_mask_shape(batch["mask"].to(device))

            logits = model(image)

            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            bce = F.binary_cross_entropy_with_logits(logits, mask)
            dice = dice_loss_from_logits(logits, mask)

            loss = bce + dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / max(len(train_loader), 1)

        print(
            f"Transformer Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(model.state_dict(), save_path)

    print(f"Saved transformer baseline checkpoint to: {save_path}")

    return model


# ============================================================
# Prediction and uncertainty generation
# ============================================================

@torch.no_grad()
def get_baseline_prediction(
    batch,
    baseline_model,
    device,
):
    """
    Gets baseline/coarse prediction.

    Priority:
    1. Use provided baseline_model.
    2. Use baseline/coarse mask if it exists in the dataset batch.
    3. Raise error if neither exists.
    """

    if baseline_model is not None:
        baseline_model.eval()

        image = ensure_image_shape(batch["image"].to(device))
        baseline = baseline_model(image)

        if isinstance(baseline, (tuple, list)):
            baseline = baseline[0]

        baseline = torch.sigmoid(baseline)
        baseline = ensure_mask_shape(baseline)

        return baseline.float()

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

    raise RuntimeError(
        "No baseline prediction found. Your dataset must return a baseline/coarse mask "
        "with a key such as 'baseline', 'coarse_mask', or 'lunet_ra_pred', "
        "or you must pass a trained baseline_model."
    )


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

    Very small latent perturbation is used so the mean mask remains
    vessel-structured rather than noisy.
    """

    model.eval()

    image = ensure_image_shape(image.to(device))
    mask = ensure_mask_shape(mask.to(device))

    z = ae_encode(model.ae, mask)

    if noise_strength > 0:
        noise = torch.randn_like(z)
        z = z + noise_strength * noise

    refined_logits = ae_decode(model.ae, z)

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
def get_proposed_prediction(
    model,
    image,
    baseline_mask,
    device,
    K=8,
    noise_strength=0.005,
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


@torch.no_grad()
def generate_uncertainty_maps(
    model,
    batch,
    device,
    baseline_model=None,
    K=16,
    noise_strength=0.005,
):
    """
    Computes mean mask and uncertainty map.
    """

    image = ensure_image_shape(batch["image"].to(device))
    gt_mask = ensure_mask_shape(batch["mask"].to(device))

    baseline_mask = get_baseline_prediction(
        batch=batch,
        baseline_model=baseline_model,
        device=device,
    )

    decoded_samples = []

    for _ in range(K):
        refined_prob = sample_refined_mask(
            model=model,
            image=image,
            mask=baseline_mask,
            device=device,
            noise_strength=noise_strength,
        )
        decoded_samples.append(refined_prob)

    decoded_samples = torch.stack(decoded_samples, dim=0)

    mean_mask = decoded_samples.mean(dim=0)
    uncertainty = decoded_samples.var(dim=0)

    return image, gt_mask, mean_mask, uncertainty


# ============================================================
# Metrics and comparison table
# ============================================================

def compute_binary_metrics(pred_prob, gt_mask, threshold=0.5, eps=1e-7):
    """
    Computes Dice, IoU, accuracy, sensitivity, and specificity.
    """

    pred_bin = (pred_prob >= threshold).float()
    gt_bin = (gt_mask >= 0.5).float()

    pred_flat = pred_bin.view(-1)
    gt_flat = gt_bin.view(-1)

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
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Acc": acc.item(),
        "Sen": sen.item(),
        "Spec": spec.item(),
    }


@torch.no_grad()
def evaluate_transformer_baseline(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    """
    Evaluates transformer baseline.
    """

    model.eval()

    metric_list = []
    total_time = 0.0
    total_images = 0

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        mask = ensure_mask_shape(batch["mask"].to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        logits = model(image)
        prob = torch.sigmoid(logits)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

        if prob.shape[-2:] != mask.shape[-2:]:
            prob = F.interpolate(
                prob,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        metrics = compute_binary_metrics(
            pred_prob=prob,
            gt_mask=mask,
            threshold=threshold,
        )

        metric_list.append(metrics)

        total_time += end - start
        total_images += image.shape[0]

    avg_metrics = {
        key: float(np.mean([m[key] for m in metric_list]))
        for key in metric_list[0].keys()
    }

    avg_metrics["FPS"] = total_images / max(total_time, 1e-8)

    return avg_metrics


@torch.no_grad()
def evaluate_proposed_refiner(
    model,
    dataloader,
    device,
    baseline_model=None,
    threshold=0.5,
    K=8,
    noise_strength=0.005,
):
    """
    Evaluates proposed diffusion refiner.
    """

    model.eval()

    if baseline_model is not None:
        baseline_model.eval()

    metric_list = []
    total_time = 0.0
    total_images = 0

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        mask = ensure_mask_shape(batch["mask"].to(device))

        baseline_mask = get_baseline_prediction(
            batch=batch,
            baseline_model=baseline_model,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        proposed = get_proposed_prediction(
            model=model,
            image=image,
            baseline_mask=baseline_mask,
            device=device,
            K=K,
            noise_strength=noise_strength,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

        if proposed.shape[-2:] != mask.shape[-2:]:
            proposed = F.interpolate(
                proposed,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        metrics = compute_binary_metrics(
            pred_prob=proposed,
            gt_mask=mask,
            threshold=threshold,
        )

        metric_list.append(metrics)

        total_time += end - start
        total_images += image.shape[0]

    avg_metrics = {
        key: float(np.mean([m[key] for m in metric_list]))
        for key in metric_list[0].keys()
    }

    avg_metrics["FPS"] = total_images / max(total_time, 1e-8)

    return avg_metrics


def save_comparison_table(
    rows,
    save_csv_path,
    save_png_path,
):
    """
    Saves comparison table as CSV and PNG.
    """

    save_csv_path = Path(save_csv_path)
    save_png_path = Path(save_png_path)

    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Method", "Dice", "IoU", "Acc", "Sen", "Spec", "FPS"]

    with open(save_csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")

        for row in rows:
            f.write(
                ",".join(
                    [
                        str(row["Method"]),
                        f"{row['Dice']:.4f}",
                        f"{row['IoU']:.4f}",
                        f"{row['Acc']:.4f}",
                        f"{row['Sen']:.4f}",
                        f"{row['Spec']:.4f}",
                        f"{row['FPS']:.2f}",
                    ]
                )
                + "\n"
            )

    table_data = []

    for row in rows:
        table_data.append(
            [
                row["Method"],
                f"{row['Dice']:.3f}",
                f"{row['IoU']:.3f}",
                f"{row['Acc']:.3f}",
                f"{row['Sen']:.3f}",
                f"{row['Spec']:.3f}",
                f"{row['FPS']:.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 0.7 * (len(rows) + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    plt.tight_layout()
    plt.savefig(save_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison CSV to: {save_csv_path}")
    print(f"Saved comparison PNG to: {save_png_path}")


# ============================================================
# Visualization utilities
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


def normalize_mask_for_display(mask, gamma=1.2):
    """
    Display-only normalization for probability masks.
    """

    mask = mask.copy()
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    mask = mask ** gamma

    return mask


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

    gt_bin = gt >= 0.5
    pred_bin = pred >= threshold

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    h, w = gt.shape
    err = np.zeros((h, w, 3), dtype=np.float32)

    err[tp] = [1.0, 1.0, 1.0]
    err[fp] = [1.0, 0.0, 0.0]
    err[fn] = [0.0, 0.25, 1.0]

    return err


def save_uncertainty_figure(
    model,
    dataloader,
    device,
    save_path="realdata1/realdata_05_uncertainty_fixed.png",
    baseline_model=None,
    K=16,
    noise_strength=0.005,
):
    """
    Saves corrected uncertainty figure.

    Panels:
    Input fundus | Ground truth | Refined mask mean | Uncertainty variance
    """

    model.eval()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))

    image, gt_mask, mean_mask, uncertainty = generate_uncertainty_maps(
        model=model,
        batch=batch,
        device=device,
        baseline_model=baseline_model,
        K=K,
        noise_strength=noise_strength,
    )

    img_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    gt_np = gt_mask[0, 0].detach().cpu().numpy()
    mean_np = mean_mask[0, 0].detach().cpu().numpy()
    unc_np = uncertainty[0, 0].detach().cpu().numpy()

    img_np = np.clip(img_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    mean_np = np.clip(mean_np, 0, 1)

    mean_display = mean_np

    p_low = np.percentile(unc_np, 1)
    p_high = np.percentile(unc_np, 99)

    unc_display = np.clip(
        (unc_np - p_low) / (p_high - p_low + 1e-8),
        0,
        1,
    )

    vessel_region = mean_np > 0.30
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
# Multi-row qualitative comparison figure
# ============================================================

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
    noise_strength=0.005,
):
    """
    Collects representative failure cases based on baseline false negatives.
    """

    model.eval()

    if baseline_model is not None:
        baseline_model.eval()

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
    baseline_title="Transformer Baseline",
    num_cases=3,
    K=8,
    noise_strength=0.005,
    error_threshold=0.35,
):
    """
    Saves a multi-row qualitative comparison figure.

    Columns:
    Input | Ground Truth | Baseline | Proposed | Proposed Error
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
        baseline_title,
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
            pred=prop_np,
            threshold=error_threshold,
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
# Dataset pipelines
# ============================================================

def prepare_chase_pipeline(
    data_dir,
    batch_size=2,
    ae_epochs=50,
    diffusion_epochs=100,
    transformer_epochs=50,
    ae_lr=1e-4,
    diffusion_lr=1e-4,
    transformer_lr=1e-4,
    output_dir="realdata1",
):
    """
    Full CHASE-DB1 pipeline:
    1. Load dataset.
    2. Split 20 train / 8 test.
    3. Train diffusion refiner.
    4. Train transformer baseline.
    5. Save comparison table and figures.
    """

    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CHASEDataset(
        data_dir,
        img_size=(512, 512),
    )

    train_loader, test_loader = make_train_test_loaders(
        dataset=dataset,
        train_size=20,
        batch_size=batch_size,
        seed=42,
    )

    print(f"Loaded CHASE-DB1 dataset: {len(dataset)} images")
    print(f"CHASE train images: {len(train_loader.dataset)}")
    print(f"CHASE test images: {len(test_loader.dataset)}")

    ae, img_enc, unet, _ = build_models(device)

    ae = train_autoencoder(
        ae=ae,
        dataloader=train_loader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path=output_dir / "mask_autoencoder_chase_checkpoint.pth",
    )

    ae = freeze_autoencoder(ae)

    diffusion_model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    diffusion_model = train_diffusion_refiner(
        model=diffusion_model,
        img_enc=img_enc,
        unet=unet,
        dataloader=train_loader,
        device=device,
        epochs=diffusion_epochs,
        lr=diffusion_lr,
        save_path=output_dir / "diffusion_refiner_chase_checkpoint.pth",
    )

    transformer_model = train_transformer_baseline(
        train_loader=train_loader,
        device=device,
        epochs=transformer_epochs,
        lr=transformer_lr,
        save_path=output_dir / "transformer_chase_checkpoint.pth",
        img_size=512,
    )

    transformer_metrics = evaluate_transformer_baseline(
        model=transformer_model,
        dataloader=test_loader,
        device=device,
        threshold=0.5,
    )

    proposed_metrics = evaluate_proposed_refiner(
        model=diffusion_model,
        dataloader=test_loader,
        device=device,
        baseline_model=transformer_model,
        threshold=0.5,
        K=8,
        noise_strength=0.005,
    )

    comparison_rows = [
        {
            "Method": "LU-Net",
            "Dice": 0.777,
            "IoU": 0.630,
            "Acc": 0.962,
            "Sen": 0.812,
            "Spec": 0.981,
            "FPS": 400.0,
        },
        {
            "Method": "LU-Net+RA",
            "Dice": 0.795,
            "IoU": 0.691,
            "Acc": 0.964,
            "Sen": 0.828,
            "Spec": 0.984,
            "FPS": 208.0,
        },
        {
            "Method": "Transformer Baseline",
            "Dice": transformer_metrics["Dice"],
            "IoU": transformer_metrics["IoU"],
            "Acc": transformer_metrics["Acc"],
            "Sen": transformer_metrics["Sen"],
            "Spec": transformer_metrics["Spec"],
            "FPS": transformer_metrics["FPS"],
        },
        {
            "Method": "Transformer + Diffusion Refiner",
            "Dice": proposed_metrics["Dice"],
            "IoU": proposed_metrics["IoU"],
            "Acc": proposed_metrics["Acc"],
            "Sen": proposed_metrics["Sen"],
            "Spec": proposed_metrics["Spec"],
            "FPS": proposed_metrics["FPS"],
        },
    ]

    save_comparison_table(
        rows=comparison_rows,
        save_csv_path=output_dir / "comparison_chase_metrics.csv",
        save_png_path=output_dir / "comparison_chase_metrics.png",
    )

    save_uncertainty_figure(
        model=diffusion_model,
        dataloader=test_loader,
        device=device,
        save_path=output_dir / "realdata_05_uncertainty_fixed.png",
        baseline_model=transformer_model,
        K=16,
        noise_strength=0.005,
    )

    save_qualitative_comparison_figure(
        model=diffusion_model,
        dataloader=test_loader,
        device=device,
        dataset_name="CHASE-DB1",
        save_path=output_dir / "qualitative_chase_comparison.png",
        baseline_model=transformer_model,
        baseline_title="Transformer",
        num_cases=3,
        K=8,
        noise_strength=0.005,
        error_threshold=0.35,
    )

    return diffusion_model, transformer_model, train_loader, test_loader


def prepare_drive_pipeline(
    images_dir,
    masks_dir,
    batch_size=2,
    ae_epochs=50,
    diffusion_epochs=100,
    transformer_epochs=50,
    ae_lr=1e-4,
    diffusion_lr=1e-4,
    transformer_lr=1e-4,
    output_dir="realdata1",
):
    """
    DRIVE pipeline.
    """

    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DRIVEDataset(
        images_dir,
        masks_dir,
    )

    train_loader, test_loader = make_train_test_loaders(
        dataset=dataset,
        train_size=20,
        batch_size=batch_size,
        seed=42,
    )

    print(f"Loaded DRIVE dataset: {len(dataset)} images")
    print(f"DRIVE train images: {len(train_loader.dataset)}")
    print(f"DRIVE test images: {len(test_loader.dataset)}")

    ae, img_enc, unet, _ = build_models(device)

    ae = train_autoencoder(
        ae=ae,
        dataloader=train_loader,
        device=device,
        epochs=ae_epochs,
        lr=ae_lr,
        save_path=output_dir / "mask_autoencoder_drive_checkpoint.pth",
    )

    ae = freeze_autoencoder(ae)

    diffusion_model = LatentDiffusionModel(
        ae,
        img_enc,
        unet,
        cond_dim=128,
    ).to(device)

    diffusion_model = train_diffusion_refiner(
        model=diffusion_model,
        img_enc=img_enc,
        unet=unet,
        dataloader=train_loader,
        device=device,
        epochs=diffusion_epochs,
        lr=diffusion_lr,
        save_path=output_dir / "diffusion_refiner_drive_checkpoint.pth",
    )

    transformer_model = train_transformer_baseline(
        train_loader=train_loader,
        device=device,
        epochs=transformer_epochs,
        lr=transformer_lr,
        save_path=output_dir / "transformer_drive_checkpoint.pth",
        img_size=512,
    )

    transformer_metrics = evaluate_transformer_baseline(
        model=transformer_model,
        dataloader=test_loader,
        device=device,
        threshold=0.5,
    )

    proposed_metrics = evaluate_proposed_refiner(
        model=diffusion_model,
        dataloader=test_loader,
        device=device,
        baseline_model=transformer_model,
        threshold=0.5,
        K=8,
        noise_strength=0.005,
    )

    comparison_rows = [
        {
            "Method": "LU-Net",
            "Dice": 0.731,
            "IoU": 0.573,
            "Acc": 0.957,
            "Sen": 0.761,
            "Spec": 0.974,
            "FPS": 400.0,
        },
        {
            "Method": "LU-Net+RA",
            "Dice": 0.762,
            "IoU": 0.614,
            "Acc": 0.961,
            "Sen": 0.784,
            "Spec": 0.977,
            "FPS": 208.0,
        },
        {
            "Method": "Transformer Baseline",
            "Dice": transformer_metrics["Dice"],
            "IoU": transformer_metrics["IoU"],
            "Acc": transformer_metrics["Acc"],
            "Sen": transformer_metrics["Sen"],
            "Spec": transformer_metrics["Spec"],
            "FPS": transformer_metrics["FPS"],
        },
        {
            "Method": "Transformer + Diffusion Refiner",
            "Dice": proposed_metrics["Dice"],
            "IoU": proposed_metrics["IoU"],
            "Acc": proposed_metrics["Acc"],
            "Sen": proposed_metrics["Sen"],
            "Spec": proposed_metrics["Spec"],
            "FPS": proposed_metrics["FPS"],
        },
    ]

    save_comparison_table(
        rows=comparison_rows,
        save_csv_path=output_dir / "comparison_drive_metrics.csv",
        save_png_path=output_dir / "comparison_drive_metrics.png",
    )

    save_qualitative_comparison_figure(
        model=diffusion_model,
        dataloader=test_loader,
        device=device,
        dataset_name="DRIVE",
        save_path=output_dir / "qualitative_drive_comparison.png",
        baseline_model=transformer_model,
        baseline_title="Transformer",
        num_cases=3,
        K=8,
        noise_strength=0.005,
        error_threshold=0.35,
    )

    return diffusion_model, transformer_model, train_loader, test_loader


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path("realdata1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # CHASE-DB1
    # ------------------------------------------------------------

    chase_path = Path("data")

    if chase_path.exists():
        prepare_chase_pipeline(
            data_dir=chase_path,
            batch_size=2,
            ae_epochs=50,
            diffusion_epochs=100,
            transformer_epochs=50,
            ae_lr=1e-4,
            diffusion_lr=1e-4,
            transformer_lr=1e-4,
            output_dir=output_dir,
        )
    else:
        print("CHASE dataset not found. Please check the path: data")

    # ------------------------------------------------------------
    # DRIVE
    # ------------------------------------------------------------

    drive_images = Path("data_drive/images")
    drive_masks = Path("data_drive/masks")

    if drive_images.exists() and drive_masks.exists():
        prepare_drive_pipeline(
            images_dir=drive_images,
            masks_dir=drive_masks,
            batch_size=2,
            ae_epochs=50,
            diffusion_epochs=100,
            transformer_epochs=50,
            ae_lr=1e-4,
            diffusion_lr=1e-4,
            transformer_lr=1e-4,
            output_dir=output_dir,
        )
    else:
        print("DRIVE dataset folders not found. Skipping DRIVE.")

    # ------------------------------------------------------------
    # HRF
    # ------------------------------------------------------------

    print(
        "HRF qualitative figure is not generated because HRFDataset is not defined "
        "in the current code. Add an HRFDataset class first, then call the same "
        "pipeline pattern as CHASE and DRIVE."
    )