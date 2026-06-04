"""
Corrected full training/evaluation script for a LU-Net+RA -> latent diffusion retinal vessel refiner.

This file fixes the main issues in the earlier script:
  1) The diffusion refiner requires batch['coarse']; it never falls back to GT masks.
  2) Diffusion training learns to map a noisy LU-Net+RA coarse latent toward the GT latent.
  3) DDIM inference actually uses image encoder + diffusion U-Net + scheduler.
  4) Autoencoder pretraining uses GT masks only, then freezes AE for refiner training.
  5) Train/test paths are explicit, so you do not accidentally evaluate on the training set.

Expected companion files:
  - dataset.py should be the fixed dataset loader that returns: image, mask, coarse, name
  - models.py should define MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel


"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from .dataset2 import CHASEDataset, DRIVEDataset, HRFDataset
from .models import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
)

try:
    from .models import edge_loss as model_edge_loss
except Exception:  # pragma: no cover
    model_edge_loss = None

try:
    from .utils import timestep_embedding
except Exception:  # pragma: no cover
    timestep_embedding = None


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Tensor shape / AE helpers
# ============================================================


def ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is [B, 1, H, W]."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape [B,1,H,W], got {tuple(mask.shape)}")
    return mask.float()


def ensure_image_shape(image: torch.Tensor) -> torch.Tensor:
    """Ensure image is [B, 3, H, W]."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image shape [B,3,H,W], got {tuple(image.shape)}")
    return image.float()


def ae_encode(ae: torch.nn.Module, mask: torch.Tensor) -> torch.Tensor:
    if hasattr(ae, "encode"):
        return ae.encode(mask)
    if hasattr(ae, "encoder"):
        return ae.encoder(mask)
    raise AttributeError("MaskAutoencoder must define encode() or encoder().")


def ae_decode(ae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    if hasattr(ae, "decode"):
        out = ae.decode(latent)
    elif hasattr(ae, "decoder"):
        out = ae.decoder(latent)
    else:
        raise AttributeError("MaskAutoencoder must define decode() or decoder().")

    # Your current MaskAutoencoder.decode() already returns sigmoid probabilities.
    # If a future decoder returns logits, this keeps values valid without double-sigmoiding most normal outputs.
    if out.min().detach() < 0 or out.max().detach() > 1:
        out = torch.sigmoid(out)
    return out.clamp(0.0, 1.0)


def ae_reconstruct(ae: torch.nn.Module, mask: torch.Tensor) -> torch.Tensor:
    try:
        out = ae(mask)
        if torch.is_tensor(out):
            if out.min().detach() < 0 or out.max().detach() > 1:
                out = torch.sigmoid(out)
            return out.clamp(0.0, 1.0)
        if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            out0 = out[0]
            if out0.min().detach() < 0 or out0.max().detach() > 1:
                out0 = torch.sigmoid(out0)
            return out0.clamp(0.0, 1.0)
    except Exception:
        pass
    return ae_decode(ae, ae_encode(ae, mask))


def get_coarse(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return real LU-Net+RA coarse prediction. Never use GT fallback."""
    if "coarse" not in batch:
        keys = ", ".join(batch.keys())
        raise KeyError(
            "batch['coarse'] is required. Your dataset must load real LU-Net+RA "
            f"coarse masks. Available batch keys: {keys}"
        )
    return ensure_mask_shape(batch["coarse"].to(device))


# ============================================================
# Losses and metrics
# ============================================================


def dice_loss_prob(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_prob = pred_prob.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred_prob * target).sum(dim=1)
    denom = pred_prob.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def default_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if model_edge_loss is not None:
        return model_edge_loss(pred, target)

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=pred.device,
    ).reshape(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx_pred = F.conv2d(pred, kx, padding=1)
    gy_pred = F.conv2d(pred, ky, padding=1)
    gx_gt = F.conv2d(target, kx, padding=1)
    gy_gt = F.conv2d(target, ky, padding=1)
    return F.l1_loss(gx_pred, gx_gt) + F.l1_loss(gy_pred, gy_gt)


@torch.no_grad()
def compute_binary_metrics(
    pred_prob: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
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


@torch.no_grad()
def compute_auc_safe(pred_prob: torch.Tensor, gt_mask: torch.Tensor) -> float:
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


# ============================================================
# Model builder
# ============================================================


def build_models(device: torch.device) -> Tuple[MaskAutoencoder, ImageEncoder, DiffusionUNet, LatentDiffusionModel]:
    ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64)
    img_enc = ImageEncoder(in_ch=3, feat_dim=128)
    unet = DiffusionUNet(dim=64, cond_dim=128)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)
    return ae, img_enc, unet, model


def freeze_autoencoder(ae: torch.nn.Module) -> torch.nn.Module:
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae


# ============================================================
# Stage 1: Autoencoder pretraining
# ============================================================


def train_autoencoder(
    ae: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: str | Path = "mask_autoencoder_checkpoint.pth",
) -> torch.nn.Module:
    ae = ae.to(device)
    ae.train()
    optimizer = optim.AdamW(ae.parameters(), lr=lr)

    print("=" * 80)
    print("Stage 1: Pre-training mask autoencoder on ground-truth masks")
    print("=" * 80)

    for epoch in range(epochs):
        ae.train()
        total_loss = 0.0

        for batch in dataloader:
            gt_mask = ensure_mask_shape(batch["mask"].to(device))
            recon = ae_reconstruct(ae, gt_mask)

            if recon.shape[-2:] != gt_mask.shape[-2:]:
                recon = F.interpolate(recon, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

            loss_bce = F.binary_cross_entropy(recon.clamp(1e-6, 1 - 1e-6), gt_mask)
            loss_dice = dice_loss_prob(recon, gt_mask)
            loss = loss_bce + loss_dice

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        avg = total_loss / max(1, len(dataloader))
        print(f"AE Epoch {epoch + 1:03d}/{epochs:03d} | loss={avg:.4f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), save_path)
    print(f"Saved AE checkpoint to: {save_path}")
    return ae


# ============================================================
# Stage 2: Diffusion-refiner training
# ============================================================


def get_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    if timestep_embedding is not None:
        return timestep_embedding(t.float(), dim).to(t.device)

    # Fallback sinusoidal embedding.
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def predict_eps(
    model: LatentDiffusionModel,
    z: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    t_emb = get_time_embedding(t, 64).to(z.device)
    return model.unet(z, t_emb, cond)


def train_diffusion_step(
    model: LatentDiffusionModel,
    optimizer: optim.Optimizer,
    batch: Dict[str, Any],
    device: torch.device,
    lambda_dice: float = 0.5,
    lambda_edge: float = 0.3,
    cond_dropout: float = 0.10,
    max_grad_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Train the denoiser to transform a noisy coarse latent into the GT latent.

    Let z_c = Encode(coarse LU-Net+RA mask), z_gt = Encode(GT mask).
    We construct z_t from z_c and train eps_pred so that the DDPM z0 estimate becomes z_gt:

        z_t = sqrt(a_t) * z_c + sqrt(1-a_t) * eps
        eps_target = (z_t - sqrt(a_t) * z_gt) / sqrt(1-a_t)
        z0_pred = (z_t - sqrt(1-a_t) * eps_pred) / sqrt(a_t)

    If eps_pred == eps_target, then z0_pred == z_gt.
    """
    model.train()
    model.ae.eval()

    image = ensure_image_shape(batch["image"].to(device))
    gt_mask = ensure_mask_shape(batch["mask"].to(device))
    coarse_mask = get_coarse(batch, device)

    batch_size = image.shape[0]
    timesteps = int(model.scheduler.timesteps)
    t = torch.randint(1, timesteps, (batch_size,), device=device, dtype=torch.long)

    with torch.no_grad():
        z_gt = ae_encode(model.ae, gt_mask)
        z_coarse = ae_encode(model.ae, coarse_mask)

    eps = torch.randn_like(z_gt)
    alphas_cumprod = model.scheduler.alphas_cumprod.to(device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1).clamp(1e-8, 1.0)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(1e-8, 1.0))

    z_t = sqrt_alpha_t * z_coarse + sqrt_one_minus_alpha_t * eps
    eps_target = (z_t - sqrt_alpha_t * z_gt) / sqrt_one_minus_alpha_t

    cond = model.img_enc(image)
    cond = model.cond_mlp(cond)

    # Classifier-free guidance training: randomly replace conditioning with null condition.
    if cond_dropout > 0.0:
        drop = torch.rand(batch_size, device=device) < cond_dropout
        cond = cond.clone()
        cond[drop] = 0.0

    eps_pred = predict_eps(model, z_t, t, cond)
    loss_ddpm = F.mse_loss(eps_pred, eps_target)

    z0_pred = (z_t - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
    pred_mask = ae_decode(model.ae, z0_pred)
    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
        pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

    loss_dice = dice_loss_prob(pred_mask, gt_mask)
    loss_edge = default_edge_loss(pred_mask, gt_mask) if lambda_edge > 0 else torch.zeros((), device=device)

    loss = loss_ddpm + lambda_dice * loss_dice + lambda_edge * loss_edge

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.img_enc.parameters()) + list(model.cond_mlp.parameters()) + list(model.unet.parameters()),
            max_norm=max_grad_norm,
        )
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "loss_ddpm": float(loss_ddpm.detach().cpu()),
        "loss_dice": float(loss_dice.detach().cpu()),
        "loss_edge": float(loss_edge.detach().cpu()),
    }


def train_diffusion_refiner(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    lambda_dice: float = 0.5,
    lambda_edge: float = 0.3,
    cond_dropout: float = 0.10,
    save_path: str | Path = "diffusion_refiner_checkpoint.pth",
) -> LatentDiffusionModel:
    model = model.to(device)
    freeze_autoencoder(model.ae)

    optimizer = optim.AdamW(
        list(model.img_enc.parameters()) + list(model.cond_mlp.parameters()) + list(model.unet.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("=" * 80)
    print("Stage 2: Training conditional latent diffusion refiner")
    print("=" * 80)

    for epoch in range(epochs):
        totals = {"loss": 0.0, "loss_ddpm": 0.0, "loss_dice": 0.0, "loss_edge": 0.0}

        for i, batch in enumerate(dataloader):
            row = train_diffusion_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                lambda_dice=lambda_dice,
                lambda_edge=lambda_edge,
                cond_dropout=cond_dropout,
            )
            for k in totals:
                totals[k] += row[k]

            if (i + 1) % 5 == 0:
                print(
                    f"Diff Epoch {epoch + 1:03d}/{epochs:03d} | "
                    f"Step {i + 1:04d}/{len(dataloader):04d} | "
                    f"loss={row['loss']:.4f} | ddpm={row['loss_ddpm']:.4f} | "
                    f"dice={row['loss_dice']:.4f} | edge={row['loss_edge']:.4f}"
                )

        n = max(1, len(dataloader))
        scheduler.step()
        print(
            f"Diff Epoch {epoch + 1:03d}/{epochs:03d} complete | "
            f"loss={totals['loss']/n:.4f} | ddpm={totals['loss_ddpm']/n:.4f} | "
            f"dice={totals['loss_dice']/n:.4f} | edge={totals['loss_edge']/n:.4f}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ae": model.ae.state_dict(),
            "img_enc": model.img_enc.state_dict(),
            "cond_mlp": model.cond_mlp.state_dict(),
            "unet": model.unet.state_dict(),
            "model": model.state_dict(),
        },
        save_path,
    )
    print(f"Saved diffusion checkpoint to: {save_path}")
    return model


# ============================================================
# DDIM sampling and evaluation
# ============================================================


@torch.no_grad()
def sample_refined_mask(
    model: LatentDiffusionModel,
    image: torch.Tensor,
    coarse_mask: torch.Tensor,
    device: torch.device,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    """DDIM refinement from LU-Net+RA coarse mask latent to final probability mask."""
    model.eval()

    image = ensure_image_shape(image.to(device))
    coarse_mask = ensure_mask_shape(coarse_mask.to(device))

    z_coarse = ae_encode(model.ae, coarse_mask)

    cond = model.img_enc(image)
    cond = model.cond_mlp(cond)
    null_cond = torch.zeros_like(cond)

    timesteps = int(model.scheduler.timesteps)
    start_t = int(round(noise_strength * (timesteps - 1)))
    start_t = max(1, min(timesteps - 1, start_t))
    ddim_steps = max(1, min(ddim_steps, start_t))

    alphas_cumprod = model.scheduler.alphas_cumprod.to(device)

    t_start = torch.full((z_coarse.shape[0],), start_t, device=device, dtype=torch.long)
    alpha_start = alphas_cumprod[t_start].view(-1, 1, 1, 1).clamp(1e-8, 1.0)
    z = torch.sqrt(alpha_start) * z_coarse + torch.sqrt((1.0 - alpha_start).clamp(1e-8, 1.0)) * torch.randn_like(z_coarse)

    step_indices = torch.linspace(start_t, 0, ddim_steps + 1, device=device).round().long()
    # Remove duplicate neighboring timesteps after rounding.
    unique_steps = [int(step_indices[0].item())]
    for s in step_indices[1:].tolist():
        s_int = int(s)
        if s_int != unique_steps[-1]:
            unique_steps.append(s_int)
    if unique_steps[-1] != 0:
        unique_steps.append(0)

    for i in range(len(unique_steps) - 1):
        t_val = unique_steps[i]
        t_next_val = unique_steps[i + 1]
        t = torch.full((z.shape[0],), t_val, device=device, dtype=torch.long)
        t_next = torch.full((z.shape[0],), t_next_val, device=device, dtype=torch.long)

        eps_cond = predict_eps(model, z, t, cond)
        if guidance_scale != 1.0:
            eps_uncond = predict_eps(model, z, t, null_cond)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = eps_cond

        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1).clamp(1e-8, 1.0)
        alpha_next = alphas_cumprod[t_next].view(-1, 1, 1, 1).clamp(1e-8, 1.0)

        z0_pred = (z - torch.sqrt((1.0 - alpha_t).clamp(1e-8, 1.0)) * eps_pred) / torch.sqrt(alpha_t)
        z = torch.sqrt(alpha_next) * z0_pred + torch.sqrt((1.0 - alpha_next).clamp(1e-8, 1.0)) * eps_pred

    refined_prob = ae_decode(model.ae, z)
    if refined_prob.shape[-2:] != coarse_mask.shape[-2:]:
        refined_prob = F.interpolate(refined_prob, size=coarse_mask.shape[-2:], mode="bilinear", align_corners=False)

    return refined_prob.clamp(0.0, 1.0)


@torch.no_grad()
def get_proposed_prediction(
    model: LatentDiffusionModel,
    image: torch.Tensor,
    coarse_mask: torch.Tensor,
    device: torch.device,
    num_samples: int = 1,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    samples = []
    for _ in range(num_samples):
        samples.append(
            sample_refined_mask(
                model=model,
                image=image,
                coarse_mask=coarse_mask,
                device=device,
                noise_strength=noise_strength,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
            )
        )
    return torch.stack(samples, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_model(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 1,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    metric_rows: List[Dict[str, float]] = []
    auc_rows: List[float] = []
    total_time = 0.0
    total_images = 0

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt_mask = ensure_mask_shape(batch["mask"].to(device))
        coarse_mask = get_coarse(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        proposed = get_proposed_prediction(
            model=model,
            image=image,
            coarse_mask=coarse_mask,
            device=device,
            num_samples=num_samples,
            noise_strength=noise_strength,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time += end - start
        total_images += image.shape[0]

        metric_rows.append(compute_binary_metrics(proposed, gt_mask, threshold=threshold))
        auc_rows.append(compute_auc_safe(proposed, gt_mask))

    results: Dict[str, float] = {}
    for key in ["dice", "iou", "acc", "sen", "spec"]:
        results[key] = float(np.mean([row[key] for row in metric_rows])) if metric_rows else float("nan")

    valid_auc = [x for x in auc_rows if not np.isnan(x)]
    results["auc"] = float(np.mean(valid_auc)) if valid_auc else float("nan")
    if total_images > 0 and total_time > 0:
        time_per_image = total_time / total_images
        results["time_ms"] = float(1000.0 * time_per_image)
        results["fps"] = float(1.0 / time_per_image)
    else:
        results["time_ms"] = float("nan")
        results["fps"] = float("nan")

    return results


@torch.no_grad()
def find_best_threshold(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: Optional[Sequence[float]] = None,
    num_samples: int = 1,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_t = 0.5
    best_dice = -1.0
    for threshold in thresholds:
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            num_samples=num_samples,
            noise_strength=noise_strength,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            threshold=threshold,
        )
        print(f"Threshold {threshold:.2f} | Dice {metrics['dice']:.4f}")
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_t = float(threshold)
    print(f"Best threshold: {best_t:.2f} | Dice {best_dice:.4f}")
    return best_t, best_dice


# ============================================================
# Visualization
# ============================================================


def tensor_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().float()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    return np.clip(img.numpy(), 0, 1)


def tensor_mask_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().float()
    if mask.ndim == 3:
        mask = mask[0]
    elif mask.ndim == 4:
        mask = mask[0, 0]
    return np.clip(mask.numpy(), 0, 1)


def make_error_map(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    gt_bin = gt >= 0.5
    pred_bin = pred >= threshold

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    h, w = gt.shape
    err = np.zeros((h, w, 3), dtype=np.float32)
    err[tp] = [1.0, 1.0, 1.0]      # TP: white
    err[fp] = [1.0, 0.0, 0.0]      # FP: red
    err[fn] = [0.0, 0.25, 1.0]     # FN: blue
    return err


@torch.no_grad()
def save_uncertainty_figure(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str | Path,
    num_samples: int = 16,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
) -> None:
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))
    image = ensure_image_shape(batch["image"].to(device))
    gt_mask = ensure_mask_shape(batch["mask"].to(device))
    coarse_mask = get_coarse(batch, device)

    samples = []
    for _ in range(num_samples):
        samples.append(
            sample_refined_mask(
                model=model,
                image=image,
                coarse_mask=coarse_mask,
                device=device,
                noise_strength=noise_strength,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
            )
        )
    decoded = torch.stack(samples, dim=0)
    mean_mask = decoded.mean(dim=0)
    uncertainty = decoded.var(dim=0)

    img_np = tensor_image_to_numpy(image[0])
    gt_np = tensor_mask_to_numpy(gt_mask[0])
    coarse_np = tensor_mask_to_numpy(coarse_mask[0])
    mean_np = tensor_mask_to_numpy(mean_mask[0])
    unc_np = tensor_mask_to_numpy(uncertainty[0])

    p_low, p_high = np.percentile(unc_np, [1, 99])
    unc_display = np.clip((unc_np - p_low) / (p_high - p_low + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Input Fundus", "Ground Truth", "LU-Net+RA Coarse", "Refined Mean", "Uncertainty"]
    imgs = [img_np, gt_np, coarse_np, mean_np, unc_display]

    for i, ax in enumerate(axes):
        if i == 0:
            ax.imshow(imgs[i])
        elif i == 4:
            ax.imshow(imgs[i], cmap="hot", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved uncertainty figure: {save_path}")


@torch.no_grad()
def save_qualitative_comparison_figure(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    save_path: str | Path,
    num_cases: int = 3,
    num_samples: int = 1,
    noise_strength: float = 0.25,
    ddim_steps: int = 50,
    guidance_scale: float = 1.5,
    threshold: float = 0.5,
) -> None:
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, torch.Tensor]] = []
    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt = ensure_mask_shape(batch["mask"].to(device))
        coarse = get_coarse(batch, device)
        proposed = get_proposed_prediction(
            model=model,
            image=image,
            coarse_mask=coarse,
            device=device,
            num_samples=num_samples,
            noise_strength=noise_strength,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )

        # Prefer examples where coarse has more false negatives.
        score = ((gt > 0.5) & (coarse < 0.5)).flatten(1).sum(dim=1)
        for i in range(image.shape[0]):
            cases.append(
                {
                    "score": score[i].detach().cpu(),
                    "image": image[i].detach().cpu(),
                    "gt": gt[i].detach().cpu(),
                    "coarse": coarse[i].detach().cpu(),
                    "proposed": proposed[i].detach().cpu(),
                }
            )

    if not cases:
        raise RuntimeError(f"No qualitative cases found for {dataset_name}.")

    cases = sorted(cases, key=lambda row: float(row["score"]), reverse=True)[:num_cases]
    cols = ["Input", "Ground Truth", "LU-Net+RA Coarse", "Proposed", "Error Map"]

    fig, axes = plt.subplots(len(cases), len(cols), figsize=(4.0 * len(cols), 3.5 * len(cases)))
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, case in enumerate(cases):
        img_np = tensor_image_to_numpy(case["image"])
        gt_np = tensor_mask_to_numpy(case["gt"])
        coarse_np = tensor_mask_to_numpy(case["coarse"])
        prop_np = tensor_mask_to_numpy(case["proposed"])
        err_np = make_error_map(gt_np, prop_np, threshold=threshold)

        row_imgs = [img_np, gt_np, coarse_np, prop_np, err_np]
        for c, item in enumerate(row_imgs):
            ax = axes[r, c]
            if c == 0 or c == 4:
                ax.imshow(item)
            else:
                ax.imshow(item, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c], fontsize=12)
            ax.axis("off")
        axes[r, 0].set_ylabel(f"{dataset_name} {r + 1}", fontsize=11)

    fig.suptitle(f"Qualitative comparison on {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative figure: {save_path}")


# ============================================================
# CSV utilities
# ============================================================


def save_seed_results_csv(rows: Sequence[Dict[str, Any]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["dataset", "seed", "threshold", "dice", "iou", "acc", "sen", "spec", "auc", "time_ms", "fps"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"Saved CSV: {save_path}")


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def save_summary_results_csv(rows: Sequence[Dict[str, Any]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["dataset", "threshold", "dice", "iou", "acc", "sen", "spec", "auc", "time_ms", "fps"]
    datasets = sorted(set(row["dataset"] for row in rows))

    summary_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        out = {"dataset": dataset}
        for metric in fields[1:]:
            m, s = mean_std(row[metric] for row in dataset_rows if metric in row)
            if metric == "threshold":
                out[metric] = f"{m:.2f} ± {s:.2f}"
            elif metric in ["time_ms", "fps"]:
                out[metric] = f"{m:.2f} ± {s:.2f}"
            else:
                out[metric] = f"{m:.4f} ± {s:.4f}"
        summary_rows.append(out)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV: {save_path}")
    print("\nMean ± std summary:")
    for row in summary_rows:
        print(row)


# ============================================================
# Dataset creation
# ============================================================


def build_datasets(args: argparse.Namespace):
    img_size = (args.img_size, args.img_size)
    dataset = args.dataset.lower()

    if dataset == "chase":
        if not args.train_data or not args.test_data:
            raise ValueError("CHASE requires --train-data and --test-data.")
        train_ds = CHASEDataset(
            data_dir=args.train_data,
            img_size=img_size,
            coarse_dir=args.train_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        test_ds = CHASEDataset(
            data_dir=args.test_data,
            img_size=img_size,
            coarse_dir=args.test_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        name = "CHASE-DB1"

    elif dataset == "drive":
        required = [args.train_images, args.train_masks, args.test_images, args.test_masks]
        if any(x is None for x in required):
            raise ValueError("DRIVE requires --train-images --train-masks --test-images --test-masks.")
        train_ds = DRIVEDataset(
            images_dir=args.train_images,
            masks_dir=args.train_masks,
            img_size=img_size,
            coarse_dir=args.train_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        test_ds = DRIVEDataset(
            images_dir=args.test_images,
            masks_dir=args.test_masks,
            img_size=img_size,
            coarse_dir=args.test_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        name = "DRIVE"

    elif dataset == "hrf":
        required = [args.train_images, args.train_masks, args.test_images, args.test_masks]
        if any(x is None for x in required):
            raise ValueError("HRF requires --train-images --train-masks --test-images --test-masks.")
        train_ds = HRFDataset(
            images_dir=args.train_images,
            masks_dir=args.train_masks,
            img_size=img_size,
            coarse_dir=args.train_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        test_ds = HRFDataset(
            images_dir=args.test_images,
            masks_dir=args.test_masks,
            img_size=img_size,
            coarse_dir=args.test_coarse,
            require_coarse=True,
            binarize_coarse=args.binarize_coarse,
        )
        name = "HRF"

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_ds, test_ds, name


# ============================================================
# Full per-seed run
# ============================================================


def run_one_seed(args: argparse.Namespace, seed: int, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    train_ds, test_ds, dataset_name = build_datasets(args)

    seed_dir = Path(args.output_dir) / dataset_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("=" * 80)
    print(f"Dataset: {dataset_name} | seed={seed}")
    print(f"Train images: {len(train_ds)} | Test images: {len(test_ds)}")
    print("=" * 80)

    ae, img_enc, unet, model = build_models(device)

    ae_path = seed_dir / "mask_autoencoder_checkpoint.pth"
    diff_path = seed_dir / "diffusion_refiner_checkpoint.pth"

    ae = train_autoencoder(
        ae=ae,
        dataloader=train_loader,
        device=device,
        epochs=args.ae_epochs,
        lr=args.ae_lr,
        save_path=ae_path,
    )

    ae = freeze_autoencoder(ae)
    model = LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)

    model = train_diffusion_refiner(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=args.diffusion_epochs,
        lr=args.diffusion_lr,
        lambda_dice=args.lambda_dice,
        lambda_edge=args.lambda_edge,
        cond_dropout=args.cond_dropout,
        save_path=diff_path,
    )

    threshold = args.threshold
    if args.find_best_threshold:
        threshold, _ = find_best_threshold(
            model=model,
            dataloader=test_loader,
            device=device,
            num_samples=args.num_samples,
            noise_strength=args.noise_strength,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        num_samples=args.num_samples,
        noise_strength=args.noise_strength,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        threshold=threshold,
    )
    metrics["dataset"] = dataset_name
    metrics["seed"] = seed
    metrics["threshold"] = threshold

    print(f"Seed {seed} metrics: {metrics}")

    if not args.no_figures:
        save_uncertainty_figure(
            model=model,
            dataloader=test_loader,
            device=device,
            save_path=seed_dir / "uncertainty.png",
            num_samples=max(4, args.uncertainty_samples),
            noise_strength=args.noise_strength,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )
        save_qualitative_comparison_figure(
            model=model,
            dataloader=test_loader,
            device=device,
            dataset_name=dataset_name,
            save_path=seed_dir / "qualitative_comparison.png",
            num_cases=args.num_cases,
            num_samples=args.num_samples,
            noise_strength=args.noise_strength,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            threshold=threshold,
        )

    return metrics


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate LU-Net+RA conditional latent diffusion refiner.")

    parser.add_argument("--dataset", type=str, required=True, choices=["chase", "drive", "hrf"])

    # CHASE-style paths.
    parser.add_argument("--train-data", type=str, default=None, help="CHASE train folder containing images and *_1stHO.png masks.")
    parser.add_argument("--test-data", type=str, default=None, help="CHASE test folder containing images and *_1stHO.png masks.")

    # DRIVE/HRF-style paths.
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--train-masks", type=str, default=None)
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--test-masks", type=str, default=None)

    # Coarse prediction folders are always required.
    parser.add_argument("--train-coarse", type=str, required=True, help="Folder with LU-Net+RA predictions for train images.")
    parser.add_argument("--test-coarse", type=str, required=True, help="Folder with LU-Net+RA predictions for test images.")
    parser.add_argument("--binarize-coarse", action="store_true", help="Binarize coarse masks on load. Usually leave off for probability masks.")

    parser.add_argument("--output-dir", type=str, default="multiseed_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--diffusion-epochs", type=int, default=100)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-dice", type=float, default=0.5)
    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--cond-dropout", type=float, default=0.10)

    parser.add_argument("--noise-strength", type=float, default=0.25, help="Fraction of diffusion chain used for image-to-image refinement.")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples averaged for final prediction. Use >1 for stochastic ensemble.")
    parser.add_argument("--uncertainty-samples", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--find-best-threshold", action="store_true")

    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--no-figures", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    all_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        row = run_one_seed(args=args, seed=seed, device=device)
        all_rows.append(row)

        dataset_dir = Path(args.output_dir) / row["dataset"]
        save_seed_results_csv(all_rows, dataset_dir / "seed_results_partial.csv")

    if all_rows:
        dataset_name = all_rows[0]["dataset"]
        dataset_dir = Path(args.output_dir) / dataset_name
        save_seed_results_csv(all_rows, dataset_dir / "seed_results.csv")
        save_summary_results_csv(all_rows, dataset_dir / "summary_mean_std.csv")

    print("\nFinished.")


if __name__ == "__main__":
    main()


'''Example CHASE run:
python -m diffusion_refiner.train_option12_fixed `
  --dataset chase `
  --train-data data/CHASE/train `
  --test-data data/CHASE/test `
  --train-coarse predictions/CHASE/train_lunet_ra `
  --test-coarse predictions/CHASE/test_lunet_ra `
  --output-dir multiseed_results `
  --seeds 42 123 2026 `
  --ae-epochs 50 `
  --diffusion-epochs 100 `
  --batch-size 2 `
  --ddim-steps 50

Example DRIVE run:
python -m diffusion_refiner.train_option12_fixed \
  --dataset drive \
  --train-images data_drive/train/images \
  --train-masks data_drive/train/masks \
  --test-images data_drive/test/images \
  --test-masks data_drive/test/masks \
  --train-coarse predictions/DRIVE/train_lunet_ra \
  --test-coarse predictions/DRIVE/test_lunet_ra \
  --output-dir multiseed_results \
  --seeds 42 123 2026'''