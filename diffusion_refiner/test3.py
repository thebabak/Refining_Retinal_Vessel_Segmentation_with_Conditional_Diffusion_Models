"""
Full integration script - connects corrected diffusion pipeline with your datasets.
Save this as: diffusion_refiner/run_corrected.py
"""

import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# Import your existing dataset classes
from diffusion_refiner.dataset2 import CHASEDataset, DRIVEDataset

# Import the corrected model components
from diffusion_refiner.models_corrected import (
    MaskAutoencoder,
    ImageEncoder,
    DiffusionUNet,
    LatentDiffusionModel,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Utilities
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_mask_shape(mask):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def ensure_image_shape(image):
    return image.float()


def logits_or_probs_to_probs(output):
    if output.min().item() >= 0.0 and output.max().item() <= 1.0:
        return output
    return torch.sigmoid(output)


# ============================================================
# Metrics
# ============================================================

def compute_binary_metrics(pred_prob, gt_mask, threshold=0.5, eps=1e-7):
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
    
    return {
        "dice": float(((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).item()),
        "iou": float(((tp + eps) / (tp + fp + fn + eps)).item()),
        "acc": float(((tp + tn + eps) / (tp + tn + fp + fn + eps)).item()),
        "sen": float(((tp + eps) / (tp + fn + eps)).item()),
        "spec": float(((tn + eps) / (tn + fp + eps)).item()),
    }


def compute_auc_safe(pred_prob, gt_mask):
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


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def edge_loss(pred_mask, gt_mask):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    
    device = pred_mask.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)
    
    pred_edge_x = F.conv2d(pred_mask, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_mask, sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
    
    gt_edge_x = F.conv2d(gt_mask, sobel_x, padding=1)
    gt_edge_y = F.conv2d(gt_mask, sobel_y, padding=1)
    gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-6)
    
    return F.l1_loss(pred_edge, gt_edge)


# ============================================================
# Training Functions
# ============================================================

def train_autoencoder(ae, dataloader, device, epochs=50, lr=1e-4, 
                      save_path="mask_autoencoder.pth"):
    """Stage 1: Pre-train mask autoencoder."""
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
            recon_logits = ae(mask)
            
            if recon_logits.shape[-2:] != mask.shape[-2:]:
                recon_logits = F.interpolate(
                    recon_logits, size=mask.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            
            bce_loss = F.binary_cross_entropy_with_logits(recon_logits, mask)
            d_loss = dice_loss_from_logits(recon_logits, mask)
            loss = bce_loss + d_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"AE Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), save_path)
    print(f"Saved autoencoder to: {save_path}")
    return ae


def train_diffusion_refiner(model, dataloader, device, epochs=100, lr=1e-4,
                           dice_weight=0.5, edge_weight=0.3,
                           save_path="diffusion_refiner.pth"):
    """Stage 2: Train conditional latent diffusion model."""
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(
        list(model.img_enc.parameters()) + list(model.unet.parameters()),
        lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("=" * 80)
    print("Stage 2: Training conditional latent diffusion refiner")
    print("=" * 80)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_diff = 0.0
        total_dice = 0.0
        total_edge = 0.0
        
        for i, batch in enumerate(dataloader):
            image = ensure_image_shape(batch["image"].to(device))
            mask = ensure_mask_shape(batch["mask"].to(device))
            batch_size = image.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, model.timesteps, (batch_size,), device=device)
            
            # Get diffusion loss
            eps_pred, eps_true, z0 = model(mask, image, t)
            diff_loss = F.mse_loss(eps_pred, eps_true)
            
            # Compute auxiliary segmentation losses at t=500
            t_aux = torch.ones(batch_size, device=device).long() * 500
            eps_pred_aux, eps_true_aux, z0_aux = model(mask, image, t_aux)
            alpha_bar_aux = model.alphas_cumprod[500].view(1, 1, 1, 1)
            
            z_noisy = (torch.sqrt(alpha_bar_aux) * model.ae.encode(mask) + 
                      torch.sqrt(1 - alpha_bar_aux) * eps_true_aux)
            z_pred = (z_noisy - torch.sqrt(1 - alpha_bar_aux) * eps_pred_aux) / torch.sqrt(alpha_bar_aux)
            recon_aux = model.ae.decode(z_pred)
            
            if recon_aux.shape[-2:] != mask.shape[-2:]:
                recon_aux = F.interpolate(
                    recon_aux, size=mask.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            
            recon_prob = logits_or_probs_to_probs(recon_aux)
            
            # Dice loss
            probs_flat = recon_prob.view(batch_size, -1)
            target_flat = mask.view(batch_size, -1)
            intersection = (probs_flat * target_flat).sum(1)
            denominator = probs_flat.sum(1) + target_flat.sum(1)
            dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
            d_loss = (1.0 - dice).mean()
            
            # Edge loss
            e_loss = edge_loss(recon_prob, mask)
            
            # Total loss
            loss = diff_loss + dice_weight * d_loss + edge_weight * e_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_diff += diff_loss.item()
            total_dice += d_loss.item()
            total_edge += e_loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(dataloader)} | "
                      f"Diff: {diff_loss:.4f} | Dice: {d_loss:.4f} | Edge: {e_loss:.4f}")
        
        avg_loss = total_loss / max(len(dataloader), 1)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} complete | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Diff: {total_diff/len(dataloader):.4f} | "
              f"Dice: {total_dice/len(dataloader):.4f} | "
              f"Edge: {total_edge/len(dataloader):.4f}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ae": model.ae.state_dict(),
        "img_enc": model.img_enc.state_dict(),
        "unet": model.unet.state_dict(),
    }, save_path)
    print(f"Saved diffusion model to: {save_path}")
    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model_full(model, dataloader, device, dataset_name, seed,
                       num_steps_list=[20, 50], noise_strength=0.5, 
                       guidance_scale=1.0, K=5, threshold=0.5):
    """
    Full evaluation with multiple DDIM step counts.
    Returns rows for CSV output.
    """
    model.eval()
    all_rows = []
    
    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt_mask = ensure_mask_shape(batch["mask"].to(device))
        batch_size = image.shape[0]
        
        # 1. AE reconstruction (sanity check)
        ae_recon = model.ae(gt_mask)
        if ae_recon.shape[-2:] != gt_mask.shape[-2:]:
            ae_recon = F.interpolate(
                ae_recon, size=gt_mask.shape[-2:],
                mode="bilinear", align_corners=False
            )
        ae_prob = logits_or_probs_to_probs(ae_recon)
        
        for b in range(batch_size):
            metrics = compute_binary_metrics(ae_prob[b:b+1], gt_mask[b:b+1], threshold)
            metrics["auc"] = compute_auc_safe(ae_prob[b:b+1], gt_mask[b:b+1])
            all_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "method": "AE reconstruction (GT input; sanity only)",
                "valid_for_main_table": "NO",
                "threshold": threshold,
                "K": K,
                "noise_strength": noise_strength,
                "num_steps": 0,
                "guidance_scale": guidance_scale,
                **metrics,
                "time_ms": float("nan"),
                "fps": float("nan"),
            })
        
        # 2. Baseline coarse mask
        coarse_mask = None
        for key in ["baseline", "coarse", "coarse_mask", "prediction", 
                    "lunet_pred", "lunet_ra_pred"]:
            if key in batch:
                coarse_mask = ensure_mask_shape(batch[key].to(device))
                break
        
        if coarse_mask is not None:
            for b in range(batch_size):
                metrics = compute_binary_metrics(
                    coarse_mask[b:b+1], gt_mask[b:b+1], threshold
                )
                metrics["auc"] = compute_auc_safe(
                    coarse_mask[b:b+1], gt_mask[b:b+1]
                )
                all_rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "Baseline coarse mask",
                    "valid_for_main_table": "YES",
                    "threshold": threshold,
                    "K": K,
                    "noise_strength": noise_strength,
                    "num_steps": 0,
                    "guidance_scale": guidance_scale,
                    **metrics,
                    "time_ms": float("nan"),
                    "fps": float("nan"),
                })
            
            # 3. Diffusion refinement with different step counts
            for num_steps in num_steps_list:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                refined_mean, uncertainty = model.ensemble_sample(
                    coarse_mask, image, K=K, num_steps=num_steps,
                    noise_strength=noise_strength, guidance_scale=guidance_scale
                )
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                for b in range(batch_size):
                    metrics = compute_binary_metrics(
                        refined_mean[b:b+1], gt_mask[b:b+1], threshold
                    )
                    metrics["auc"] = compute_auc_safe(
                        refined_mean[b:b+1], gt_mask[b:b+1]
                    )
                    all_rows.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": f"LU-Net+RA+Diff({num_steps})",
                        "valid_for_main_table": "YES",
                        "threshold": threshold,
                        "K": K,
                        "noise_strength": noise_strength,
                        "num_steps": num_steps,
                        "guidance_scale": guidance_scale,
                        **metrics,
                        "time_ms": (elapsed / batch_size) * 1000,
                        "fps": batch_size / elapsed if elapsed > 0 else float("nan"),
                    })
    
    return all_rows


def aggregate_and_save(all_rows, output_path):
    """Aggregate metrics across seeds and save as CSV."""
    # Group by dataset, method, and num_steps
    from collections import defaultdict
    
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["dataset"], row["method"], row["num_steps"])
        groups[key].append(row)
    
    summary_rows = []
    for (dataset, method, num_steps), rows in groups.items():
        summary = {
            "dataset": dataset,
            "method": method,
            "num_steps": num_steps,
            "n_seeds": len(set(r["seed"] for r in rows)),
        }
        
        for metric in ["dice", "iou", "acc", "sen", "spec", "auc", "time_ms", "fps"]:
            values = [r[metric] for r in rows if not np.isnan(r[metric])]
            if len(values) == 0:
                summary[metric] = "nan"
            elif len(values) == 1:
                summary[metric] = f"{np.mean(values):.4f}"
            else:
                summary[metric] = f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
        
        summary_rows.append(summary)
    
    # Save detailed rows
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    
    # Save summary
    summary_path = output_path.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Saved detailed results to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    
    return summary_rows


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Configuration
    SEEDS = [42, 123, 2026]
    AE_EPOCHS = 50
    DIFF_EPOCHS = 100
    BATCH_SIZE = 2
    LR = 1e-4
    NUM_STEPS_LIST = [20, 50]  # DDIM steps to evaluate
    NOISE_STRENGTH = 0.5       # Starting noise level for refinement
    GUIDANCE_SCALE = 1.5       # Classifier-free guidance
    K = 5                      # Ensemble samples for uncertainty
    
    OUTPUT_ROOT = Path("seed_outputs_corrected")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print()
    
    all_rows = []
    
    # ============================================================
    # CHASE-DB1
    # ============================================================
    CHASE_DATA_DIR = Path("data")
    
    if CHASE_DATA_DIR.exists():
        print("=" * 80)
        print("CHASE-DB1 Dataset")
        print("=" * 80)
        
        for seed in SEEDS:
            print(f"\n{'='*80}")
            print(f"CHASE-DB1 | Seed {seed}")
            print(f"{'='*80}")
            
            set_seed(seed)
            
            # Load dataset
            dataset = CHASEDataset(CHASE_DATA_DIR, img_size=(512, 512))
            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True,
                generator=make_generator(seed),
                worker_init_fn=seed_worker,
                num_workers=0
            )
            print(f"Loaded {len(dataset)} images")
            
            # Build models
            ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
            img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
            unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
            
            # Stage 1: Train autoencoder
            output_dir = OUTPUT_ROOT / "CHASE-DB1" / f"seed_{seed}"
            ae_path = output_dir / "mask_autoencoder.pth"
            
            ae = train_autoencoder(
                ae, dataloader, device,
                epochs=AE_EPOCHS, lr=LR,
                save_path=str(ae_path)
            )
            
            # Freeze autoencoder
            for param in ae.parameters():
                param.requires_grad = False
            
            # Stage 2: Train diffusion refiner
            model = LatentDiffusionModel(
                ae=ae, img_enc=img_enc, unet=unet,
                cond_dim=128, timesteps=1000
            ).to(device)
            
            diff_path = output_dir / "diffusion_refiner.pth"
            model = train_diffusion_refiner(
                model, dataloader, device,
                epochs=DIFF_EPOCHS, lr=LR,
                dice_weight=0.5, edge_weight=0.3,
                save_path=str(diff_path)
            )
            
            # Evaluate
            print("\nEvaluating...")
            test_loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0
            )
            
            rows = evaluate_model_full(
                model, test_loader, device,
                dataset_name="CHASE-DB1",
                seed=seed,
                num_steps_list=NUM_STEPS_LIST,
                noise_strength=NOISE_STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                K=K
            )
            all_rows.extend(rows)
            
            print(f"Seed {seed} complete. Generated {len(rows)} evaluation rows.")
    
    else:
        print(f"CHASE-DB1 data directory not found: {CHASE_DATA_DIR}")
    
    # ============================================================
    # DRIVE
    # ============================================================
    DRIVE_IMAGES = Path("data_drive/images")
    DRIVE_MASKS = Path("data_drive/masks")
    
    if DRIVE_IMAGES.exists() and DRIVE_MASKS.exists():
        print("\n" + "=" * 80)
        print("DRIVE Dataset")
        print("=" * 80)
        
        for seed in SEEDS:
            print(f"\n{'='*80}")
            print(f"DRIVE | Seed {seed}")
            print(f"{'='*80}")
            
            set_seed(seed)
            
            dataset = DRIVEDataset(DRIVE_IMAGES, DRIVE_MASKS)
            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True,
                generator=make_generator(seed),
                worker_init_fn=seed_worker,
                num_workers=0
            )
            print(f"Loaded {len(dataset)} images")
            
            ae = MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
            img_enc = ImageEncoder(in_ch=3, feat_dim=128).to(device)
            unet = DiffusionUNet(dim=64, cond_dim=128).to(device)
            
            output_dir = OUTPUT_ROOT / "DRIVE" / f"seed_{seed}"
            
            ae = train_autoencoder(
                ae, dataloader, device,
                epochs=AE_EPOCHS, lr=LR,
                save_path=str(output_dir / "mask_autoencoder.pth")
            )
            
            for param in ae.parameters():
                param.requires_grad = False
            
            model = LatentDiffusionModel(
                ae=ae, img_enc=img_enc, unet=unet,
                cond_dim=128, timesteps=1000
            ).to(device)
            
            model = train_diffusion_refiner(
                model, dataloader, device,
                epochs=DIFF_EPOCHS, lr=LR,
                dice_weight=0.5, edge_weight=0.3,
                save_path=str(output_dir / "diffusion_refiner.pth")
            )
            
            test_loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0
            )
            
            rows = evaluate_model_full(
                model, test_loader, device,
                dataset_name="DRIVE",
                seed=seed,
                num_steps_list=NUM_STEPS_LIST,
                noise_strength=NOISE_STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                K=K
            )
            all_rows.extend(rows)
    
    else:
        print("DRIVE dataset not found. Skipping.")
    
    # ============================================================
    # Save results
    # ============================================================
    if all_rows:
        output_csv = OUTPUT_ROOT / "multiseed_results.csv"
        aggregate_and_save(all_rows, str(output_csv))
        
        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        # Group by dataset and method
        from collections import defaultdict
        groups = defaultdict(list)
        for row in all_rows:
            if row["valid_for_main_table"] == "YES":
                key = (row["dataset"], row["method"])
                groups[key].append(row["dice"])
        
        for (dataset, method), dice_values in sorted(groups.items()):
            mean_dice = np.mean(dice_values)
            std_dice = np.std(dice_values, ddof=1) if len(dice_values) > 1 else 0.0
            print(f"{dataset:12s} | {method:30s} | Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    
    else:
        print("No results generated. Check dataset paths.")
    
    print("\nDone!")