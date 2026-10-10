"""
Evaluate the stored LU-Net+RA coarse baseline predictions on the SAME
8-image CHASE-DB1 test split used in the paper, so that the baseline
row in the results table comes from the same protocol as the proposed
method (response to Reviewer Comment 1).

Split reproduction: sorted list of the 28 CHASE-DB1 images,
torch.utils.data.random_split([20, 8], generator seed=42) as in
make_train_test_loaders (tarinWithOp.py).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BASELINE_DIR = ROOT / "predictions" / "CHASE" / "lunet_ra_debug"
IMG_SIZE = (512, 512)
TRAIN_SIZE = 20
SEED = 42


def load_binary_mask(path: Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize(IMG_SIZE, Image.Resampling.NEAREST)
    mask = TF.to_tensor(mask).float()
    return (mask > 0.5).float()


def load_probability_mask(path: Path) -> torch.Tensor:
    pred = Image.open(path).convert("L")
    pred = pred.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    pred = TF.to_tensor(pred).float()
    return pred.clamp(0.0, 1.0)


def binary_metrics(pred_prob: torch.Tensor, gt: torch.Tensor, threshold: float, eps: float = 1e-7):
    pred_bin = (pred_prob >= threshold).float().reshape(-1)
    gt_bin = (gt >= 0.5).float().reshape(-1)
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
    return {
        "dice": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),
        "iou": ((tp + eps) / (tp + fp + fn + eps)).item(),
        "acc": ((tp + tn + eps) / (tp + tn + fp + fn + eps)).item(),
        "sen": ((tp + eps) / (tp + fn + eps)).item(),
        "spec": ((tn + eps) / (tn + fp + eps)).item(),
    }


def auc_safe(pred_prob: torch.Tensor, gt: torch.Tensor):
    try:
        from sklearn.metrics import roc_auc_score
        p = pred_prob.reshape(-1).numpy()
        g = (gt.reshape(-1).numpy() >= 0.5).astype(np.uint8)
        if len(np.unique(g)) < 2:
            return float("nan")
        return float(roc_auc_score(g, p))
    except Exception:
        return float("nan")


def main():
    stems = sorted(p.stem for p in DATA_DIR.glob("*.jpg"))
    assert len(stems) == 28, f"Expected 28 CHASE-DB1 images, found {len(stems)}"

    gen = torch.Generator().manual_seed(SEED)
    _, test_subset = torch.utils.data.random_split(stems, [TRAIN_SIZE, len(stems) - TRAIN_SIZE], generator=gen)
    test_stems = sorted(test_subset.indices and [stems[i] for i in test_subset.indices])
    print("Test split (8 images):", test_stems)

    for threshold in (0.5, 0.25):
        rows, aucs = [], []
        print(f"\n=== LU-Net+RA baseline on the same 8 test images (threshold={threshold}) ===")
        for stem in test_stems:
            gt = load_binary_mask(DATA_DIR / f"{stem}_1stHO.png")
            pred = load_probability_mask(BASELINE_DIR / f"{stem}.png")
            m = binary_metrics(pred, gt, threshold)
            a = auc_safe(pred, gt)
            rows.append(m)
            aucs.append(a)
            print(f"{stem}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()) + f" auc={a:.4f}")

        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        valid_auc = [a for a in aucs if not np.isnan(a)]
        mean["auc"] = float(np.mean(valid_auc)) if valid_auc else float("nan")
        print("\nMEAN over 8 test images:")
        print(" ".join(f"{k}={v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
