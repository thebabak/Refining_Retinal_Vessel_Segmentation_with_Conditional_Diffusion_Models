import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import optim

from .dataset2 import CHASEDataset, DRIVEDataset


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


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice


@torch.no_grad()
def compute_metrics_from_logits(logits, target, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    tn = ((1.0 - pred) * (1.0 - target)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
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


# ============================================================
# Model blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class TransformerBottleneck(nn.Module):
    """
    Transformer encoder applied at the bottleneck feature map.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim=256,
        num_heads=8,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        max_tokens=4096,
    ):
        super().__init__()

        self.dim = dim
        self.max_tokens = max_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        n = tokens.shape[1]

        if n > self.max_tokens:
            raise RuntimeError(
                f"Too many tokens: {n}. Increase max_tokens."
            )

        tokens = tokens + self.pos_embed[:, :n, :]
        tokens = self.transformer(tokens)

        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        return x


class RetinalTransUNet(nn.Module):
    """
    Lightweight TransUNet-style model.

    Encoder:
        CNN downsampling path

    Bottleneck:
        Transformer encoder

    Decoder:
        U-Net style upsampling path
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        base=32,
        transformer_dim=256,
        transformer_depth=4,
        transformer_heads=8,
    ):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, transformer_dim)

        self.transformer = TransformerBottleneck(
            dim=transformer_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            mlp_ratio=4.0,
            dropout=0.1,
            max_tokens=4096,
        )

        self.up4 = UpBlock(transformer_dim, base * 8, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = self.pool1(e1)

        e2 = self.enc2(x)
        x = self.pool2(e2)

        e3 = self.enc3(x)
        x = self.pool3(e3)

        e4 = self.enc4(x)
        x = self.pool4(e4)

        x = self.bottleneck(x)
        x = self.transformer(x)

        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)

        logits = self.out_conv(x)

        return logits


# ============================================================
# Training / validation
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0

    for batch in dataloader:
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

        loss = segmentation_loss(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0

    metric_sums = {
        "dice": 0.0,
        "iou": 0.0,
        "acc": 0.0,
        "sen": 0.0,
        "spec": 0.0,
    }

    count = 0

    for batch in dataloader:
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

        loss = segmentation_loss(logits, mask)
        total_loss += loss.item()

        metrics = compute_metrics_from_logits(logits, mask)

        for key in metric_sums:
            metric_sums[key] += metrics[key]

        count += 1

    avg_metrics = {
        key: value / max(count, 1)
        for key, value in metric_sums.items()
    }

    avg_loss = total_loss / max(len(dataloader), 1)

    return avg_loss, avg_metrics


def train_transformer_baseline(
    dataset,
    save_path,
    epochs=100,
    batch_size=2,
    lr=1e-4,
    train_ratio=0.8,
    num_workers=0,
):
    device = get_device()

    print("=" * 80)
    print("Training Transformer baseline: RetinalTransUNet")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Dataset size: {len(dataset)}")

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = RetinalTransUNet(
        in_ch=3,
        out_ch=1,
        base=32,
        transformer_dim=256,
        transformer_depth=4,
        transformer_heads=8,
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

    best_dice = -1.0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Acc: {val_metrics['acc']:.4f} | "
            f"Sen: {val_metrics['sen']:.4f} | "
            f"Spec: {val_metrics['spec']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_dice": best_dice,
                    "epoch": epoch + 1,
                },
                save_path,
            )

            print(f"Saved best model to: {save_path}")

    total_time = time.time() - start_time

    print("=" * 80)
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print("=" * 80)

    return model


# ============================================================
# Prediction visualization
# ============================================================

@torch.no_grad()
def save_prediction_examples(
    model,
    dataloader,
    device,
    save_path,
    num_cases=3,
    threshold=0.5,
):
    model.eval()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cases = []

    for batch in dataloader:
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

        prob = torch.sigmoid(logits)

        batch_size = image.shape[0]

        for i in range(batch_size):
            cases.append(
                {
                    "image": image[i].detach().cpu(),
                    "gt": mask[i].detach().cpu(),
                    "pred": prob[i].detach().cpu(),
                }
            )

            if len(cases) >= num_cases:
                break

        if len(cases) >= num_cases:
            break

    columns = ["Input", "Ground Truth", "Transformer Prediction", "Binary Prediction"]

    fig, axes = plt.subplots(
        len(cases),
        len(columns),
        figsize=(4 * len(columns), 3.6 * len(cases)),
    )

    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, case in enumerate(cases):
        image_np = case["image"].permute(1, 2, 0).numpy()
        gt_np = case["gt"][0].numpy()
        pred_np = case["pred"][0].numpy()
        pred_bin = (pred_np >= threshold).astype(np.float32)

        image_np = np.clip(image_np, 0, 1)

        row_items = [
            image_np,
            gt_np,
            pred_np,
            pred_bin,
        ]

        for col_idx, item in enumerate(row_items):
            ax = axes[row_idx, col_idx]

            if col_idx == 0:
                ax.imshow(item)
            else:
                ax.imshow(item, cmap="gray", vmin=0, vmax=1)

            ax.axis("off")

            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved transformer prediction examples to: {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    device = get_device()

    output_dir = Path("transformer_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # CHASE-DB1 Transformer baseline
    # ------------------------------------------------------------

    chase_path = Path("data")

    if chase_path.exists():
        chase_dataset = CHASEDataset(
            chase_path,
            img_size=(512, 512),
        )

        transformer_model = train_transformer_baseline(
            dataset=chase_dataset,
            save_path=output_dir / "transformer_chase_best.pth",
            epochs=100,
            batch_size=2,
            lr=1e-4,
            train_ratio=0.8,
            num_workers=0,
        )

        chase_loader = DataLoader(
            chase_dataset,
            batch_size=2,
            shuffle=False,
        )

        save_prediction_examples(
            model=transformer_model,
            dataloader=chase_loader,
            device=device,
            save_path=output_dir / "transformer_chase_examples.png",
            num_cases=3,
            threshold=0.5,
        )

    else:
        print("CHASE dataset not found at path: data")

    # ------------------------------------------------------------
    # DRIVE Transformer baseline
    # ------------------------------------------------------------

    drive_images = Path("data_drive/images")
    drive_masks = Path("data_drive/masks")

    if drive_images.exists() and drive_masks.exists():
        drive_dataset = DRIVEDataset(
            drive_images,
            drive_masks,
            img_size=(512, 512),
        )

        drive_model = train_transformer_baseline(
            dataset=drive_dataset,
            save_path=output_dir / "transformer_drive_best.pth",
            epochs=100,
            batch_size=2,
            lr=1e-4,
            train_ratio=0.8,
            num_workers=0,
        )

        drive_loader = DataLoader(
            drive_dataset,
            batch_size=2,
            shuffle=False,
        )

        save_prediction_examples(
            model=drive_model,
            dataloader=drive_loader,
            device=device,
            save_path=output_dir / "transformer_drive_examples.png",
            num_cases=3,
            threshold=0.5,
        )

    else:
        print("DRIVE dataset folders not found. Skipping DRIVE transformer baseline.")