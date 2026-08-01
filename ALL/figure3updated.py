from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_refiner.dataset import CHASEDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_image_shape(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image shape [B,3,H,W], got {tuple(image.shape)}")
    return image.float()


def ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape [B,1,H,W], got {tuple(mask.shape)}")
    return mask.float()


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.clamp(0.0, 1.0).numpy()


def make_error_map(gt_mask: np.ndarray, pred_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    gt = gt_mask >= 0.5
    pred = pred_prob >= threshold

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)
    tn = (~gt) & (~pred)

    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    out[tp] = (1.0, 1.0, 1.0)
    out[fp] = (1.0, 0.0, 0.0)
    out[fn] = (0.0, 0.0, 1.0)
    out[tn] = (0.0, 0.0, 0.0)
    return out


def dice_from_probs(gt_mask: torch.Tensor, pred_prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    gt_bin = (gt_mask >= 0.5).float()
    pred_bin = (pred_prob >= threshold).float()
    inter = (gt_bin * pred_bin).flatten(1).sum(dim=1)
    denom = gt_bin.flatten(1).sum(dim=1) + pred_bin.flatten(1).sum(dim=1)
    return (2.0 * inter + 1e-7) / (denom + 1e-7)


def select_zoom_boxes(
    gt_mask: np.ndarray,
    coarse_prob: np.ndarray,
    prop_prob: np.ndarray,
    input_img: np.ndarray,
    threshold: float,
    max_regions: int = 2,
    crop_size: int = 96,
    min_distance: int = 72,
    min_gt_pixels: int = 180,
    min_input_brightness: float = 0.08,
) -> List[Tuple[int, int, int, int]]:
    gt = gt_mask >= 0.5
    coarse = coarse_prob >= threshold
    proposed = prop_prob >= threshold

    recovered = gt & (~coarse) & proposed
    candidate_mask = recovered
    if candidate_mask.sum() == 0:
        candidate_mask = gt & (~coarse)
    if candidate_mask.sum() == 0:
        candidate_mask = gt

    coords = np.argwhere(candidate_mask)
    if len(coords) == 0:
        h, w = gt.shape
        y0 = max(0, h // 2 - crop_size // 2)
        x0 = max(0, w // 2 - crop_size // 2)
        return [(x0, y0, crop_size, crop_size)]

    sample_count = min(len(coords), 512)
    if len(coords) > sample_count:
        idx = np.linspace(0, len(coords) - 1, sample_count).astype(int)
        coords = coords[idx]

    h, w = gt.shape
    scored_boxes: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for y, x in coords:
        y0 = int(np.clip(y - crop_size // 2, 0, max(0, h - crop_size)))
        x0 = int(np.clip(x - crop_size // 2, 0, max(0, w - crop_size)))

        recovered_crop = recovered[y0:y0 + crop_size, x0:x0 + crop_size]
        gt_crop = gt_mask[y0:y0 + crop_size, x0:x0 + crop_size]
        img_crop = input_img[y0:y0 + crop_size, x0:x0 + crop_size]
        gt_pixels = int((gt_crop >= 0.5).sum())
        recovered_pixels = int(recovered_crop.sum())
        brightness = float(img_crop.mean())

        if gt_pixels >= min_gt_pixels and brightness >= min_input_brightness:
            # Favor regions with true recovered vessels and enough visibility.
            score = 2.0 * recovered_pixels + 0.5 * gt_pixels + 200.0 * brightness
            scored_boxes.append((float(score), (x0, y0, crop_size, crop_size)))

    scored_boxes.sort(key=lambda item: item[0], reverse=True)
    boxes: List[Tuple[int, int, int, int]] = []
    for _, box in scored_boxes:
        if not boxes:
            boxes.append(box)
            if len(boxes) >= max_regions:
                break
            continue

        bx, by, bw, bh = box
        ok = True
        for sx, sy, sw, sh in boxes:
            cx1, cy1 = bx + bw / 2.0, by + bh / 2.0
            cx2, cy2 = sx + sw / 2.0, sy + sh / 2.0
            if (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 < float(min_distance ** 2):
                ok = False
                break
        if ok:
            boxes.append(box)
            if len(boxes) >= max_regions:
                break

    if not boxes:
        for y, x in coords[:max_regions]:
            y0 = int(np.clip(y - crop_size // 2, 0, max(0, h - crop_size)))
            x0 = int(np.clip(x - crop_size // 2, 0, max(0, w - crop_size)))
            boxes.append((x0, y0, crop_size, crop_size))
    return boxes


def crop_array(arr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, width, height = box
    return arr[y0:y0 + height, x0:x0 + width]


def build_candidates(device: torch.device) -> List[Tuple[str, torch.nn.Module, Callable[..., torch.Tensor]]]:
    candidates: List[Tuple[str, torch.nn.Module, Callable[..., torch.Tensor]]] = []

    try:
        from diffusion_refiner import models_corrected

        model, _ = models_corrected.build_model(device)

        def corrected_sampler(
            model: torch.nn.Module,
            coarse: torch.Tensor,
            image: torch.Tensor,
            ddim_steps: int,
            noise_strength: float,
            guidance_scale: float,
            num_samples: int,
        ) -> torch.Tensor:
            if num_samples > 1 and hasattr(model, "ensemble_sample"):
                mean_mask, _ = model.ensemble_sample(
                    coarse,
                    image,
                    K=num_samples,
                    num_steps=ddim_steps,
                    noise_strength=noise_strength,
                    guidance_scale=guidance_scale,
                )
                return mean_mask
            return model.ddim_sample(
                coarse,
                image,
                num_steps=ddim_steps,
                noise_strength=noise_strength,
                guidance_scale=guidance_scale,
            )

        candidates.append(("models_corrected", model, corrected_sampler))
    except Exception:
        pass

    try:
        from diffusion_refiner import models_corrected2

        ae = models_corrected2.MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
        img_enc = models_corrected2.ImageEncoder(in_ch=3, feat_dim=128).to(device)
        unet = models_corrected2.DiffusionUNet(dim=64, cond_dim=128, time_emb_dim=256).to(device)
        model = models_corrected2.LatentDiffusionModel(ae, img_enc, unet, cond_dim=128, timesteps=1000).to(device)

        def corrected2_sampler(
            model: torch.nn.Module,
            coarse: torch.Tensor,
            image: torch.Tensor,
            ddim_steps: int,
            noise_strength: float,
            guidance_scale: float,
            num_samples: int,
        ) -> torch.Tensor:
            if num_samples > 1 and hasattr(model, "ensemble_sample"):
                mean_mask, _ = model.ensemble_sample(
                    coarse,
                    image,
                    K=num_samples,
                    num_steps=ddim_steps,
                    noise_strength=noise_strength,
                    guidance_scale=guidance_scale,
                )
                return mean_mask
            return model.ddim_sample(
                coarse,
                image,
                num_steps=ddim_steps,
                noise_strength=noise_strength,
                guidance_scale=guidance_scale,
            )

        candidates.append(("models_corrected2", model, corrected2_sampler))
    except Exception:
        pass

    try:
        from diffusion_refiner import models
        from diffusion_refiner.train_option12_fixed import get_proposed_prediction

        ae = models.MaskAutoencoder(in_ch=1, base=32, latent_dim=64).to(device)
        img_enc = models.ImageEncoder(in_ch=3, feat_dim=128).to(device)
        unet = models.DiffusionUNet(dim=64, cond_dim=128).to(device)
        model = models.LatentDiffusionModel(ae, img_enc, unet, cond_dim=128).to(device)

        def current_sampler(
            model: torch.nn.Module,
            coarse: torch.Tensor,
            image: torch.Tensor,
            ddim_steps: int,
            noise_strength: float,
            guidance_scale: float,
            num_samples: int,
        ) -> torch.Tensor:
            return get_proposed_prediction(
                model=model,
                image=image,
                coarse_mask=coarse,
                device=image.device,
                num_samples=num_samples,
                noise_strength=noise_strength,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
            )

        candidates.append(("models", model, current_sampler))
    except Exception:
        pass

    return candidates


def try_load_model(checkpoint_path: Path, device: torch.device) -> Tuple[str, torch.nn.Module, Callable[..., torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")

    last_error = None
    for name, model, sampler in build_candidates(device):
        try:
            model.ae.load_state_dict(checkpoint["ae"], strict=True)
            model.img_enc.load_state_dict(checkpoint["img_enc"], strict=True)
            model.unet.load_state_dict(checkpoint["unet"], strict=True)
            model.eval()
            return name, model, sampler
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not match the checkpoint to any known model variant. "
        f"Last error: {last_error}"
    )


@torch.no_grad()
def build_cases(
    model: torch.nn.Module,
    sampler: Callable[..., torch.Tensor],
    dataloader: DataLoader,
    device: torch.device,
    num_cases: int,
    case_offset: int,
    num_samples: int,
    noise_strength: float,
    ddim_steps: int,
    guidance_scale: float,
    threshold: float,
    positive_only: bool,
) -> List[dict]:
    cases = []

    for batch in dataloader:
        image = ensure_image_shape(batch["image"].to(device))
        gt = ensure_mask_shape(batch["mask"].to(device))
        coarse = ensure_mask_shape(batch["coarse"].to(device))

        proposed = sampler(
            model,
            coarse,
            image,
            ddim_steps,
            noise_strength,
            guidance_scale,
            num_samples,
        )

        if proposed.shape[-2:] != gt.shape[-2:]:
            proposed = torch.nn.functional.interpolate(
                proposed,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        coarse_dice = dice_from_probs(gt, coarse, threshold=threshold)
        proposed_dice = dice_from_probs(gt, proposed, threshold=threshold)
        delta_dice = proposed_dice - coarse_dice

        hard_score = ((gt > 0.5) & (coarse < 0.5)).flatten(1).sum(dim=1)
        for i in range(image.shape[0]):
            cases.append(
                {
                    "score": float(delta_dice[i].detach().cpu()),
                    "delta_dice": float(delta_dice[i].detach().cpu()),
                    "baseline_dice": float(coarse_dice[i].detach().cpu()),
                    "proposed_dice": float(proposed_dice[i].detach().cpu()),
                    "hard_score": float(hard_score[i].detach().cpu()),
                    "image": image[i].detach().cpu(),
                    "gt": gt[i].detach().cpu(),
                    "coarse": coarse[i].detach().cpu(),
                    "proposed": proposed[i].detach().cpu(),
                }
            )

    if not cases:
        raise RuntimeError("No cases were collected from the dataloader.")

    cases.sort(key=lambda row: (row["score"], row["hard_score"]), reverse=True)
    if positive_only:
        positive_cases = [row for row in cases if row["delta_dice"] > 0]
        if positive_cases:
            cases = positive_cases
    start = max(0, case_offset)
    end = start + num_cases
    selected = cases[start:end]
    if not selected:
        raise RuntimeError(
            f"No cases available for case_offset={case_offset} and num_cases={num_cases}. "
            f"Collected only {len(cases)} cases."
        )

    positive = sum(1 for row in cases if row["delta_dice"] > 0)
    print(f"Found {positive}/{len(cases)} cases with positive Dice improvement.")
    for idx, row in enumerate(selected, start=1):
        print(
            f"Selected case {idx + start}: "
            f"baseline_dice={row['baseline_dice']:.4f}, "
            f"proposed_dice={row['proposed_dice']:.4f}, "
            f"delta={row['delta_dice']:+.4f}"
        )
    return selected


def save_figure(cases: List[dict], save_path: Path, threshold: float, title: str) -> None:
    cols = ["Input Image", "Ground Truth", "LU-Net+RA (Baseline)", "Proposed Method", "Error Map"]

    fig, axes = plt.subplots(len(cases), len(cols), figsize=(4.0 * len(cols), 3.6 * len(cases)))
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, case in enumerate(cases):
        img_np = tensor_image_to_numpy(case["image"])
        gt_np = tensor_mask_to_numpy(case["gt"])
        coarse_np = tensor_mask_to_numpy(case["coarse"])
        prop_np = tensor_mask_to_numpy(case["proposed"])
        err_np = make_error_map(gt_np, prop_np, threshold=threshold)
        boxes = select_zoom_boxes(gt_np, coarse_np, prop_np, img_np, threshold)
        box_colors = ["yellow", "cyan"]

        row_imgs = [img_np, gt_np, coarse_np, prop_np, err_np]
        for c, item in enumerate(row_imgs):
            ax = axes[r, c]
            if c in (0, 4):
                ax.imshow(item)
            else:
                ax.imshow(item, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c], fontsize=12)
            for idx, box in enumerate(boxes):
                color = box_colors[idx % len(box_colors)]
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2],
                        box[3],
                        linewidth=1.2,
                        edgecolor=color,
                        facecolor="none",
                    )
                )
            ax.axis("off")

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_zoom_figure(
    cases: List[dict],
    save_path: Path,
    threshold: float,
    title: str,
    crop_size: int,
    zooms_per_case: int,
) -> None:
    cols = ["Input Crop", "Ground Truth Crop", "Baseline Crop", "Proposed Crop", "Error Crop"]
    rows = []

    for case_idx, case in enumerate(cases, start=1):
        img_np = tensor_image_to_numpy(case["image"])
        gt_np = tensor_mask_to_numpy(case["gt"])
        coarse_np = tensor_mask_to_numpy(case["coarse"])
        prop_np = tensor_mask_to_numpy(case["proposed"])
        err_np = make_error_map(gt_np, prop_np, threshold=threshold)
        boxes = select_zoom_boxes(
            gt_np,
            coarse_np,
            prop_np,
            img_np,
            threshold,
            max_regions=zooms_per_case,
            crop_size=crop_size,
        )

        for zoom_idx, box in enumerate(boxes, start=1):
            rows.append(
                {
                    "label": f"Case {case_idx}, ROI {zoom_idx}",
                    "images": [
                        crop_array(img_np, box),
                        crop_array(gt_np, box),
                        crop_array(coarse_np, box),
                        crop_array(prop_np, box),
                        crop_array(err_np, box),
                    ],
                }
            )

    if not rows:
        raise RuntimeError("No zoom rows were created.")

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 3.0 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, row in enumerate(rows):
        for c, item in enumerate(row["images"]):
            ax = axes[r, c]
            if c in (0, 4):
                ax.imshow(item)
            else:
                ax.imshow(item, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            ax.axis("off")
        axes[r, 0].set_ylabel(row["label"], fontsize=10)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate updated Figure 3 with baseline vs proposed comparison.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--coarse-dir", type=Path, default=root / "predictions" / "CHASE" / "lunet_ra_debug")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "seed_outputs_corrected" / "CHASE-DB1" / "seed_42" / "diffusion_refiner.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "realdata" / "qualitative_chase_comparison_updated.png",
    )
    parser.add_argument(
        "--zoom-output",
        type=Path,
        default=root / "realdata" / "qualitative_chase_comparison_zoomed.png",
    )
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--noise-strength", type=float, default=0.25)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--title", type=str, default="Qualitative comparison on CHASE-DB1")
    parser.add_argument("--zoom-title", type=str, default="Zoomed vessel improvements on CHASE-DB1")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--zooms-per-case", type=int, default=2)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Select only cases where proposed Dice is better than baseline Dice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    dataset = CHASEDataset(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        coarse_dir=args.coarse_dir,
        require_coarse=True,
        binarize_coarse=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_name, model, sampler = try_load_model(args.checkpoint, device)
    print(f"Loaded checkpoint with model variant: {model_name}")

    cases = build_cases(
        model=model,
        sampler=sampler,
        dataloader=dataloader,
        device=device,
        num_cases=args.num_cases,
        case_offset=args.case_offset,
        num_samples=args.num_samples,
        noise_strength=args.noise_strength,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        threshold=args.threshold,
        positive_only=args.positive_only,
    )
    save_figure(cases, args.output, args.threshold, args.title)
    save_zoom_figure(
        cases,
        args.zoom_output,
        args.threshold,
        args.zoom_title,
        args.crop_size,
        args.zooms_per_case,
    )
    print(f"Saved updated figure to: {args.output}")
    print(f"Saved zoomed figure to: {args.zoom_output}")


if __name__ == "__main__":
    main()