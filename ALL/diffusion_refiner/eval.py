import torch
import numpy as np
from pathlib import Path
import csv
from sklearn.metrics import roc_auc_score, jaccard_score
from torch.utils.data import DataLoader


class SegmentationMetrics:
    """Compute standard segmentation metrics for vessel segmentation."""
    
    @staticmethod
    def dice(pred, target, eps=1e-6):
        """Dice coefficient (F1 score for binary segmentation)."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        intersection = np.sum(pred * target)
        return (2 * intersection + eps) / (np.sum(pred) + np.sum(target) + eps)
    
    @staticmethod
    def iou(pred, target, eps=1e-6):
        """Intersection over Union (Jaccard)."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        pred_bin = (pred > 0.5).astype(np.int32)
        target_bin = target.astype(np.int32)
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin) - intersection
        return (intersection + eps) / (union + eps)
    
    @staticmethod
    def accuracy(pred, target, threshold=0.5):
        """Pixel-wise accuracy."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        pred_bin = (pred > threshold).astype(np.int32)
        target_bin = target.astype(np.int32)
        return np.mean(pred_bin == target_bin)
    
    @staticmethod
    def sensitivity(pred, target, threshold=0.5):
        """Recall / True Positive Rate."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        pred_bin = (pred > threshold).astype(np.int32)
        target_bin = target.astype(np.int32)
        tp = np.sum((pred_bin == 1) & (target_bin == 1))
        fn = np.sum((pred_bin == 0) & (target_bin == 1))
        return tp / (tp + fn + 1e-6)
    
    @staticmethod
    def specificity(pred, target, threshold=0.5):
        """True Negative Rate."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        pred_bin = (pred > threshold).astype(np.int32)
        target_bin = target.astype(np.int32)
        tn = np.sum((pred_bin == 0) & (target_bin == 0))
        fp = np.sum((pred_bin == 1) & (target_bin == 0))
        return tn / (tn + fp + 1e-6)
    
    @staticmethod
    def auc(pred, target):
        """Area Under ROC Curve."""
        pred = pred.flatten().numpy() if isinstance(pred, torch.Tensor) else pred.flatten()
        target = target.flatten().numpy() if isinstance(target, torch.Tensor) else target.flatten()
        try:
            return roc_auc_score(target, pred)
        except:
            return 0.5


def evaluate_model(model, dataloader, device, num_steps=50, guidance_scale=1.5, save_results=None):
    """Evaluate model on dataset and compute metrics.
    
    Args:
        model: LatentDiffusionModel
        dataloader: PyTorch DataLoader
        device: torch device
        num_steps: DDIM sampling steps
        guidance_scale: guidance scale for refinement
        save_results: path to save CSV results
    
    Returns:
        dict with aggregated metrics
    """
    from .inference import refine_mask
    
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            target_mask = batch['mask'].to(device)
            coarse_mask = batch.get('baseline', None)
            if coarse_mask is None:
                coarse_mask = target_mask + 0.1 * torch.randn_like(target_mask)
                coarse_mask = torch.clamp(coarse_mask, 0, 1)
            else:
                coarse_mask = coarse_mask.to(device)

            B = image.shape[0]
            
            # Refine each mask in batch
            for i in range(B):
                img_i = image[i]
                coarse_i = coarse_mask[i]
                target_i = target_mask[i]
                
                # Refine
                refined = refine_mask(
                    model,
                    img_i,
                    coarse_i,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    device=device
                )
                
                # Compute metrics
                pred_np = refined.detach().cpu().squeeze().numpy()
                target_np = target_i.detach().cpu().squeeze().numpy()
                
                metrics = {
                    'image_idx': batch_idx * len(dataloader) + i,
                    'dice': SegmentationMetrics.dice(pred_np, target_np),
                    'iou': SegmentationMetrics.iou(pred_np, target_np),
                    'accuracy': SegmentationMetrics.accuracy(pred_np, target_np),
                    'sensitivity': SegmentationMetrics.sensitivity(pred_np, target_np),
                    'specificity': SegmentationMetrics.specificity(pred_np, target_np),
                    'auc': SegmentationMetrics.auc(pred_np, target_np),
                }
                metrics_list.append(metrics)
            
            if (batch_idx + 1) % 2 == 0:
                print(f"Evaluated {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Aggregate results
    if metrics_list:
        agg_metrics = {
            'dice_mean': np.mean([m['dice'] for m in metrics_list]),
            'dice_std': np.std([m['dice'] for m in metrics_list]),
            'iou_mean': np.mean([m['iou'] for m in metrics_list]),
            'iou_std': np.std([m['iou'] for m in metrics_list]),
            'accuracy_mean': np.mean([m['accuracy'] for m in metrics_list]),
            'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
            'sensitivity_mean': np.mean([m['sensitivity'] for m in metrics_list]),
            'sensitivity_std': np.std([m['sensitivity'] for m in metrics_list]),
            'specificity_mean': np.mean([m['specificity'] for m in metrics_list]),
            'specificity_std': np.std([m['specificity'] for m in metrics_list]),
            'auc_mean': np.mean([m['auc'] for m in metrics_list]),
            'auc_std': np.std([m['auc'] for m in metrics_list]),
        }
    else:
        agg_metrics = {}
    
    # Save detailed results
    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys() if metrics_list else [])
            writer.writeheader()
            writer.writerows(metrics_list)
        print(f"Saved detailed results to {save_path}")
    
    return agg_metrics, metrics_list


def print_metrics(agg_metrics):
    """Pretty-print aggregated metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in agg_metrics.items():
        print(f"{key:25s}: {value:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    from .dataset2 import DRIVEDataset, HRFDataset
    from .inference import load_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(
        description="Evaluate the diffusion refiner on DRIVE or HRF datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["drive", "hrf"], help="Dataset type to evaluate.")
    parser.add_argument("--images", type=str, required=True, help="Folder containing input fundus images.")
    parser.add_argument("--masks", type=str, required=True, help="Folder containing ground-truth vessel masks.")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Optional folder containing coarse baseline predictions for refinement.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion refiner checkpoint.")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Path to save detailed metrics CSV.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20, help="DDIM sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="Guidance scale for refinement.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = load_checkpoint(ckpt_path, device)

    ds_kwargs = {
        "img_size": (args.img_size, args.img_size),
        "baseline_dir": args.baseline_dir,
    }

    if args.dataset == "drive":
        ds = DRIVEDataset(args.images, args.masks, **ds_kwargs)
    else:
        ds = HRFDataset(args.images, args.masks, **ds_kwargs)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Evaluating on {len(ds)} images from {args.dataset.upper()}...")
    if args.baseline_dir is None:
        print("Warning: no baseline directory provided. Using target mask as coarse input for refinement.")

    agg_metrics, detailed = evaluate_model(
        model,
        dl,
        device,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        save_results=args.output,
    )

    print_metrics(agg_metrics)
