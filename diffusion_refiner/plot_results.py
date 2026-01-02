"""
Simple plotting utilities for evaluation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(csv_path, save_path=None):
    """Plot evaluation metrics from CSV."""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Vessel Segmentation Evaluation Metrics', fontsize=16)
    
    metrics = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity', 'auc']
    for ax, metric in zip(axes.flat, metrics):
        ax.hist(df[metric], bins=10, edgecolor='black', alpha=0.7)
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Frequency')
        ax.axvline(df[metric].mean(), color='r', linestyle='--', label=f'μ={df[metric].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    csv_path = Path("evaluation_results.csv")
    if csv_path.exists():
        plot_metrics(csv_path, save_path="evaluation_metrics_plot.png")
    else:
        print(f"CSV not found at {csv_path}")
