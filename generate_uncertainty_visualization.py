"""
Generate visualization of ensemble sampling and uncertainty quantification.
Creates a comprehensive figure showing the uncertainty workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_ensemble_visualization():
    """Create figure showing ensemble sampling workflow and uncertainty."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid spec for complex layout
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Uncertainty Quantification via Ensemble Sampling\nDiffusion-based Retinal Vessel Refinement', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== LEFT COLUMN: Input =====
    ax_input = fig.add_subplot(gs[:2, 0])
    ax_input.set_xlim(0, 10)
    ax_input.set_ylim(0, 10)
    ax_input.axis('off')
    ax_input.text(5, 9, 'Input', fontsize=12, fontweight='bold', ha='center')
    
    # Draw boxes for input
    for i, label in enumerate(['RGB Image', 'Coarse Mask']):
        y_pos = 7 - i*2.5
        box = FancyBboxPatch((0.5, y_pos-0.8), 9, 1.5, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='lightblue', linewidth=2)
        ax_input.add_patch(box)
        ax_input.text(5, y_pos, label, fontsize=11, ha='center', va='center', fontweight='bold')
    
    # ===== CENTER COLUMNS: Ensemble Sampling =====
    # Title
    ax_title = fig.add_subplot(gs[0, 1:3])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Ensemble Sampling (k=5 Independent Samples)', 
                 fontsize=13, fontweight='bold', ha='center', transform=ax_title.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.5))
    
    # Individual samples
    np.random.seed(42)
    for i in range(5):
        ax = fig.add_subplot(gs[1, 1 + i//3])
        
        # Create synthetic prediction (varies slightly)
        sample = np.random.randn(64, 64) * 0.3 + 0.6
        sample = np.clip(sample, 0, 1)
        
        # Add vessel-like structure
        y, x = np.mgrid[0:64, 0:64]
        vessels = np.exp(-((y - 32)**2 + (x - 32)**2) / 400) * 0.5
        sample = np.clip(sample + vessels, 0, 1)
        
        ax.imshow(sample, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Sample {i+1}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # ===== BOTTOM LEFT: Mean =====
    ax_mean = fig.add_subplot(gs[2, 0:2])
    mean_pred = np.random.randn(64, 64) * 0.15 + 0.6
    mean_pred = np.clip(mean_pred, 0, 1)
    y, x = np.mgrid[0:64, 0:64]
    vessels = np.exp(-((y - 32)**2 + (x - 32)**2) / 400) * 0.5
    mean_pred = np.clip(mean_pred + vessels, 0, 1)
    
    im_mean = ax_mean.imshow(mean_pred, cmap='gray', vmin=0, vmax=1)
    ax_mean.set_title('Mean Prediction\n(Refined Mask)', fontsize=11, fontweight='bold')
    ax_mean.axis('off')
    plt.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
    
    # ===== BOTTOM CENTER: Variance =====
    ax_var = fig.add_subplot(gs[2, 2:4])
    variance = np.random.rand(64, 64) * 0.05
    # Higher variance at boundaries
    variance += np.exp(-((y - 32)**2 + (x - 32)**2) / 600) * 0.02
    variance = np.clip(variance, 0, 0.1)
    
    im_var = ax_var.imshow(variance, cmap='hot', vmin=0, vmax=0.1)
    ax_var.set_title('Pixel-wise Variance\n(Uncertainty Map)', fontsize=11, fontweight='bold')
    ax_var.axis('off')
    cbar = plt.colorbar(im_var, ax=ax_var, fraction=0.046, pad=0.04)
    cbar.set_label('Variance', fontsize=9)
    
    plt.savefig(OUTPUT_DIR / 'ensemble_sampling_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ensemble_sampling_workflow.png")
    plt.close()


def generate_uncertainty_comparison():
    """Create comparison of single vs ensemble uncertainty estimates."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Single Sample vs. Ensemble Sampling\nUncertainty Quantification Comparison', 
                 fontsize=14, fontweight='bold')
    
    np.random.seed(42)
    
    # Row 1: Single sample (no uncertainty)
    ax = axes[0, 0]
    single = np.random.randn(128, 128) * 0.2 + 0.6
    single = np.clip(single, 0, 1)
    ax.imshow(single, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Single Sample\nRefined Mask', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.text(0.5, 0.7, 'Single Refinement', fontsize=12, fontweight='bold', ha='center', 
           transform=ax.transAxes)
    ax.text(0.5, 0.4, '✗ No uncertainty estimate\n✗ No confidence measure\n✗ Cannot identify ambiguous regions', 
           fontsize=10, ha='center', transform=ax.transAxes, family='monospace',
           bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    ax.axis('off')
    
    ax = axes[0, 2]
    ax.text(0.5, 0.5, 'N/A\n(Single output only)', fontsize=11, ha='center', 
           transform=ax.transAxes, style='italic', color='gray')
    ax.axis('off')
    
    # Row 2: Ensemble samples
    ax = axes[1, 0]
    ensemble_mean = np.random.randn(128, 128) * 0.15 + 0.6
    ensemble_mean = np.clip(ensemble_mean, 0, 1)
    ax.imshow(ensemble_mean, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Ensemble Mean\nRefined Mask', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.text(0.5, 0.7, 'Ensemble Refinement (k=5)', fontsize=12, fontweight='bold', ha='center', 
           transform=ax.transAxes)
    ax.text(0.5, 0.4, '✓ Pixel-wise uncertainty\n✓ Confidence map\n✓ Identify ambiguous regions', 
           fontsize=10, ha='center', transform=ax.transAxes, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.axis('off')
    
    ax = axes[1, 2]
    uncertainty = np.random.rand(128, 128) * 0.05
    # Higher at boundaries
    y, x = np.mgrid[0:128, 0:128]
    uncertainty += np.exp(-((y - 64)**2 + (x - 64)**2) / 1000) * 0.015
    uncertainty = np.clip(uncertainty, 0, 0.1)
    
    im = ax.imshow(uncertainty, cmap='hot', vmin=0, vmax=0.1)
    ax.set_title('Uncertainty Map\n(Variance)', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ensemble_vs_single_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: ensemble_vs_single_comparison.png")
    plt.close()


def generate_clinical_workflow():
    """Create clinical decision support workflow diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(5, 11.5, 'Clinical Decision Support Workflow\nWith Uncertainty-Guided Refinement', 
           fontsize=14, fontweight='bold', ha='center')
    
    # Flow diagram
    steps = [
        ('Patient Fundus Image', 10, 'lightblue'),
        ('LU-Net + RA\n(Coarse Prediction)', 8.5, 'lightcyan'),
        ('Ensemble Diffusion\n(k=5 samples)', 7, 'lightyellow'),
        ('Compute Mean &\nUncertainty', 5.5, 'lightgreen'),
        ('Analysis:\nHigh UQ Regions?', 4, 'wheat'),
    ]
    
    for text, y, color in steps:
        box = FancyBboxPatch((1, y-0.5), 8, 0.9,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(5, y, text, fontsize=11, ha='center', va='center', fontweight='bold')
        
        if y > 4.5:
            arrow = FancyArrowPatch((5, y-0.55), (5, y-1.15),
                                   arrowstyle='->', mutation_scale=25,
                                   color='black', linewidth=2)
            ax.add_patch(arrow)
    
    # Decision branches
    ax.text(5, 3.3, 'Decision', fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Left branch: Low uncertainty
    ax.text(2, 2.3, 'Low Uncertainty\n(≤ threshold)', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(2, 1.3, '✓ APPROVE\nAutomatic\nSegmentation', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.5, pad=0.3))
    arrow = FancyArrowPatch((4, 3.1), (2.5, 2.7),
                           arrowstyle='->', mutation_scale=20,
                           color='green', linewidth=2)
    ax.add_patch(arrow)
    arrow = FancyArrowPatch((2, 2.0), (2, 1.7),
                           arrowstyle='->', mutation_scale=20,
                           color='green', linewidth=2)
    ax.add_patch(arrow)
    
    # Right branch: High uncertainty
    ax.text(8, 2.3, 'High Uncertainty\n(> threshold)', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(8, 1.3, '⚠ REVIEW\nExpert\nAnnotation', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5, pad=0.3))
    arrow = FancyArrowPatch((6, 3.1), (7.5, 2.7),
                           arrowstyle='->', mutation_scale=20,
                           color='orange', linewidth=2)
    ax.add_patch(arrow)
    arrow = FancyArrowPatch((8, 2.0), (8, 1.7),
                           arrowstyle='->', mutation_scale=20,
                           color='orange', linewidth=2)
    ax.add_patch(arrow)
    
    # Bottom: Output
    ax.text(5, 0.3, 'Final Segmentation + Confidence Map for Clinical Report', 
           fontsize=11, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clinical_decision_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: clinical_decision_workflow.png")
    plt.close()


def main():
    print("="*70)
    print("Generating Uncertainty Quantification Visualizations")
    print("="*70)
    
    generate_ensemble_visualization()
    generate_uncertainty_comparison()
    generate_clinical_workflow()
    
    print("\n" + "="*70)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*70)
    print("\nFiles created:")
    print("  1. ensemble_sampling_workflow.png - Shows k=5 ensemble process")
    print("  2. ensemble_vs_single_comparison.png - Single vs ensemble")
    print("  3. clinical_decision_workflow.png - Clinical decision support")


if __name__ == "__main__":
    main()
