"""
Generate a comparison table and figures (baseline vs. diffusion).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_comparison_table():
    """Create comparison dataframe."""
    data = {
        'Aspect': [
            'Dice (CHASE)',
            'IoU (CHASE)',
            'Sensitivity',
            'Specificity',
            'Inference Time (ms)',
            'Parameters (M)',
            'GFLOPs',
            'Thin Vessel Focus',
            'Uncertainty Est.',
            'Clinical Ready'
        ],
        'Baseline LU-Net': [
            '0.795',
            '0.691',
            '0.822',
            '0.984',
            '4.8',
            '1.94',
            '12.2',
            'RA Module',
            '❌ No',
            '✅ Yes'
        ],
        'Diffusion Refiner': [
            'TBD*',
            'TBD*',
            'TBD*',
            'TBD*',
            '50-100',
            '+1.2',
            'TBD',
            'Iter. Refinement',
            '✅ Sampling Var.',
            '⚠ Slower'
        ],
        'Combined': [
            '0.82-0.85?',
            '0.72-0.75?',
            '0.85-0.88?',
            '~0.98',
            '55-105',
            '3.1',
            'TBD',
            '✅✅✅ Both',
            '✅ Full',
            '⚠ With tuning'
        ]
    }
    df = pd.DataFrame(data)
    return df


def plot_comparison():
    """Plot inference speed vs. accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Speed vs. Dice
    methods = ['LU-Net', 'Diffusion\n(20 steps)', 'Diffusion\n(50 steps)', 'Combined\n(Estimated)']
    speed_ms = [4.8, 50, 100, 55]
    dice = [0.795, 0.80, 0.825, 0.835]  # Rough estimates
    
    ax1.scatter(speed_ms, dice, s=300, alpha=0.7, c=['blue', 'orange', 'red', 'green'])
    for i, method in enumerate(methods):
        ax1.annotate(method, (speed_ms[i], dice[i]), xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Inference Time (ms/image)', fontsize=12)
    ax1.set_ylabel('Dice Coefficient', fontsize=12)
    ax1.set_title('Speed vs. Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0.78, 0.85)
    
    # Metrics comparison
    metrics = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    lunet = [0.795, 0.691, 0.822, 0.984]
    diffusion_est = [0.825, 0.73, 0.87, 0.98]  # Estimates
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, lunet, width, label='LU-Net+RA', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, diffusion_est, width, label='Combined (Est.)', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics (CHASE-DB1)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0.65, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_baseline_vs_diffusion.png', dpi=150, bbox_inches='tight')
    print("Saved plot: comparison_baseline_vs_diffusion.png")
    plt.close()


def create_latex_table():
    """Generate LaTeX table for paper."""
    df = create_comparison_table()
    latex = df.to_latex(index=False)
    
    # Save to file with UTF-8 encoding
    with open('comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex)
    print("Saved LaTeX table: comparison_table.tex")
    
    # Print to console
    print("\n" + "="*80)
    print("COMPARISON TABLE (for paper)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    print("Generating comparison tables and figures...")
    create_comparison_table()
    plot_comparison()
    create_latex_table()
    print("\nDone!")
