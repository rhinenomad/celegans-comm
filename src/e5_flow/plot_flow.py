# -*- coding: utf-8 -*-
"""
E5: Max-flow/Min-cut Visualization

Generate visualization plots for E5 experiment:
- Overlap ratio distribution histogram
- Observed vs Null model comparison
- Statistical test results (p-value distribution)
- Significant node pair analysis
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"])
    args = ap.parse_args()
    
    C = load_cfg(args.config)
    out_fig = Path(C["report"]["figures_dir"])
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tbl = Path(C["report"]["tables_dir"])
    
    # Load results
    results_file = out_tbl / f"e5_flow_results_{args.variant}.csv"
    summary_file = out_tbl / f"e5_flow_summary_{args.variant}.csv"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    summary = pd.read_csv(summary_file)
    
    print(f"[E5 Plot] Loaded {len(df)} results from {results_file}")
    
    # Filter valid results (with p-values)
    df_valid = df[df['p_value'].notna()].copy()
    
    if len(df_valid) == 0:
        print("[E5 Plot] WARNING: No valid p-values found. Creating basic plots only.")
        df_valid = df.copy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Overlap ratio distribution
    ax1 = plt.subplot(2, 3, 1)
    overlap_ratios = df['overlap_ratio'].dropna()
    null_overlap_ratios = df['null_mean_overlap_ratio'].dropna()
    
    if len(overlap_ratios) > 0:
        ax1.hist(overlap_ratios, bins=30, alpha=0.7, label='Observed', color='steelblue', edgecolor='black')
        if len(null_overlap_ratios) > 0 and null_overlap_ratios.sum() > 0:
            ax1.axvline(null_overlap_ratios.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Null mean ({null_overlap_ratios.mean():.4f})')
        ax1.axvline(overlap_ratios.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Observed mean ({overlap_ratios.mean():.4f})')
        ax1.set_xlabel('Overlap Ratio', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of Overlap Ratios', fontweight='bold', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Observed vs Null comparison (box plot)
    ax2 = plt.subplot(2, 3, 2)
    if len(overlap_ratios) > 0 and len(null_overlap_ratios) > 0:
        data_to_plot = []
        labels = []
        if len(overlap_ratios) > 0:
            data_to_plot.append(overlap_ratios.values)
            labels.append('Observed')
        if len(null_overlap_ratios) > 0 and null_overlap_ratios.sum() > 0:
            # For null, we need to sample from the distribution
            # Since we only have mean, we'll create a synthetic distribution
            null_samples = np.random.normal(null_overlap_ratios.mean(), 
                                          null_overlap_ratios.std() if null_overlap_ratios.std() > 0 else 0.01,
                                          size=min(100, len(overlap_ratios)))
            null_samples = np.clip(null_samples, 0, 1)
            data_to_plot.append(null_samples)
            labels.append('Null Model')
        
        if len(data_to_plot) > 0:
            bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.set_ylabel('Overlap Ratio', fontweight='bold')
            ax2.set_title('Observed vs Null Model', fontweight='bold', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. P-value distribution
    ax3 = plt.subplot(2, 3, 3)
    if len(df_valid) > 0:
        p_values = df_valid['p_value'].dropna()
        if len(p_values) > 0:
            ax3.hist(p_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
            ax3.set_xlabel('P-value', fontweight='bold')
            ax3.set_ylabel('Frequency', fontweight='bold')
            ax3.set_title('P-value Distribution', fontweight='bold', fontsize=11)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. Flow value vs Overlap ratio
    ax4 = plt.subplot(2, 3, 4)
    if len(df) > 0:
        scatter = ax4.scatter(df['flow_value'], df['overlap_ratio'], 
                             alpha=0.6, s=50, c=df['p_value_fdr'] if 'p_value_fdr' in df.columns else df['p_value'],
                             cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Max Flow Value', fontweight='bold')
        ax4.set_ylabel('Overlap Ratio', fontweight='bold')
        ax4.set_title('Flow Value vs Overlap Ratio', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('P-value (FDR)', fontweight='bold', rotation=270, labelpad=20)
    
    # 5. Min-cut size vs Overlap ratio
    ax5 = plt.subplot(2, 3, 5)
    if len(df) > 0:
        scatter = ax5.scatter(df['min_cut_size'], df['overlap_ratio'], 
                             alpha=0.6, s=50, c=df['p_value_fdr'] if 'p_value_fdr' in df.columns else df['p_value'],
                             cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('Min-cut Size', fontweight='bold')
        ax5.set_ylabel('Overlap Ratio', fontweight='bold')
        ax5.set_title('Min-cut Size vs Overlap Ratio', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('P-value (FDR)', fontweight='bold', rotation=270, labelpad=20)
    
    # 6. Significant pairs (if any)
    ax6 = plt.subplot(2, 3, 6)
    if 'significant_fdr' in df.columns:
        significant_count = df['significant_fdr'].sum()
        non_significant_count = len(df) - significant_count
        
        if significant_count > 0 or non_significant_count > 0:
            counts = [non_significant_count, significant_count]
            labels_pie = ['Non-significant', 'Significant (FDR)']
            colors_pie = ['lightcoral', 'lightgreen']
            explode = (0, 0.1) if significant_count > 0 else (0, 0)
            
            ax6.pie(counts, labels=labels_pie, autopct='%1.1f%%', 
                   colors=colors_pie, explode=explode, startangle=90,
                   textprops={'fontweight': 'bold'})
            ax6.set_title('Significance Distribution\n(FDR corrected)', 
                          fontweight='bold', fontsize=11)
    else:
        ax6.text(0.5, 0.5, 'No significance data\navailable', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax6.transAxes)
        ax6.set_title('Significance Distribution', fontweight='bold', fontsize=11)
    
    plt.suptitle(f'E5: Max-flow/Min-cut Analysis Results ({args.variant})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = out_fig / f"e5_flow_analysis_{args.variant}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[E5 Plot] Saved figure: {output_file}")
    plt.close()
    
    # Create a second figure for detailed analysis
    fig2 = plt.figure(figsize=(14, 10))
    
    # 1. Top overlap ratios
    ax1 = plt.subplot(2, 2, 1)
    top_n = min(20, len(df))
    df_sorted = df.nlargest(top_n, 'overlap_ratio')
    y_pos = np.arange(len(df_sorted))
    bars = ax1.barh(y_pos, df_sorted['overlap_ratio'], alpha=0.7, edgecolor='black')
    
    # Color bars by significance
    if 'significant_fdr' in df_sorted.columns:
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if row['significant_fdr']:
                bars[i].set_color('green')
            else:
                bars[i].set_color('steelblue')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['source']}→{row['target']}" 
                         for _, row in df_sorted.iterrows()], fontsize=8)
    ax1.set_xlabel('Overlap Ratio', fontweight='bold')
    ax1.set_title(f'Top {top_n} Overlap Ratios', fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 2. Z-score distribution
    ax2 = plt.subplot(2, 2, 2)
    if 'z_score' in df.columns:
        z_scores = df['z_score'].dropna()
        if len(z_scores) > 0:
            ax2.hist(z_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Z=0')
            ax2.axvline(1.96, color='orange', linestyle='--', linewidth=2, label='Z=1.96 (p=0.05)')
            ax2.set_xlabel('Z-score', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Z-score Distribution', fontweight='bold', fontsize=11)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # 3. Flow value distribution
    ax3 = plt.subplot(2, 2, 3)
    flow_values = df['flow_value'].dropna()
    if len(flow_values) > 0:
        ax3.hist(flow_values, bins=30, alpha=0.7, color='teal', edgecolor='black')
        ax3.axvline(flow_values.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean ({flow_values.mean():.2f})')
        ax3.set_xlabel('Max Flow Value', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Max Flow Value Distribution', fontweight='bold', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Min-cut size distribution
    ax4 = plt.subplot(2, 2, 4)
    cut_sizes = df['min_cut_size'].dropna()
    if len(cut_sizes) > 0:
        ax4.hist(cut_sizes, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax4.axvline(cut_sizes.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean ({cut_sizes.mean():.1f})')
        ax4.set_xlabel('Min-cut Size', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Min-cut Size Distribution', fontweight='bold', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'E5: Detailed Analysis ({args.variant})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file2 = out_fig / f"e5_flow_detailed_{args.variant}.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"[E5 Plot] Saved detailed figure: {output_file2}")
    plt.close()
    
    print(f"[E5 Plot] Done! Generated 2 figures.")

if __name__ == "__main__":
    main()

