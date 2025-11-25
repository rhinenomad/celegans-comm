# -*- coding: utf-8 -*-
"""
E5: Results Analysis

分析E5实验结果，找出显著的重合率节点对和关键发现
"""
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"])
    args = ap.parse_args()
    
    C = load_cfg(args.config)
    out_tbl = Path(C["report"]["tables_dir"])
    
    # Load results
    results_file = out_tbl / f"e5_flow_results_{args.variant}.csv"
    summary_file = out_tbl / f"e5_flow_summary_{args.variant}.csv"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    summary = pd.read_csv(summary_file)
    
    print("=" * 80)
    print(f"E5 Results Analysis ({args.variant})")
    print("=" * 80)
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    print(f"Total pairs tested: {len(df)}")
    print(f"Mean overlap ratio: {df['overlap_ratio'].mean():.4f}")
    print(f"Std overlap ratio: {df['overlap_ratio'].std():.4f}")
    print(f"Median overlap ratio: {df['overlap_ratio'].median():.4f}")
    print(f"Min overlap ratio: {df['overlap_ratio'].min():.4f}")
    print(f"Max overlap ratio: {df['overlap_ratio'].max():.4f}")
    print()
    
    print(f"Mean flow value: {df['flow_value'].mean():.2f}")
    print(f"Mean min-cut size: {df['min_cut_size'].mean():.1f}")
    print()
    
    # Significance analysis
    if 'significant_fdr' in df.columns:
        n_significant = df['significant_fdr'].sum()
        print(f"Significant pairs (FDR corrected): {n_significant} ({n_significant/len(df)*100:.1f}%)")
    else:
        print("No significance data available")
    print()
    
    # Top overlap ratios
    print("Top 10 Overlap Ratios:")
    print("-" * 80)
    top_overlap = df.nlargest(10, 'overlap_ratio')[['source', 'target', 'overlap_ratio', 
                                                     'flow_value', 'min_cut_size', 
                                                     'p_value_fdr' if 'p_value_fdr' in df.columns else 'p_value']]
    for idx, row in top_overlap.iterrows():
        p_val = row.get('p_value_fdr', row.get('p_value', 'N/A'))
        p_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) and not np.isnan(p_val) else "N/A"
        print(f"  {row['source']:8s} → {row['target']:8s}: "
              f"overlap={row['overlap_ratio']:.4f}, "
              f"flow={row['flow_value']:.1f}, "
              f"cut_size={row['min_cut_size']:.0f}, "
              f"p={p_str}")
    print()
    
    # Zero overlap pairs
    zero_overlap = df[df['overlap_ratio'] == 0]
    print(f"Pairs with zero overlap: {len(zero_overlap)} ({len(zero_overlap)/len(df)*100:.1f}%)")
    if len(zero_overlap) > 0:
        print("  Sample zero-overlap pairs:")
        for idx, row in zero_overlap.head(5).iterrows():
            print(f"    {row['source']} → {row['target']}: flow={row['flow_value']:.1f}, cut_size={row['min_cut_size']:.0f}")
    print()
    
    # High overlap pairs (top 10%)
    threshold = df['overlap_ratio'].quantile(0.9)
    high_overlap = df[df['overlap_ratio'] >= threshold]
    print(f"High overlap pairs (top 10%, threshold={threshold:.4f}): {len(high_overlap)}")
    if len(high_overlap) > 0:
        print("  Characteristics:")
        print(f"    Mean flow value: {high_overlap['flow_value'].mean():.2f}")
        print(f"    Mean min-cut size: {high_overlap['min_cut_size'].mean():.1f}")
        print(f"    Mean overlap ratio: {high_overlap['overlap_ratio'].mean():.4f}")
    print()
    
    # Flow value analysis
    print("Flow Value Analysis:")
    print("-" * 80)
    print(f"  Min: {df['flow_value'].min():.1f}")
    print(f"  Max: {df['flow_value'].max():.1f}")
    print(f"  Mean: {df['flow_value'].mean():.2f}")
    print(f"  Median: {df['flow_value'].median():.2f}")
    print()
    
    # Min-cut size analysis
    print("Min-cut Size Analysis:")
    print("-" * 80)
    print(f"  Min: {df['min_cut_size'].min():.0f}")
    print(f"  Max: {df['min_cut_size'].max():.0f}")
    print(f"  Mean: {df['min_cut_size'].mean():.1f}")
    print(f"  Median: {df['min_cut_size'].median():.1f}")
    print()
    
    # Correlation analysis
    print("Correlations:")
    print("-" * 80)
    if 'p_value' in df.columns:
        valid_p = df[df['p_value'].notna()]
        if len(valid_p) > 0:
            corr_flow_overlap = df['flow_value'].corr(df['overlap_ratio'])
            corr_cut_overlap = df['min_cut_size'].corr(df['overlap_ratio'])
            corr_flow_cut = df['flow_value'].corr(df['min_cut_size'])
            
            print(f"  Flow value vs Overlap ratio: {corr_flow_overlap:.4f}")
            print(f"  Min-cut size vs Overlap ratio: {corr_cut_overlap:.4f}")
            print(f"  Flow value vs Min-cut size: {corr_flow_cut:.4f}")
    print()
    
    # Null model comparison
    if 'null_mean_overlap_ratio' in df.columns:
        null_mean = df['null_mean_overlap_ratio'].mean()
        observed_mean = df['overlap_ratio'].mean()
        print("Null Model Comparison:")
        print("-" * 80)
        print(f"  Observed mean overlap ratio: {observed_mean:.4f}")
        print(f"  Null model mean overlap ratio: {null_mean:.4f}")
        if null_mean > 0:
            fold_change = observed_mean / null_mean if null_mean > 0 else np.inf
            print(f"  Fold change: {fold_change:.2f}x")
        else:
            print("  Null model has zero overlap (cannot compute fold change)")
    print()
    
    # Key findings
    print("Key Findings:")
    print("-" * 80)
    if 'significant_fdr' in df.columns and df['significant_fdr'].sum() > 0:
        print(f"  ✓ Found {df['significant_fdr'].sum()} significant pairs (FDR corrected)")
        sig_pairs = df[df['significant_fdr']]
        print(f"    Mean overlap ratio of significant pairs: {sig_pairs['overlap_ratio'].mean():.4f}")
    else:
        print("  ✗ No significant pairs found after FDR correction")
        print("    This may indicate:")
        print("    - Min-cut nodes do not significantly overlap with betweenness centrality top-k")
        print("    - The network structure does not show the expected bottleneck pattern")
        print("    - Need to adjust k value or null model parameters")
    
    if df['overlap_ratio'].max() > 0:
        max_pair = df.loc[df['overlap_ratio'].idxmax()]
        print(f"  • Highest overlap ratio: {max_pair['overlap_ratio']:.4f} "
              f"({max_pair['source']} → {max_pair['target']})")
    
    if len(zero_overlap) > len(df) * 0.5:
        print(f"  • Warning: {len(zero_overlap)/len(df)*100:.1f}% of pairs have zero overlap")
        print("    This suggests min-cut nodes are generally not in betweenness centrality top-k")
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

