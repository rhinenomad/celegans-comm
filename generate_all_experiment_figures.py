#!/usr/bin/env python3
"""
Generate all experiment figures with consistent 5-color scatter plot style
Uses the same color scheme: blue, purple, pink, red, green
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import networkx as nx
from scipy.stats import spearmanr
from scipy.sparse import load_npz

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Output directory
out_dir = Path("results/figures/report_figures")
out_dir.mkdir(parents=True, exist_ok=True)

# RGB colors - consistent across all experiments
blue_rgb = (113/255, 194/255, 255/255)   # #71C2FF
purple_rgb = (184/255, 148/255, 219/255)  # #B894DB
pink_rgb = (255/255, 148/255, 219/255)   # #FF94DB
red_rgb = (255/255, 87/255, 50/255)      # #FF5732
green_rgb = (146/255, 238/255, 84/255)   # #92EE54

color_palette = [blue_rgb, purple_rgb, pink_rgb, red_rgb, green_rgb]

print("=" * 80)
print("Generating All Experiment Figures with Consistent 5-Color Style")
print("=" * 80)

# ============================================================================
# E1: BFS Coverage - Layer vs Cumulative Coverage
# ============================================================================
print("\n[1/5] Generating E1: BFS Coverage Scatter Plot...")

try:
    bfs_data = pd.read_csv("results/tables/e1_bfs_layers_V2.csv")
    total_nodes = 272
    bfs_data["fraction"] = bfs_data["cum_covered"] / total_nodes
    
    # Group by layer and calculate statistics
    layer_stats = bfs_data.groupby("layer")["fraction"].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample individual seed data for scatter (one point per seed per layer)
    np.random.seed(42)
    sample_size = min(200, len(bfs_data))
    sampled = bfs_data.sample(n=sample_size, random_state=42) if len(bfs_data) > sample_size else bfs_data
    
    # Assign colors based on layer ranges and create legend entries
    layer_ranges = {
        'Layer ≤ 1': (1, blue_rgb),
        'Layer = 2': (2, purple_rgb),
        'Layer = 3': (3, pink_rgb),
        'Layer = 4': (4, red_rgb),
        'Layer ≥ 5': (float('inf'), green_rgb)
    }
    
    layers = sampled['layer'].values
    fractions = sampled['fraction'].values
    
    point_colors = []
    for layer in layers:
        if layer <= 1:
            color = blue_rgb
        elif layer <= 2:
            color = purple_rgb
        elif layer <= 3:
            color = pink_rgb
        elif layer <= 4:
            color = red_rgb
        else:
            color = green_rgb
        point_colors.append(color)
    
    # Add slight jitter for visibility
    x_jitter = np.random.normal(0, 0.05, len(layers))
    y_jitter = np.random.normal(0, 0.01, len(fractions))
    
    # Plot each layer range separately for legend
    for label, (threshold, color) in layer_ranges.items():
        if threshold == 1:
            mask = layers <= 1
        elif threshold == 2:
            mask = layers == 2
        elif threshold == 3:
            mask = layers == 3
        elif threshold == 4:
            mask = layers == 4
        else:
            mask = layers >= 5
        
        if np.any(mask):
            ax.scatter(layers[mask] + x_jitter[mask], fractions[mask] + y_jitter[mask],
                      c=[color], alpha=0.5, s=30, edgecolors='none', label=label)
    
    # Plot mean line
    ax.plot(layer_stats['layer'], layer_stats['mean'],
           'o-', linewidth=3, markersize=10,
           color='#2E86AB', label='Mean coverage', zorder=5)
    
    # Add shaded region for std (only near the mean line, not full range)
    # Use a narrow band around the mean line
    std_band = layer_stats['std'] * 0.3  # Make it narrower (30% of std)
    ax.fill_between(layer_stats['layer'],
                   layer_stats['mean'] - std_band,
                   layer_stats['mean'] + std_band,
                   alpha=0.3, color='#2E86AB', zorder=1, label='±0.3 std')
    
    ax.set_xlabel("BFS Layer (hops)", fontweight='bold', fontsize=16)
    ax.set_ylabel("Cumulative Coverage (fraction)", fontweight='bold', fontsize=16)
    ax.set_title("E1: BFS Coverage by Layer", fontweight='bold', fontsize=17)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95, ncol=1)
    
    plt.tight_layout()
    plt.savefig(out_dir / "e1_result_coverage.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir / 'e1_result_coverage.png'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# E2: Alpha-Sweep (already exists, but update to use V4 data)
# ============================================================================
print("\n[2/5] Generating E2: Alpha-sweep Scatter Plot...")

try:
    e2_paths = pd.read_csv("results/tables/e2_dijkstra_paths_V4.csv")
    e2_summary = pd.read_csv("results/tables/e2_dijkstra_alpha_summary_V4.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data points
    np.random.seed(42)
    sample_size = min(5000, len(e2_paths))
    sampled = e2_paths.sample(n=sample_size, random_state=42) if len(e2_paths) > sample_size else e2_paths
    
    alphas = sampled['alpha'].values
    costs = sampled['cost'].values
    finite_mask = np.isfinite(costs)
    alphas = alphas[finite_mask]
    costs = costs[finite_mask]
    
    # Assign colors based on alpha ranges and create legend entries
    alpha_ranges = {
        'α = 0.0 (pure topological)': (0.0, blue_rgb),
        'α = 0.25': (0.25, purple_rgb),
        'α = 0.5': (0.5, pink_rgb),
        'α = 0.75': (0.75, red_rgb),
        'α = 1.0 (pure geometric)': (1.0, green_rgb)
    }
    
    point_colors = []
    for alpha in alphas:
        if alpha == 0.0:
            color = blue_rgb
        elif alpha <= 0.25:
            color = purple_rgb
        elif alpha <= 0.5:
            color = pink_rgb
        elif alpha <= 0.75:
            color = red_rgb
        else:
            color = green_rgb
        point_colors.append(color)
    
    point_colors = np.array(point_colors)
    
    # Add very slight jitter to avoid exact overlaps
    np.random.seed(42)
    alpha_jitter = np.random.normal(0, 0.003, len(alphas))  # Very small jitter on x-axis (0.3% of range)
    cost_range = costs.max() - costs.min()
    cost_jitter = np.random.normal(0, cost_range * 0.003, len(costs))  # Very small jitter on y-axis (0.3% of range)
    
    # Plot each alpha group separately for legend
    for label, (alpha_val, color) in alpha_ranges.items():
        if alpha_val == 0.0:
            mask = alphas == 0.0
        elif alpha_val == 0.25:
            mask = (alphas > 0.0) & (alphas <= 0.25)
        elif alpha_val == 0.5:
            mask = (alphas > 0.25) & (alphas <= 0.5)
        elif alpha_val == 0.75:
            mask = (alphas > 0.5) & (alphas <= 0.75)
        else:
            mask = alphas > 0.75
        
        if np.any(mask):
            ax.scatter(alphas[mask] + alpha_jitter[mask], costs[mask] + cost_jitter[mask],
                      c=[color], alpha=0.6, s=20, edgecolors='none', label=label)
    
    # Overlay summary line
    ax.plot(e2_summary['alpha'], e2_summary['mean_cost'],
           'o-', linewidth=3, markersize=10,
           color='#A23B72', label='Mean cost', zorder=5)
    
    ax.set_xlabel("α (geometric weight, dimensionless)", fontweight='bold', fontsize=16)
    ax.set_ylabel("Path Cost (weighted distance)", fontweight='bold', fontsize=16)
    ax.set_title("E2: Alpha-sweep Cost Distribution", fontweight='bold', fontsize=17)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, ncol=1)
    
    plt.tight_layout()
    plt.savefig(out_dir / "e2_result_alpha_sweep.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir / 'e2_result_alpha_sweep.png'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# E3: Greedy Navigation - Path Length vs Shortest Path Length
# ============================================================================
print("\n[3/5] Generating E3: Navigation Path Length Comparison...")

try:
    e3_results = pd.read_csv("results/tables/e3_nav_results.csv")
    
    # Calculate stretch for successful navigations
    e3_results = e3_results.copy()
    e3_results['stretch'] = e3_results['path_len'] / e3_results['sp_len']
    successful = e3_results[e3_results['success'] == True].copy()
    
    if len(successful) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        path_lens = successful['path_len'].values
        sp_lens = successful['sp_len'].values
        stretches = successful['stretch'].values
        
        # Sample for visualization if too many points
        np.random.seed(42)
        sample_size = min(2000, len(successful))
        if len(successful) > sample_size:
            indices = np.random.choice(len(successful), sample_size, replace=False)
            path_lens = path_lens[indices]
            sp_lens = sp_lens[indices]
            stretches = stretches[indices]
        
        # Assign colors based on stretch ranges
        stretch_ranges = {
            'Stretch ≤ 2x': (2, blue_rgb),
            '2x < Stretch ≤ 4x': (4, purple_rgb),
            '4x < Stretch ≤ 6x': (6, pink_rgb),
            '6x < Stretch ≤ 8x': (8, red_rgb),
            'Stretch > 8x': (float('inf'), green_rgb)
        }
        
        point_colors = []
        for stretch in stretches:
            if stretch <= 2:
                color = blue_rgb
            elif stretch <= 4:
                color = purple_rgb
            elif stretch <= 6:
                color = pink_rgb
            elif stretch <= 8:
                color = red_rgb
            else:
                color = green_rgb
            point_colors.append(color)
        
        # Add slight jitter
        x_jitter = np.random.normal(0, 0.2, len(path_lens))
        y_jitter = np.random.normal(0, 0.2, len(sp_lens))
        
        # Plot each stretch range separately for legend
        for label, (threshold, color) in stretch_ranges.items():
            if threshold == 2:
                mask = stretches <= 2
            elif threshold == 4:
                mask = (stretches > 2) & (stretches <= 4)
            elif threshold == 6:
                mask = (stretches > 4) & (stretches <= 6)
            elif threshold == 8:
                mask = (stretches > 6) & (stretches <= 8)
            else:
                mask = stretches > 8
            
            if np.any(mask):
                # Swap x and y: now x = greedy path, y = shortest path
                ax.scatter(path_lens[mask] + x_jitter[mask], sp_lens[mask] + y_jitter[mask],
                          c=[color], alpha=0.6, s=30, edgecolors='none', label=label)
        
        # Add diagonal line (y = x, optimal case) - swapped
        max_val = max(sp_lens.max(), path_lens.max())
        ax.plot([0, max_val], [0, max_val],
               '--', linewidth=2, color='gray', alpha=0.5,
               label='Optimal (stretch = 1.0)', zorder=4)
        
        # Add median stretch lines - swapped: y = x / median_stretch
        median_stretch = np.median(stretches)
        ax.plot([0, max_val], [0, max_val / median_stretch],
               '--', linewidth=2, color='#A23B72',
               label=f'Median stretch: {median_stretch:.2f}x', zorder=5)
        
        ax.set_xlabel("Greedy Path Length (hops)", fontweight='bold', fontsize=16)
        ax.set_ylabel("Shortest Path Length (hops)", fontweight='bold', fontsize=16)
        ax.set_title("E3: Greedy Navigation Path Length vs Optimal", fontweight='bold', fontsize=17)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 10])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95, ncol=1)
        
        plt.tight_layout()
        plt.savefig(out_dir / "e3_result_navigation.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_dir / 'e3_result_navigation.png'}")
    else:
        print("  ✗ No successful navigations found")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# E4: Diffusion vs Geodesic (already exists, just verify)
# ============================================================================
print("\n[4/5] E4: Diffusion vs Geodesic (already generated by generate_e2_e4_scatter.py)")
print("  → Using existing: e4_result_diffusion.png")

# ============================================================================
# E5: Max-Flow Min-Cut - Flow Value vs Overlap Ratio
# ============================================================================
print("\n[5/5] Generating E5: Flow Analysis Scatter Plot...")

try:
    e5_results = pd.read_csv("results/tables/e5_flow_results_V2.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    flow_values = e5_results['flow_value'].values
    overlap_ratios = e5_results['overlap_ratio'].values
    min_cut_sizes = e5_results['min_cut_size'].values
    
    # Assign colors based on min_cut_size ranges
    point_colors = []
    for cut_size in min_cut_sizes:
        if cut_size <= 50:
            color = blue_rgb
        elif cut_size <= 75:
            color = purple_rgb
        elif cut_size <= 100:
            color = pink_rgb
        elif cut_size <= 125:
            color = red_rgb
        else:
            color = green_rgb
        point_colors.append(color)
    
    # Add slight jitter
    np.random.seed(42)
    x_jitter = np.random.normal(0, 0.5, len(flow_values))
    y_jitter = np.random.normal(0, 0.005, len(overlap_ratios))
    
    ax.scatter(flow_values + x_jitter, overlap_ratios + y_jitter,
              c=point_colors, alpha=0.6, s=50, edgecolors='none')
    
    # Add mean lines
    mean_flow = np.mean(flow_values)
    mean_overlap = np.mean(overlap_ratios)
    ax.axvline(mean_flow, color='#A23B72', linestyle='--', linewidth=2,
              label=f'Mean flow: {mean_flow:.2f}', zorder=5)
    ax.axhline(mean_overlap, color='#A23B72', linestyle='--', linewidth=2,
              label=f'Mean overlap: {mean_overlap:.3f}', zorder=5)
    
    ax.set_xlabel("Max Flow Value", fontweight='bold', fontsize=14)
    ax.set_ylabel("Overlap Ratio (min-cut ∩ betweenness)", fontweight='bold', fontsize=14)
    ax.set_title("E5: Max-Flow vs Min-Cut/Betweenness Overlap", fontweight='bold', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_dir / "e5_result_flow_overlap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir / 'e5_result_flow_overlap.png'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 80)
print("All experiment figures generated!")
print("=" * 80)
print(f"\nOutput directory: {out_dir}")
print("\nGenerated figures:")
print("  - e1_result_coverage.png")
print("  - e2_result_alpha_sweep.png")
print("  - e3_result_stretch.png")
print("  - e4_result_diffusion.png (existing)")
print("  - e5_result_flow_overlap.png")

