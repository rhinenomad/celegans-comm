#!/usr/bin/env python3
"""
Generate E1 BFS coverage as scatter plot (similar to E2/E3 style)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Output directory
out_dir = Path("results/figures/report_figures")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Generating E1: BFS Coverage Scatter Plot")
print("=" * 80)

# Load E1 data (try V2 first, fallback to V1)
try:
    bfs_data = pd.read_csv("results/tables/e1_bfs_layers_V2.csv")
except:
    bfs_data = pd.read_csv("results/tables/e1_bfs_layers_V1.csv")

print(f"Loaded {len(bfs_data)} rows from {len(bfs_data['seed'].unique())} seeds")

# Calculate fraction (272 total nodes)
total_nodes = 272
bfs_data["fraction"] = bfs_data["cum_covered"] / total_nodes

# Get unique seeds
seeds = bfs_data['seed'].unique()
n_seeds = len(seeds)

# Create scatter plot
fig, ax = plt.subplots(figsize=(12, 7))

# Add jitter for better visibility
np.random.seed(42)

# Color palette (similar to E2/E3 style)
if n_seeds == 1:
    # Single seed: use different colors for different layers
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D', '#6A4C93']
    seed_data = bfs_data[bfs_data['seed'] == seeds[0]]
    layers = seed_data['layer'].values
    fractions = seed_data['fraction'].values
    
    # Scatter plot with different colors for each layer
    for i, (layer, fraction) in enumerate(zip(layers, fractions)):
        ax.scatter(layer, fraction,
                  alpha=0.7, s=100,
                  color=colors[i % len(colors)],
                  edgecolors='black', linewidth=0.5,
                  label=f'Layer {int(layer)}' if i < 6 else '',
                  zorder=3)
    
    # Add connecting line
    ax.plot(layers, fractions, 
            '--', linewidth=2, alpha=0.5, color='gray', zorder=1)
    
    # Label the seed
    ax.text(0.02, 0.98, f'Seed: {seeds[0]}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # For single seed, use the seed data for xlim
    max_layer = max(layers)
    bfs_avg = seed_data.copy()
else:
    # Multiple seeds: plot each seed as a curve (not scatter)
    # Use a colormap for different seeds
    colors = plt.cm.viridis(np.linspace(0, 1, n_seeds))
    
    # Plot individual seed curves (lighter, semi-transparent)
    for i, seed in enumerate(seeds):
        seed_data = bfs_data[bfs_data['seed'] == seed]
        layers = seed_data['layer'].values
        fractions = seed_data['fraction'].values
        
        # Plot as curve (line) instead of scatter
        ax.plot(layers, fractions,
                '-', linewidth=1.5, alpha=0.5,
                color=colors[i],
                label=f'{seed}' if n_seeds <= 10 else '',  # Only label if <= 10 seeds
                zorder=2)
    
    # Calculate and plot average coverage curve (thicker, more prominent)
    bfs_avg = bfs_data.groupby("layer")["fraction"].agg(['mean', 'std']).reset_index()
    ax.plot(bfs_avg["layer"], bfs_avg["mean"], 
            'o-', linewidth=3, markersize=10,
            color='#2E86AB', markerfacecolor='#C73E1D', 
            markeredgecolor='white', markeredgewidth=2,
            label='Average', zorder=4)
    
    # Add confidence interval (if std available)
    if 'std' in bfs_avg.columns:
        ax.fill_between(bfs_avg["layer"], 
                        bfs_avg["mean"] - bfs_avg["std"],
                        bfs_avg["mean"] + bfs_avg["std"],
                        alpha=0.2, color='#2E86AB', zorder=1)

# Set labels and title
ax.set_xlabel("BFS Layer", fontweight='bold', fontsize=12)
ax.set_ylabel("Fraction of Nodes Reached", fontweight='bold', fontsize=12)
ax.set_title("E1: BFS Coverage by Layer", fontweight='bold', fontsize=13)
if n_seeds == 1:
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=1)
else:
    # Only show average in legend if many seeds (to avoid clutter)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3, axis='both')
ax.set_ylim([0, 1.05])
max_layer = max(bfs_avg["layer"]) if 'layer' in bfs_avg.columns else max(bfs_data['layer'])
ax.set_xlim([-0.3, max_layer + 0.5])

# Add statistics text
if n_seeds == 1:
    max_coverage = max(fractions)
    max_layer_val = max(layers)
    stats_text = f"Max coverage: {max_coverage:.1%} at layer {int(max_layer_val)}"
else:
    max_layer_val = bfs_avg["layer"].max()
    max_coverage = bfs_avg["mean"].max()
    stats_text = f"Max coverage: {max_coverage:.1%} at layer {int(max_layer_val)}"

if n_seeds > 1:  # Only add stats text if multiple seeds (single seed already has label)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(out_dir / "e1_bfs_coverage_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {out_dir / 'e1_bfs_coverage_scatter.png'}")

print()
print("=" * 80)
print("E1 scatter plot generated successfully!")
print("=" * 80)

