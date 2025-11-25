# -*- coding: utf-8 -*-
"""
Generate all figures for the report according to the specification.
All figures are in English for the English report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path
import networkx as nx
import pickle
from scipy.stats import spearmanr, gaussian_kde, beta
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Output directory
out_dir = Path("results/figures/report_figures")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Generating Report Figures")
print("=" * 80)
print()

# ============================================================================
# Figure 1: Network Overview
# ============================================================================
print("[1/12] Generating Figure 1: Network Overview...")

# Load graph
try:
    from networkx.readwrite.gpickle import read_gpickle
    G = read_gpickle("data/processed/V2/graph_V2_directed_weighted.gpickle")
except:
    with open("data/processed/V2/graph_V2_directed_weighted.gpickle", "rb") as f:
        G = pickle.load(f)

# Load node types
nodes_df = pd.read_csv("data/processed/nodes_wormwiring_ISM.csv")
node_type_map = dict(zip(nodes_df['node_id'], nodes_df['neuron_type']))

# Get positions
pos = {}
for n in G.nodes():
    if 'pos' in G.nodes[n] and G.nodes[n]['pos']:
        x, y = G.nodes[n]['pos']
        pos[n] = (float(x), float(y))
    elif 'x' in G.nodes[n] and 'y' in G.nodes[n]:
        pos[n] = (float(G.nodes[n]['x']), float(G.nodes[n]['y']))

if not pos:
    # Generate layout if not available
    pos = nx.spring_layout(G.to_undirected(), seed=42, k=0.5, iterations=50)

# Sample edges for visualization (show ~10% of edges)
all_edges = list(G.edges())
np.random.seed(42)
sample_edges = np.random.choice(len(all_edges), size=min(500, len(all_edges)), replace=False)
edges_to_show = [all_edges[i] for i in sample_edges]

# Color map
type_colors = {'S': '#FF6B6B', 'I': '#4ECDC4', 'M': '#95E1D3', 'P': '#F38181'}

fig, ax = plt.subplots(figsize=(8, 6))

# Draw edges
for u, v in edges_to_show:
    if u in pos and v in pos:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
               'gray', alpha=0.1, linewidth=0.3, zorder=1)

# Draw nodes by type
for node_type in ['S', 'I', 'M']:
    nodes_of_type = [n for n in G.nodes() if node_type_map.get(n) == node_type]
    x_coords = [pos[n][0] for n in nodes_of_type if n in pos]
    y_coords = [pos[n][1] for n in nodes_of_type if n in pos]
    if x_coords:
        ax.scatter(x_coords, y_coords, c=type_colors[node_type], 
                  s=20, alpha=0.7, label=f'{node_type} ({len(nodes_of_type)})', 
                  edgecolors='black', linewidth=0.3, zorder=2)

ax.set_title("Figure 1. C. elegans Connectome Network", fontweight='bold', fontsize=12)
ax.set_xlabel("X coordinate", fontweight='bold')
ax.set_ylabel("Y coordinate", fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.axis('off')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(out_dir / "figure1_network_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'figure1_network_overview.png'}")

# ============================================================================
# Pipeline Flowchart
# ============================================================================
print("[2/12] Generating Pipeline Flowchart...")

fig, ax = plt.subplots(figsize=(10, 3))

# Boxes
boxes = [
    ("Connectome\n(272 nodes, 3,995 edges)", 0.5, 0.5),
    ("E1\nBFS Cascades", 2.0, 0.5),
    ("E2\nα-sweep\nShortest Paths", 3.5, 0.5),
    ("E3\nGreedy\nNavigation", 5.0, 0.5),
    ("E4\nDiffusion", 6.5, 0.5),
    ("E5\nMax-flow/\nMin-cut", 8.0, 0.5),
]

# Draw boxes
for i, (label, x, y) in enumerate(boxes):
    if i == 0:
        color = '#E8F4F8'
        width, height = 1.2, 0.8
    else:
        color = '#FFF4E6'
        width, height = 1.0, 0.8
    
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", facecolor=color,
                         edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=9)

# Draw arrows
for i in range(len(boxes) - 1):
    x1 = boxes[i][1] + 0.6
    x2 = boxes[i+1][1] - 0.5
    y = boxes[i][2]
    arrow = FancyArrowPatch((x1, y), (x2, y),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.set_xlim(0, 9)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("Experimental Pipeline", fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(out_dir / "pipeline_flowchart.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'pipeline_flowchart.png'}")

# ============================================================================
# Figure E1: BFS Coverage Curve
# ============================================================================
print("[3/12] Generating Figure E1: BFS Coverage Curve...")

# Load BFS data
bfs_data = pd.read_csv("results/tables/e1_bfs_layers_V2.csv")

# Get ASIL data (example sensory neuron)
asil_data = bfs_data[bfs_data['seed'] == 'ASIL'].copy()
if len(asil_data) == 0:
    # Use first seed if ASIL not found
    first_seed = bfs_data['seed'].iloc[0]
    asil_data = bfs_data[bfs_data['seed'] == first_seed].copy()

# Get all seeds for shading
all_seeds = bfs_data['seed'].unique()
max_layer = int(bfs_data['layer'].max())
n_nodes = G.number_of_nodes()

# Calculate coverage for all seeds
all_coverage = {}
for seed in all_seeds:
    seed_data = bfs_data[bfs_data['seed'] == seed]
    layers = seed_data['layer'].values
    cum_covered = seed_data['cum_covered'].values
    coverage_frac = cum_covered / n_nodes
    all_coverage[seed] = (layers, coverage_frac)

fig, ax = plt.subplots(figsize=(6, 4.5))

# Shade area for all sensory neurons
if len(all_coverage) > 1:
    all_layers = np.arange(0, max_layer + 1)
    min_coverage = np.zeros(max_layer + 1)
    max_coverage = np.zeros(max_layer + 1)
    
    for layer in all_layers:
        coverages_at_layer = []
        for seed, (layers, coverage) in all_coverage.items():
            if layer in layers:
                idx = np.where(layers == layer)[0][0]
                coverages_at_layer.append(coverage[idx])
        if coverages_at_layer:
            min_coverage[layer] = min(coverages_at_layer)
            max_coverage[layer] = max(coverages_at_layer)
    
    ax.fill_between(all_layers, min_coverage, max_coverage, 
                   alpha=0.2, color='gray', label='All sensory neurons')

# Plot ASIL curve
if len(asil_data) > 0:
    layers = asil_data['layer'].values
    cum_covered = asil_data['cum_covered'].values
    coverage_frac = cum_covered / n_nodes
    ax.plot(layers, coverage_frac, 'o-', linewidth=2, markersize=6,
           color='#2E86AB', label='ASIL (example)', zorder=3)

ax.set_xlabel("BFS Layer", fontweight='bold')
ax.set_ylabel("Fraction of Nodes Reached", fontweight='bold')
ax.set_title("Figure E1. BFS Layer-wise Coverage", fontweight='bold', fontsize=11)
ax.set_xlim(-0.2, max_layer + 0.5)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "figure_e1_bfs_coverage.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'figure_e1_bfs_coverage.png'}")

# ============================================================================
# Figure E2: Alpha-sweep Cost
# ============================================================================
print("[4/12] Generating Figure E2: Alpha-sweep Cost...")

alpha_data = pd.read_csv("results/tables/e2_dijkstra_alpha_summary_V2.csv")

fig, ax = plt.subplots(figsize=(6, 4.5))

# Normalize cost to [0, 1] for display
costs = alpha_data['mean_cost'].values
normalized_costs = (costs - costs.min()) / (costs.max() - costs.min()) if costs.max() > costs.min() else costs

ax.plot(alpha_data['alpha'], normalized_costs, 'o-', linewidth=2, markersize=8,
       color='#A23B72', label='Normalized path cost')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='α=0 (topology only)')

ax.set_xlabel("α (geometric weight)", fontweight='bold')
ax.set_ylabel("Normalized Mean Path Cost", fontweight='bold')
ax.set_title("Figure E2. Cost vs α (Topology-Geometry Trade-off)", fontweight='bold', fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "figure_e2_alpha_sweep.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'figure_e2_alpha_sweep.png'}")

# ============================================================================
# Figure E3a: Navigation Schematic
# ============================================================================
print("[5/12] Generating Figure E3a: Navigation Schematic...")

fig, ax = plt.subplots(figsize=(5, 5))

# Draw a simple schematic
# Current node
current = (0.3, 0.5)
target = (0.8, 0.5)

# Neighbors
neighbors = [
    (0.2, 0.3, 'N1'),
    (0.2, 0.7, 'N2'),
    (0.4, 0.3, 'N3'),
    (0.4, 0.7, 'N4'),
]

# Draw edges to neighbors
for nx, ny, label in neighbors:
    ax.plot([current[0], nx], [current[1], ny], 'gray', linewidth=1.5, alpha=0.5, zorder=1)

# Draw neighbors
for nx, ny, label in neighbors:
    # Calculate distance to target
    dist = np.sqrt((nx - target[0])**2 + (ny - target[1])**2)
    circle = Circle((nx, ny), 0.05, color='lightblue', edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(circle)
    ax.text(nx, ny - 0.1, f'{label}\nd={dist:.2f}', ha='center', va='top', fontsize=8)

# Draw current node
circle = Circle(current, 0.06, color='orange', edgecolor='black', linewidth=2, zorder=3)
ax.add_patch(circle)
ax.text(current[0], current[1], 'Current', ha='center', va='center', fontweight='bold', fontsize=9, zorder=4)

# Draw target
circle = Circle(target, 0.06, color='red', edgecolor='black', linewidth=2, zorder=3)
ax.add_patch(circle)
ax.text(target[0], target[1], 'Target', ha='center', va='center', fontweight='bold', fontsize=9, color='white', zorder=4)

# Highlight closest neighbor (N3)
closest = (0.4, 0.3)
ax.plot([current[0], closest[0]], [current[1], closest[1]], 
       'green', linewidth=3, alpha=0.7, zorder=2, label='Pick closest')
ax.annotate('', xy=closest, xytext=current,
           arrowprops=dict(arrowstyle='->', color='green', lw=2.5, alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Figure E3a. Greedy Navigation Schematic", fontweight='bold', fontsize=11, pad=20)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(out_dir / "figure_e3a_navigation_schematic.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'figure_e3a_navigation_schematic.png'}")

# ============================================================================
# Figure E4: Matrix Thumbnails
# ============================================================================
print("[6/12] Generating Figure E4: Matrix Thumbnails...")

# Load diffusion and geodesic matrices
try:
    e4_data = np.load("results/tables/e4_diffusion_matrix.npz")
    diffusion_matrix = e4_data["diffusion"]
    geodesic_matrix = e4_data["geodesic"]
    
    # Sample for visualization (use every nth element)
    n = len(diffusion_matrix)
    sample_rate = max(1, n // 100)  # Show ~100x100
    
    diff_sample = diffusion_matrix[::sample_rate, ::sample_rate]
    geo_sample = geodesic_matrix[::sample_rate, ::sample_rate]
    
    # Normalize for display
    diff_norm = (diff_sample - diff_sample.min()) / (diff_sample.max() - diff_sample.min()) if diff_sample.max() > diff_sample.min() else diff_sample
    geo_norm = (geo_sample - geo_sample.min()) / (geo_sample.max() - geo_sample.min()) if geo_sample.max() > geo_sample.min() else geo_sample
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    im1 = ax1.imshow(geo_norm, cmap='viridis_r', aspect='auto', interpolation='nearest')
    ax1.set_title("Geodesic Distance Matrix", fontweight='bold', fontsize=10)
    ax1.set_xlabel("Node index", fontweight='bold')
    ax1.set_ylabel("Node index", fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Distance (normalized)')
    
    im2 = ax2.imshow(diff_norm, cmap='plasma', aspect='auto', interpolation='nearest')
    ax2.set_title("Diffusion Matrix (Heat Kernel)", fontweight='bold', fontsize=10)
    ax2.set_xlabel("Node index", fontweight='bold')
    ax2.set_ylabel("Node index", fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Diffusion (normalized)')
    
    plt.suptitle("Figure E4. Distance vs Diffusion Matrices", fontweight='bold', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_e4_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir / 'figure_e4_matrices.png'}")
except Exception as e:
    print(f"  ⚠️  Could not generate E4 matrices: {e}")

# ============================================================================
# Figure E5: Max-flow/Min-cut Schematic
# ============================================================================
print("[7/12] Generating Figure E5: Max-flow/Min-cut Schematic...")

fig, ax = plt.subplots(figsize=(8, 4))

# Draw super-source
source_box = FancyBboxPatch((0.1, 0.3), 0.15, 0.4,
                           boxstyle="round,pad=0.05", facecolor='#4ECDC4',
                           edgecolor='black', linewidth=2)
ax.add_patch(source_box)
ax.text(0.175, 0.5, 'Super-\nSource\n(Sensory)', ha='center', va='center', 
       fontweight='bold', fontsize=9)

# Draw network (middle)
network_box = FancyBboxPatch((0.4, 0.2), 0.4, 0.6,
                            boxstyle="round,pad=0.05", facecolor='#FFF4E6',
                            edgecolor='black', linewidth=2)
ax.add_patch(network_box)
ax.text(0.6, 0.5, 'Network\n(272 nodes)', ha='center', va='center',
       fontweight='bold', fontsize=10)

# Draw min-cut (red line)
ax.plot([0.8, 0.8], [0.2, 0.8], 'red', linewidth=4, alpha=0.7, label='Min-cut')
ax.text(0.82, 0.5, 'Min-cut\nnodes', ha='left', va='center', 
       fontweight='bold', fontsize=9, color='red')

# Draw super-sink
sink_box = FancyBboxPatch((0.95, 0.3), 0.15, 0.4,
                         boxstyle="round,pad=0.05", facecolor='#FF6B6B',
                         edgecolor='black', linewidth=2)
ax.add_patch(sink_box)
ax.text(1.025, 0.5, 'Super-\nSink\n(Motor)', ha='center', va='center',
       fontweight='bold', fontsize=9, color='white')

# Draw flow arrows
arrow1 = FancyArrowPatch((0.25, 0.5), (0.4, 0.5),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='blue', alpha=0.7)
ax.add_patch(arrow1)
ax.text(0.325, 0.55, 'Max-flow', ha='center', fontsize=8, color='blue', fontweight='bold')

arrow2 = FancyArrowPatch((0.8, 0.5), (0.95, 0.5),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='blue', alpha=0.7)
ax.add_patch(arrow2)

ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("Figure E5. Max-flow/Min-cut Schematic", fontweight='bold', fontsize=11, pad=20)
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(out_dir / "figure_e5_schematic.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'figure_e5_schematic.png'}")

# ============================================================================
# E1 Result: Coverage Curve (Result version)
# ============================================================================
print("[8/12] Generating E1 Result: Coverage Curve...")

fig, ax = plt.subplots(figsize=(6, 4.5))

# Same as E1 but with result annotations
if len(asil_data) > 0:
    layers = asil_data['layer'].values
    cum_covered = asil_data['cum_covered'].values
    coverage_frac = cum_covered / n_nodes
    ax.plot(layers, coverage_frac, 'o-', linewidth=2.5, markersize=7,
           color='#2E86AB', label='Coverage', zorder=3)
    
    # Highlight 6 hops
    if 5 in layers:
        idx = np.where(layers == 5)[0][0]
        ax.plot(5, coverage_frac[idx], 'ro', markersize=12, zorder=4)
        ax.annotate(f'{coverage_frac[idx]*100:.1f}%', 
                   xy=(5, coverage_frac[idx]), xytext=(5.5, coverage_frac[idx] + 0.1),
                   fontsize=10, fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.set_xlabel("BFS Layer (hops)", fontweight='bold')
ax.set_ylabel("Fraction of Nodes Reached", fontweight='bold')
ax.set_title("E1 Result: Fast Layered Coverage", fontweight='bold', fontsize=11)
ax.set_xlim(-0.2, max_layer + 0.5)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "e1_result_coverage.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'e1_result_coverage.png'}")

# ============================================================================
# E2 Result: Cost vs Alpha (Result version)
# ============================================================================
print("[9/12] Generating E2 Result: Cost vs Alpha...")

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(alpha_data['alpha'], normalized_costs, 'o-', linewidth=2.5, markersize=8,
       color='#A23B72', label='Normalized path cost', zorder=3)
ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
          label='α=0 (topology only, minimum cost)', zorder=2)

# Add reachability (secondary y-axis)
ax2 = ax.twinx()
reach_rates = alpha_data['reach_rate'].values
ax2.plot(alpha_data['alpha'], reach_rates, 's--', linewidth=1.5, markersize=6,
        color='green', alpha=0.6, label='Reachability', zorder=1)
ax2.set_ylabel("Reachability Rate", fontweight='bold', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0.85, 0.95)

ax.set_xlabel("α (geometric weight)", fontweight='bold')
ax.set_ylabel("Normalized Mean Path Cost", fontweight='bold', color='#A23B72')
ax.tick_params(axis='y', labelcolor='#A23B72')
ax.set_title("E2 Result: Topology Dominates α-sweep", fontweight='bold', fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "e2_result_alpha_sweep.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'e2_result_alpha_sweep.png'}")

# ============================================================================
# E3 Result: Success/Failure and Stretch
# ============================================================================
print("[10/12] Generating E3 Result: Success/Failure and Stretch...")

# Load E3 results
try:
    e3_results = pd.read_csv("results/tables/e3_nav_results_V2.csv")
except:
    # Try alternative path
    e3_results = pd.read_csv("results/tables/e3_nav_results.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Success/Failure pie
# Check column name
if 'success' in e3_results.columns:
    success_col = 'success'
    success_count = e3_results[success_col].sum()
elif 'Success' in e3_results.columns:
    success_col = 'Success'
    success_count = e3_results[success_col].sum()
else:
    # Infer from other columns
    if 'stretch' in e3_results.columns:
        success_mask = e3_results['stretch'].notna() & (e3_results['stretch'] != np.inf)
        success_count = success_mask.sum()
    else:
        success_count = 0

failure_count = len(e3_results) - success_count
success_rate = success_count / len(e3_results) if len(e3_results) > 0 else 0

colors_pie = ['#4ECDC4', '#FF6B6B']
explode = (0.05, 0)
ax1.pie([success_count, failure_count], 
       labels=[f'Success\n({success_rate*100:.1f}%)', f'Failure\n({(1-success_rate)*100:.1f}%)'],
       autopct='%1.1f%%', colors=colors_pie, explode=explode,
       startangle=90, textprops={'fontweight': 'bold', 'fontsize': 10})
ax1.set_title("E3 Result: Success vs Failure", fontweight='bold', fontsize=11)

# Right: Stretch distribution (successful paths only) - using KDE curve
if 'success' in e3_results.columns:
    successful = e3_results[e3_results['success'] == True]
elif 'Success' in e3_results.columns:
    successful = e3_results[e3_results['Success'] == True]
else:
    # Use stretch column to infer success
    successful = e3_results[e3_results['stretch'].notna() & (e3_results['stretch'] != np.inf)]
stretches = successful['stretch'].dropna()
stretches = stretches[stretches < 10]  # Filter outliers for better visualization

if len(stretches) > 1:
    # Create KDE (Kernel Density Estimation) curve
    try:
        kde = gaussian_kde(stretches)
        x_range = np.linspace(stretches.min(), stretches.max(), 200)
        density = kde(x_range)
        
        # Normalize to make it look like a probability density
        density = density / density.max()  # Normalize to [0, 1] for better visualization
        
        ax2.plot(x_range, density, linewidth=2.5, color='#2E86AB', label='Density curve')
        ax2.fill_between(x_range, density, alpha=0.3, color='#2E86AB')
    except:
        # Fallback to histogram if KDE fails
        ax2.hist(stretches, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black', 
                linewidth=0.5, density=True)
    
    ax2.axvline(stretches.median(), color='red', linestyle='--', linewidth=2,
               label=f'Median = {stretches.median():.2f}')
    ax2.axvline(1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Optimal (stretch=1.0)')
    ax2.set_xlabel("Stretch Ratio", fontweight='bold')
    ax2.set_ylabel("Density", fontweight='bold')
    ax2.set_title("E3 Result: Stretch Distribution (Successful Paths)", fontweight='bold', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "e3_result_navigation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'e3_result_navigation.png'}")

# ============================================================================
# E4 Result: Diffusion vs Geodesic Scatter
# ============================================================================
print("[11/12] Generating E4 Result: Diffusion vs Geodesic...")

try:
    # Sample to get approximately 1000 valid points with uniform distribution across distance ranges
    n = len(diffusion_matrix)
    target_points = 1000
    points_per_range = target_points // 5  # ~200 points per color range
    
    # Collect all valid points first, grouped by distance range
    # Blue: 2-3, Purple: 3-5, Pink: 5-7, Red: 7-20, Green: 20-355
    distance_ranges = [2, 3, 5, 7, 20, 355]
    all_valid_by_range = {i: {'diff': [], 'geo': []} for i in range(5)}
    
    for i in range(n):
        for j in range(i + 1, n):
            diff_val = diffusion_matrix[i, j]
            geo_val = geodesic_matrix[i, j]
            
            if np.isfinite(geo_val) and geo_val > 0 and np.isfinite(diff_val):
                # Assign to appropriate range
                if geo_val < 3:
                    range_idx = 0  # Blue - 2-3
                elif geo_val < 5:
                    range_idx = 1  # Purple - 3-5
                elif geo_val < 7:
                    range_idx = 2  # Pink - 5-7
                elif geo_val < 20:
                    range_idx = 3  # Red - 7-20
                else:
                    range_idx = 4  # Green - 20-355
                
                all_valid_by_range[range_idx]['diff'].append(diff_val)
                all_valid_by_range[range_idx]['geo'].append(geo_val)
    
    # Sample uniformly from each range
    diffusion_vals_list = []
    geodesic_vals_list = []
    for range_idx in range(5):
        range_diff = all_valid_by_range[range_idx]['diff']
        range_geo = all_valid_by_range[range_idx]['geo']
        
        if len(range_diff) > points_per_range:
            # Sample points_per_range points from this range
            indices = np.random.choice(len(range_diff), points_per_range, replace=False)
            diffusion_vals_list.extend([range_diff[i] for i in indices])
            geodesic_vals_list.extend([range_geo[i] for i in indices])
        else:
            # Use all points from this range
            diffusion_vals_list.extend(range_diff)
            geodesic_vals_list.extend(range_geo)
    
    diffusion_vals = np.array(diffusion_vals_list)
    geodesic_vals = np.array(geodesic_vals_list)
    inverse_geodesic_vals = 1.0 / geodesic_vals
    
    # Calculate correlation
    rho, p_val = spearmanr(diffusion_vals, inverse_geodesic_vals)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Use E2 color scheme for different Geodesic Distance ranges
    # E2 colors: Blue, Purple, Pink, Red, Green
    e2_colors = {
        0: '#71C2FF',    # Blue (113, 194, 255)
        1: '#B894DB',    # Purple (184, 148, 219)
        2: '#FF94DB',    # Pink (255, 148, 219)
        3: '#FF5732',    # Red (255, 87, 50)
        4: '#92EE54'     # Green (146, 238, 84)
    }
    
    e2_edge_colors = {
        0: '#4A9FCC',    # Darker blue
        1: '#8B6FA8',    # Darker purple
        2: '#CC6FAF',    # Darker pink
        3: '#CC3F1F',    # Darker red
        4: '#6FBB2F'     # Darker green
    }
    
    # Calculate geodesic_vals from inverse_geodesic_vals for grouping
    geodesic_vals = 1.0 / inverse_geodesic_vals
    
    # Divide geodesic_vals into 5 fixed distance ranges
    # Blue: 2-3, Purple: 3-5, Pink: 5-7, Red: 7-20, Green: 20-355
    distance_ranges = [2, 3, 5, 7, 20, 355]
    
    # Assign colors based on fixed distance ranges
    point_colors = []
    point_edge_colors = []
    for geo_val in geodesic_vals:
        if geo_val < 3:
            color_idx = 0  # Blue - 2-3
        elif geo_val < 5:
            color_idx = 1  # Purple - 3-5
        elif geo_val < 7:
            color_idx = 2  # Pink - 5-7
        elif geo_val < 20:
            color_idx = 3  # Red - 7-20
        else:
            color_idx = 4  # Green - 20-355
        point_colors.append(e2_colors[color_idx])
        point_edge_colors.append(e2_edge_colors[color_idx])
    
    point_colors = np.array(point_colors)
    point_edge_colors = np.array(point_edge_colors)
    
    # Adjust alpha and size - make points larger (fewer points now, can use larger size)
    n_points = len(diffusion_vals)
    if n_points > 2000:
        alpha_val = 0.5
        size_val = 30  # Larger points
    elif n_points > 1000:
        alpha_val = 0.6
        size_val = 35
    else:
        alpha_val = 0.7
        size_val = 40
    
    # Add very slight jitter - just enough to avoid exact overlaps
    np.random.seed(42)  # For reproducibility
    # Very tiny jitter (0.01% of range) - minimal offset to avoid exact overlaps
    x_jitter = (inverse_geodesic_vals.max() - inverse_geodesic_vals.min()) * 0.0001  # 0.01% of inverse geodesic range
    y_jitter = (diffusion_vals.max() - diffusion_vals.min()) * 0.0001  # 0.01% of diffusion range
    
    # Plot each color group separately for better legend control
    for color_idx in range(5):
        mask = point_colors == e2_colors[color_idx]
        if np.any(mask):
            # Format label based on range (last one is >20)
            if color_idx == 4:  # Last group (green, 20-355)
                label = 'Distance >20'
            else:
                label = f'Distance {distance_ranges[color_idx]}-{distance_ranges[color_idx+1]}'
            
            # Swap axes: X is diffusion, Y is inverse geodesic (then invert Y)
            x_vals = diffusion_vals[mask] + np.random.normal(0, y_jitter, np.sum(mask))
            y_vals = inverse_geodesic_vals[mask] + np.random.normal(0, x_jitter, np.sum(mask))
            
            ax.scatter(x_vals, y_vals, 
                      alpha=alpha_val, s=size_val, 
                      color=e2_colors[color_idx], 
                      edgecolors=e2_edge_colors[color_idx], 
                      linewidth=0.3,
                      label=label)
    
    # Remove correlation text box (user requested to remove bottom-right data)
    
    # Swap axes: X is diffusion, Y is inverse geodesic distance (then invert both axes)
    ax.set_xlabel("Diffusion Strength (Heat Kernel)", fontweight='bold', fontsize=16)
    ax.set_ylabel("Inverse Geodesic Distance (1/d)", fontweight='bold', fontsize=16)
    ax.set_title("E4 Result: Diffusion vs Geodesic Distance", fontweight='bold', fontsize=17)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Invert both axes
    ax.invert_yaxis()  # Flip Y axis vertically - larger inverse distance (smaller distance) at top
    ax.invert_xaxis()  # Flip X axis horizontally
    
    # Add legend for the 5 color groups (lower left, same font size as E2)
    ax.legend(loc='lower left', fontsize=20, framealpha=0.95, ncol=1)
    
    plt.tight_layout()
    plt.savefig(out_dir / "e4_result_diffusion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir / 'e4_result_diffusion.png'}")
except Exception as e:
    print(f"  ⚠️  Could not generate E4 result: {e}")

# ============================================================================
# E5 Result: Overlap and Min-cut Size Distributions
# ============================================================================
print("[12/12] Generating E5 Result: Overlap and Min-cut Size...")

e5_results = pd.read_csv("results/tables/e5_flow_results_V2.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Overlap ratio distribution - Histogram + KDE with boundary handling
overlap_ratios = e5_results['overlap_ratio'].dropna()
if len(overlap_ratios) > 1:
    # Method: Histogram + KDE overlay (more scientific for bounded [0,1] data)
    # Use logit transform to handle boundary issues
    eps = 1e-6
    overlap_clipped = np.clip(overlap_ratios, eps, 1 - eps)  # Avoid exact 0/1
    logit_ratios = np.log(overlap_clipped / (1 - overlap_clipped))
    
    # Histogram (main visualization)
    n_bins = min(30, len(overlap_ratios) // 3)
    counts, bins, patches = ax1.hist(overlap_ratios, bins=n_bins, color='#A23B72', 
                                     alpha=0.5, edgecolor='black', linewidth=0.5, 
                                     density=True, label='Histogram')
    
    # KDE on logit-transformed data (smoother, handles boundaries better)
    try:
        kde_logit = gaussian_kde(logit_ratios)
        # Generate x values in original space [0, 1]
        x_range = np.linspace(eps, 1 - eps, 200)
        logit_x = np.log(x_range / (1 - x_range))
        density_logit = kde_logit(logit_x)
        # Transform back: need to account for Jacobian of logit transform
        # d/dx logit(x) = 1/(x*(1-x)), so density in original space = density_logit / (x*(1-x))
        density_original = density_logit / (x_range * (1 - x_range))
        # Normalize to match histogram scale
        density_original = density_original / density_original.max() * counts.max()
        
        ax1.plot(x_range, density_original, linewidth=2.5, color='#A23B72', 
                label='Smoothed density (KDE, logit)', linestyle='--')
    except:
        # Fallback: simple KDE on original data
        try:
            kde = gaussian_kde(overlap_ratios)
            x_range = np.linspace(overlap_ratios.min(), overlap_ratios.max(), 200)
            density = kde(x_range)
            density = density / density.max() * counts.max()
            ax1.plot(x_range, density, linewidth=2.5, color='#A23B72', 
                    label='Smoothed density (KDE)', linestyle='--')
        except:
            pass
    
    # Add rug ticks to show actual data points
    ax1.plot(overlap_ratios, np.zeros_like(overlap_ratios) - 0.01 * counts.max(), 
            '|', color='black', markersize=8, alpha=0.3, label='Data points')
    
    ax1.axvline(overlap_ratios.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {overlap_ratios.mean():.4f}')
    ax1.set_xlabel("Overlap Ratio", fontweight='bold')
    ax1.set_ylabel("Density", fontweight='bold')
    ax1.set_title("E5 Result: Min-cut vs Betweenness Overlap", fontweight='bold', fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

# Right: Min-cut size distribution - ECDF/CCDF (better for discrete, heavy-tailed data)
cut_sizes = e5_results['min_cut_size'].dropna()
if len(cut_sizes) > 1:
    # Method: ECDF (Empirical Cumulative Distribution Function) - more appropriate for discrete/heavy-tailed
    sorted_sizes = np.sort(cut_sizes)
    n = len(sorted_sizes)
    ecdf = np.arange(1, n + 1) / n
    ccdf = 1 - ecdf  # Complementary CDF (shows tail better)
    
    # Plot ECDF (left y-axis)
    ax2_twin = ax2.twinx()
    ax2.plot(sorted_sizes, ecdf, linewidth=2.5, color='#2E86AB', 
            label='ECDF: P(X ≤ x)', alpha=0.7)
    ax2_twin.plot(sorted_sizes, ccdf, linewidth=2.5, color='#FF6B6B', 
                 label='CCDF: P(X > x)', linestyle='--', alpha=0.7)
    
    # Also show histogram for reference (frequency polygon style)
    n_bins = min(30, len(cut_sizes) // 3)
    counts, bins = np.histogram(cut_sizes, bins=n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Normalize counts to match ECDF scale
    counts_norm = counts / counts.sum() * ecdf.max()
    ax2.plot(bin_centers, counts_norm, 'o-', linewidth=1.5, color='gray', 
            markersize=4, alpha=0.5, label='Frequency polygon', zorder=1)
    
    ax2.axvline(cut_sizes.mean(), color='red', linestyle=':', linewidth=2,
               label=f'Mean = {cut_sizes.mean():.1f}', zorder=3)
    ax2.axvline(np.median(cut_sizes), color='orange', linestyle=':', linewidth=2,
               label=f'Median = {np.median(cut_sizes):.1f}', zorder=3)
    
    ax2.set_xlabel("Min-cut Size", fontweight='bold')
    ax2.set_ylabel("ECDF: P(X ≤ x)", fontweight='bold', color='#2E86AB')
    ax2_twin.set_ylabel("CCDF: P(X > x)", fontweight='bold', color='#FF6B6B')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    ax2_twin.tick_params(axis='y', labelcolor='#FF6B6B')
    ax2.set_title("E5 Result: Min-cut Size Distribution (ECDF/CCDF)", fontweight='bold', fontsize=11)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "e5_result_bottlenecks.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_dir / 'e5_result_bottlenecks.png'}")

print()
print("=" * 80)
print("All figures generated successfully!")
print(f"Output directory: {out_dir}")
print("=" * 80)

