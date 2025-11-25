#!/usr/bin/env python3
"""
Generate E1, E3, and E4 experiment figures with consistent 5-color scatter plot style
Uses the same color scheme: blue, purple, pink, red, green
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.legend_handler import HandlerPatch
from pathlib import Path
import pickle
import networkx as nx
from scipy.stats import spearmanr
from scipy.sparse import load_npz
from scipy.interpolate import interp1d

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
print("Generating E1, E3, and E4 Experiment Figures")
print("=" * 80)

# ============================================================================
# E1: BFS Coverage - Layer vs Cumulative Coverage
# ============================================================================
print("\n[1/3] Generating E1: BFS Coverage Scatter Plot...")

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
    print(f"  [OK] Saved: {out_dir / 'e1_result_coverage.png'}")
except Exception as e:
    print(f"  [ERROR] Error: {e}")

# ============================================================================
# E3: Greedy Navigation - Path Length vs Shortest Path Length
# ============================================================================
print("\n[2/3] Generating E3: Navigation Path Length Comparison...")

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
        print(f"  [OK] Saved: {out_dir / 'e3_result_navigation.png'}")
    else:
        print("  [ERROR] No successful navigations found")
except Exception as e:
    print(f"  [ERROR] Error: {e}")

# ============================================================================
# E4: Diffusion vs Geodesic Distance Scatter Plot
# ============================================================================
print("\n[3/3] Generating E4: Diffusion vs Geodesic Distance Scatter Plot...")

# Load diffusion and geodesic matrices
try:
    data = np.load("results/tables/e4_diffusion_matrix.npz", allow_pickle=True)
    diffusion_matrix = data['diffusion']
    geodesic_matrix = data['geodesic']
    nodes = data['nodes']
    print(f"  Loaded matrices: {diffusion_matrix.shape}")
except Exception as e:
    print(f"  [WARNING] Could not load matrices: {e}")
    print("  Running E4 to generate matrices...")
    import subprocess
    subprocess.run(["python3", "src/e4_diffusion/run_diff.py", "--config", "configs/params.yaml", "--variant", "V2"])
    data = np.load("results/tables/e4_diffusion_matrix.npz", allow_pickle=True)
    diffusion_matrix = data['diffusion']
    geodesic_matrix = data['geodesic']
    nodes = data['nodes']

# Sample points for visualization
n = len(nodes)
target_points = 1000
points_per_range = target_points // 5

# Collect valid points by distance range
# New ranges: 1, 2, 3, 4-5, >6
all_valid_by_range = {i: {'diff': [], 'geo': []} for i in range(5)}

for i in range(n):
    for j in range(i + 1, n):
        diff_val = diffusion_matrix[i, j]
        geo_val = geodesic_matrix[i, j]
        
        if np.isfinite(geo_val) and geo_val > 0 and np.isfinite(diff_val):
            if geo_val == 1:
                range_idx = 0  # Distance = 1
            elif geo_val == 2:
                range_idx = 1  # Distance = 2
            elif geo_val == 3:
                range_idx = 2  # Distance = 3
            elif 4 <= geo_val <= 5:
                range_idx = 3  # Distance = 4-5
            elif 6 <= geo_val <= 10:
                range_idx = 4  # Distance 6-10 (green, omit >10)
            else:
                # Distance > 10, skip
                continue
            
            all_valid_by_range[range_idx]['diff'].append(diff_val)
            all_valid_by_range[range_idx]['geo'].append(geo_val)

# Sample uniformly from each range
diffusion_vals_list = []
geodesic_vals_list = []
for range_idx in range(5):
    range_diff = all_valid_by_range[range_idx]['diff']
    range_geo = all_valid_by_range[range_idx]['geo']
    
    if len(range_diff) > points_per_range:
        indices = np.random.choice(len(range_diff), points_per_range, replace=False)
        diffusion_vals_list.extend([range_diff[i] for i in indices])
        geodesic_vals_list.extend([range_geo[i] for i in indices])
    else:
        diffusion_vals_list.extend(range_diff)
        geodesic_vals_list.extend(range_geo)

diffusion_vals = np.array(diffusion_vals_list)
geodesic_vals = np.array(geodesic_vals_list)

# Calculate correlation (diffusion vs direct geodesic distance)
# Expected: negative correlation (higher diffusion ↔ shorter distance)
# This is correct: as distance increases, diffusion should decrease
rho, p_val = spearmanr(diffusion_vals, geodesic_vals)

# Create scatter plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xscale('log')
ax.set_yscale('log')

# Color scheme
e4_colors = {
    0: blue_rgb,      # Blue
    1: purple_rgb,    # Purple
    2: pink_rgb,      # Pink
    3: red_rgb,       # Red
    4: green_rgb      # Green
}

# Assign colors based on geodesic distance ranges
# Ranges: 1, 2, 3, 4-5, >6
point_colors = []
for geo_val in geodesic_vals:
    if geo_val == 1:
        color_idx = 0  # Distance = 1
    elif geo_val == 2:
        color_idx = 1  # Distance = 2
    elif geo_val == 3:
        color_idx = 2  # Distance = 3
    elif 4 <= geo_val <= 5:
        color_idx = 3  # Distance = 4-5
    elif 6 <= geo_val <= 10:
        color_idx = 4  # Distance 6-10 (green, omit >10)
    else:
        # Distance > 10, should not happen after filtering
        color_idx = 0  # Default
    point_colors.append(e4_colors[color_idx])

point_colors = np.array(point_colors)

# Plot each color group separately
# Range labels: 1, 2, 3, 4-5, 6-10
range_labels = ['d=1', 'd=2', 'd=3', 'd=4-5', 'd=6-10']

# Store statistics for each range
range_stats = {}

# Collect Q25 and Q75 points for all ranges (to connect them later)
q25_points = []  # List of (diffusion, geodesic) tuples
q75_points = []  # List of (diffusion, geodesic) tuples

for color_idx in range(5):
    # Create mask by comparing colors (need to compare arrays properly)
    target_color = e4_colors[color_idx]
    mask = np.array([np.allclose(pc, target_color, atol=0.01) if isinstance(pc, (list, tuple, np.ndarray)) else False 
                     for pc in point_colors])
    
    if np.any(mask):
        # Get data for this range
        range_diffusion = diffusion_vals[mask]
        range_geodesic = geodesic_vals[mask]
        
        # Calculate robust statistics for this range
        mean_diff = np.mean(range_diffusion)
        median_diff = np.median(range_diffusion)
        q25_diff = np.percentile(range_diffusion, 25)
        q75_diff = np.percentile(range_diffusion, 75)
        iqr_diff = q75_diff - q25_diff
        mean_geo = np.mean(range_geodesic)
        median_geo = np.median(range_geodesic)
        
        # Adjust median_geo for d=4-5 and d=6-10 to be between the two rows of points
        if color_idx == 3:  # d=4-5 (red)
            # Place between 4 and 5
            median_geo = 4.5
        elif color_idx == 4:  # d=6-10 (green)
            # Place between 6 and 10, maybe around 7-8
            median_geo = 7.5
        
        # Store statistics
        range_stats[color_idx] = {
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'q25_diff': q25_diff,
            'q75_diff': q75_diff,
            'iqr_diff': iqr_diff,
            'mean_geo': mean_geo,
            'median_geo': median_geo,
            'color': e4_colors[color_idx],
            'label': range_labels[color_idx]
        }
        
        # Collect Q25 and Q75 points (using median_geo as Y coordinate)
        q25_points.append((q25_diff, median_geo))
        q75_points.append((q75_diff, median_geo))
        
        # Add very slight jitter to avoid exact overlaps
        np.random.seed(42)  # For reproducibility
        n_points = np.sum(mask)
        
        # Calculate jitter size (very small percentage, only on X-axis)
        # Y-axis (distance) is discrete, so minimal or no jitter
        x_range = diffusion_vals.max() - diffusion_vals.min()
        
        # Reduce jitter for all points to make them closer together
        if color_idx == 0:  # d=1 - blue, minimal jitter
            x_jitter = x_range * 0.0002  # 0.02% of X range (very small for blue)
            y_jitter = 0.01  # Very small Y jitter for blue
        else:
            x_jitter = x_range * 0.0003  # 0.03% of X range (reduced)
            y_jitter = 0.02  # Smaller Y jitter (reduced)
        
        # Apply jitter (only X-axis, minimal Y)
        x_vals = diffusion_vals[mask] + np.random.normal(0, x_jitter, n_points)
        y_vals = geodesic_vals[mask] + np.random.normal(0, y_jitter, n_points)
        
        # Ensure Y values don't cross distance boundaries
        # For discrete distances, keep them within their range (tighter bounds)
        if color_idx == 0:  # d=1 - tighter bounds for blue
            y_vals = np.clip(y_vals, 0.95, 1.05)  # Very tight range for blue
        elif color_idx == 1:  # d=2
            y_vals = np.clip(y_vals, 1.9, 2.1)  # Tighter
        elif color_idx == 2:  # d=3
            y_vals = np.clip(y_vals, 2.9, 3.1)  # Tighter
        elif color_idx == 3:  # d=4-5
            y_vals = np.clip(y_vals, 3.9, 5.1)  # Tighter
        elif color_idx == 4:  # d=6-10
            y_vals = np.clip(y_vals, 5.9, 10.1)  # Keep within 6-10 range
        
        # Add label for legend
        ax.scatter(x_vals, y_vals,
                  alpha=0.6, s=30,
                  color=e4_colors[color_idx],
                  edgecolors='none',
                  label=range_labels[color_idx])
        
        # Plot Q25, median, and Q75 points
        # Q25 point
        ax.scatter(q25_diff, median_geo, color=e4_colors[color_idx], 
                  s=80, marker='o', edgecolors='white', linewidths=2,
                  zorder=6, alpha=0.95)
        
        # Median point (primary statistic)
        ax.scatter(median_diff, median_geo, color=e4_colors[color_idx], 
                  s=120, marker='o', edgecolors='white', linewidths=2.5,
                  zorder=7, alpha=0.95, label=range_labels[color_idx])  # Simplified label
        
        # Q75 point
        ax.scatter(q75_diff, median_geo, color=e4_colors[color_idx], 
                  s=80, marker='o', edgecolors='white', linewidths=2,
                  zorder=6, alpha=0.95)
        
        # Add text annotation with median and IQR near the median point
        # Position text far to the right of Q75 (right edge of diamond)
        # In log scale, use a large multiplicative offset to place it far to the right
        # Adjust X position: use median-based positioning for consistency
        # Since median values are similar, use median * multiplier for consistent positioning
        if color_idx == 3:  # d=4-5 (red)
            text_x_offset = median_diff * 4.5  # Based on median for red
        elif color_idx == 4:  # d=6-10 (green)
            text_x_offset = median_diff * 4.5  # Based on median for green (same multiplier)
        else:
            text_x_offset = q75_diff * 2.5  # Far to the right of Q75 in log scale
        # Adjust Y position: move all annotations much higher
        if color_idx == 3:  # d=4-5 (red)
            text_y_offset = median_geo * 1.25   # Lower for red
        elif color_idx == 4:  # d=6-10 (green)
            text_y_offset = median_geo * 1.30   # Much higher for green
        else:
            text_y_offset = median_geo * 1.28   # Much higher for other groups
        # More detailed annotation text
        annotation_text = f'{range_labels[color_idx]}\nmedian={median_diff:.5f}\nIQR=[{q25_diff:.5f}, {q75_diff:.5f}]'
        ax.text(text_x_offset, text_y_offset, annotation_text,
               fontsize=12, color='black',  # Text color changed to black
               verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=e4_colors[color_idx], linewidth=1),
               zorder=7)

# Plot: X = h_uv (diffusion, tau=0.5), Y = d(uv) (geodesic distance)
# Expected: negative correlation (higher diffusion ↔ shorter distance)
# Both axes use log scale

ax.set_xlabel("Heat-kernel communicability H_uv (τ = 0.5) log scale", fontweight='bold', fontsize=20)
ax.set_ylabel("Geodesic distance d(u,v) log scale", fontweight='bold', fontsize=20)
ax.set_title("E4: Diffusion vs Geodesic Distance", fontweight='bold', fontsize=21)
ax.set_ylim(bottom=None, top=12)  # Set Y-axis maximum to 12 (omit >12)

# Invert both axes
ax.invert_xaxis()  # Flip X axis horizontally (left-right)
ax.invert_yaxis()  # Flip Y axis vertically (up-down)

ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Connect median points with lines (trend line using robust statistics)
if len(range_stats) > 1:
    sorted_stats = sorted(range_stats.items(), key=lambda x: x[1]['median_geo'])
    median_diffs = [s[1]['median_diff'] for s in sorted_stats]
    median_geos = [s[1]['median_geo'] for s in sorted_stats]
    
    # Draw connecting line through median points (primary trend)
    ax.plot(median_diffs, median_geos, color='black', linewidth=2.5, alpha=0.5, 
           zorder=2, linestyle='-', label='Median trend')
    
    # Connect all Q25 points and all Q75 points, fill between them
    if len(q25_points) > 1 and len(q75_points) > 1:
        # Sort Q25 and Q75 points by geodesic distance (Y coordinate)
        q25_sorted = sorted(q25_points, key=lambda x: x[1])
        q75_sorted = sorted(q75_points, key=lambda x: x[1])
        
        # Extract coordinates
        q25_x = [p[0] for p in q25_sorted]
        q25_y = [p[1] for p in q25_sorted]
        q75_x = [p[0] for p in q75_sorted]
        q75_y = [p[1] for p in q75_sorted]
        
        # Connect Q25 points with a line (no label to avoid duplicate legend entries)
        ax.plot(q25_x, q25_y, color='gray', linewidth=2, alpha=0.7, 
               zorder=3, linestyle='--')
        
        # Connect Q75 points with a line (no label to avoid duplicate legend entries)
        ax.plot(q75_x, q75_y, color='gray', linewidth=2, alpha=0.7, 
               zorder=3, linestyle='--')
        
        # Fill the area between Q25 and Q75 lines with light gray
        # Use fill_between with interpolated points
        # Create a common Y grid for interpolation
        all_y = sorted(set(q25_y + q75_y))
        if len(all_y) > 1:
            # Interpolate Q25 and Q75 values at common Y points
            try:
                q25_interp = interp1d(q25_y, q25_x, kind='linear', 
                                     fill_value='extrapolate', bounds_error=False)
                q75_interp = interp1d(q75_y, q75_x, kind='linear', 
                                     fill_value='extrapolate', bounds_error=False)
                
                q25_interp_x = [q25_interp(y) for y in all_y]
                q75_interp_x = [q75_interp(y) for y in all_y]
                
                # Fill between the two lines
                ax.fill_betweenx(all_y, q25_interp_x, q75_interp_x,
                               color='lightgray', alpha=0.3, zorder=1)
            except:
                # Fallback: simple fill if interpolation fails
                for i in range(len(q25_sorted)):
                    if i < len(q75_sorted):
                        ax.fill_betweenx([q25_y[i], q75_y[i]], 
                                        [q25_x[i], q75_x[i]],
                                        color='lightgray', alpha=0.3, zorder=1)

# Create legend with distance ranges
legend_elements = []
for color_idx in range(5):
    if color_idx in range_stats:
        stats = range_stats[color_idx]
        # Add scatter point for legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=stats['color'], 
                                          markersize=8, alpha=0.6,
                                          label=stats['label']))

# Add legend entry for median point (blue with white border and gray outer ring)
# Create a custom legend entry with two circles: gray outer, white middle, blue center
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Create three circles: light gray outer, white middle, dark gray center
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        # Outer light gray circle (even larger)
        light_gray = (0.8, 0.8, 0.8)  # Light gray color
        dark_gray = (0.4, 0.4, 0.4)  # Dark gray color
        p1 = Circle(center, radius=height/0.9, facecolor=light_gray, edgecolor='none', transform=trans)
        # Middle white circle (much larger)
        p2 = Circle(center, radius=height/1.5, facecolor='white', edgecolor='none', transform=trans)
        # Inner dark gray circle (much larger)
        p3 = Circle(center, radius=height/2.0, facecolor=dark_gray, edgecolor='none', alpha=0.6, transform=trans)
        return [p1, p2, p3]

# Create a dummy patch for the legend with label
median_patch = Circle((0, 0), 1, facecolor='gray', edgecolor='white', linewidth=2)
median_patch.set_label('Median (Q25 and Q75)')  # Add label with 25% and 75% percentiles
legend_elements.append(median_patch)

# Add legend entry for IQR diamond
legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='gray', 
                                     alpha=0.25, edgecolor='gray', 
                                     linestyle='--', linewidth=1, label='IQR range'))

# Create legend with custom handler for median
handler_map = {median_patch: HandlerCircle()}
ax.legend(handles=legend_elements, handler_map=handler_map, loc='lower left', fontsize=14, framealpha=0.95, ncol=1)

plt.tight_layout()
plt.savefig(out_dir / "e4_result_diffusion.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {out_dir / 'e4_result_diffusion.png'}")
print(f"  Correlation: ρ = {rho:.4f}, p = {p_val:.2e}")

print("\n" + "=" * 80)
print("All experiment figures generated!")
print("=" * 80)
print(f"\nOutput directory: {out_dir}")
print("\nGenerated figures:")
print("  - e1_result_coverage.png")
print("  - e3_result_navigation.png")
print("  - e4_result_diffusion.png")
