#!/usr/bin/env python3
"""
Generate E2 and E4 scatter plots with specified RGB colors
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

print("=" * 80)
print("Generating E2 and E4 Scatter Plots")
print("=" * 80)

# RGB colors (pink, red, purple) - normalized to [0, 1]
pink_rgb = (255/255, 148/255, 219/255)   # #FF94DB
red_rgb = (255/255, 87/255, 50/255)      # #FF5732
purple_rgb = (184/255, 148/255, 219/255)  # #B894DB

# Also include blue and green for E4
blue_rgb = (113/255, 194/255, 255/255)   # #71C2FF
green_rgb = (146/255, 238/255, 84/255)   # #92EE54

# ============================================================================
# E2: Alpha-sweep Scatter Plot
# ============================================================================
print("\n[1/2] Generating E2: Alpha-sweep Scatter Plot...")

# Load E2 data
e2_paths = pd.read_csv("results/tables/e2_dijkstra_paths_V2.csv")
e2_summary = pd.read_csv("results/tables/e2_dijkstra_alpha_summary_V2.csv")

# Create scatter plot: alpha vs cost (with color coding by alpha value)
fig, ax = plt.subplots(figsize=(8, 6))

# Sample data points for scatter (not all points, to avoid overcrowding)
np.random.seed(42)
sample_size = min(5000, len(e2_paths))
if len(e2_paths) > sample_size:
    sampled = e2_paths.sample(n=sample_size, random_state=42)
else:
    sampled = e2_paths

# Color points by alpha value
alphas = sampled['alpha'].values
costs = sampled['cost'].values
finite_mask = np.isfinite(costs)
alphas = alphas[finite_mask]
costs = costs[finite_mask]

# Assign colors based on alpha ranges
point_colors = []
for alpha in alphas:
    if alpha == 0.0:
        color = blue_rgb  # Blue for pure topology
    elif alpha <= 0.25:
        color = purple_rgb  # Purple
    elif alpha <= 0.5:
        color = pink_rgb  # Pink
    elif alpha <= 0.75:
        color = red_rgb  # Red
    else:
        color = green_rgb  # Green for high geometry

    point_colors.append(color)

point_colors = np.array(point_colors)

# Plot scatter
ax.scatter(alphas, costs, c=point_colors, alpha=0.6, s=20, edgecolors='none')

# Overlay summary line
ax.plot(e2_summary['alpha'], e2_summary['mean_cost'], 
       'o-', linewidth=3, markersize=10,
       color='#A23B72', label='Mean cost', zorder=5)

ax.set_xlabel("α (geometric weight)", fontweight='bold', fontsize=16)
ax.set_ylabel("Path Cost", fontweight='bold', fontsize=16)
ax.set_title("E2: Alpha-sweep Cost Distribution", fontweight='bold', fontsize=17)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(out_dir / "e2_result_alpha_sweep.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {out_dir / 'e2_result_alpha_sweep.png'}")

# ============================================================================
# E4: Diffusion vs Geodesic Distance Scatter Plot
# ============================================================================
print("\n[2/2] Generating E4: Diffusion vs Geodesic Distance Scatter Plot...")

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
            from scipy.interpolate import interp1d
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
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch

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
from matplotlib.patches import Rectangle
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
print("Done!")
print("=" * 80)

