#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析E3实验的几何一致性(GC)和Greedy Routing Efficiency (GRE)

基于Cannistraci et al. (2022)的理论框架
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from geometric_congruence import (
    compute_geometrical_congruence,
    compute_greedy_routing_efficiency,
    analyze_geometric_topology_relationship
)
from run_nav import load_graph, get_pos, read_list, load_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/processed/V2/graph_V2_directed_weighted.gpickle")
    ap.add_argument("--results", default="results/tables/e3_nav_results.csv")
    ap.add_argument("--output_dir", default="results/figures")
    ap.add_argument("--sample_size", type=int, default=500, help="采样节点对数量（用于GC计算和散点图）")
    ap.add_argument("--config", default="configs/params.yaml")
    args = ap.parse_args()
    
    # 加载配置
    C = load_cfg(args.config)
    
    # 加载图
    print("Loading graph...")
    G = load_graph(args.graph)
    pos = get_pos(G)
    
    if pos is None:
        print("ERROR: No position data found in graph!")
        return
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Position data: {len(pos)} nodes")
    
    # 加载E3结果
    print(f"Loading E3 results from {args.results}...")
    results_df = pd.read_csv(args.results)
    print(f"Results: {len(results_df)} node pairs")
    
    # 计算GRE
    print("\n" + "="*60)
    print("Computing Greedy Routing Efficiency (GRE)...")
    print("="*60)
    gre, success_rate, avg_stretch, normalized_stretch = compute_greedy_routing_efficiency(results_df)
    print(f"Success Rate: {success_rate*100:.2f}%")
    print(f"Average Stretch (successful): {avg_stretch:.2f}")
    print(f"Normalized Stretch: {normalized_stretch:.4f}")
    print(f"GRE (Greedy Routing Efficiency): {gre:.4f}")
    
    # 计算GC
    print("\n" + "="*60)
    print(f"Computing Geometrical Congruence (GC) (sampling {args.sample_size} pairs)...")
    print("="*60)
    
    # 从results_df提取节点对
    node_pairs_from_results = list(zip(results_df['seed'], results_df['target']))
    
    # 如果结果数据少，从图中采样更多节点对并运行导航实验
    if len(node_pairs_from_results) < args.sample_size:
        print(f"Results have only {len(node_pairs_from_results)} pairs, sampling more from graph and running navigation...")
        import random
        random.seed(42)
        
        # 获取sensory和motor节点
        sensory_nodes = read_list(C["data"]["sensory_list"]) if "data" in C and "sensory_list" in C["data"] else None
        motor_nodes = read_list(C["data"]["motor_list"]) if "data" in C and "motor_list" in C["data"] else None
        
        if sensory_nodes is None or motor_nodes is None:
            # 从节点类型推断
            sensory_nodes = [n for n, d in G.nodes(data=True) if d.get('neuron_type') == 'S']
            motor_nodes = [n for n, d in G.nodes(data=True) if d.get('neuron_type') == 'M']
        
        # 生成所有可能的sensory-motor对
        all_pairs = [(s, t) for s in sensory_nodes if s in G for t in motor_nodes if t in G and s != t]
        
        # 检查可达性并采样
        print(f"  Total possible pairs: {len(all_pairs)}")
        reachable_pairs = []
        for s, t in all_pairs:
            if nx.has_path(G, s, t):
                reachable_pairs.append((s, t))
        
        print(f"  Reachable pairs: {len(reachable_pairs)}")
        
        # 采样
        sample_size = min(args.sample_size, len(reachable_pairs))
        if sample_size > len(node_pairs_from_results):
            additional_pairs = random.sample(reachable_pairs, sample_size - len(node_pairs_from_results))
            print(f"  Sampling {len(additional_pairs)} additional pairs")
            print(f"  Running navigation experiments on additional pairs...")
            
            # 对额外采样的节点对运行导航实验
            from run_nav import greedy_route
            step_limit = max(1, G.number_of_nodes())
            additional_results = []
            
            existing_pairs_set = set(node_pairs_from_results)
            for i, (s, t) in enumerate(additional_pairs):
                if (s, t) not in existing_pairs_set:
                    if (i + 1) % 100 == 0:
                        print(f"    Processed {i+1}/{len(additional_pairs)} pairs...")
                    
                    success, path, reason = greedy_route(G, s, t, pos, step_limit, scoring_weights=None)
                    plen = max(0, len(path) - 1)
                    
                    # 计算最短路径长度
                    try:
                        sp = nx.shortest_path_length(G, s, t)
                    except:
                        sp = np.inf
                    
                    stretch = (plen / sp) if success and np.isfinite(sp) and sp > 0 else (1.0 if success and sp == 0 else np.inf)
                    
                    additional_results.append({
                        'seed': s,
                        'target': t,
                        'success': success,
                        'path_len': plen,
                        'sp_len': sp,
                        'stretch': stretch,
                        'failure_reason': None if success else reason
                    })
            
            # 合并结果
            if additional_results:
                additional_df = pd.DataFrame(additional_results)
                results_df = pd.concat([results_df, additional_df], ignore_index=True)
                print(f"  ✓ Added {len(additional_results)} navigation results")
                print(f"  Total results now: {len(results_df)} pairs")
            
            node_pairs = node_pairs_from_results + additional_pairs
        else:
            node_pairs = node_pairs_from_results
    else:
        node_pairs = node_pairs_from_results[:args.sample_size] if len(node_pairs_from_results) > args.sample_size else node_pairs_from_results
    
    print(f"  Total pairs to analyze: {len(node_pairs)}")
    
    gc_values, gc_mean, gc_std, gc_details = compute_geometrical_congruence(
        G, pos, node_pairs=node_pairs
    )
    
    print(f"GC Mean: {gc_mean:.4f}")
    print(f"GC Std: {gc_std:.4f}")
    print(f"GC Min: {np.min(gc_values) if len(gc_values) > 0 else np.nan:.4f}")
    print(f"GC Max: {np.max(gc_values) if len(gc_values) > 0 else np.nan:.4f}")
    print(f"GC Median: {np.median(gc_values) if len(gc_values) > 0 else np.nan:.4f}")
    
    # 分析几何-拓扑关系
    print("\n" + "="*60)
    print("Analyzing Geometric-Topology Relationship...")
    print("="*60)
    
    analysis = analyze_geometric_topology_relationship(
        G, pos, results_df, node_pairs=node_pairs
    )
    
    if 'high_gc_success_rate' in analysis and not np.isnan(analysis['high_gc_success_rate']):
        print(f"\nGC分组分析（中位数分割）:")
        print(f"  High GC (>median) Success Rate: {analysis['high_gc_success_rate']*100:.2f}%")
        print(f"  Low GC (≤median) Success Rate: {analysis['low_gc_success_rate']*100:.2f}%")
        improvement = (analysis['high_gc_success_rate'] - analysis['low_gc_success_rate']) * 100
        print(f"  Improvement: {improvement:+.2f} percentage points")
        
        if 'q1_success_rate' in analysis:
            print(f"\nGC分组分析（四分位数）:")
            print(f"  Q1 (GC ≤ {analysis.get('gc_q25', 0):.4f}): {analysis['q1_success_rate']*100:.2f}% (n={analysis['q1_count']})")
            print(f"  Q2 (GC {analysis.get('gc_q25', 0):.4f} - {analysis.get('gc_median', 0):.4f}): {analysis['q2_success_rate']*100:.2f}% (n={analysis['q2_count']})")
            print(f"  Q3 (GC {analysis.get('gc_median', 0):.4f} - {analysis.get('gc_q75', 0):.4f}): {analysis['q3_success_rate']*100:.2f}% (n={analysis['q3_count']})")
            print(f"  Q4 (GC > {analysis.get('gc_q75', 0):.4f}): {analysis['q4_success_rate']*100:.2f}% (n={analysis['q4_count']})")
    
    if 'gc_stretch_correlation' in analysis and not np.isnan(analysis['gc_stretch_correlation']):
        print(f"GC-Stretch Correlation (Spearman): {analysis['gc_stretch_correlation']:.4f}")
        print(f"P-value: {analysis['gc_stretch_pvalue']:.4e}")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: GC分布 - 使用更多bins
    if len(gc_values) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        n_bins_hist = min(50, max(20, len(gc_values) // 5))  # 更多bins
        ax.hist(gc_values, bins=n_bins_hist, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax.axvline(gc_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {gc_mean:.4f}')
        ax.axvline(np.median(gc_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(gc_values):.4f}')
        if 'gc_q25' in analysis and not np.isnan(analysis['gc_q25']):
            ax.axvline(analysis['gc_q25'], color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1: {analysis["gc_q25"]:.4f}')
            ax.axvline(analysis['gc_q75'], color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3: {analysis["gc_q75"]:.4f}')
        ax.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax.set_title(f'Geometrical Congruence Distribution (n={len(gc_values)}, {n_bins_hist} bins)', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "e3_gc_distribution.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'e3_gc_distribution.png'}")
        plt.close()
    
    # Figure 2: GC vs Success Rate
    if 'merged' in analysis and len(analysis['merged']) > 0:
        merged = analysis['merged']
        
        # 如果有额外的GC数据（没有导航结果的），也加入散点图
        if 'gc_df' in analysis and len(analysis['gc_df']) > len(merged):
            # 合并所有GC数据
            all_gc_data = analysis['gc_df'].merge(
                results_df,
                on=['seed', 'target'],
                how='left'
            )
            # 标记哪些有导航结果
            all_gc_data['has_nav_result'] = all_gc_data['success'].notna()
        else:
            all_gc_data = merged
            all_gc_data['has_nav_result'] = True
        
        # 分箱分析 - 使用更多bins
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：GC vs Success Rate (分箱) - 使用更多bins
        ax1 = axes[0]
        # 使用所有GC数据（包括没有导航结果的）
        gc_data_for_binning = all_gc_data[all_gc_data['has_nav_result']] if 'has_nav_result' in all_gc_data.columns else merged
        
        # 根据数据量动态调整bins数量，但至少10个bins
        n_bins = min(30, max(10, len(gc_data_for_binning) // 5))  # 更激进的bins数量
        bins = np.linspace(gc_data_for_binning['gc'].min(), gc_data_for_binning['gc'].max(), n_bins + 1)
        gc_data_for_binning['gc_bin'] = pd.cut(gc_data_for_binning['gc'], bins=bins, include_lowest=True)
        bin_stats = gc_data_for_binning.groupby('gc_bin', observed=True)['success'].agg(['mean', 'count'])
        bin_centers = [interval.mid for interval in bin_stats.index]
        
        # 只显示有足够样本的bins（至少2个样本，降低阈值以显示更多点）
        min_samples = max(1, len(gc_data_for_binning) // 50)  # 动态调整最小样本数
        valid_bins = bin_stats['count'] >= min_samples
        bin_centers = [bc for bc, valid in zip(bin_centers, valid_bins) if valid]
        bin_means = bin_stats['mean'][valid_bins].values * 100
        bin_counts = bin_stats['count'][valid_bins].values
        
        ax1.plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=6, color='#2E86AB', alpha=0.8)
        # 添加样本量标注（只标注部分点，避免太拥挤）
        if len(bin_centers) <= 20:
            for i, (x, y, count) in enumerate(zip(bin_centers, bin_means, bin_counts)):
                if i % max(1, len(bin_centers) // 8) == 0:  # 标注更多点
                    ax1.text(x, y + 2, f'n={int(count)}', ha='center', va='bottom', fontsize=7, alpha=0.7)
        ax1.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
        ax1.set_title(f'(a) GC vs Success Rate (n={len(gc_data_for_binning)} pairs, {len(bin_centers)} bins)', fontweight='bold', fontsize=12)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # 添加四分位数分割线
        if 'gc_q25' in analysis and 'gc_q75' in analysis and 'gc_median' in analysis:
            ax1.axvline(analysis['gc_q25'], color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax1.axvline(analysis['gc_median'], color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax1.axvline(analysis['gc_q75'], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # 右图：GC vs Stretch (成功路径) - 显示所有数据点
        ax2 = axes[1]
        successful = merged[merged['success'] == True]
        if len(successful) > 0:
            gc_success = successful['gc'].values
            stretch_success = successful['stretch'].values
            stretch_success = stretch_success[np.isfinite(stretch_success)]
            gc_success = gc_success[:len(stretch_success)]
            
            if len(gc_success) > 0:
                # 根据数据点数量调整散点大小和透明度
                n_points = len(gc_success)
                if n_points > 100:
                    alpha = 0.3
                    s = 15
                elif n_points > 50:
                    alpha = 0.4
                    s = 20
                else:
                    alpha = 0.5
                    s = 25
                
                ax2.scatter(gc_success, stretch_success, alpha=alpha, s=s, color='#A23B72', edgecolors='black', linewidth=0.1)
                ax2.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
                ax2.set_ylabel('Stretch Ratio', fontweight='bold', fontsize=11)
                ax2.set_title(f'(b) GC vs Stretch (n={len(gc_success)} successful paths)', fontweight='bold', fontsize=12)
                ax2.grid(alpha=0.3)
                
                # 添加趋势线
                if len(gc_success) > 1:
                    z = np.polyfit(gc_success, stretch_success, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(gc_success.min(), gc_success.max(), 100)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                    ax2.legend()
        else:
            # 如果没有成功路径，显示所有路径的GC值
            ax2.scatter(merged['gc'], merged['success'].astype(int), alpha=0.3, s=20, color='gray')
            ax2.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Success (1) / Failure (0)', fontweight='bold', fontsize=11)
            ax2.set_title(f'(b) GC vs Success (n={len(merged)} pairs)', fontweight='bold', fontsize=12)
            ax2.grid(alpha=0.3)
        
        # 创建一个单独的散点图显示所有GC值（现在所有点都有导航结果）
        if 'gc_df' in analysis and len(analysis['gc_df']) > 0:
            fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
            
            # 显示所有GC值（现在所有点都有导航结果）
            all_gc = all_gc_data['gc'].values
            # 现在所有点都应该有导航结果
            has_result = all_gc_data['has_nav_result'].values if 'has_nav_result' in all_gc_data.columns else np.ones(len(all_gc), dtype=bool)
            
            # 左图：GC分布散点图（所有数据点）
            ax3 = axes2[0]
            
            # 添加少量随机y偏移，避免点重叠
            np.random.seed(42)
            y_offset = np.random.normal(0, 0.02, len(all_gc))
            
            # 有导航结果的用不同颜色
            gc_with_result = all_gc[has_result]
            gc_without_result = all_gc[~has_result]
            y_offset_with = y_offset[has_result]
            y_offset_without = y_offset[~has_result]
            
            # 绘制散点图（现在所有点都有导航结果，所以不需要显示灰色点）
            # if len(gc_without_result) > 0:
            #     ax3.scatter(gc_without_result, y_offset_without, 
            #                alpha=0.4, s=20, color='lightgray', edgecolors='gray', linewidth=0.3,
            #                label=f'No nav result (n={len(gc_without_result)})')
            if len(gc_with_result) > 0:
                # 如果有导航结果，显示成功/失败（成功用绿色叠在上面，失败用红色在下面）
                result_data = all_gc_data[has_result]
                success_mask = result_data['success'] == True
                gc_success_all = result_data.loc[success_mask, 'gc'].values
                gc_fail_all = result_data.loc[~success_mask, 'gc'].values
                y_success = y_offset_with[success_mask]
                y_fail = y_offset_with[~success_mask]
                
                # 先绘制失败的点（在下面）
                if len(gc_fail_all) > 0:
                    ax3.scatter(gc_fail_all, y_fail, 
                               alpha=0.5, s=25, color='red', edgecolors='darkred', linewidth=0.3,
                               label=f'Failure (n={len(gc_fail_all)})', zorder=1)
                # 再绘制成功的点（叠在上面）
                if len(gc_success_all) > 0:
                    ax3.scatter(gc_success_all, y_success, 
                               alpha=0.5, s=25, color='green', edgecolors='darkgreen', linewidth=0.3,
                               label=f'Success (n={len(gc_success_all)})', zorder=2)
            
            ax3.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
            ax3.set_ylabel('Random Offset (for visibility)', fontweight='bold', fontsize=11)
            ax3.set_title(f'(a) GC Distribution Scatter (n={len(all_gc)} pairs)', fontweight='bold', fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3, axis='x')
            ax3.set_ylim(-0.1, 0.1)
            
            # 右图：GC vs Success Rate（分箱显示趋势，更清楚地显示GC越高成功率越高）
            ax4 = axes2[1]
            if len(gc_with_result) > 0:
                result_data = all_gc_data[has_result].copy()
                gc_all = result_data['gc'].values
                success_all = result_data['success'].values
                
                # 使用分位数分箱，确保每个bin的样本量大致相等（增加bin数量，使区间更细分）
                n_bins = 40  # 从20增加到40，使区间更细分
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = [result_data['gc'].quantile(q) for q in quantiles]
                bin_edges[0] = result_data['gc'].min() - 0.001
                bin_edges[-1] = result_data['gc'].max() + 0.001
                
                result_data['gc_bin'] = pd.cut(result_data['gc'], bins=bin_edges, include_lowest=True)
                bin_stats = result_data.groupby('gc_bin', observed=True)['success'].agg(['mean', 'count', 'std'])
                bin_centers = [interval.mid for interval in bin_stats.index]
                
                min_samples = max(3, len(result_data) // 50)  # 降低最小样本数要求，以适应更多bins
                valid_bins = bin_stats['count'] >= min_samples
                bin_centers = [bc for bc, valid in zip(bin_centers, valid_bins) if valid]
                bin_means = bin_stats['mean'][valid_bins].values * 100
                bin_counts = bin_stats['count'][valid_bins].values
                bin_stds = bin_stats['std'][valid_bins].values * 100
                
                # 计算95%置信区间
                bin_sem = bin_stds / np.sqrt(bin_counts)
                bin_ci = 1.96 * bin_sem
                
                # 绘制分箱成功率曲线（使用更小的marker以适应更多点）
                ax4.plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=5, 
                        color='#2E86AB', alpha=0.9, label='Success Rate')
                ax4.fill_between(bin_centers, bin_means - bin_ci, bin_means + bin_ci, 
                                alpha=0.2, color='#2E86AB')
                
                # 添加原始数据点（透明度较低）
                y_offset_success = np.random.normal(0.95, 0.02, len(gc_all))
                y_offset_fail = np.random.normal(0.05, 0.02, len(gc_all))
                
                success_mask = result_data['success'] == True
                gc_success = gc_all[success_mask]
                gc_fail = gc_all[~success_mask]
                y_success = y_offset_success[success_mask]
                y_fail = y_offset_fail[~success_mask]
                
                # 先绘制失败的点（在下面）
                if len(gc_fail) > 0:
                    ax4.scatter(gc_fail, y_fail * 100, alpha=0.15, s=10, color='red', 
                               edgecolors='none', label='Failure points', zorder=1)
                # 再绘制成功的点（叠在上面）
                if len(gc_success) > 0:
                    ax4.scatter(gc_success, y_success * 100, alpha=0.15, s=10, color='green', 
                               edgecolors='none', label='Success points', zorder=2)
                
                # 添加直线拟合和R²
                from scipy.stats import spearmanr
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                # Spearman相关性
                corr, p_val = spearmanr(gc_all, success_all.astype(int))
                
                # 线性回归拟合（使用分箱后的数据）
                if len(bin_centers) > 2:
                    # 准备数据
                    X_fit = np.array(bin_centers).reshape(-1, 1)
                    y_fit = bin_means
                    
                    # 线性回归
                    lr = LinearRegression()
                    lr.fit(X_fit, y_fit)
                    y_pred = lr.predict(X_fit)
                    r2 = r2_score(y_fit, y_pred)
                    
                    # 绘制拟合线
                    x_fit_line = np.linspace(min(bin_centers), max(bin_centers), 200)
                    y_fit_line = lr.predict(x_fit_line.reshape(-1, 1))
                    ax4.plot(x_fit_line, y_fit_line, 'r--', linewidth=2.5, alpha=0.8, 
                            label=f'Linear fit (R²={r2:.3f})')
                    
                    # 也可以添加平滑趋势线（可选）
                    from scipy.interpolate import UnivariateSpline
                    try:
                        spline = UnivariateSpline(bin_centers, bin_means, s=len(bin_centers)*10)
                        x_smooth = np.linspace(min(bin_centers), max(bin_centers), 200)
                        y_smooth = spline(x_smooth)
                        ax4.plot(x_smooth, y_smooth, 'orange', linewidth=2, alpha=0.6, 
                                linestyle=':', label=f'Smooth trend')
                    except:
                        pass
                
                ax4.set_xlabel('Geometrical Congruence (GC)', fontweight='bold', fontsize=11)
                ax4.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
                ax4.set_title(f'(b) GC vs Success Rate (n={len(gc_all)} pairs)', fontweight='bold', fontsize=12)
                ax4.legend(fontsize=9)
                ax4.grid(alpha=0.3)
                ax4.set_ylim(0, 105)
                
                # 添加统计信息（包括R²）
                if len(bin_centers) > 2:
                    X_fit = np.array(bin_centers).reshape(-1, 1)
                    y_fit = bin_means
                    lr = LinearRegression()
                    lr.fit(X_fit, y_fit)
                    y_pred = lr.predict(X_fit)
                    r2 = r2_score(y_fit, y_pred)
                    
                    stats_text = f'Spearman r={corr:.3f}\np={p_val:.2e}\nR²={r2:.3f}'
                else:
                    stats_text = f'Spearman r={corr:.3f}\np={p_val:.2e}'
                
                ax4.text(0.05, 0.95, stats_text, 
                        transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(output_dir / "e3_gc_all_scatter.png", dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_dir / 'e3_gc_all_scatter.png'}")
            plt.close()
        
        plt.tight_layout()
        plt.savefig(output_dir / "e3_gc_analysis.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'e3_gc_analysis.png'}")
        plt.close()
    
    # 保存结果
    summary = {
        'gre': gre,
        'success_rate': success_rate,
        'avg_stretch': avg_stretch,
        'normalized_stretch': normalized_stretch,
        'gc_mean': gc_mean,
        'gc_std': gc_std,
        'gc_median': np.median(gc_values) if len(gc_values) > 0 else np.nan,
        'gc_min': np.min(gc_values) if len(gc_values) > 0 else np.nan,
        'gc_max': np.max(gc_values) if len(gc_values) > 0 else np.nan,
        'high_gc_success_rate': analysis.get('high_gc_success_rate', np.nan),
        'low_gc_success_rate': analysis.get('low_gc_success_rate', np.nan),
        'gc_stretch_correlation': analysis.get('gc_stretch_correlation', np.nan),
        'gc_stretch_pvalue': analysis.get('gc_stretch_pvalue', np.nan)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = Path("results/tables/e3_gc_gre_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✓ Saved summary: {summary_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"\nKey Findings:")
    print(f"  - GRE (Greedy Routing Efficiency): {gre:.4f}")
    print(f"  - GC Mean (Geometrical Congruence): {gc_mean:.4f}")
    if 'high_gc_success_rate' in analysis and not np.isnan(analysis['high_gc_success_rate']):
        print(f"  - High GC paths have {improvement:+.2f}% higher success rate")

if __name__ == "__main__":
    main()

