# -*- coding: utf-8 -*-
"""
Geometrical Congruence (GC) and Greedy Routing Efficiency (GRE) analysis

Based on theoretical framework from Cannistraci et al. (2022):
- Geometrical Congruence (GC): "Straightness" of topological shortest paths in geometric space
- Greedy Routing Efficiency (GRE): Comprehensive metric combining success rate and path quality
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import math

def euclid(a, b):
    """Calculate Euclidean distance"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_geometrical_congruence(G, pos, node_pairs=None, sample_size=None):
    """
    Compute Geometrical Congruence (GC) of the network
    
    Definition (Cannistraci et al. 2022):
    GC = geometric straight-line distance / geometric path length (topological shortest path projected to geometric space)
    GC close to 1 indicates "topological geodesic ≈ geometric geodesic"
    
    Parameters:
        G: NetworkX graph
        pos: Position dictionary {node: (x, y)}
        node_pairs: List of node pairs to analyze [(s, t), ...], if None analyze all reachable pairs
        sample_size: If node_pairs is None, randomly sample this many pairs (None means all)
    
    Returns:
        gc_values: List of GC values for each node pair
        gc_mean: Mean GC value
        gc_std: Standard deviation of GC values
        details: Detailed information dictionary
    """
    if pos is None:
        return [], np.nan, np.nan, {}
    
    gc_values = []
    details = {
        'geo_straight': [],      # Geometric straight-line distance
        'geo_path_length': [],   # Geometric path length (projected)
        'topo_path_length': [],  # Topological path length (hop count)
        'pairs': []              # Node pairs
    }
    
    # Determine node pairs to analyze
    if node_pairs is None:
        # Compute all reachable node pairs
        nodes = list(G.nodes())
        if sample_size is not None and len(nodes) > sample_size:
            # Random sampling
            import random
            random.seed(42)
            pairs = []
            for _ in range(min(sample_size, len(nodes) * (len(nodes) - 1) // 2)):
                s, t = random.sample(nodes, 2)
                if nx.has_path(G, s, t):
                    pairs.append((s, t))
            node_pairs = pairs
        else:
            # All reachable pairs
            node_pairs = []
            for s in nodes:
                for t in nodes:
                    if s != t and nx.has_path(G, s, t):
                        node_pairs.append((s, t))
    
    # Compute GC for each node pair
    for s, t in node_pairs:
        if s not in pos or t not in pos:
            continue
        
        try:
            # Topological shortest path
            tsp = nx.shortest_path(G, s, t)
            tsp_length = len(tsp) - 1  # hop count
            
            # Geometric path length (projected to geometric space)
            geo_path_length = 0.0
            for i in range(len(tsp) - 1):
                u, v = tsp[i], tsp[i + 1]
                if u in pos and v in pos:
                    geo_path_length += euclid(pos[u], pos[v])
            
            # Geometric straight-line distance
            geo_straight = euclid(pos[s], pos[t])
            
            # GC = geometric straight-line distance / geometric path length
            if geo_path_length > 0:
                gc = geo_straight / geo_path_length
                gc_values.append(gc)
                details['geo_straight'].append(geo_straight)
                details['geo_path_length'].append(geo_path_length)
                details['topo_path_length'].append(tsp_length)
                details['pairs'].append((s, t))
        except (nx.NetworkXNoPath, KeyError):
            continue
    
    if len(gc_values) == 0:
        return [], np.nan, np.nan, details
    
    gc_mean = np.mean(gc_values)
    gc_std = np.std(gc_values)
    
    return gc_values, gc_mean, gc_std, details

def compute_greedy_routing_efficiency(results_df):
    """
    计算Greedy Routing Efficiency (GRE)
    
    定义（Cannistraci et al. 2022）：
    GRE = success_rate × normalized_stretch
    其中normalized_stretch = 1 / average_stretch
    
    参数:
        results_df: E3实验结果DataFrame，包含'success'和'stretch'列
    
    返回:
        gre: GRE值 [0, 1]
        success_rate: 成功率
        avg_stretch: 平均stretch（成功路径）
        normalized_stretch: 归一化stretch
    """
    if len(results_df) == 0:
        return 0.0, 0.0, np.nan, 0.0
    
    # 成功率
    success_rate = results_df['success'].mean()
    
    # 成功路径的stretch
    successful = results_df[results_df['success'] == True]
    if len(successful) == 0:
        return 0.0, success_rate, np.nan, 0.0
    
    # 平均stretch（只考虑有限值）
    stretch_values = successful['stretch'].values
    stretch_values = stretch_values[np.isfinite(stretch_values)]
    
    if len(stretch_values) == 0:
        return 0.0, success_rate, np.nan, 0.0
    
    avg_stretch = np.mean(stretch_values)
    
    # 归一化stretch（越大越好，所以用倒数）
    normalized_stretch = 1.0 / avg_stretch if avg_stretch > 0 else 0.0
    
    # GRE = success_rate × normalized_stretch
    gre = success_rate * normalized_stretch
    
    return gre, success_rate, avg_stretch, normalized_stretch

def analyze_geometric_topology_relationship(G, pos, results_df, node_pairs=None):
    """
    分析几何-拓扑关系
    
    参数:
        G: NetworkX图
        pos: 位置字典
        results_df: E3实验结果
        node_pairs: 要分析的节点对（如果None，从results_df提取）
    
    返回:
        analysis: 分析结果字典
    """
    # 从results_df提取节点对
    if node_pairs is None:
        node_pairs = list(zip(results_df['seed'], results_df['target']))
    
    # 计算每个节点对的GC
    gc_values, gc_mean, gc_std, gc_details = compute_geometrical_congruence(
        G, pos, node_pairs=node_pairs
    )
    
    # 创建GC DataFrame
    gc_df = pd.DataFrame({
        'seed': [s for s, t in gc_details['pairs']],
        'target': [t for s, t in gc_details['pairs']],
        'gc': gc_values,
        'geo_straight': gc_details['geo_straight'],
        'geo_path_length': gc_details['geo_path_length'],
        'topo_path_length': gc_details['topo_path_length']
    })
    
    # 合并results_df和gc_df
    merged = results_df.merge(
        gc_df,
        on=['seed', 'target'],
        how='inner'
    )
    
    # 分析GC与导航成功的关系
    analysis = {
        'gc_mean': gc_mean,
        'gc_std': gc_std,
        'gc_values': gc_values,
        'gc_df': gc_df,
        'merged': merged
    }
    
    if len(merged) > 0:
        # GC与成功率的关系 - 使用四分位数分组（更细的分组）
        gc_median = merged['gc'].median()
        gc_q25 = merged['gc'].quantile(0.25)
        gc_q75 = merged['gc'].quantile(0.75)
        
        # 分成4组：Q1, Q2, Q3, Q4
        q1 = merged[merged['gc'] <= gc_q25]
        q2 = merged[(merged['gc'] > gc_q25) & (merged['gc'] <= gc_median)]
        q3 = merged[(merged['gc'] > gc_median) & (merged['gc'] <= gc_q75)]
        q4 = merged[merged['gc'] > gc_q75]
        
        # 计算各组的成功率
        analysis['q1_success_rate'] = q1['success'].mean() if len(q1) > 0 else np.nan
        analysis['q2_success_rate'] = q2['success'].mean() if len(q2) > 0 else np.nan
        analysis['q3_success_rate'] = q3['success'].mean() if len(q3) > 0 else np.nan
        analysis['q4_success_rate'] = q4['success'].mean() if len(q4) > 0 else np.nan
        analysis['q1_count'] = len(q1)
        analysis['q2_count'] = len(q2)
        analysis['q3_count'] = len(q3)
        analysis['q4_count'] = len(q4)
        
        # 保留原来的中位数分组（向后兼容）
        high_gc = merged[merged['gc'] > gc_median]
        low_gc = merged[merged['gc'] <= gc_median]
        
        analysis['high_gc_success_rate'] = high_gc['success'].mean() if len(high_gc) > 0 else np.nan
        analysis['low_gc_success_rate'] = low_gc['success'].mean() if len(low_gc) > 0 else np.nan
        analysis['gc_q25'] = gc_q25
        analysis['gc_q75'] = gc_q75
        analysis['gc_median'] = gc_median
        
        # GC与stretch的关系（成功路径）
        successful = merged[merged['success'] == True]
        if len(successful) > 0:
            from scipy.stats import spearmanr
            gc_success = successful['gc'].values
            stretch_success = successful['stretch'].values
            stretch_success = stretch_success[np.isfinite(stretch_success)]
            gc_success = gc_success[:len(stretch_success)]
            
            if len(gc_success) > 1 and len(stretch_success) > 1:
                corr, p_val = spearmanr(gc_success, stretch_success)
                analysis['gc_stretch_correlation'] = corr
                analysis['gc_stretch_pvalue'] = p_val
            else:
                analysis['gc_stretch_correlation'] = np.nan
                analysis['gc_stretch_pvalue'] = np.nan
    else:
            analysis['gc_stretch_correlation'] = np.nan
            analysis['gc_stretch_pvalue'] = np.nan
    
    return analysis
