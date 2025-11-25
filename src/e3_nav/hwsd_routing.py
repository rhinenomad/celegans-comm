#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HWSD (Hybrid Weighted Spatial Distance) Routing
基于 Barjuan et al. 2024 的方法
混合几何距离和边权重的路由策略
"""

import networkx as nx
import numpy as np
import math
from typing import Dict, Tuple, Optional

def euclidean_distance(pos: Dict, u: str, v: str) -> float:
    """计算两个节点之间的欧氏距离"""
    if u not in pos or v not in pos:
        return np.inf
    p1 = pos[u]
    p2 = pos[v]
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def hwsd_route(G: nx.DiGraph, s: str, t: str, pos: Dict, 
                step_limit: int, lambda_param: float = 0.5,
                weight_key: str = 'weight') -> Tuple[bool, list, Optional[str]]:
    """
    HWSD (Hybrid Weighted Spatial Distance) Routing
    
    混合几何距离和边权重的贪心路由策略
    
    评分函数：
    score(v) = λ * d_geo(v, target) + (1-λ) * (1 / w(u, v))
    
    其中：
    - d_geo(v, target): 节点v到目标target的几何距离
    - w(u, v): 边(u, v)的权重
    - λ: 混合参数（0=纯权重，1=纯几何）
    
    Args:
        G: NetworkX图
        s: 源节点
        t: 目标节点
        pos: 节点位置字典 {node: (x, y)}
        step_limit: 最大步数
        lambda_param: 混合参数（0-1），控制几何vs权重的权重
        weight_key: 边权重属性名
    
    Returns:
        (success, path, reason)
    """
    if (pos is None) or (s not in pos) or (t not in pos):
        return False, [s], "no_geometry"
    
    path, visited = [s], {s}
    
    # 获取最大权重用于归一化
    all_weights = [d.get(weight_key, 1.0) for u, v, d in G.edges(data=True)]
    max_weight = max(all_weights) if all_weights else 1.0
    min_weight = min([w for w in all_weights if w > 0]) if all_weights else 1.0
    
    def compute_hwsd_score(v: str) -> float:
        """计算HWSD评分（越小越好）"""
        # 几何距离部分（归一化）
        geo_dist = euclidean_distance(pos, v, t)
        if not np.isfinite(geo_dist):
            return np.inf
        
        # 归一化几何距离（假设最大距离约为2.0，基于spring_layout范围）
        max_geo_dist = 2.0
        normalized_geo = geo_dist / max_geo_dist if max_geo_dist > 0 else geo_dist
        
        # 边权重部分（归一化）
        edge_weight = G.get_edge_data(path[-1], v, {}).get(weight_key, 1.0)
        if edge_weight <= 0:
            return np.inf
        
        # 使用逆权重（权重越大，成本越小）
        # 归一化：1 / (normalized_weight)
        normalized_weight = edge_weight / max_weight if max_weight > 0 else 1.0
        inverse_weight = 1.0 / normalized_weight if normalized_weight > 0 else np.inf
        
        # 混合评分
        score = lambda_param * normalized_geo + (1.0 - lambda_param) * inverse_weight
        
        return score
    
    while len(path) - 1 < step_limit:
        u = path[-1]
        if u == t:
            return True, path, None
        
        nbrs = [v for v in G.successors(u) if v not in visited]
        if not nbrs:
            return False, path, "dead_end"
        
        # 选择评分最小的邻居
        v = min(nbrs, key=compute_hwsd_score)
        visited.add(v)
        path.append(v)
    
    return False, path, "step_limit"

def hwsd_sweep(G: nx.DiGraph, pos: Dict, seed_target_pairs: list,
                lambda_values: list = None, step_limit: int = None,
                weight_key: str = 'weight') -> Dict:
    """
    对lambda参数进行sweep，找到最优的混合参数
    
    Args:
        G: NetworkX图
        pos: 节点位置字典
        seed_target_pairs: 节点对列表 [(seed, target), ...]
        lambda_values: lambda参数值列表，如果为None则使用默认值
        step_limit: 最大步数
        weight_key: 边权重属性名
    
    Returns:
        包含每个lambda值的结果的字典
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if step_limit is None:
        step_limit = max(1, G.number_of_nodes())
    
    results = {}
    
    for lambda_val in lambda_values:
        successes = 0
        total_stretch = 0.0
        successful_stretches = []
        path_lengths = []
        
        for s, t in seed_target_pairs:
            if s not in G or t not in G or s == t:
                continue
            
            success, path, reason = hwsd_route(G, s, t, pos, step_limit, lambda_val, weight_key)
            path_len = max(0, len(path) - 1)
            
            # 计算最短路径长度
            try:
                sp_len = nx.shortest_path_length(G, s, t)
            except nx.NetworkXNoPath:
                sp_len = np.inf
            
            if success and np.isfinite(sp_len) and sp_len > 0:
                stretch = path_len / sp_len
                successes += 1
                total_stretch += stretch
                successful_stretches.append(stretch)
                path_lengths.append(path_len)
        
        n_total = len(seed_target_pairs)
        success_rate = successes / n_total if n_total > 0 else 0.0
        mean_stretch = np.mean(successful_stretches) if successful_stretches else np.inf
        median_stretch = np.median(successful_stretches) if successful_stretches else np.inf
        mean_path_len = np.mean(path_lengths) if path_lengths else np.inf
        
        results[lambda_val] = {
            "lambda": lambda_val,
            "success_rate": success_rate,
            "n_success": successes,
            "n_total": n_total,
            "mean_stretch": mean_stretch,
            "median_stretch": median_stretch,
            "mean_path_length": mean_path_len
        }
    
    return results

