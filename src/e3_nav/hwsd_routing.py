#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HWSD (Hybrid Weighted Spatial Distance) Routing
Based on Barjuan et al. 2024 method
Hybrid routing strategy combining geometric distance and edge weights
"""

import networkx as nx
import numpy as np
import math
from typing import Dict, Tuple, Optional

def euclidean_distance(pos: Dict, u: str, v: str) -> float:
    """Compute Euclidean distance between two nodes"""
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
    
    Greedy routing strategy that combines geometric distance and edge weights
    
    Scoring function:
    score(v) = λ * d_geo(v, target) + (1-λ) * (1 / w(u, v))
    
    Where:
    - d_geo(v, target): geometric distance from node v to target
    - w(u, v): weight of edge (u, v)
    - λ: mixing parameter (0=pure weight, 1=pure geometric)
    
    Args:
        G: NetworkX graph
        s: source node
        t: target node
        pos: node position dictionary {node: (x, y)}
        step_limit: maximum number of steps
        lambda_param: mixing parameter (0-1), controls weight of geometry vs weights
        weight_key: edge weight attribute name
    
    Returns:
        (success, path, reason)
    """
    if (pos is None) or (s not in pos) or (t not in pos):
        return False, [s], "no_geometry"
    
    path, visited = [s], {s}
    
    # Get maximum weight for normalization
    all_weights = [d.get(weight_key, 1.0) for u, v, d in G.edges(data=True)]
    max_weight = max(all_weights) if all_weights else 1.0
    min_weight = min([w for w in all_weights if w > 0]) if all_weights else 1.0
    
    def compute_hwsd_score(v: str) -> float:
        """Compute HWSD score (smaller is better)"""
        # Geometric distance component (normalized)
        geo_dist = euclidean_distance(pos, v, t)
        if not np.isfinite(geo_dist):
            return np.inf
        
        # Normalize geometric distance (assume max distance ~2.0, based on spring_layout range)
        max_geo_dist = 2.0
        normalized_geo = geo_dist / max_geo_dist if max_geo_dist > 0 else geo_dist
        
        # Edge weight component (normalized)
        edge_weight = G.get_edge_data(path[-1], v, {}).get(weight_key, 1.0)
        if edge_weight <= 0:
            return np.inf
        
        # Use inverse weight (larger weight means smaller cost)
        # Normalize: 1 / (normalized_weight)
        normalized_weight = edge_weight / max_weight if max_weight > 0 else 1.0
        inverse_weight = 1.0 / normalized_weight if normalized_weight > 0 else np.inf
        
        # Hybrid score
        score = lambda_param * normalized_geo + (1.0 - lambda_param) * inverse_weight
        
        return score
    
    while len(path) - 1 < step_limit:
        u = path[-1]
        if u == t:
            return True, path, None
        
        nbrs = [v for v in G.successors(u) if v not in visited]
        if not nbrs:
            return False, path, "dead_end"
        
        # Select neighbor with minimum score
        v = min(nbrs, key=compute_hwsd_score)
        visited.add(v)
        path.append(v)
    
    return False, path, "step_limit"

def hwsd_sweep(G: nx.DiGraph, pos: Dict, seed_target_pairs: list,
                lambda_values: list = None, step_limit: int = None,
                weight_key: str = 'weight') -> Dict:
    """
    Sweep lambda parameter to find optimal mixing parameter
    
    Args:
        G: NetworkX graph
        pos: node position dictionary
        seed_target_pairs: list of node pairs [(seed, target), ...]
        lambda_values: list of lambda parameter values, if None use default values
        step_limit: maximum number of steps
        weight_key: edge weight attribute name
    
    Returns:
        dictionary containing results for each lambda value
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
            
            # Compute shortest path length
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

