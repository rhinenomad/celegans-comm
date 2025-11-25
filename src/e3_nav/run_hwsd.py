#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行HWSD (Hybrid Weighted Spatial Distance) Routing实验
基于 Barjuan et al. 2024 的方法
"""

import argparse
import pickle
import yaml
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from hwsd_routing import hwsd_route, hwsd_sweep

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_graph(path):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def read_list(p):
    if p and p != "null" and Path(p).exists():
        return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    return None

def get_pos(G):
    pos = {}
    for n, d in G.nodes(data=True):
        if "pos" in d and d["pos"] is not None:
            x, y = d["pos"]
            pos[n] = (float(x), float(y))
        elif "x" in d and "y" in d:
            pos[n] = (float(d["x"]), float(d["y"]))
    return pos if pos else None

def sp_hops(G, s, t):
    try:
        return nx.shortest_path_length(G, s, t)
    except Exception:
        return np.inf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1", "V2", "V3", "V4"])
    ap.add_argument("--lambda_values", nargs="+", type=float, 
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    help="Lambda values for HWSD sweep")
    ap.add_argument("--max_seeds", type=int, default=None)
    ap.add_argument("--max_targets", type=int, default=None)
    args = ap.parse_args()

    C = load_cfg(args.config)
    out_tbl = Path(C["report"]["tables_dir"])
    out_tbl.mkdir(parents=True, exist_ok=True)

    G = load_graph(C["data"]["graphs"][args.variant])
    
    # Load seeds and targets
    seeds = read_list(C["data"]["sensory_list"])
    targets = read_list(C["data"]["motor_list"])
    
    if seeds is None or targets is None:
        # Fallback to degree-based selection
        if args.max_seeds:
            seeds = [n for n, _ in sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:args.max_seeds]]
        else:
            seeds = list(G.nodes())[:10]
        
        if args.max_targets:
            targets = [n for n, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:args.max_targets]]
        else:
            targets = list(G.nodes())[:20]

    pos = get_pos(G)
    if pos is None:
        print("Error: No position data found in graph")
        return

    step_limit = max(1, G.number_of_nodes())
    
    # Prepare seed-target pairs
    seed_target_pairs = [(s, t) for s in seeds if s in G 
                         for t in targets if t in G and s != t]
    
    print(f"Running HWSD routing on {len(seed_target_pairs)} pairs...")
    print(f"Lambda values: {args.lambda_values}")
    
    # Run HWSD sweep
    sweep_results = hwsd_sweep(G, pos, seed_target_pairs, 
                               lambda_values=args.lambda_values,
                               step_limit=step_limit)
    
    # Convert to DataFrame
    sweep_df = pd.DataFrame(list(sweep_results.values()))
    sweep_df = sweep_df.sort_values('lambda')
    sweep_df.to_csv(out_tbl / "e3_hwsd_sweep_results.csv", index=False)
    
    # Find optimal lambda (highest success rate, then lowest stretch)
    optimal_lambda = sweep_df.loc[sweep_df['success_rate'].idxmax(), 'lambda']
    if sweep_df['success_rate'].max() > 0:
        # Among high success rates, find lowest stretch
        high_success = sweep_df[sweep_df['success_rate'] >= sweep_df['success_rate'].max() * 0.95]
        optimal_lambda = high_success.loc[high_success['mean_stretch'].idxmin(), 'lambda']
    
    print(f"\nOptimal lambda: {optimal_lambda:.2f}")
    print(f"  Success rate: {sweep_df.loc[sweep_df['lambda'] == optimal_lambda, 'success_rate'].values[0]:.4f}")
    print(f"  Mean stretch: {sweep_df.loc[sweep_df['lambda'] == optimal_lambda, 'mean_stretch'].values[0]:.4f}")
    
    # Run detailed analysis with optimal lambda
    print(f"\nRunning detailed analysis with lambda={optimal_lambda:.2f}...")
    detailed_rows = []
    
    for s, t in seed_target_pairs:
        success, path, reason = hwsd_route(G, s, t, pos, step_limit, optimal_lambda)
        plen = max(0, len(path) - 1)
        sp = sp_hops(G, s, t)
        stretch = (plen / sp) if success and np.isfinite(sp) and sp > 0 else (
            1.0 if success and sp == 0 else np.inf)
        
        detailed_rows.append({
            "seed": s,
            "target": t,
            "lambda": optimal_lambda,
            "success": success,
            "path_len": plen,
            "sp_len": sp,
            "stretch": stretch,
            "failure_reason": None if success else reason
        })
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(out_tbl / f"e3_hwsd_detailed_lambda_{optimal_lambda:.2f}.csv", index=False)
    
    print(f"\nHWSD routing done ->")
    print(f"  Sweep results: {out_tbl / 'e3_hwsd_sweep_results.csv'}")
    print(f"  Detailed results (lambda={optimal_lambda:.2f}): {out_tbl / f'e3_hwsd_detailed_lambda_{optimal_lambda:.2f}.csv'}")

if __name__ == "__main__":
    main()

