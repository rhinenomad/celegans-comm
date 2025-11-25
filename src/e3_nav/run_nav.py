# -*- coding: utf-8 -*-
import argparse, pickle, yaml, math
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
try:
    from .geometric_congruence import (
        compute_geometrical_congruence,
        compute_greedy_routing_efficiency,
        analyze_geometric_topology_relationship
    )
except ImportError:
    # If running as script, import directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from geometric_congruence import (
        compute_geometrical_congruence,
        compute_greedy_routing_efficiency,
        analyze_geometric_topology_relationship
    )

# ---------- utilities ----------
def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def load_graph(path):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(path)
    except Exception:
        with open(path, "rb") as f: return pickle.load(f)

def read_list(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()] if p and p!="null" and Path(p).exists() else None

def get_pos(G):
    pos = {}
    for n,d in G.nodes(data=True):
        if "pos" in d and d["pos"] is not None:
            x,y = d["pos"]; pos[n]=(float(x),float(y))
        elif "x" in d and "y" in d:
            pos[n]=(float(d["x"]), float(d["y"]))
    return pos if pos else None

def euclid(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def successors(G, u):
    try: return list(G.successors(u))
    except Exception: return list(G.neighbors(u))

def sp_hops(G, s, t):
    try: return nx.shortest_path_length(G, s, t)
    except Exception: return np.inf

# ---------- greedy geometric ----------
def greedy_route(G, s, t, pos, step_limit, scoring_weights=None):
    """
    Greedy routing with multi-cue navigation support.
    
    Args:
        G: NetworkX graph
        s: source node
        t: target node
        pos: position dictionary {node: (x, y)}
        step_limit: maximum steps
        scoring_weights: dict with keys 'degree', 'weight', 'homophily', 'dist_to_target'
                        If None, uses pure geometric distance (original behavior)
    """
    if (pos is None) or (s not in pos) or (t not in pos): 
        return False, [s], "no_geometry"
    
    # Default to pure geometric if no weights specified
    use_multi_cue = scoring_weights is not None and any(v > 0 for v in scoring_weights.values())
    
    path, visited = [s], {s}
    
    # Pre-compute node degrees and types for efficiency
    if use_multi_cue:
        node_degrees = dict(G.degree())
        node_in_degrees = dict(G.in_degree())
        node_out_degrees = dict(G.out_degree())
        node_types = {n: G.nodes[n].get('neuron_type', 'Unknown') for n in G.nodes()}
        target_type = node_types.get(t, 'Unknown')
        # Pre-compute max weight for normalization
        all_weights = [d.get('weight', 0) for u, v2, d in G.edges(data=True)]
        max_weight = max(all_weights) if all_weights else 1
    
    def compute_score(v):
        """Compute multi-cue score for neighbor v (lower is better)."""
        if not use_multi_cue:
            # Pure geometric distance (original behavior)
            return euclid(pos[v], pos[t]) if v in pos and t in pos else np.inf
        
        score = 0.0
        
        # 1. Distance to target (normalized, lower is better)
        if scoring_weights.get('dist_to_target', 0) > 0:
            if v in pos and t in pos:
                dist = euclid(pos[v], pos[t])
                # Normalize by max possible distance in layout (approximate)
                max_dist = 2.0  # spring_layout typically in [-1, 1] range
                normalized_dist = dist / max_dist if max_dist > 0 else dist
                score += scoring_weights['dist_to_target'] * normalized_dist
            else:
                score += scoring_weights['dist_to_target'] * 10.0  # Penalty for missing pos
        
        # 2. Degree (normalized, higher degree is better, so we use inverse)
        if scoring_weights.get('degree', 0) > 0:
            deg = node_degrees.get(v, 0)
            max_deg = max(node_degrees.values()) if node_degrees else 1
            # Inverse: prefer higher degree nodes (lower score)
            normalized_deg = 1.0 - (deg / max_deg) if max_deg > 0 else 1.0
            score += scoring_weights['degree'] * normalized_deg
        
        # 3. Edge weight (normalized, higher weight is better, so we use inverse)
        if scoring_weights.get('weight', 0) > 0:
            edge_weight = G.get_edge_data(path[-1], v, {}).get('weight', 0)
            # Inverse: prefer higher weight edges (lower score)
            normalized_weight = 1.0 - (edge_weight / max_weight) if max_weight > 0 else 1.0
            score += scoring_weights['weight'] * normalized_weight
        
        # 4. Homophily (prefer same type as target, lower score for same type)
        if scoring_weights.get('homophily', 0) > 0:
            v_type = node_types.get(v, 'Unknown')
            # Same type as target = 0 (preferred), different = 1
            homophily_score = 0.0 if v_type == target_type else 1.0
            score += scoring_weights['homophily'] * homophily_score
        
        return score
    
    while len(path) - 1 < step_limit:
        u = path[-1]
        if u == t: return True, path, None
        nbrs = [v for v in successors(G, u) if v not in visited]
        if not nbrs:
            return False, path, "dead_end"
        v = min(nbrs, key=compute_score)
        visited.add(v); path.append(v)

    return False, path, "step_limit"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V4", choices=["V1","V2", "V3", "V4", "V5"])
    ap.add_argument("--max_seeds", type=int, default=5)
    ap.add_argument("--max_targets", type=int, default=20)
    args = ap.parse_args()

    C = load_cfg(args.config)
    out_tbl = Path(C["report"]["tables_dir"]); out_tbl.mkdir(parents=True, exist_ok=True)

    G = load_graph(C["data"]["graphs"][args.variant])
    seeds = read_list(C["data"]["sensory_list"]) or \
            [n for n,_ in sorted(G.in_degree(), key=lambda x:x[1], reverse=True)[:args.max_seeds]] or \
            [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)[:args.max_seeds]]
    targets = read_list(C["data"]["motor_list"]) or \
              [n for n,_ in sorted(G.out_degree(), key=lambda x:x[1], reverse=True)[:args.max_targets]] or \
              [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)[:args.max_targets]]

    pos = get_pos(G)
    step_limit = max(1, G.number_of_nodes())
    
    # Load multi-cue scoring weights from config
    scoring_weights = None
    if "navigation" in C.get("analysis", {}) and "local_scoring_weights" in C["analysis"]["navigation"]:
        scoring_weights = C["analysis"]["navigation"]["local_scoring_weights"]
        # Normalize weights to sum to 1.0
        total = sum(scoring_weights.values())
        if total > 0:
            scoring_weights = {k: v / total for k, v in scoring_weights.items()}
        print(f"Using multi-cue navigation with weights: {scoring_weights}")

    rows = []
    for s in seeds:
        if s not in G: continue
        for t in targets:
            if t not in G or s == t: continue
            success, path, reason = greedy_route(G, s, t, pos, step_limit, scoring_weights)
            plen = max(0, len(path)-1)
            sp = sp_hops(G, s, t)
            stretch = (plen / sp) if success and np.isfinite(sp) and sp > 0 else (1.0 if success and sp == 0 else np.inf)
            rows.append({"seed":s, "target":t, "success":success, "path_len":plen, "sp_len":sp, "stretch":stretch, "failure_reason":None if success else reason})

    df = pd.DataFrame(rows)
    df.to_csv(out_tbl/"e3_nav_results.csv", index=False)

    # summary
    def med_stretch(x):
        vals = [v for v in x if np.isfinite(v)]
        return float(np.median(vals)) if vals else np.nan

    summary = pd.DataFrame([{
        "strategy":"greedy",
        "n_pairs": int(len(df)),
        "success_rate": float(df["success"].mean()) if len(df) else 0.0,
        "median_stretch": med_stretch(df["stretch"]) if len(df) else np.nan
    }])
    summary.to_csv(out_tbl/"e3_nav_summary.csv", index=False)
    
    # Compute geometric congruence and GRE
    print("\n=== Computing Geometric Congruence (GC) and Greedy Routing Efficiency (GRE) ===")
    
    # Prepare sample pairs for GC calculation
    sample_pairs = [(row["seed"], row["target"]) for _, row in df.iterrows()]
    
    # Compute GC (limit sample size if too many pairs)
    sample_size = min(1000, len(sample_pairs)) if len(sample_pairs) > 1000 else None
    gc_values, gc_mean, gc_std, gc_details = compute_geometrical_congruence(G, pos, node_pairs=sample_pairs, sample_size=sample_size)
    print(f"Geometrical Congruence (GC):")
    print(f"  Mean: {gc_mean:.4f}")
    print(f"  Median: {np.median(gc_values) if len(gc_values) > 0 else np.nan:.4f}")
    print(f"  Std: {gc_std:.4f}")
    print(f"  Valid pairs: {len(gc_values)}")
    
    # Compute GRE
    gre, success_rate, avg_stretch, normalized_stretch = compute_greedy_routing_efficiency(df)
    print(f"\nGreedy Routing Efficiency (GRE):")
    print(f"  GRE: {gre:.4f}")
    print(f"  Success rate: {success_rate:.4f}")
    print(f"  Mean stretch: {avg_stretch:.4f}")
    print(f"  Normalized stretch: {normalized_stretch:.4f}")
    
    # Save GC and GRE results
    gc_df = pd.DataFrame({
        'gc_mean': [gc_mean],
        'gc_std': [gc_std],
        'gc_median': [np.median(gc_values) if len(gc_values) > 0 else np.nan],
        'n_valid_pairs': [len(gc_values)]
    })
    gc_df.to_csv(out_tbl/"e3_gc_results.csv", index=False)
    
    gre_df = pd.DataFrame({
        'gre': [gre],
        'success_rate': [success_rate],
        'mean_stretch': [avg_stretch],
        'normalized_stretch': [normalized_stretch]
    })
    gre_df.to_csv(out_tbl/"e3_gre_results.csv", index=False)
    
    # Extended analysis: geometric-topology relationship
    print("\n=== Analyzing Geometric-Topology Relationship ===")
    sample_size_ext = min(1000, len(sample_pairs)) if len(sample_pairs) > 1000 else None
    extended_results = analyze_geometric_topology_relationship(G, pos, df, node_pairs=sample_pairs)
    
    if "distance_correlation" in extended_results:
        corr = extended_results["distance_correlation"]
        print(f"Geometric-Topological Distance Correlation:")
        print(f"  Spearman ρ: {corr.get('spearman_rho', np.nan):.4f} (p={corr.get('spearman_p', np.nan):.4f})")
        print(f"  Pearson r: {corr.get('pearson_r', np.nan):.4f} (p={corr.get('pearson_p', np.nan):.4f})")
        print(f"  Pairs analyzed: {corr.get('n_pairs', 0)}")
    
    # Save extended results
    import json
    with open(out_tbl/"e3_extended_analysis.json", "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        json.dump(convert_to_serializable(extended_results), f, indent=2)
    
    print("\nE3 greedy done ->", out_tbl/"e3_nav_results.csv", "|", out_tbl/"e3_nav_summary.csv")
    print("  Extended analysis ->", out_tbl/"e3_gc_results.csv", "|", out_tbl/"e3_gre_results.csv", "|", out_tbl/"e3_extended_analysis.json")

if __name__ == "__main__":
    main()