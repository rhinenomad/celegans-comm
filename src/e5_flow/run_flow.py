# -*- coding: utf-8 -*-
"""
E5: Max-flow/Min-cut + Bottleneck Analysis

According to the research proposal, E5 objectives are:
- Compute max-flow between sensory-motor node pairs
- Find min-cut (set of nodes in minimum cut)
- Compute betweenness centrality
- Compare overlap between min-cut members and betweenness centrality top-k
- Compare with configuration model null model
- Statistical testing (p < 0.05, FDR corrected)

Success criterion:
- Overlap between min-cut members and betweenness centrality top-k is significantly higher than null model (p < 0.05, FDR corrected)

References:
- Ford-Fulkerson algorithm: max-flow min-cut theorem
- Varshney et al. (2011): C. elegans structural analysis
"""
import argparse, pickle, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------- utilities ----------
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
    """Read node list file"""
    if not p or p == "null" or not Path(p).exists():
        return None
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

# ---------- max-flow and min-cut ----------
def compute_max_flow_min_cut(G, source, target, capacity='weight'):
    """
    Compute max-flow and min-cut from source to target
    
    Parameters:
        G: NetworkX graph (directed or undirected)
        source: source node
        target: target node
        capacity: edge capacity attribute name (default 'weight')
    
    Returns:
        flow_value: max-flow value
        min_cut_nodes: set of nodes in min-cut (source side)
        flow_dict: flow value dictionary
    """
    if source not in G or target not in G:
        return 0.0, set(), {}
    
    if source == target:
        return 0.0, set(), {}
    
    # For undirected graphs, convert to directed graph (bidirectional edges)
    if not G.is_directed():
        G_directed = G.to_directed()
    else:
        G_directed = G
    
    # Check if path exists
    if not nx.has_path(G_directed, source, target):
        return 0.0, set(), {}
    
    # Compute max-flow
    try:
        flow_value, flow_dict = nx.maximum_flow(G_directed, source, target, capacity=capacity)
    except Exception as e:
        # If failed (e.g., unweighted graph), use unit capacity
        try:
            flow_value, flow_dict = nx.maximum_flow(G_directed, source, target, capacity=1.0)
        except Exception:
            return 0.0, set(), {}
    
    # Compute min-cut (set of nodes reachable from source)
    # Use residual graph to find nodes reachable from source
    residual = nx.DiGraph()
    for u in G_directed:
        for v in G_directed[u]:
            # Get capacity
            if capacity == 'weight' and 'weight' in G_directed[u][v]:
                cap = G_directed[u][v]['weight']
            else:
                cap = 1.0
            
            # Compute residual capacity
            flow_uv = flow_dict.get(u, {}).get(v, 0.0)
            residual_cap = cap - flow_uv
            
            if residual_cap > 0:
                residual.add_edge(u, v, capacity=residual_cap)
    
    # Find nodes reachable from source (source side of min-cut)
    if source in residual:
        reachable = set(nx.descendants(residual, source))
        reachable.add(source)
    else:
        reachable = {source}
    
    return flow_value, reachable, flow_dict

# ---------- betweenness centrality ----------
def compute_betweenness_centrality(G, k=None, normalized=True):
    """
    Compute betweenness centrality
    
    Parameters:
        G: NetworkX graph
        k: number of nodes to sample (None means all nodes)
        normalized: whether to normalize
    
    Returns:
        bc_dict: dictionary mapping nodes to betweenness centrality values
    """
    if G.is_directed():
        bc = nx.betweenness_centrality(G, k=k, normalized=normalized, weight='weight')
    else:
        bc = nx.betweenness_centrality(G, k=k, normalized=normalized, weight='weight')
    
    return bc

# ---------- configuration model null model ----------
def generate_configuration_model(G, seed=None, preserve_in_out=True):
    """
    Generate configuration model null model (degree-preserving random graph)
    
    Parameters:
        G: original graph
        seed: random seed
        preserve_in_out: for directed graphs, whether to preserve in-degree and out-degree
    
    Returns:
        G_null: randomly generated graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    if G.is_directed():
        if preserve_in_out:
            # Preserve in-degree and out-degree
            in_degree_seq = [d for n, d in G.in_degree()]
            out_degree_seq = [d for n, d in G.out_degree()]
            try:
                G_null = nx.directed_configuration_model(in_degree_seq, out_degree_seq, seed=seed)
                # Convert to DiGraph (remove self-loops and multi-edges)
                G_null = nx.DiGraph(G_null)
                G_null.remove_edges_from(nx.selfloop_edges(G_null))
            except Exception:
                # If failed, use simpler method
                G_null = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), 
                                            directed=True, seed=seed)
        else:
            # Only preserve total degree
            degree_seq = [d for n, d in G.degree()]
            G_null = nx.configuration_model(degree_seq, seed=seed)
            G_null = nx.DiGraph(G_null)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
    else:
        # Undirected graph
        degree_seq = [d for n, d in G.degree()]
        try:
            G_null = nx.configuration_model(degree_seq, seed=seed)
            G_null = nx.Graph(G_null)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
        except Exception:
            # If failed, use simpler method
            G_null = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), 
                                        directed=False, seed=seed)
    
    # If original graph has weights, add weights to random graph (random assignment)
    if nx.is_weighted(G):
        for u, v in G_null.edges():
            # Randomly select from original graph's weight distribution
            weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
            if weights:
                G_null[u][v]['weight'] = np.random.choice(weights)
            else:
                G_null[u][v]['weight'] = 1.0
    
    return G_null

# ---------- overlap analysis ----------
def compute_overlap(min_cut_nodes, bc_top_k_nodes):
    """
    Compute overlap between two node sets
    
    Parameters:
        min_cut_nodes: min-cut node set
        bc_top_k_nodes: betweenness centrality top-k node set
    
    Returns:
        overlap: number of overlapping nodes
        overlap_ratio: overlap ratio (overlapping nodes / min_cut nodes)
        jaccard: Jaccard similarity coefficient
    """
    if not min_cut_nodes:
        return 0, 0.0, 0.0
    
    overlap = len(min_cut_nodes & bc_top_k_nodes)
    overlap_ratio = overlap / len(min_cut_nodes) if min_cut_nodes else 0.0
    
    union = len(min_cut_nodes | bc_top_k_nodes)
    jaccard = overlap / union if union > 0 else 0.0
    
    return overlap, overlap_ratio, jaccard

# ---------- statistical testing ----------
def test_significance(observed_overlap_ratio, null_overlap_ratios, alpha=0.05):
    """
    Statistical testing: compare observed value with null model distribution
    
    Parameters:
        observed_overlap_ratio: observed overlap ratio
        null_overlap_ratios: list of null model overlap ratios
        alpha: significance level
    
    Returns:
        p_value: p-value
        z_score: z-score
        significant: whether significant
    """
    if not null_overlap_ratios:
        return np.nan, np.nan, False
    
    null_mean = np.mean(null_overlap_ratios)
    null_std = np.std(null_overlap_ratios)
    
    if null_std == 0:
        # If null model has no variation, cannot perform test
        return np.nan, np.nan, False
    
    # Z-score
    z_score = (observed_overlap_ratio - null_mean) / null_std if null_std > 0 else 0.0
    
    # One-tailed test: whether observed value is significantly higher than null model
    # Use Wilcoxon signed-rank test or t-test
    if len(null_overlap_ratios) >= 3:
        # Use one-sample t-test (one-tailed)
        t_stat, p_value = stats.ttest_1samp(null_overlap_ratios, observed_overlap_ratio)
        # One-tailed test: whether observed value is significantly higher than null
        if observed_overlap_ratio > null_mean:
            p_value = p_value / 2  # One-tailed
        else:
            p_value = 1.0 - p_value / 2
    else:
        # Sample size too small, use simple comparison
        p_value = np.sum(np.array(null_overlap_ratios) >= observed_overlap_ratio) / len(null_overlap_ratios)
    
    significant = p_value < alpha
    
    return p_value, z_score, significant

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"])
    ap.add_argument("--max_pairs", type=int, default=100, 
                    help="Maximum number of sensory-motor pairs to test")
    args = ap.parse_args()
    
    C = load_cfg(args.config)
    out_tbl = Path(C["report"]["tables_dir"])
    out_tbl.mkdir(parents=True, exist_ok=True)
    out_fig = Path(C["report"]["figures_dir"])
    out_fig.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    G = load_graph(C["data"]["graphs"][args.variant])
    print(f"[E5] Loaded {args.variant}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load sensory and motor node lists
    sensory_list_path = C["data"].get("sensory_list") or "data/interim/sensory_nodes.txt"
    motor_list_path = C["data"].get("motor_list") or "data/interim/motor_nodes.txt"
    
    sensory_nodes = read_list(sensory_list_path)
    motor_nodes = read_list(motor_list_path)
    
    # If lists don't exist, infer from graph
    if not sensory_nodes:
        # Try to infer from node attributes
        sensory_nodes = [n for n, d in G.nodes(data=True) 
                        if d.get('neuron_type') == 'S' or d.get('neuron_type') == 'SENSORY']
    if not motor_nodes:
        motor_nodes = [n for n, d in G.nodes(data=True) 
                      if d.get('neuron_type') == 'M' or d.get('neuron_type') == 'MOTOR']
    
    # Filter: only keep nodes that exist in graph
    sensory_nodes = [n for n in sensory_nodes if n in G]
    motor_nodes = [n for n in motor_nodes if n in G]
    
    print(f"[E5] Found {len(sensory_nodes)} sensory nodes, {len(motor_nodes)} motor nodes")
    
    if not sensory_nodes or not motor_nodes:
        print("[E5] ERROR: No sensory or motor nodes found!")
        return
    
    # Limit number of node pairs to test
    max_pairs = min(args.max_pairs, len(sensory_nodes) * len(motor_nodes))
    print(f"[E5] Testing up to {max_pairs} sensory-motor pairs")
    
    # Get configuration
    flow_cfg = C["analysis"]["flow"]
    capacity_key = flow_cfg.get("capacity_key", "weight")
    if args.variant == "V1" and flow_cfg.get("unit_capacity_on_V1", True):
        capacity_key = 1.0  # V1 is unweighted graph, use unit capacity
    
    null_cfg = C["analysis"]["null_model"]
    n_null_graphs = null_cfg.get("n_graphs", 20)
    preserve_in_out = null_cfg.get("preserve_in_out", True)
    
    # Compute betweenness centrality (global)
    print(f"[E5] Computing betweenness centrality...")
    bc = compute_betweenness_centrality(G)
    
    # Select top-k nodes (k = average min_cut size)
    # First compute some min-cuts to estimate k
    sample_pairs = min(10, len(sensory_nodes) * len(motor_nodes))
    sample_cut_sizes = []
    for i, s in enumerate(sensory_nodes[:min(5, len(sensory_nodes))]):
        for j, t in enumerate(motor_nodes[:min(2, len(motor_nodes))]):
            if s != t and s in G and t in G:
                _, min_cut_nodes, _ = compute_max_flow_min_cut(G, s, t, capacity=capacity_key)
                if min_cut_nodes:
                    sample_cut_sizes.append(len(min_cut_nodes))
    
    avg_cut_size = int(np.mean(sample_cut_sizes)) if sample_cut_sizes else 20
    k = max(10, min(avg_cut_size * 2, G.number_of_nodes() // 10))  # top-k = 2x average cut size, but not more than 10% of nodes
    
    print(f"[E5] Using top-k={k} for betweenness centrality comparison")
    
    # Get top-k betweenness centrality nodes
    bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    bc_top_k_nodes = set([n for n, _ in bc_sorted[:k]])
    
    # Generate null models
    print(f"[E5] Generating {n_null_graphs} null model graphs...")
    null_graphs = []
    for i in range(n_null_graphs):
        G_null = generate_configuration_model(G, seed=C["project"]["seed"] + i, 
                                               preserve_in_out=preserve_in_out)
        null_graphs.append(G_null)
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{n_null_graphs} null graphs")
    
    # Compute betweenness centrality top-k for each null graph
    print(f"[E5] Computing betweenness centrality for null models...")
    null_bc_top_k_sets = []
    for i, G_null in enumerate(null_graphs):
        bc_null = compute_betweenness_centrality(G_null)
        bc_null_sorted = sorted(bc_null.items(), key=lambda x: x[1], reverse=True)
        bc_null_top_k = set([n for n, _ in bc_null_sorted[:k]])
        null_bc_top_k_sets.append(bc_null_top_k)
    
    # Test sensory-motor node pairs
    print(f"[E5] Computing max-flow/min-cut for sensory-motor pairs...")
    results = []
    
    pair_count = 0
    for s in sensory_nodes:
        if pair_count >= max_pairs:
            break
        for t in motor_nodes:
            if pair_count >= max_pairs:
                break
            if s == t or s not in G or t not in G:
                continue
            
            # Compute max-flow and min-cut
            flow_value, min_cut_nodes, flow_dict = compute_max_flow_min_cut(
                G, s, t, capacity=capacity_key
            )
            
            if not min_cut_nodes:
                continue
            
            # Compute overlap with real graph's betweenness centrality top-k
            overlap, overlap_ratio, jaccard = compute_overlap(min_cut_nodes, bc_top_k_nodes)
            
            # Compute overlap with null models
            null_overlap_ratios = []
            for null_bc_top_k in null_bc_top_k_sets:
                null_overlap, null_overlap_ratio, _ = compute_overlap(min_cut_nodes, null_bc_top_k)
                null_overlap_ratios.append(null_overlap_ratio)
            
            # Statistical testing
            p_value, z_score, significant = test_significance(
                overlap_ratio, null_overlap_ratios, alpha=C["report"]["stats"]["fdr"]
            )
            
            results.append({
                "source": s,
                "target": t,
                "flow_value": float(flow_value),
                "min_cut_size": len(min_cut_nodes),
                "min_cut_nodes": ",".join(sorted(min_cut_nodes)),
                "overlap_with_bc_topk": overlap,
                "overlap_ratio": float(overlap_ratio),
                "jaccard": float(jaccard),
                "null_mean_overlap_ratio": float(np.mean(null_overlap_ratios)),
                "null_std_overlap_ratio": float(np.std(null_overlap_ratios)),
                "z_score": float(z_score) if not np.isnan(z_score) else np.nan,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                "significant": significant
            })
            
            pair_count += 1
            if pair_count % 10 == 0:
                print(f"  Processed {pair_count} pairs...")
    
    if not results:
        print("[E5] ERROR: No valid results computed!")
        return
    
    df_results = pd.DataFrame(results)
    
    # FDR correction
    print(f"[E5] Applying FDR correction...")
    valid_p_values = df_results['p_value'].dropna()
    if len(valid_p_values) > 0:
        rejected, p_adjusted, _, _ = multipletests(
            valid_p_values, 
            alpha=C["report"]["stats"]["fdr"],
            method='fdr_bh'  # Benjamini-Hochberg
        )
        
        # Create adjusted p-value dictionary
        p_adjusted_dict = dict(zip(valid_p_values.index, p_adjusted))
        rejected_dict = dict(zip(valid_p_values.index, rejected))
        
        # Add adjusted p-values and FDR-corrected significance
        df_results['p_value_fdr'] = df_results['p_value'].map(p_adjusted_dict).fillna(df_results['p_value'])
        df_results['significant_fdr'] = df_results.apply(
            lambda row: rejected_dict.get(row.name, False) if not np.isnan(row['p_value']) else False,
            axis=1
        )
    else:
        df_results['p_value_fdr'] = df_results['p_value']
        df_results['significant_fdr'] = df_results['significant']
    
    # Save detailed results (compatible with makefile expected filenames)
    df_results.to_csv(out_tbl / f"e5_flow_results_{args.variant}.csv", index=False)
    # Also save with makefile-expected names for backward compatibility
    df_results[["source", "target", "flow_value"]].to_csv(out_tbl / "e5_flow_values.csv", index=False)
    df_results[["source", "target", "min_cut_nodes"]].to_csv(out_tbl / "e5_mincut_edges.csv", index=False)
    df_results[["source", "target", "overlap_ratio", "null_mean_overlap_ratio", "p_value_fdr", "significant_fdr"]].to_csv(out_tbl / "e5_flow_ablation_by_mincut.csv", index=False)
    print(f"[E5] Saved detailed results: {out_tbl / f'e5_flow_results_{args.variant}.csv'}")
    
    # Compute summary statistics
    n_pairs = len(df_results)
    n_significant = df_results['significant_fdr'].sum()
    mean_overlap_ratio = df_results['overlap_ratio'].mean()
    mean_null_overlap_ratio = df_results['null_mean_overlap_ratio'].mean()
    mean_flow_value = df_results['flow_value'].mean()
    mean_cut_size = df_results['min_cut_size'].mean()
    
    summary = pd.DataFrame([{
        "variant": args.variant,
        "n_pairs_tested": n_pairs,
        "n_significant_fdr": int(n_significant),
        "significance_rate": float(n_significant / n_pairs) if n_pairs > 0 else 0.0,
        "mean_overlap_ratio": float(mean_overlap_ratio),
        "mean_null_overlap_ratio": float(mean_null_overlap_ratio),
        "mean_flow_value": float(mean_flow_value),
        "mean_min_cut_size": float(mean_cut_size),
        "k_betweenness": k,
        "n_null_graphs": n_null_graphs,
        "fdr_alpha": C["report"]["stats"]["fdr"],
        "success_criterion_met": n_significant > 0  # At least one significant result
    }])
    
    summary.to_csv(out_tbl / f"e5_flow_summary_{args.variant}.csv", index=False)
    print(f"[E5] Saved summary: {out_tbl / f'e5_flow_summary_{args.variant}.csv'}")
    
    # Print results
    print(f"\n[E5] Results Summary:")
    print(f"  Pairs tested: {n_pairs}")
    print(f"  Significant (FDR corrected): {n_significant} ({n_significant/n_pairs*100:.1f}%)")
    print(f"  Mean overlap ratio: {mean_overlap_ratio:.4f}")
    print(f"  Mean null overlap ratio: {mean_null_overlap_ratio:.4f}")
    print(f"  Mean flow value: {mean_flow_value:.4f}")
    print(f"  Mean min-cut size: {mean_cut_size:.1f}")
    print(f"  Success criterion met: {'✓' if n_significant > 0 else '✗'}")
    
    print(f"\n[E5] Done!")

if __name__ == "__main__":
    main()
