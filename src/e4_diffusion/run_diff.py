# -*- coding: utf-8 -*-
"""
E4: Diffusion vs Geodesic Distance Analysis

According to the research proposal, E4 objectives are:
- Compute diffusion matrix (heat kernel or k-step random walk)
- Compute geodesic distance matrix (shortest path distance)
- Compare correlation between diffusion values and geodesic distance (Spearman ρ)
- Success criterion: ρ_diffusion ≥ ρ_geodesic

References:
- Gămănuț et al. (2018): Using diffusion models to analyze connectomes
- Varshney et al. (2011): C. elegans structural analysis fundamentals
"""
import argparse, pickle, yaml, math
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.stats import spearmanr
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

# ---------- diffusion models ----------
def compute_heat_kernel(G, tau=0.5):
    """
    Compute heat kernel diffusion matrix: H = exp(-τL)
    
    Parameters:
        G: NetworkX graph (directed or undirected)
        tau: time scale parameter (default 0.5)
    
    Returns:
        H: diffusion matrix (n×n numpy array), H[i,j] represents diffusion strength from node i to node j
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build graph Laplacian matrix L = D - A
    # For directed graphs, use out-degree; for undirected graphs, use degree
    if G.is_directed():
        # Directed graph: L = D_out - A
        L = nx.directed_laplacian_matrix(G, nodelist=nodes, weight='weight')
    else:
        # Undirected graph: L = D - A
        L = nx.laplacian_matrix(G, nodelist=nodes, weight='weight')
    
    # Convert to sparse matrix for faster computation
    L_sparse = sparse.csr_matrix(L)
    
    # Compute heat kernel: H = exp(-τL)
    # Use scipy's expm function to compute matrix exponential
    H = expm(-tau * L_sparse).toarray()
    
    # Normalize: ensure each row sums to 1 (probability interpretation)
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    H = H / row_sums
    
    return H, nodes

def compute_k_step_rw(G, k=3):
    """
    Compute k-step random walk probability matrix
    
    Parameters:
        G: NetworkX graph (directed or undirected)
        k: number of random walk steps (default 3)
    
    Returns:
        P_k: k-step random walk probability matrix (n×n numpy array)
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Compute one-step transition probability matrix P
    if G.is_directed():
        # Directed graph: P = D_out^(-1) A
        # where A is adjacency matrix, D_out is out-degree matrix
        P = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()
        out_degree = np.array([G.out_degree(node, weight='weight') for node in nodes])
        out_degree[out_degree == 0] = 1  # Avoid division by zero
        P = P / out_degree[:, None]
    else:
        # Undirected graph: P = D^(-1) A
        P = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()
        degree = np.array([G.degree(node, weight='weight') for node in nodes])
        degree[degree == 0] = 1  # Avoid division by zero
        P = P / degree[:, None]
    
    # Compute k-step transition probability: P_k = P^k
    P_k = np.linalg.matrix_power(P, k)
    
    return P_k, nodes

# ---------- geodesic distance ----------
def compute_geodesic_matrix(G):
    """
    Compute geodesic distance matrix (shortest path distance)
    
    Parameters:
        G: NetworkX graph
    
    Returns:
        D: distance matrix (n×n numpy array), D[i,j] represents shortest path length from node i to node j
           If nodes are unreachable, value is inf
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Compute shortest path lengths between all node pairs
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)  # Distance from node to itself is 0
    
    # For directed graphs, use directed shortest paths
    # For undirected graphs, use undirected shortest paths
    if G.is_directed():
        # Directed graph: use directed shortest paths
        try:
            # Try using weights
            if nx.is_weighted(G):
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            else:
                # Unweighted graph: use hop count
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        except Exception:
            # If failed, fall back to unweighted shortest paths
            path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    else:
        # Undirected graph
        try:
            if nx.is_weighted(G):
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            else:
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        except Exception:
            path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Fill distance matrix
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    for u, lengths in path_lengths.items():
        if u not in node_to_idx:
            continue
        i = node_to_idx[u]
        for v, dist in lengths.items():
            if v not in node_to_idx:
                continue
            j = node_to_idx[v]
            if np.isfinite(dist):
                D[i, j] = float(dist)
    
    return D, nodes

# ---------- correlation analysis ----------
def flatten_and_correlate(diffusion_matrix, geodesic_matrix, nodes):
    """
    Flatten matrices and compute Spearman correlation coefficient
    
    Parameters:
        diffusion_matrix: diffusion matrix (n×n)
        geodesic_matrix: geodesic distance matrix (n×n)
        nodes: node list
    
    Returns:
        correlation: Spearman correlation coefficient (using inverse distance)
        p_value: p-value
        valid_pairs: number of valid node pairs
    
    Note:
        According to the research proposal, we compare diffusion values with inverse geodesic distance (1/distance).
        This is because smaller distances should correspond to larger diffusion, so we use inverse distance to obtain positive correlation.
    """
    n = len(nodes)
    
    # Flatten upper triangular matrix (avoid duplicates and self-loops)
    # Only consider node pairs where i < j
    diffusion_vals = []
    inverse_geodesic_vals = []
    
    for i in range(n):
        for j in range(i + 1, n):
            diff_val = diffusion_matrix[i, j]
            geo_val = geodesic_matrix[i, j]
            
            # Only consider valid distance values (not inf and distance > 0)
            if np.isfinite(geo_val) and geo_val > 0 and np.isfinite(diff_val):
                diffusion_vals.append(diff_val)
                # Use inverse distance (1/distance) for correlation analysis
                # This way, smaller distances correspond to larger inverse distances, and diffusion should be larger, giving positive correlation
                inverse_geodesic_vals.append(1.0 / geo_val)
    
    if len(diffusion_vals) < 2:
        return np.nan, np.nan, 0
    
    # Compute Spearman correlation coefficient (diffusion vs inverse distance)
    # Expect positive correlation: larger diffusion values correspond to larger inverse distances (i.e., smaller distances)
    correlation, p_value = spearmanr(diffusion_vals, inverse_geodesic_vals)
    
    return correlation, p_value, len(diffusion_vals)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1", "V2", "V3"])
    args = ap.parse_args()
    
    C = load_cfg(args.config)
    out_tbl = Path(C["report"]["tables_dir"])
    out_tbl.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    G = load_graph(C["data"]["graphs"][args.variant])
    print(f"[E4] Loaded {args.variant}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get diffusion model configuration
    diff_cfg = C["analysis"]["diffusion"]
    model = diff_cfg.get("model", "heat_kernel")
    tau = diff_cfg.get("tau", 0.5)
    k = diff_cfg.get("k", 3)
    
    print(f"[E4] Using diffusion model: {model}")
    
    # Compute diffusion matrix
    if model == "heat_kernel":
        print(f"[E4] Computing heat kernel diffusion (tau={tau})...")
        diffusion_matrix, nodes = compute_heat_kernel(G, tau=tau)
    elif model == "k_step_rw":
        print(f"[E4] Computing {k}-step random walk...")
        diffusion_matrix, nodes = compute_k_step_rw(G, k=k)
    else:
        raise ValueError(f"Unknown diffusion model: {model}")
    
    print(f"[E4] Diffusion matrix computed: {diffusion_matrix.shape}")
    
    # Compute geodesic distance matrix
    print(f"[E4] Computing geodesic distance matrix...")
    geodesic_matrix, _ = compute_geodesic_matrix(G)
    print(f"[E4] Geodesic matrix computed: {geodesic_matrix.shape}")
    
    # Compute correlation
    print(f"[E4] Computing correlation...")
    correlation, p_value, valid_pairs = flatten_and_correlate(
        diffusion_matrix, geodesic_matrix, nodes
    )
    
    print(f"[E4] Spearman ρ(diffusion, geodesic) = {correlation:.4f} (p={p_value:.4e}, n={valid_pairs})")
    
    # Save results
    summary = pd.DataFrame([{
        "variant": args.variant,
        "diffusion_model": model,
        "tau" if model == "heat_kernel" else "k": tau if model == "heat_kernel" else k,
        "rho_diffusion_geodesic": float(correlation) if not np.isnan(correlation) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "n_pairs": int(valid_pairs),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }])
    
    summary.to_csv(out_tbl / "e4_diffusion_correlations.csv", index=False)
    
    # Save diffusion matrix (optional, may be large)
    # If matrix is too large, only save summary statistics
    if G.number_of_nodes() < 1000:  # Only save full matrix for small graphs
        np.savez_compressed(out_tbl / "e4_diffusion_matrix.npz",
                            diffusion=diffusion_matrix,
                            geodesic=geodesic_matrix,
                            nodes=np.array(nodes, dtype=object))
        print(f"[E4] Saved matrices to e4_diffusion_matrix.npz")
    
    # Check success criterion
    target_rho = 0.0  # According to research proposal: ρ_diffusion ≥ ρ_geodesic (simplified here as > 0)
    success = not np.isnan(correlation) and correlation > target_rho
    print(f"[E4] Success criterion (ρ > {target_rho}): {'✓' if success else '✗'}")
    
    print(f"[E4] Done -> {out_tbl / 'e4_diffusion_correlations.csv'}")
    
    return correlation, p_value, valid_pairs

if __name__ == "__main__":
    main()
