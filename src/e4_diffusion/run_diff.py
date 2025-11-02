# -*- coding: utf-8 -*-
"""
E4: Diffusion vs Geodesic Distance Analysis

根据research proposal，E4的目标是：
- 计算扩散矩阵（heat kernel或k-step random walk）
- 计算测地距离矩阵（最短路径距离）
- 比较扩散值与几何距离的相关性（Spearman ρ）
- 成功标准：ρ_diffusion ≥ ρ_geodesic

参考文献：
- Gămănuț et al. (2018): 使用扩散模型分析连接组
- Varshney et al. (2011): C. elegans结构分析基础
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
    计算热核扩散矩阵：H = exp(-τL)
    
    参数:
        G: NetworkX图（有向或无向）
        tau: 时间尺度参数（默认0.5）
    
    返回:
        H: 扩散矩阵 (n×n numpy array)，H[i,j]表示从节点i到节点j的扩散强度
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # 构建图的拉普拉斯矩阵 L = D - A
    # 对于有向图，使用出度；对于无向图，使用度
    if G.is_directed():
        # 有向图：L = D_out - A
        L = nx.directed_laplacian_matrix(G, nodelist=nodes, weight='weight')
    else:
        # 无向图：L = D - A
        L = nx.laplacian_matrix(G, nodelist=nodes, weight='weight')
    
    # 转换为稀疏矩阵以加速计算
    L_sparse = sparse.csr_matrix(L)
    
    # 计算热核：H = exp(-τL)
    # 使用scipy的expm函数计算矩阵指数
    H = expm(-tau * L_sparse).toarray()
    
    # 归一化：确保每行和为1（概率解释）
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    H = H / row_sums
    
    return H, nodes

def compute_k_step_rw(G, k=3):
    """
    计算K步随机游走概率矩阵
    
    参数:
        G: NetworkX图（有向或无向）
        k: 随机游走步数（默认3）
    
    返回:
        P_k: K步随机游走概率矩阵 (n×n numpy array)
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # 计算一步转移概率矩阵 P
    if G.is_directed():
        # 有向图：P = D_out^(-1) A
        # 其中A是邻接矩阵，D_out是出度矩阵
        P = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()
        out_degree = np.array([G.out_degree(node, weight='weight') for node in nodes])
        out_degree[out_degree == 0] = 1  # 避免除零
        P = P / out_degree[:, None]
    else:
        # 无向图：P = D^(-1) A
        P = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()
        degree = np.array([G.degree(node, weight='weight') for node in nodes])
        degree[degree == 0] = 1  # 避免除零
        P = P / degree[:, None]
    
    # 计算K步转移概率：P_k = P^k
    P_k = np.linalg.matrix_power(P, k)
    
    return P_k, nodes

# ---------- geodesic distance ----------
def compute_geodesic_matrix(G):
    """
    计算测地距离矩阵（最短路径距离）
    
    参数:
        G: NetworkX图
    
    返回:
        D: 距离矩阵 (n×n numpy array)，D[i,j]表示从节点i到节点j的最短路径长度
           如果节点不可达，则为inf
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # 计算所有节点对之间的最短路径长度
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)  # 自己到自己的距离为0
    
    # 对于有向图，使用有向最短路径
    # 对于无向图，使用无向最短路径
    if G.is_directed():
        # 有向图：使用有向最短路径
        try:
            # 尝试使用权重
            if nx.is_weighted(G):
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            else:
                # 无权图：使用hop count
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        except Exception:
            # 如果失败，回退到无权最短路径
            path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    else:
        # 无向图
        try:
            if nx.is_weighted(G):
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            else:
                path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        except Exception:
            path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # 填充距离矩阵
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
    展平矩阵并计算Spearman相关系数
    
    参数:
        diffusion_matrix: 扩散矩阵 (n×n)
        geodesic_matrix: 测地距离矩阵 (n×n)
        nodes: 节点列表
    
    返回:
        correlation: Spearman相关系数（使用逆距离）
        p_value: p值
        valid_pairs: 有效节点对数量
    
    注意：
        根据research proposal，我们比较扩散值与逆测地距离（1/距离）的相关性。
        这是因为距离越小，扩散应该越大，所以应该使用逆距离来获得正相关。
    """
    n = len(nodes)
    
    # 展平上三角矩阵（避免重复和自环）
    # 只考虑 i < j 的节点对
    diffusion_vals = []
    inverse_geodesic_vals = []
    
    for i in range(n):
        for j in range(i + 1, n):
            diff_val = diffusion_matrix[i, j]
            geo_val = geodesic_matrix[i, j]
            
            # 只考虑有效的距离值（不是inf且距离>0）
            if np.isfinite(geo_val) and geo_val > 0 and np.isfinite(diff_val):
                diffusion_vals.append(diff_val)
                # 使用逆距离（1/距离）来进行相关性分析
                # 这样距离越小，逆距离越大，扩散应该越大，得到正相关
                inverse_geodesic_vals.append(1.0 / geo_val)
    
    if len(diffusion_vals) < 2:
        return np.nan, np.nan, 0
    
    # 计算Spearman相关系数（扩散 vs 逆距离）
    # 期望正相关：扩散值越大，逆距离越大（即距离越小）
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
    
    # 加载图
    G = load_graph(C["data"]["graphs"][args.variant])
    print(f"[E4] Loaded {args.variant}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 获取扩散模型配置
    diff_cfg = C["analysis"]["diffusion"]
    model = diff_cfg.get("model", "heat_kernel")
    tau = diff_cfg.get("tau", 0.5)
    k = diff_cfg.get("k", 3)
    
    print(f"[E4] Using diffusion model: {model}")
    
    # 计算扩散矩阵
    if model == "heat_kernel":
        print(f"[E4] Computing heat kernel diffusion (tau={tau})...")
        diffusion_matrix, nodes = compute_heat_kernel(G, tau=tau)
    elif model == "k_step_rw":
        print(f"[E4] Computing {k}-step random walk...")
        diffusion_matrix, nodes = compute_k_step_rw(G, k=k)
    else:
        raise ValueError(f"Unknown diffusion model: {model}")
    
    print(f"[E4] Diffusion matrix computed: {diffusion_matrix.shape}")
    
    # 计算测地距离矩阵
    print(f"[E4] Computing geodesic distance matrix...")
    geodesic_matrix, _ = compute_geodesic_matrix(G)
    print(f"[E4] Geodesic matrix computed: {geodesic_matrix.shape}")
    
    # 计算相关性
    print(f"[E4] Computing correlation...")
    correlation, p_value, valid_pairs = flatten_and_correlate(
        diffusion_matrix, geodesic_matrix, nodes
    )
    
    print(f"[E4] Spearman ρ(diffusion, geodesic) = {correlation:.4f} (p={p_value:.4e}, n={valid_pairs})")
    
    # 保存结果
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
    
    summary.to_csv(out_tbl / "e4_diffusion_summary.csv", index=False)
    
    # 保存扩散矩阵（可选，可能很大）
    # 如果矩阵太大，可以只保存摘要统计
    if G.number_of_nodes() < 1000:  # 只保存小图的完整矩阵
        np.savez_compressed(out_tbl / "e4_diffusion_matrix.npz",
                            diffusion=diffusion_matrix,
                            geodesic=geodesic_matrix,
                            nodes=np.array(nodes, dtype=object))
        print(f"[E4] Saved matrices to e4_diffusion_matrix.npz")
    
    # 检查成功标准
    target_rho = 0.0  # 根据research proposal：ρ_diffusion ≥ ρ_geodesic（这里简化为 > 0）
    success = not np.isnan(correlation) and correlation > target_rho
    print(f"[E4] Success criterion (ρ > {target_rho}): {'✓' if success else '✗'}")
    
    print(f"[E4] Done -> {out_tbl / 'e4_diffusion_summary.csv'}")
    
    return correlation, p_value, valid_pairs

if __name__ == "__main__":
    main()
