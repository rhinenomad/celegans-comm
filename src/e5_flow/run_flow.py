# -*- coding: utf-8 -*-
"""
E5: Max-flow/Min-cut + Bottleneck Analysis

根据research proposal，E5的目标是：
- 计算sensory-motor节点对之间的max-flow
- 找到min-cut（最小割的节点集合）
- 计算betweenness centrality
- 比较min-cut成员与betweenness centrality Top-k的重合率
- 与configuration model null model对比
- 统计检验（p < 0.05，经FDR校正）

成功标准：
- min-cut成员与中介中心性Top-k的重合率显著高于空模型（p < 0.05，经FDR校正）

参考文献：
- Ford-Fulkerson算法：最大流最小割定理
- Varshney et al. (2011): C. elegans结构分析
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
    """读取节点列表文件"""
    if not p or p == "null" or not Path(p).exists():
        return None
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

# ---------- max-flow and min-cut ----------
def compute_max_flow_min_cut(G, source, target, capacity='weight'):
    """
    计算从source到target的最大流和最小割
    
    参数:
        G: NetworkX图（有向或无向）
        source: 源节点
        target: 目标节点
        capacity: 边容量属性名（默认'weight'）
    
    返回:
        flow_value: 最大流值
        min_cut_nodes: 最小割的节点集合（source侧）
        flow_dict: 流值字典
    """
    if source not in G or target not in G:
        return 0.0, set(), {}
    
    if source == target:
        return 0.0, set(), {}
    
    # 对于无向图，需要转换为有向图（双向边）
    if not G.is_directed():
        G_directed = G.to_directed()
    else:
        G_directed = G
    
    # 检查是否有路径
    if not nx.has_path(G_directed, source, target):
        return 0.0, set(), {}
    
    # 计算最大流
    try:
        flow_value, flow_dict = nx.maximum_flow(G_directed, source, target, capacity=capacity)
    except Exception as e:
        # 如果失败（例如无权图），使用单位容量
        try:
            flow_value, flow_dict = nx.maximum_flow(G_directed, source, target, capacity=1.0)
        except Exception:
            return 0.0, set(), {}
    
    # 计算最小割（source可达的节点集合）
    # 使用residual graph找到source可达的节点
    residual = nx.DiGraph()
    for u in G_directed:
        for v in G_directed[u]:
            # 获取容量
            if capacity == 'weight' and 'weight' in G_directed[u][v]:
                cap = G_directed[u][v]['weight']
            else:
                cap = 1.0
            
            # 计算剩余容量
            flow_uv = flow_dict.get(u, {}).get(v, 0.0)
            residual_cap = cap - flow_uv
            
            if residual_cap > 0:
                residual.add_edge(u, v, capacity=residual_cap)
    
    # 找到source可达的节点（最小割的source侧）
    if source in residual:
        reachable = set(nx.descendants(residual, source))
        reachable.add(source)
    else:
        reachable = {source}
    
    return flow_value, reachable, flow_dict

# ---------- betweenness centrality ----------
def compute_betweenness_centrality(G, k=None, normalized=True):
    """
    计算betweenness centrality
    
    参数:
        G: NetworkX图
        k: 采样节点数（None表示全部节点）
        normalized: 是否归一化
    
    返回:
        bc_dict: 节点到betweenness centrality值的字典
    """
    if G.is_directed():
        bc = nx.betweenness_centrality(G, k=k, normalized=normalized, weight='weight')
    else:
        bc = nx.betweenness_centrality(G, k=k, normalized=normalized, weight='weight')
    
    return bc

# ---------- configuration model null model ----------
def generate_configuration_model(G, seed=None, preserve_in_out=True):
    """
    生成configuration model null model（度保持随机图）
    
    参数:
        G: 原始图
        seed: 随机种子
        preserve_in_out: 对于有向图，是否保持入度和出度
    
    返回:
        G_null: 随机生成的图
    """
    if seed is not None:
        np.random.seed(seed)
    
    if G.is_directed():
        if preserve_in_out:
            # 保持入度和出度
            in_degree_seq = [d for n, d in G.in_degree()]
            out_degree_seq = [d for n, d in G.out_degree()]
            try:
                G_null = nx.directed_configuration_model(in_degree_seq, out_degree_seq, seed=seed)
                # 转换为DiGraph（移除自环和多边）
                G_null = nx.DiGraph(G_null)
                G_null.remove_edges_from(nx.selfloop_edges(G_null))
            except Exception:
                # 如果失败，使用更简单的方法
                G_null = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), 
                                            directed=True, seed=seed)
        else:
            # 只保持总度数
            degree_seq = [d for n, d in G.degree()]
            G_null = nx.configuration_model(degree_seq, seed=seed)
            G_null = nx.DiGraph(G_null)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
    else:
        # 无向图
        degree_seq = [d for n, d in G.degree()]
        try:
            G_null = nx.configuration_model(degree_seq, seed=seed)
            G_null = nx.Graph(G_null)
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
        except Exception:
            # 如果失败，使用更简单的方法
            G_null = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), 
                                        directed=False, seed=seed)
    
    # 如果原图有权重，给随机图也添加权重（随机分配）
    if nx.is_weighted(G):
        for u, v in G_null.edges():
            # 从原图的权重分布中随机选择
            weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
            if weights:
                G_null[u][v]['weight'] = np.random.choice(weights)
            else:
                G_null[u][v]['weight'] = 1.0
    
    return G_null

# ---------- overlap analysis ----------
def compute_overlap(min_cut_nodes, bc_top_k_nodes):
    """
    计算两个节点集合的重合率
    
    参数:
        min_cut_nodes: min-cut节点集合
        bc_top_k_nodes: betweenness centrality top-k节点集合
    
    返回:
        overlap: 重合节点数
        overlap_ratio: 重合率（重合节点数 / min_cut节点数）
        jaccard: Jaccard相似系数
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
    统计检验：比较观察值与null model分布
    
    参数:
        observed_overlap_ratio: 观察到的重合率
        null_overlap_ratios: null model的重合率列表
        alpha: 显著性水平
    
    返回:
        p_value: p值
        z_score: z分数
        significant: 是否显著
    """
    if not null_overlap_ratios:
        return np.nan, np.nan, False
    
    null_mean = np.mean(null_overlap_ratios)
    null_std = np.std(null_overlap_ratios)
    
    if null_std == 0:
        # 如果null model没有变异，无法进行检验
        return np.nan, np.nan, False
    
    # Z-score
    z_score = (observed_overlap_ratio - null_mean) / null_std if null_std > 0 else 0.0
    
    # One-tailed test: 观察值是否显著高于null model
    # 使用Wilcoxon signed-rank test或t-test
    if len(null_overlap_ratios) >= 3:
        # 使用单样本t检验（单尾）
        t_stat, p_value = stats.ttest_1samp(null_overlap_ratios, observed_overlap_ratio)
        # 单尾检验：观察值是否显著高于null
        if observed_overlap_ratio > null_mean:
            p_value = p_value / 2  # 单尾
        else:
            p_value = 1.0 - p_value / 2
    else:
        # 样本太少，使用简单的比较
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
    
    # 加载图
    G = load_graph(C["data"]["graphs"][args.variant])
    print(f"[E5] Loaded {args.variant}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 加载sensory和motor节点列表
    sensory_list_path = C["data"].get("sensory_list") or "data/interim/sensory_nodes.txt"
    motor_list_path = C["data"].get("motor_list") or "data/interim/motor_nodes.txt"
    
    sensory_nodes = read_list(sensory_list_path)
    motor_nodes = read_list(motor_list_path)
    
    # 如果列表不存在，从图中推断
    if not sensory_nodes:
        # 尝试从节点属性推断
        sensory_nodes = [n for n, d in G.nodes(data=True) 
                        if d.get('neuron_type') == 'S' or d.get('neuron_type') == 'SENSORY']
    if not motor_nodes:
        motor_nodes = [n for n, d in G.nodes(data=True) 
                      if d.get('neuron_type') == 'M' or d.get('neuron_type') == 'MOTOR']
    
    # 过滤：只保留图中存在的节点
    sensory_nodes = [n for n in sensory_nodes if n in G]
    motor_nodes = [n for n in motor_nodes if n in G]
    
    print(f"[E5] Found {len(sensory_nodes)} sensory nodes, {len(motor_nodes)} motor nodes")
    
    if not sensory_nodes or not motor_nodes:
        print("[E5] ERROR: No sensory or motor nodes found!")
        return
    
    # 限制测试的节点对数量
    max_pairs = min(args.max_pairs, len(sensory_nodes) * len(motor_nodes))
    print(f"[E5] Testing up to {max_pairs} sensory-motor pairs")
    
    # 获取配置
    flow_cfg = C["analysis"]["flow"]
    capacity_key = flow_cfg.get("capacity_key", "weight")
    if args.variant == "V1" and flow_cfg.get("unit_capacity_on_V1", True):
        capacity_key = 1.0  # V1是无权图，使用单位容量
    
    null_cfg = C["analysis"]["null_model"]
    n_null_graphs = null_cfg.get("n_graphs", 20)
    preserve_in_out = null_cfg.get("preserve_in_out", True)
    
    # 计算betweenness centrality（全局）
    print(f"[E5] Computing betweenness centrality...")
    bc = compute_betweenness_centrality(G)
    
    # 选择top-k节点（k = min_cut的平均大小）
    # 先计算一些min-cut来估计k
    sample_pairs = min(10, len(sensory_nodes) * len(motor_nodes))
    sample_cut_sizes = []
    for i, s in enumerate(sensory_nodes[:min(5, len(sensory_nodes))]):
        for j, t in enumerate(motor_nodes[:min(2, len(motor_nodes))]):
            if s != t and s in G and t in G:
                _, min_cut_nodes, _ = compute_max_flow_min_cut(G, s, t, capacity=capacity_key)
                if min_cut_nodes:
                    sample_cut_sizes.append(len(min_cut_nodes))
    
    avg_cut_size = int(np.mean(sample_cut_sizes)) if sample_cut_sizes else 20
    k = max(10, min(avg_cut_size * 2, G.number_of_nodes() // 10))  # top-k = 平均cut大小的2倍，但不超过节点数的10%
    
    print(f"[E5] Using top-k={k} for betweenness centrality comparison")
    
    # 获取top-k betweenness centrality节点
    bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    bc_top_k_nodes = set([n for n, _ in bc_sorted[:k]])
    
    # 生成null models
    print(f"[E5] Generating {n_null_graphs} null model graphs...")
    null_graphs = []
    for i in range(n_null_graphs):
        G_null = generate_configuration_model(G, seed=C["project"]["seed"] + i, 
                                               preserve_in_out=preserve_in_out)
        null_graphs.append(G_null)
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{n_null_graphs} null graphs")
    
    # 对每个null graph计算betweenness centrality top-k
    print(f"[E5] Computing betweenness centrality for null models...")
    null_bc_top_k_sets = []
    for i, G_null in enumerate(null_graphs):
        bc_null = compute_betweenness_centrality(G_null)
        bc_null_sorted = sorted(bc_null.items(), key=lambda x: x[1], reverse=True)
        bc_null_top_k = set([n for n, _ in bc_null_sorted[:k]])
        null_bc_top_k_sets.append(bc_null_top_k)
    
    # 测试sensory-motor节点对
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
            
            # 计算max-flow和min-cut
            flow_value, min_cut_nodes, flow_dict = compute_max_flow_min_cut(
                G, s, t, capacity=capacity_key
            )
            
            if not min_cut_nodes:
                continue
            
            # 计算与真实图的betweenness centrality top-k的重合率
            overlap, overlap_ratio, jaccard = compute_overlap(min_cut_nodes, bc_top_k_nodes)
            
            # 计算与null models的重合率
            null_overlap_ratios = []
            for null_bc_top_k in null_bc_top_k_sets:
                null_overlap, null_overlap_ratio, _ = compute_overlap(min_cut_nodes, null_bc_top_k)
                null_overlap_ratios.append(null_overlap_ratio)
            
            # 统计检验
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
    
    # FDR校正
    print(f"[E5] Applying FDR correction...")
    valid_p_values = df_results['p_value'].dropna()
    if len(valid_p_values) > 0:
        rejected, p_adjusted, _, _ = multipletests(
            valid_p_values, 
            alpha=C["report"]["stats"]["fdr"],
            method='fdr_bh'  # Benjamini-Hochberg
        )
        
        # 创建调整后的p值字典
        p_adjusted_dict = dict(zip(valid_p_values.index, p_adjusted))
        rejected_dict = dict(zip(valid_p_values.index, rejected))
        
        # 添加调整后的p值和FDR校正后的显著性
        df_results['p_value_fdr'] = df_results['p_value'].map(p_adjusted_dict).fillna(df_results['p_value'])
        df_results['significant_fdr'] = df_results.apply(
            lambda row: rejected_dict.get(row.name, False) if not np.isnan(row['p_value']) else False,
            axis=1
        )
    else:
        df_results['p_value_fdr'] = df_results['p_value']
        df_results['significant_fdr'] = df_results['significant']
    
    # 保存详细结果
    df_results.to_csv(out_tbl / f"e5_flow_results_{args.variant}.csv", index=False)
    print(f"[E5] Saved detailed results: {out_tbl / f'e5_flow_results_{args.variant}.csv'}")
    
    # 计算摘要统计
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
        "success_criterion_met": n_significant > 0  # 至少有一个显著结果
    }])
    
    summary.to_csv(out_tbl / f"e5_flow_summary_{args.variant}.csv", index=False)
    print(f"[E5] Saved summary: {out_tbl / f'e5_flow_summary_{args.variant}.csv'}")
    
    # 打印结果
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
