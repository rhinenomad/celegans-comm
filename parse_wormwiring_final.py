#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 WormWiring Excel 文件中提取神经元连接数据，构建图结构
- 只考虑雌雄同体（hermaphrodite）
- 化学突触（chemical）和 gap junction 非对称（asymmetric）
- 只考虑 I（interneuron）、S（sensory）、M（motor）神经元

工作表结构：
- 行1: 类型分组标签（列标题位置）
- 行3: 神经元名称（列标题，从第4列开始）
- 列1: PHARYNX等非神经元
- 列3: 神经元名称（行标题，从第4行开始）
- 数据区域：从第4行第4列开始
"""
import openpyxl
import networkx as nx
from pathlib import Path
import pickle
import pandas as pd

def find_type_boundaries(ws):
    """找到类型分组的边界（列和行）"""
    # 查找第1行中的类型分组列索引
    type_cols = {}  # {列索引: 类型名称}
    for j in range(1, ws.max_column + 1):
        val = ws.cell(1, j).value
        if val is not None:
            val_str = str(val).strip().upper()
            if 'SENSORY NEURONS' in val_str:
                type_cols[j] = 'S'
            elif 'INTERNEURONS' in val_str or ('INTER' in val_str and 'NEURON' in val_str):
                type_cols[j] = 'I'
            elif 'MOTOR NEURONS' in val_str:
                type_cols[j] = 'M'
    
    # 查找第1列中的类型分组行索引
    type_rows = {}  # {行索引: 类型名称}
    for i in range(1, ws.max_row + 1):
        val = ws.cell(i, 1).value
        if val is not None:
            val_str = str(val).strip().upper()
            if 'SENSORY NEURONS' in val_str:
                type_rows[i] = 'S'
            elif 'INTERNEURONS' in val_str or ('INTER' in val_str and 'NEURON' in val_str):
                type_rows[i] = 'I'
            elif 'MOTOR NEURONS' in val_str:
                type_rows[i] = 'M'
    
    return type_cols, type_rows

def extract_neurons(ws):
    """提取神经元列表及其类型"""
    # 找到类型分组边界
    type_cols, type_rows = find_type_boundaries(ws)
    
    # 提取列标题（第3行，从第4列开始）- 目标神经元
    neuron_cols = {}  # {神经元名称: (列索引, 类型)}
    
    # 确定列的类型分组边界
    col_boundaries = sorted(type_cols.items())
    
    for j in range(4, ws.max_column + 1):
        # 确定当前列属于哪个类型
        neuron_type = None
        for idx, (col_idx, ntype) in enumerate(col_boundaries):
            if j >= col_idx:
                neuron_type = ntype
            if idx + 1 < len(col_boundaries) and j < col_boundaries[idx + 1][0]:
                break
        
        if neuron_type is None:
            continue
        
        val = ws.cell(3, j).value  # 第3行是列标题
        if val is not None:
            neuron_name = str(val).strip()
            # 排除明显的非神经元名称
            if neuron_name and len(neuron_name) < 20:
                neuron_cols[neuron_name] = (j, neuron_type)
    
    # 提取行标题（第3列，从第4行开始）- 源神经元
    neuron_rows = {}  # {神经元名称: (行索引, 类型)}
    
    # 确定行的类型分组边界
    row_boundaries = sorted(type_rows.items())
    
    for i in range(4, ws.max_row + 1):
        # 确定当前行属于哪个类型
        neuron_type = None
        for idx, (row_idx, ntype) in enumerate(row_boundaries):
            if i >= row_idx:
                neuron_type = ntype
            if idx + 1 < len(row_boundaries) and i < row_boundaries[idx + 1][0]:
                break
        
        if neuron_type is None:
            continue
        
        val = ws.cell(i, 3).value  # 第3列是行标题
        if val is not None:
            neuron_name = str(val).strip()
            # 排除明显的非神经元名称
            if neuron_name and len(neuron_name) < 20:
                neuron_rows[neuron_name] = (i, neuron_type)
    
    return neuron_rows, neuron_cols

def build_graph(excel_path):
    """构建图"""
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    
    # 找到工作表
    ws_chemical = None
    ws_gap = None
    for name in wb.sheetnames:
        if 'hermaphrodite' in name.lower() and 'chemical' in name.lower():
            ws_chemical = wb[name]
        if 'hermaphrodite' in name.lower() and 'gap' in name.lower() and 'asymmetric' in name.lower():
            ws_gap = wb[name]
    
    if ws_chemical is None or ws_gap is None:
        raise ValueError(f"找不到所需的工作表")
    
    print("=" * 80)
    print("解析 WormWiring Excel 文件")
    print("=" * 80)
    print(f"使用工作表: {ws_chemical.title}, {ws_gap.title}")
    
    # 提取神经元列表
    print("\n1. 提取神经元列表...")
    neurons_rows_chem, neurons_cols_chem = extract_neurons(ws_chemical)
    neurons_rows_gap, neurons_cols_gap = extract_neurons(ws_gap)
    
    # 合并神经元列表（只保留I/S/M类型）
    all_neurons = {}
    for name, (row_idx, ntype) in neurons_rows_chem.items():
        if ntype in ['S', 'I', 'M']:
            all_neurons[name] = ntype
    for name, (col_idx, ntype) in neurons_cols_chem.items():
        if ntype in ['S', 'I', 'M']:
            if name not in all_neurons:
                all_neurons[name] = ntype
    
    print(f"  找到 {len(all_neurons)} 个神经元 (I/S/M类型)")
    print(f"  S={sum(1 for t in all_neurons.values() if t=='S')}, "
          f"I={sum(1 for t in all_neurons.values() if t=='I')}, "
          f"M={sum(1 for t in all_neurons.values() if t=='M')}")
    
    # 创建图
    G = nx.DiGraph()
    for name, ntype in all_neurons.items():
        G.add_node(name, neuron_type=ntype)
    
    # 从化学突触表添加边
    print("\n2. 添加化学突触边...")
    chemical_count = 0
    for source_name, (source_row, _) in neurons_rows_chem.items():
        if source_name not in all_neurons:
            continue
        for target_name, (target_col, _) in neurons_cols_chem.items():
            if target_name not in all_neurons:
                continue
            
            weight = ws_chemical.cell(source_row, target_col).value
            if weight is not None:
                try:
                    w = float(weight)
                    if w > 0:
                        G.add_edge(source_name, target_name,
                                 chemical_weight=w, edge_type='chemical')
                        chemical_count += 1
                except (ValueError, TypeError):
                    pass
    
    print(f"  添加了 {chemical_count} 条化学突触边")
    
    # 从gap非对称表添加边
    print("\n3. 添加 gap junction 边...")
    gap_count = 0
    both_count = 0
    for source_name, (source_row, _) in neurons_rows_gap.items():
        if source_name not in all_neurons:
            continue
        for target_name, (target_col, _) in neurons_cols_gap.items():
            if target_name not in all_neurons:
                continue
            
            weight = ws_gap.cell(source_row, target_col).value
            if weight is not None:
                try:
                    w = float(weight)
                    if w > 0:
                        if G.has_edge(source_name, target_name):
                            G[source_name][target_name]['gap_weight'] = w
                            G[source_name][target_name]['edge_type'] = 'both'
                            both_count += 1
                        else:
                            G.add_edge(source_name, target_name,
                                     gap_weight=w, edge_type='gap')
                            gap_count += 1
                except (ValueError, TypeError):
                    pass
    
    print(f"  添加了 {gap_count} 条 gap junction 边")
    print(f"  其中 {both_count} 条边同时有化学和gap连接")
    
    return G, all_neurons

def main():
    excel_path = Path("data/raw/wormwiring_SI5_connectome_adjacency_corrected_2020.xlsx")
    
    if not excel_path.exists():
        print(f"错误: 文件不存在 {excel_path}")
        return
    
    G, neurons = build_graph(excel_path)
    
    # 统计信息
    print("\n" + "=" * 80)
    print("图统计信息")
    print("=" * 80)
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    
    type_counts = {'S': 0, 'I': 0, 'M': 0}
    for node, data in G.nodes(data=True):
        ntype = data.get('neuron_type', '')
        if ntype in type_counts:
            type_counts[ntype] += 1
    
    print(f"\n节点类型分布:")
    for ntype, count in type_counts.items():
        print(f"  {ntype}: {count}")
    
    # 统计边类型
    edge_type_counts = {'chemical': 0, 'gap': 0, 'both': 0}
    for u, v, d in G.edges(data=True):
        etype = d.get('edge_type', 'unknown')
        if etype in edge_type_counts:
            edge_type_counts[etype] += 1
    
    print(f"\n边类型分布:")
    print(f"  仅化学突触: {edge_type_counts['chemical']}")
    print(f"  仅 gap junction: {edge_type_counts['gap']}")
    print(f"  同时有化学和gap: {edge_type_counts['both']}")
    
    # 保存
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图
    graph_path = output_dir / "graph_wormwiring_hermaphrodite_ISM.pickle"
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"\n✓ 图已保存: {graph_path}")
    
    # 保存CSV
    edges_data = []
    for u, v, d in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'source_type': G.nodes[u]['neuron_type'],
            'target_type': G.nodes[v]['neuron_type'],
            'chemical_weight': d.get('chemical_weight', 0),
            'gap_weight': d.get('gap_weight', 0),
            'edge_type': d.get('edge_type', 'unknown')
        })
    
    df_edges = pd.DataFrame(edges_data)
    edges_csv = output_dir / "edges_wormwiring_ISM.csv"
    df_edges.to_csv(edges_csv, index=False)
    print(f"✓ 边列表已保存: {edges_csv}")
    
    nodes_data = [{'node_id': n, 'neuron_type': d['neuron_type']} 
                  for n, d in G.nodes(data=True)]
    df_nodes = pd.DataFrame(nodes_data)
    nodes_csv = output_dir / "nodes_wormwiring_ISM.csv"
    df_nodes.to_csv(nodes_csv, index=False)
    print(f"✓ 节点列表已保存: {nodes_csv}")
    
    # 显示示例
    print("\n" + "=" * 80)
    print("示例节点（前15个）:")
    print("=" * 80)
    for i, (node, data) in enumerate(list(G.nodes(data=True))[:15]):
        print(f"  {node}: {data['neuron_type']}")
    
    print("\n示例边（前10条）:")
    print("=" * 80)
    for i, (u, v, d) in enumerate(list(G.edges(data=True))[:10]):
        chem = d.get('chemical_weight', 0)
        gap = d.get('gap_weight', 0)
        print(f"  {u} -> {v}: chemical={chem}, gap={gap}, type={d.get('edge_type')}")

if __name__ == "__main__":
    main()

