# -*- coding: utf-8 -*-
"""
Process raw graph data (CSV or GML) into three versions:
V1: Directed-unweighted (topological baseline)
V2: Directed-weighted (edge weight = synapse count/strength)
V3: Undirected (gap layer; if no gap, symmetrize chemical layer)
Output summary statistics to results/tables/summary_V1_V2_V3.csv
"""
import os, argparse
from pathlib import Path
import yaml
import pandas as pd
import networkx as nx
import pickle
try:
    from networkx.readwrite.gpickle import write_gpickle as nx_write_gpickle
except Exception:
    nx_write_gpickle = None



def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)

def summarize(G, weighted=False):
    N = G.number_of_nodes()
    M = G.number_of_edges()
    dens = nx.density(G.to_undirected()) if G.is_directed() else nx.density(G)
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    num_comp = len(comps)
    lcc = max((len(c) for c in comps), default=0)
    wstats = {}
    if weighted:
        ws = [float(d.get("weight", 1.0)) for _,_,d in G.edges(data=True)]
        if ws:
            s = pd.Series(ws, dtype=float)
            wstats = dict(weight_sum=float(s.sum()),
                          weight_mean=float(s.mean()),
                          weight_median=float(s.median()),
                          weight_max=float(s.max()))
    return dict(N=N, M=M, density=float(dens),
                num_components=num_comp, largest_component=lcc, **wstats)

def read_from_csv(csv_path: Path):
    """CSV: Required columns source,target; optional weight,type (chemical/gap)"""
    df = pd.read_csv(csv_path)
    need = {"source","target"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {need}; actual columns: {list(df.columns)}")
    # Chemical layer (directed)
    chem = df if "type" not in df.columns else df[df["type"].astype(str).str.lower().eq("chemical")]
    Gc = nx.DiGraph()
    for _, r in chem.iterrows():
        u, v = str(r["source"]), str(r["target"])
        w = float(r["weight"]) if "weight" in r and pd.notna(r["weight"]) else 1.0
        if u == v or w <= 0:  # Remove self-loops and non-positive weights
            continue
        if Gc.has_edge(u, v):
            Gc[u][v]["weight"] += w
        else:
            Gc.add_edge(u, v, weight=w)
    # Gap layer (undirected) optional
    Gg = None
    if "type" in df.columns:
        gap = df[df["type"].astype(str).str.lower().eq("gap")]
        if len(gap):
            Gg = nx.Graph()
            for _, r in gap.iterrows():
                u, v = str(r["source"]), str(r["target"])
                w = float(r["weight"]) if "weight" in r and pd.notna(r["weight"]) else 1.0
                if u == v or w <= 0:
                    continue
                if Gg.has_edge(u, v):
                    Gg[u][v]["weight"] += w
                else:
                    Gg.add_edge(u, v, weight=w)
    return Gc, Gg

def read_from_gml(gml_path: Path):
    G0 = nx.read_gml(gml_path)
    # Convert to directed graph
    Gc = nx.DiGraph()
    Gc.add_nodes_from([str(n) for n in G0.nodes()])
    for u, v, d in G0.edges(data=True):
        u, v = str(u), str(v)
        w = float(d.get("weight", 1.0))
        if u == v or w <= 0:
            continue
        if Gc.has_edge(u, v):
            Gc[u][v]["weight"] += w
        else:
            Gc.add_edge(u, v, weight=w)
    return Gc, None  # No dedicated gap layer

def build_variants(Gc: nx.DiGraph, Gg: nx.Graph|None):
    # V1: Directed-unweighted
    V1 = nx.DiGraph()
    V1.add_nodes_from(Gc.nodes(data=True))
    for u, v in Gc.edges():
        V1.add_edge(u, v, weight=1.0)
    # V2: Directed-weighted
    V2 = Gc.copy()
    # V3: Undirected (if gap layer missing, symmetrize chemical layer)
    if Gg is None:
        V3 = nx.Graph()
        for u, v, d in Gc.edges(data=True):
            w = float(d.get("weight", 1.0))
            if V3.has_edge(u, v):
                V3[u][v]["weight"] += w
            else:
                V3.add_edge(u, v, weight=w)
    else:
        V3 = Gg
    return V1, V2, V3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, help="data/raw/edges.csv or .gml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outV1 = Path(cfg["data"]["graphs"]["V1"])
    outV2 = Path(cfg["data"]["graphs"]["V2"])
    outV3 = Path(cfg["data"]["graphs"]["V3"])
    for p in [outV1, outV2, outV3]:
        p.parent.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    ext = in_path.suffix.lower()
    if ext == ".csv":
        Gc, Gg = read_from_csv(in_path)
    elif ext == ".gml":
        Gc, Gg = read_from_gml(in_path)
    else:
        raise SystemExit(f"Unsupported input format: {ext} (please provide .csv or .gml)")

    V1, V2, V3 = build_variants(Gc, Gg)
    def _dump_graph(G, path):
        if nx_write_gpickle is not None:
            nx_write_gpickle(G, str(path))
        else:
            with open(path, "wb") as f:
                pickle.dump(G, f)

    _dump_graph(V1, outV1)
    _dump_graph(V2, outV2)
    _dump_graph(V3, outV3)


    # Summary statistics
    tbl_dir = Path(cfg["report"]["tables_dir"]); tbl_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {"variant":"V1", **summarize(V1, weighted=False)},
        {"variant":"V2", **summarize(V2, weighted=True)},
        {"variant":"V3", **summarize(V3, weighted=True)},
    ]
    pd.DataFrame(summary).to_csv(tbl_dir/"summary_V1_V2_V3.csv", index=False)

    print("Saved graphs:\n ", outV1, "\n ", outV2, "\n ", outV3)
    print("Summary table:", tbl_dir/"summary_V1_V2_V3.csv")

if __name__ == "__main__":
    main()
