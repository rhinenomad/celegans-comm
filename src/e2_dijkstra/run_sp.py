# -*- coding: utf-8 -*-
import argparse, pickle, yaml, math
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def build_cost_graph(G, topo_mode="inverse_weight", use_geo=False, alpha=0.0, pos=None):
    """Return DiGraph H with edge attribute 'cost'."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u,v,d in G.edges(data=True):
        w = float(d.get("weight",1.0))
        topo = 1.0/max(w,1e-9) if (topo_mode=="inverse_weight") else 1.0
        if use_geo and pos and (u in pos) and (v in pos):
            dist = euclid(pos[u], pos[v])
            cost = alpha*dist + (1.0-alpha)*topo
        else:
            cost = topo
        H.add_edge(u,v, cost=cost)
    return H

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1","V2"], help="V2 primary; V1 is unweighted control")
    args = ap.parse_args()

    C = load_cfg(args.config)
    out_fig = Path(C["report"]["figures_dir"]); out_tbl = Path(C["report"]["tables_dir"])
    out_fig.mkdir(parents=True, exist_ok=True); out_tbl.mkdir(parents=True, exist_ok=True)

    # graph & seeds/targets
    G = load_graph(C["data"]["graphs"][args.variant])
    seeds = read_list(C["data"]["sensory_list"])
    targets = read_list(C["data"]["motor_list"])

    if not seeds:
        # fallback: top-k indegree as seeds
        seeds = [n for n,_ in sorted(G.in_degree(), key=lambda x:x[1], reverse=True)[:5]] \
                or [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)[:5]]
    if not targets:
        # fallback: top-k outdegree as targets
        targets = [n for n,_ in sorted(G.out_degree(), key=lambda x:x[1], reverse=True)[:20]] \
                  or [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)[:20]]

    topo_mode = C["analysis"]["dijkstra"].get("topological_cost", "inverse_weight")
    alpha_grid = C["analysis"].get("alpha_grid", [0.0,0.25,0.5,0.75,1.0])
    # use geometry only if distance_key set AND positions exist
    use_geo = bool(C["analysis"]["dijkstra"].get("distance_key"))
    pos = get_pos(G) if use_geo else None
    if use_geo and not pos:
        print("[warn] distance_key configured but no coordinates found; fall back to topological cost only.")
        use_geo = False

    # sweep
    all_rows = []
    for a in alpha_grid:
        H = build_cost_graph(G, topo_mode=topo_mode, use_geo=use_geo, alpha=float(a), pos=pos)
        for s in seeds:
            if s not in H: continue
            sp = nx.single_source_dijkstra_path_length(H, s, weight="cost")
            for t in targets:
                cost = sp.get(t, np.inf)
                all_rows.append({"alpha":a, "seed":s, "target":t, "cost":cost})

    df = pd.DataFrame(all_rows)
    df.to_csv(out_tbl/f"e2_dijkstra_paths_{args.variant}.csv", index=False)

    # reach_rate & mean_cost per alpha
    def reach_rate(col): 
        x = np.isfinite(col.values)
        return float(x.mean()) if len(x)>0 else 0.0
    agg = df.groupby("alpha").agg(reach_rate=("cost", lambda x: np.isfinite(x).mean()),
                                  mean_cost=("cost", lambda x: np.nanmean([v for v in x if np.isfinite(v)]) if np.any(np.isfinite(x)) else np.nan)).reset_index()
    agg.to_csv(out_tbl/f"e2_dijkstra_alpha_summary_{args.variant}.csv", index=False)

    # quick plot: alpha vs mean_cost (only finite)
    plt.figure()
    plt.plot(agg["alpha"], agg["mean_cost"], marker="o")
    plt.xlabel("alpha (geometry weight)")
    plt.ylabel("mean shortest-path cost")
    plt.title(f"E2 Dijkstra α-sweep ({args.variant})")
    plt.tight_layout()
    plt.savefig(out_fig/f"fig_e2_alpha_vs_cost_{args.variant}.png", dpi=160)

    print("E2 done ->",
          out_tbl/f"e2_dijkstra_paths_{args.variant}.csv", "|",
          out_tbl/f"e2_dijkstra_alpha_summary_{args.variant}.csv", "|",
          out_fig/f"fig_e2_alpha_vs_cost_{args.variant}.png")

if __name__ == "__main__":
    main()
