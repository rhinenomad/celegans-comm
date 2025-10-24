# -*- coding: utf-8 -*-
import argparse, pickle, yaml, math
from pathlib import Path
import networkx as nx
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

def bfs_layers_directed(G, seed):
    dist = nx.single_source_shortest_path_length(G, seed)  # respects direction
    rows, cum = [], 0
    bylayer = {}
    for _,d in dist.items():
        bylayer[d] = bylayer.get(d, 0) + 1
    for L in sorted(bylayer):
        cum += bylayer[L]
        rows.append({"seed":seed, "layer":L, "new_nodes":bylayer[L], "cum_covered":cum})
    return pd.DataFrame(rows)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V1", choices=["V1","V2","V3"], help="which graph variant to use")
    ap.add_argument("--max_seeds", type=int, default=5, help="if no sensory_list provided, pick top-k indegree as seeds")
    args = ap.parse_args()

    C = load_cfg(args.config)
    out_fig = Path(C["report"]["figures_dir"]); out_tbl = Path(C["report"]["tables_dir"])
    out_fig.mkdir(parents=True, exist_ok=True); out_tbl.mkdir(parents=True, exist_ok=True)

    G = load_graph(C["data"]["graphs"][args.variant])

    # seeds: sensory_list if present, else top-k in-degree nodes
    seeds = read_list(C["data"]["sensory_list"])
    if not seeds:
        seeds = [n for n,_ in sorted(G.in_degree(), key=lambda x:x[1], reverse=True)[:args.max_seeds] if hasattr(G, "in_degree")] \
                or [n for n,_ in sorted(G.degree(), key=lambda x:x[1], reverse=True)[:args.max_seeds]]

    # run BFS per seed
    dfs = []
    for s in seeds:
        if s in G:
            dfs.append(bfs_layers_directed(G, s))
    if not dfs:
        raise SystemExit("No valid seeds found in the graph.")

    cov = pd.concat(dfs, ignore_index=True)
    cov.to_csv(out_tbl/f"e1_bfs_layers_{args.variant}.csv", index=False)

    # average coverage curve
    avg = cov.groupby("layer")["cum_covered"].mean().reset_index()
    plt.figure()
    plt.plot(avg["layer"], avg["cum_covered"], marker="o")
    plt.xlabel("BFS layer")
    plt.ylabel("Average cumulative coverage")
    plt.title(f"E1 BFS coverage ({args.variant})")
    plt.tight_layout()
    plt.savefig(out_fig/f"fig_bfs_coverage_{args.variant}.png", dpi=160)
    print("E1 done ->", out_tbl/f"e1_bfs_layers_{args.variant}.csv", "|", out_fig/f"fig_bfs_coverage_{args.variant}.png")

if __name__ == "__main__":
    main()
