# -*- coding: utf-8 -*-
import argparse, pickle, yaml, math
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

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
def greedy_route(G, s, t, pos, step_limit):
    if (pos is None) or (s not in pos) or (t not in pos): 
        return False, [s], "no_geometry"
    path, visited = [s], {s}
    def d(x): return euclid(pos[x], pos[t]) if x in pos and t in pos else np.inf
    while len(path) - 1 < step_limit:
        u = path[-1]
        if u == t: return True, path, None
        nbrs = [v for v in successors(G, u) if v not in visited]
        if not nbrs:
            return False, path, "dead_end"
        # curr_d = d(u)
        v = min(nbrs, key=d)
        # if d(v) < curr_d:
        visited.add(v); path.append(v)

    return False, path, "step_limit"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--variant", default="V2", choices=["V1","V2", "V3", "V4"])
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

    rows = []
    for s in seeds:
        if s not in G: continue
        for t in targets:
            if t not in G or s == t: continue
            success, path, reason = greedy_route(G, s, t, pos, step_limit)
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
    summary.to_csv(out_tbl/"e3_nav_summary.csv", index=False)
    print("E3 greedy done ->", out_tbl/"e3_nav_results.csv", "|", out_tbl/"e3_nav_summary.csv")

if __name__ == "__main__":
    main()