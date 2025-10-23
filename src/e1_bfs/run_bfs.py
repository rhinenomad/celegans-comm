import argparse, pickle, yaml, networkx as nx
from pathlib import Path
import pandas as pd

def load_cfg(p): 
    with open(p) as f: return yaml.safe_load(f)

def load_graph(p):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(p)
    except Exception:
        with open(p, "rb") as f: return pickle.load(f)

def main(cfg_path):
    C = load_cfg(cfg_path)
    out_dir = Path(C["report"]["tables_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    G = load_graph(C["data"]["graphs"]["V1"])
    # 最小占位：输出节点边数，确认能读到 V1
    df = pd.DataFrame([{"variant":"V1", "nodes":G.number_of_nodes(), "edges":G.number_of_edges()}])
    df.to_csv(out_dir/"e1_bfs_ok.csv", index=False)
    print("E1 OK ->", out_dir/"e1_bfs_ok.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    args = ap.parse_args()
    main(args.config)
