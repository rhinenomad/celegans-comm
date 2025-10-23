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
    G = load_graph(C["data"]["graphs"]["V2"])
    df = pd.DataFrame([{"variant":"V2", "flow_placeholder":"ready", "edges":G.number_of_edges()}])
    df.to_csv(out_dir/"e5_flow_ok.csv", index=False)
    print("E5 OK ->", out_dir/"e5_flow_ok.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    args = ap.parse_args()
    main(args.config)
