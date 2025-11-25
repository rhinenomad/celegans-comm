#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate V4 graph variant: V2 (directed, weighted) with algorithmically generated coordinates
V4 is based on V2 but includes geometric coordinates computed using spring layout algorithm
"""
import argparse
import pickle
import yaml
from pathlib import Path
import networkx as nx

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_graph(path):
    """Load graph from pickle file"""
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def main():
    ap = argparse.ArgumentParser(description="Generate V4: V2 with algorithmically generated coordinates")
    ap.add_argument("--config", default="configs/params.yaml", help="Configuration file path")
    ap.add_argument("--base-variant", default="V2", choices=["V1", "V2", "V3"],
                    help="Base variant to use (default: V2)")
    ap.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai", "spectral"],
                    help="Layout algorithm to use (default: spring)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for layout (default: 42)")
    args = ap.parse_args()

    C = load_cfg(args.config)
    
    # Load base graph (V2 by default)
    base_graph_path = C["data"]["graphs"][args.base_variant]
    if not Path(base_graph_path).exists():
        raise FileNotFoundError(f"Base graph not found: {base_graph_path}")
    
    print(f"Loading base graph: {args.base_variant} from {base_graph_path}")
    G = load_graph(base_graph_path)
    print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check if graph already has coordinates
    nodes_with_pos = sum(1 for n in G.nodes() if 'pos' in G.nodes[n] or ('x' in G.nodes[n] and 'y' in G.nodes[n]))
    if nodes_with_pos > 0:
        print(f"  Warning: {nodes_with_pos} nodes already have coordinates. Overwriting...")
    
    # Generate coordinates using layout algorithm
    print(f"\nGenerating coordinates using {args.layout} layout (seed={args.seed})...")
    
    # Convert to undirected for layout computation (if directed)
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Compute layout
    if args.layout == "spring":
        pos = nx.spring_layout(G_undirected, seed=args.seed, iterations=50)
    elif args.layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G_undirected, seed=args.seed)
        except Exception as e:
            print(f"  Warning: Kamada-Kawai layout failed ({e}), falling back to spring layout")
            pos = nx.spring_layout(G_undirected, seed=args.seed, iterations=50)
    elif args.layout == "spectral":
        try:
            pos = nx.spectral_layout(G_undirected)
        except Exception as e:
            print(f"  Warning: Spectral layout failed ({e}), falling back to spring layout")
            pos = nx.spring_layout(G_undirected, seed=args.seed, iterations=50)
    
    # Attach coordinates to nodes
    for n, (x, y) in pos.items():
        G.nodes[n]['pos'] = (float(x), float(y))
    
    print(f"  Generated coordinates for {len(pos)} nodes")
    
    # Save V4 graph
    v4_path = Path(C["data"]["graphs"].get("V4", "data/processed/V4/graph_V4_directed_weighted.gpickle"))
    v4_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save graph
    try:
        from networkx.readwrite.gpickle import write_gpickle
        write_gpickle(G, str(v4_path))
        print(f"\n✓ Saved V4 graph to: {v4_path}")
    except Exception:
        with open(v4_path, "wb") as f:
            pickle.dump(G, f)
        print(f"\n✓ Saved V4 graph to: {v4_path}")
    
    # Show sample positions
    sample_nodes = list(G.nodes())[:5]
    print(f"\nSample node positions:")
    for n in sample_nodes:
        if 'pos' in G.nodes[n]:
            print(f"  {n}: {G.nodes[n]['pos']}")
    
    # Update config file if V4 path not present
    if "V4" not in C["data"]["graphs"]:
        print(f"\n⚠ Note: V4 path not in config file. Add this line to configs/params.yaml:")
        print(f'    V4: "{v4_path}"')
    
    print(f"\nV4 graph summary:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Directed: {G.is_directed()}")
    print(f"  Weighted: {nx.is_weighted(G)}")
    print(f"  Has coordinates: {sum(1 for n in G.nodes() if 'pos' in G.nodes[n])} nodes")

if __name__ == "__main__":
    main()

