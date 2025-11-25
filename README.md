# Algorithmic Communication Models on the C. elegans Connectome

## Pre-registered success criteria
- **E3 (Navigation)** on V2: `success rate ≥ 70%` and `median stretch ≤ 1.25`.
- **E4 (Diffusion vs Geodesic)**: Compare communication matrices with target labels/matrices (if available) using Spearman ρ, satisfying `ρ_diffusion ≥ ρ_geodesic`.
- **E5 (Bottlenecks)**: Overlap rate between min-cut members and betweenness centrality Top-k is **significantly higher** than degree-preserving null model (p < 0.05, FDR corrected).

## Reproducibility quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Preprocessing script: raw → interim → processed/V1,V2,V3
python src/pipeline_prepare_graphs.py --config configs/params.yaml
# Run experiments
python src/e1_bfs/run_bfs.py       --config configs/params.yaml
python src/e2_dijkstra/run_sp.py   --config configs/params.yaml
python src/e3_nav/run_nav.py       --config configs/params.yaml
python src/e4_diffusion/run_diff.py --config configs/params.yaml
python src/e5_flow/run_flow.py     --config configs/params.yaml
```