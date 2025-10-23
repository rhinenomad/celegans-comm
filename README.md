# Algorithmic Communication Models on the C. elegans Connectome

## Pre-registered success criteria
- **E3 (Navigation)** on V2: `success rate ≥ 70%` 且 `median stretch ≤ 1.25`.
- **E4 (Diffusion vs Geodesic)**: 以 Spearman ρ 比较通信矩阵与目标标签/矩阵（如可得），满足 `ρ_diffusion ≥ ρ_geodesic`.
- **E5 (Bottlenecks)**: min-cut 成员与中介中心性 Top-k 的重合率 **显著高于** 度序保持的空模型（p < 0.05，经 FDR 校正）。

## Reproducibility quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 预处理脚本将 raw → interim → processed/V1,V2,V3
python src/pipeline_prepare_graphs.py --config configs/params.yaml
# 运行各实验
python src/e1_bfs/run_bfs.py       --config configs/params.yaml
python src/e2_dijkstra/run_sp.py   --config configs/params.yaml
python src/e3_nav/run_nav.py       --config configs/params.yaml
python src/e4_diffusion/run_diff.py --config configs/params.yaml
python src/e5_flow/run_flow.py     --config configs/params.yaml