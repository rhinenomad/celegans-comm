# ===== C. elegans communication models — Makefile =====

# ---- Basics (override in CLI if needed) ----
PY        ?= python3
CFG       ?= configs/params.yaml
INPUT     ?= data/raw/edges.csv
PORT      ?= 8000

# Which variant to use in E1/E2 (V1|V2|V3 / V1|V2)
E1_VARIANT ?= V1
E2_VARIANT ?= V2

# Portable "open" command (macOS=open, Linux=xdg-open, else echo path)
OPEN := $(shell if command -v open >/dev/null 2>&1; then echo open; \
               elif command -v xdg-open >/dev/null 2>&1; then echo xdg-open; \
               else echo printf; fi)

# ---- Outputs produced by preprocess ----
V1 := data/processed/V1/graph_V1_directed_unweighted.gpickle
V2 := data/processed/V2/graph_V2_directed_weighted.gpickle
V3 := data/processed/V3/graph_V3_undirected.gpickle
V4 := data/processed/V4/graph_V4_directed_unweighted.gpickle
SUMMARY := results/tables/summary_V1_V2_V3.csv

# ---- Figures/Tables (patterns) ----
E1_FIG := results/figures/fig_bfs_coverage_$(E1_VARIANT).png
E1_TBL := results/tables/e1_bfs_layers_$(E1_VARIANT).csv

E2_FIG := results/figures/fig_e2_alpha_vs_cost_$(E2_VARIANT).png
E2_TBL1 := results/tables/e2_dijkstra_paths_$(E2_VARIANT).csv
E2_TBL2 := results/tables/e2_dijkstra_alpha_summary_$(E2_VARIANT).csv

E3_TBL1 := results/tables/e3_nav_results.csv
E3_TBL2 := results/tables/e3_nav_summary.csv

E4_TBL := results/tables/e4_diffusion_correlations.csv

E5_TBL1 := results/tables/e5_flow_values.csv
E5_TBL2 := results/tables/e5_mincut_edges.csv
E5_TBL3 := results/tables/e5_flow_ablation_by_mincut.csv

# ---- Help (default) ----
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "  preprocess          # build V1/V2/V3 from $(INPUT)"
	@echo "  e1                  # BFS (variant=$(E1_VARIANT))"
	@echo "  e2                  # Dijkstra alpha-sweep (variant=$(E2_VARIANT))"
	@echo "  e3                  # Decentralized navigation"
	@echo "  e4                  # Diffusion correlations"
	@echo "  e5                  # Max-flow/Min-cut + ablation"
	@echo "  all                 # preprocess + e1..e5"
	@echo "  open-e1|open-e2     # open latest figures"
	@echo "  api                 # serve Swagger UI on :$(PORT)"
	@echo "  clean / distclean   # remove results / results+processed graphs"
	@echo ""
	@echo "Override examples:"
	@echo "  make e1 E1_VARIANT=V2"
	@echo "  make preprocess INPUT=data/raw/edges.csv"

# ---- Preprocess (produces all 3 graphs + summary) ----
.PHONY: preprocess
preprocess: $(V1) $(V2) $(V3) $(SUMMARY)

$(V1) $(V2) $(V3) $(SUMMARY):
	$(PY) src/pipeline_prepare_graphs.py --config $(CFG) --input $(INPUT)

# ---- E1: BFS ----
.PHONY: e1
e1: $(E1_FIG)

$(E1_FIG): preprocess
	$(PY) src/e1_bfs/run_bfs.py --config $(CFG) --variant $(E1_VARIANT)

.PHONY: open-e1
open-e1: e1
	$(OPEN) "$(E1_FIG)"

# ---- E2: Dijkstra alpha-sweep ----
.PHONY: e2
e2: $(E2_FIG)

$(E2_FIG): preprocess
	$(PY) src/e2_dijkstra/run_sp.py --config $(CFG) --variant $(E2_VARIANT)

.PHONY: open-e2
open-e2: e2
	$(OPEN) "$(E2_FIG)"

# ---- E3: Decentralized navigation ----
.PHONY: e3
e3: $(E3_TBL1) $(E3_TBL2)

$(E3_TBL1) $(E3_TBL2): preprocess
	$(PY) src/e3_nav/run_nav.py --config $(CFG) --variant V4

# ---- E4: Diffusion baseline ----
.PHONY: e4
e4: $(E4_TBL)

$(E4_TBL): preprocess
	$(PY) src/e4_diffusion/run_diff.py --config $(CFG)

# ---- E5: Max-flow / Min-cut + ablation ----
.PHONY: e5
e5: $(E5_TBL1) $(E5_TBL2) $(E5_TBL3)

$(E5_TBL1) $(E5_TBL2) $(E5_TBL3): preprocess
	$(PY) src/e5_flow/run_flow.py --config $(CFG)

# ---- All ----
.PHONY: all
all: preprocess e1 e2 e3 e4 e5

# ---- API (Swagger/OpenAPI) ----
.PHONY: api
api:
	uvicorn api.serve_swagger:app --reload --port $(PORT)

# ---- Clean ----
.PHONY: clean distclean
clean:
	@echo "Remove result CSV/figures…"
	@rm -rf results/tables/*.csv results/figures/*.png || true

distclean: clean
	@echo "Remove processed graphs…"
	@rm -rf data/processed/V1 data/processed/V2 data/processed/V3 data/processed/V4
