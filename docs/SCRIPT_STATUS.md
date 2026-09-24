# Analysis script status

## Canonical active scripts

The active scripts live directly in `Code/analysis/` and all share paths from `analysis_config.py`.

### TabArena stability

- `lodo_score.py` — Score aggregation and leave-one-dataset-out baseline.
- `current-dataset_removal_search.py` — exact + beam dataset-removal search.
- `current_extract_removal_influence.py` — extracts influence measures from saved removal-search evaluations.
- `current_dataset_influence.py` — combines LODO and adversarial influence measures.
- `current_fold_stability.py` — fold-based stability analysis.
- `current_fold_vs_dataset_stability.py` — fold-vs-dataset stability comparison.
- `current_dataset_removal_search_elo.py` — Elo dataset-removal search.
- `compare_score_elo_same_subsets.py` — Score-vs-Elo comparison on shared subsets.
- `compare_score_elo_removal.py` — broader Score-vs-Elo removal comparison.

### Cross-benchmark Phase 1

- `current_simple_crossbenchmark.py` — canonical LODO analysis for GLUE, OpenLLM, and ImageNet using BenchBench-native loaders.
- `current_glue_lodo.py` — GLUE compatibility entry point.
- `current_openllm_analysis.py` — OpenLLM compatibility entry point.
- `current_imageNet_analysis.py` — ImageNet compatibility entry point.
- `plot_crossbenchmark_lodo.py` — common LODO figures.
- `benchmark_comparison.py` — final cross-benchmark comparison table.

### Performance gaps / figures

- `performance_gap_analysis.py`
- `current_glue_performance_gap.py`
- `glue_top_leaderboard_gap_analysis.py`
- `glue_performance_gap_plots.py`
- `plot_fold_stability_figures.py`
- `plot_score_elo_removal.py`

### Repository utilities

- `analysis_config.py` — single source of truth for repository, data, BenchBench, and result paths.
- `organize_results.py` — safe consolidation of the old `Code/analysis/results/` directory into root `results/`.
- `run_current_pipeline.py` — safe orchestrator. Reporting is the default; refresh/expensive recomputation is explicit.

## Legacy / provenance

Older exploratory or duplicate scripts remain under `Code/analysis/legacy/`. Original pre-cleanup versions are under `docs/original_scripts/`.
