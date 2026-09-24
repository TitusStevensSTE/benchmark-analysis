# Analysis script status

## Current / thesis pipeline

- `lodo_score.py` — TabArena Score aggregation + LODO baseline.
- `current-dataset_removal_search.py` — TabArena Score joint dataset removal; exact search followed by beam search.
- `current_extract_removal_influence.py` — extracts influential/adversarial datasets from the joint-removal output.
- `current_dataset_influence.py` — combines LODO, winner-change and tau-break influence measures.
- `current_fold_stability.py` — fold-level stability analysis for TabArena.
- `current_fold_vs_dataset_stability.py` — compares fold-level and dataset-level instability.
- `current_dataset_removal_search_elo.py` — TabArena Elo joint-removal analysis.
- `compare_score_elo_same_subsets.py` — Score-vs-Elo comparison on the same removal subsets.
- `compare_score_elo_removal.py` — broader Score-vs-Elo removal comparison/summary.
- `plot_score_elo_removal.py` — Score-vs-Elo figures.
- `plot_fold_stability_figures.py` — fold-stability figures.
- `current_simple_crossbenchmark.py` — canonical Phase 1 LODO analysis for GLUE, OpenLLM and ImageNet using BenchBench-native loaders.
- `current_glue_lodo.py` / `current_openllm_analysis.py` / `current_imageNet_analysis.py` — compatibility entry points to the canonical cross-benchmark implementation.
- `current_glue_performance_gap.py` — GLUE pairwise performance-gap vs LODO reversal analysis.
- `glue_top_leaderboard_gap_analysis.py` — top-of-leaderboard adjacent-gap analysis.
- `glue_performance_gap_plots.py` — plots the GLUE performance-gap analyses.
- `plot_crossbenchmark_lodo.py` — common plots for the canonical Phase 1 LODO results.
- `benchmark_comparison.py` — canonical cross-benchmark comparison table.
- `run_current_pipeline.py` — optional orchestrator for the current pipeline.

## Legacy / exploratory

The following are retained under `Code/analysis/legacy/` because they are older or duplicate experiments and should not be used as the thesis pipeline without a deliberate reason:

- `current_elo_1_51.py`
- `current_elo_lodo.py`
- `current_elo_winners.py`
- `current_tabarena_elo_removal.py`
- `glue_analysis.py`
- `lodo_elo.py`
- `winners_elo.py`
- `winners_elo_plot.py`
- `winners_lodo_score.py`

The legacy files are preserved for provenance. Some contain machine-specific paths or older output locations.
