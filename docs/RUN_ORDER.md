# Reproducible analysis order

Run from the repository root:

```text
python Code/analysis/lodo_score.py
python Code/analysis/current-dataset_removal_search.py
python Code/analysis/current_extract_removal_influence.py
python Code/analysis/current_dataset_influence.py
python Code/analysis/performance_gap_analysis.py
python Code/analysis/current_fold_stability.py
python Code/analysis/current_fold_vs_dataset_stability.py
python Code/analysis/current_dataset_removal_search_elo.py
python Code/analysis/compare_score_elo_same_subsets.py
python Code/analysis/compare_score_elo_removal.py
python Code/analysis/current_simple_crossbenchmark.py
python Code/analysis/current_glue_performance_gap.py
python Code/analysis/glue_top_leaderboard_gap_analysis.py
python Code/analysis/glue_performance_gap_plots.py
python Code/analysis/plot_crossbenchmark_lodo.py
python Code/analysis/plot_fold_stability_figures.py
python Code/analysis/plot_score_elo_removal.py
python Code/analysis/benchmark_comparison.py
```

Alternatively:

```text
python Code/analysis/run_current_pipeline.py
```

The joint Score search now defaults to the analysis configuration used for the latest thesis results: exhaustive removal depths 1–5 and heuristic/beam exploration through depth 8. Existing checkpoints can still be resumed.

GLUE, OpenLLM and ImageNet are loaded through the local BenchBench repository. TabArena remains on the TabArena-specific input pipeline.
