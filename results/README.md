# Analysis results

This directory is the single canonical location for generated analysis outputs and figures.

Typical subdirectories:

```text
results/
├── phase1_crossbenchmark/
├── dataset_removal_search/
├── dataset_influence/
├── fold_stability/
├── performance_gap/
├── stability_dimensions/
├── score_vs_elo/
└── _archive/
```

Do not create a second `Code/analysis/results/` directory. The utility
`Code/analysis/organize_results.py` can migrate older outputs into this location.
