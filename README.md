# Benchmark Analysis — Clean Thesis Pipeline

This repository contains the analysis code for the benchmark-stability thesis.

## Repository structure

```text
benchmark-analysis/
├── Code/
│   └── analysis/              # analysis source code only
│       ├── data/              # input datasets
│       └── legacy/             # older exploratory scripts
├── benchbench/                 # local BenchBench checkout
├── results/                    # ALL generated analysis results and figures
├── docs/                       # documentation and preserved originals
└── README.md
```

The canonical output location is **`results/` at the repository root**. Active scripts must not create `Code/analysis/results/`.

## Current analysis pipeline

- TabArena uses the existing TabArena-specific data pipeline.
- GLUE, OpenLLM, and ImageNet use the local BenchBench-native loaders.
- `current_simple_crossbenchmark.py` is the canonical Phase 1 cross-benchmark LODO analysis.
- `plot_crossbenchmark_lodo.py` creates the common Phase 1 LODO figures.
- `benchmark_comparison.py` creates the cross-benchmark comparison table.
- Older exploratory/duplicate scripts are retained under `Code/analysis/legacy/`; their original versions are preserved under `docs/original_scripts/`.

## Important safety rule

`run_current_pipeline.py` is intentionally safe by default.

With no arguments, it only reads existing result files and creates plots/summaries:

```powershell
python Code/analysis/run_current_pipeline.py
```

This default mode does **not** start the exhaustive removal or fold/Elo searches.

To recompute the non-exhaustive analyses explicitly:

```powershell
python Code/analysis/run_current_pipeline.py --refresh
```

To run one expensive analysis deliberately:

```powershell
python Code/analysis/run_current_pipeline.py --stage dataset_search --expensive
```

To run the complete pipeline, including expensive recomputation, the permission must be explicit:

```powershell
python Code/analysis/run_current_pipeline.py --full --expensive
```

## Consolidating old result locations

Older scripts stored outputs in `Code/analysis/results/`. The migration utility consolidates these into the root `results/` directory.

First inspect the migration plan:

```powershell
python Code/analysis/organize_results.py
```

Then apply it:

```powershell
python Code/analysis/organize_results.py --apply
```

Conflicting existing root files are preserved under `results/_archive/pre_cleanup_root_results/` rather than silently overwritten.

## Existing expensive analyses

Existing result files can be reused. Do **not** rerun the 24+ hour analyses merely because the code was reorganized. Only invoke an expensive stage deliberately when a new computation is actually required.
