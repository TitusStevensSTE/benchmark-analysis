# Reproduction Guide

## 1. Run from the repository root

All commands below assume the working directory is the repository root.

### Restore required source checkouts on a fresh clone

Initialize the TabArena submodule at the revision recorded by the main repository:

```powershell
git submodule update --init --recursive
```

BenchBench is a separate checkout ignored by the main repository. The validated revision is `1dbce6d655570c5095d62a68fa3430800df46ca8` from `https://github.com/socialfoundations/benchbench.git`.

If `benchbench/` does not exist, clone it from the repository root:

```powershell
git clone https://github.com/socialfoundations/benchbench.git benchbench
git -C benchbench checkout 1dbce6d655570c5095d62a68fa3430800df46ca8
```


### Create a fresh Python environment (optional)

The audited environment used Python 3.12.10. From the repository root:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
python -m pip install -e .\Code\tabarena\packages\bencheval
python -m pip install -e .\benchbench
```

The requirements file pins the direct numerical and plotting dependencies from the audited environment; it is not a complete transitive lockfile. The local `bencheval` and BenchBench checkouts are installed separately.

Alternatively, activate the existing thesis environment as described below.

Activate the project environment used for the thesis:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\Code\tabarena\.venv\Scripts\Activate.ps1
```

Confirm the interpreter if needed:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

The active analysis code uses repository-relative paths via `Code/analysis/analysis_config.py`.

## 2. Report and figure validation

Use this before submission:

```powershell
python Code/analysis/run_current_pipeline.py --report
```

A successful run should end with:

```text
Report validation passed: required saved outputs are present.
Pipeline complete.
```

Report mode regenerates configured report and figure outputs under `results/`, including the nine consolidated thesis figures and the benchmark-comparison CSV/Markdown files. It then validates the canonical result tables and final figure pairs. It does not rerun the expensive benchmark searches. Because generated files can be overwritten, this is not a read-only operation.

## 3. Regenerate derived analyses

To recompute the configured non-exhaustive stages and regenerate report plots:

```powershell
python Code/analysis/run_current_pipeline.py --refresh
```

This is intentionally different from `--report`: refresh mode can rewrite saved derived outputs and figures.

## 4. Run one expensive stage deliberately

Example:

```powershell
python Code/analysis/run_current_pipeline.py --stage dataset_search --expensive
```

The explicit `--expensive` flag is required for stages classified as expensive by the pipeline.

## 5. Full recomputation

Use only when a complete recomputation is actually needed:

```powershell
python Code/analysis/run_current_pipeline.py --full --expensive
```

The full pipeline includes long-running TabArena / removal-search / fold / Elo-related stages as configured in `run_current_pipeline.py`.

## 6. Final thesis figures

The canonical thesis figure script is:

```powershell
python Code/analysis/plot_final_thesis_figures.py
```

It writes `.png` and `.pdf` versions to:

```text
results/final_figures/
```

The final figure set contains nine figures; see `docs/PLOT_GUIDE.md`.

## 7. Important canonical result files

The pipeline report checks these current saved result tables:

```text
results/lodo_score_results.csv
results/benchbench_metrics/benchbench_metrics.csv
results/random_subset_winner_frequency/random_subset_repetitions.csv
results/dataset_influence_final/dataset_influence_ranked.csv
results/performance_gap/performance_gap_vs_lodo.csv
results/dataset_removal_search/score/depth_summary.csv
results/phase1_crossbenchmark/glue/lodo_results.csv
results/phase1_crossbenchmark/openllm/unique_model_mean/lodo_results.csv
results/phase1_crossbenchmark/imagenet/lodo_results.csv
```

## 8. Canonical benchmark loading

TabArena retains its TabArena-specific data-loading pipeline.

The cross-benchmark analysis uses the local BenchBench loaders for GLUE, OpenLLM, and ImageNet through the active cross-benchmark analysis code.

## 9. Legacy / historical material

The directory:

```text
docs/original_scripts/
```

contains preserved historical or superseded scripts. These files are retained for provenance and should not be mistaken for the canonical current execution path.

## 10. Reproducibility caveat for Score versus Elo

The current repository supports the full 51-dataset Score and Elo LODO comparison as the primary reproducible comparison.

The historical 93-shared-subset comparison is not reproducible from the currently committed coordinated-removal artifacts and is therefore intentionally not part of the primary result path.
