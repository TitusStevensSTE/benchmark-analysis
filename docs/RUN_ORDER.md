# Reproducible run order

## 1. Consolidate result directories once

Run a dry run first:

```powershell
python Code/analysis/organize_results.py
```

Then apply it:

```powershell
python Code/analysis/organize_results.py --apply
```

This makes the repository root `results/` the canonical result location.

## 2. Safe reporting from existing results

After migration, the safest normal command is:

```powershell
python Code/analysis/run_current_pipeline.py
```

This generates plots and the benchmark-comparison summary from existing CSVs. It does not launch exhaustive recomputation.

## 3. Refresh non-exhaustive analyses

```powershell
python Code/analysis/run_current_pipeline.py --refresh
```

This explicitly recomputes the Phase 1 cross-benchmark and other non-exhaustive analyses, then regenerates reports.

## 4. Expensive analyses

The following stages are protected and require `--expensive`:

```text
lodo_score.py
current-dataset_removal_search.py
current_fold_stability.py
current_dataset_removal_search_elo.py
compare_score_elo_same_subsets.py
compare_score_elo_removal.py
```

Run a single stage deliberately, for example:

```powershell
python Code/analysis/run_current_pipeline.py --stage dataset_search --expensive
```

Do not rerun the existing exhaustive analyses simply to reproduce the reorganized repository. Their saved outputs should be reused.
