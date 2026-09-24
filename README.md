# Benchmark Analysis — Updated Script Bundle

This bundle is the cleaned current analysis pipeline for the benchmark-stability thesis work.

## What is current

The canonical pipeline is under `Code/analysis/`.

- TabArena uses the existing TabArena-specific score/LODO/removal pipeline.
- GLUE, OpenLLM, and ImageNet are loaded through the local BenchBench repository via its native loaders.
- Phase 1 cross-benchmark LODO is centralized in `current_simple_crossbenchmark.py`.
- `current_glue_lodo.py`, `current_openllm_analysis.py`, and `current_imageNet_analysis.py` are compatibility entry points that call the canonical implementation.
- Cross-benchmark summaries are written to `Code/analysis/results/phase1_crossbenchmark/`.
- `run_current_pipeline.py` provides a reproducible end-to-end execution order.

## Important cleanup

Older exploratory/duplicate scripts are under `Code/analysis/legacy/` and are not part of the current thesis pipeline. The original versions are preserved under `docs/original_scripts/` for provenance.

The repository-root path handling uses `Path(__file__).resolve().parents[2]`; the previous Elo removal script contained an off-by-one root calculation that could produce `Code/Code/...` paths. That is corrected here.

The current joint Score removal defaults match the latest analysis configuration: exhaustive depths 1–5 and heuristic/beam exploration through depth 8.

## Running

Run from the repository root:

```text
python Code/analysis/run_current_pipeline.py
```

Or follow `docs/RUN_ORDER.md` to run stages individually.

The local checkout must contain the BenchBench repository at `<repo-root>/benchbench` and the required Python dependencies. The uploaded bundle contains the analysis scripts, not the BenchBench checkout itself.
