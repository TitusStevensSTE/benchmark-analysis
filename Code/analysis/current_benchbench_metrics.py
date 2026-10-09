from __future__ import annotations

"""Reproduce the BenchBench cardinal analysis and add TabArena.

Reference benchmarks are handled exactly through BenchBench's native
``load_cardinal_benchmark`` loader and its native metric functions. The
ImageNet, Random, and Constant controls are averaged over five seeds, as in
the authors' cardinal.ipynb. TabArena is added as a model-by-dataset matrix
using the existing higher-is-better ``dataset_score`` values.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"
if str(BENCHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHBENCH_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "benchbench_metrics"
OUTPUT_CSV = OUTPUT_DIR / "benchbench_metrics.csv"
TABARENA_SOURCE = RESULTS_ROOT / "score_method_dataset.csv"

# Seven cardinal benchmarks from the authors' notebook, plus the three
# five-seed reference controls used in their cardinal figure.
REFERENCE_BENCHMARKS = [
    ("GLUE", "GLUE"),
    ("SuperGLUE", "SuperGLUE"),
    ("BIG-Bench-Hard", "BigBenchHard"),
    ("MTEB", "MTEB"),
    ("OpenLLM", "OpenLLM"),
    ("MMLU", "MMLU"),
    ("VTAB", "VTAB"),
    ("ImageNet", "ImageNet"),
    ("Random", "Random"),
    ("Constant", "Constant"),
]
SEEDED_BENCHMARKS = {"ImageNet", "Random", "Constant"}
NUM_SEEDS = 5

RESULT_COLUMNS = [
    "benchmark",
    "representation",
    "num_models",
    "num_task_columns",
    "diversity_W",
    "diversity_MRC",
    "sensitivity_tau",
    "sensitivity_MRC",
]


def benchbench_functions():
    from benchbench.data import load_cardinal_benchmark
    from benchbench.measures.cardinal import get_diversity, get_sensitivity

    return load_cardinal_benchmark, get_diversity, get_sensitivity


def prepare_tabarena() -> tuple[pd.DataFrame, list[str]]:
    """Build the TabArena model-by-dataset matrix from native score output."""
    long_data = pd.read_csv(TABARENA_SOURCE)
    matrix = long_data.pivot(
        index="method",
        columns="dataset",
        values="dataset_score",
    )
    matrix.columns.name = None
    matrix = matrix.reset_index().rename(columns={"method": "Model"})
    task_columns = [column for column in matrix.columns if column != "Model"]

    # Keep model rows ordered by descending mean score, matching the prior
    # TabArena preparation. The BenchBench metric functions are unchanged.
    mean_score = matrix[task_columns].to_numpy(dtype=float).mean(axis=1)
    order = np.argsort(-mean_score, kind="stable")
    matrix = matrix.iloc[order].reset_index(drop=True)
    return matrix, task_columns


def get_native_metrics(data, task_columns, get_diversity, get_sensitivity):
    """Call the authors' functions with their default parameters."""
    diversity_w, diversity_mrc = get_diversity(data, task_columns)
    sensitivity_tau, sensitivity_mrc = get_sensitivity(data, task_columns)
    return {
        "diversity_W": float(diversity_w),
        "diversity_MRC": float(diversity_mrc),
        "sensitivity_tau": float(sensitivity_tau),
        "sensitivity_MRC": float(sensitivity_mrc),
    }


def get_seed_averaged_metrics(
    loader_name,
    load_cardinal_benchmark,
    get_diversity,
    get_sensitivity,
):
    """Average both native measures over the five seeds used in the paper."""
    diversity_values = []
    sensitivity_values = []
    data_for_shape = None
    task_columns_for_shape = None

    for seed in range(NUM_SEEDS):
        data, task_columns = load_cardinal_benchmark(loader_name, seed=seed)
        data_for_shape = data
        task_columns_for_shape = list(task_columns)
        diversity_values.append(
            get_diversity(data, task_columns_for_shape)
        )
        sensitivity_values.append(
            get_sensitivity(data, task_columns_for_shape)
        )

    diversity_mean = np.asarray(diversity_values, dtype=float).mean(axis=0)
    sensitivity_mean = np.asarray(sensitivity_values, dtype=float).mean(axis=0)
    metrics = {
        "diversity_W": float(diversity_mean[0]),
        "diversity_MRC": float(diversity_mean[1]),
        "sensitivity_tau": float(sensitivity_mean[0]),
        "sensitivity_MRC": float(sensitivity_mean[1]),
    }
    return data_for_shape, task_columns_for_shape, metrics


def print_result(name: str, data, task_columns, metrics) -> None:
    print(
        f"{name:16s} models={len(data):4d} "
        f"score_columns={len(task_columns):4d} "
        f"W={metrics['diversity_W']:.6f} "
        f"D_MRC={metrics['diversity_MRC']:.6f} "
        f"S_tau={metrics['sensitivity_tau']:.6f} "
        f"S_MRC={metrics['sensitivity_MRC']:.6f}"
    )


def main() -> None:
    load_cardinal_benchmark, get_diversity, get_sensitivity = benchbench_functions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    tabarena, tabarena_tasks = prepare_tabarena()
    tabarena_metrics = get_native_metrics(
        tabarena, tabarena_tasks, get_diversity, get_sensitivity
    )
    print_result("TabArena", tabarena, tabarena_tasks, tabarena_metrics)
    rows.append({
        "benchmark": "TabArena",
        "representation": "score_matrix",
        "num_models": len(tabarena),
        "num_task_columns": len(tabarena_tasks),
        **tabarena_metrics,
    })

    for name, loader_name in REFERENCE_BENCHMARKS:
        if name in SEEDED_BENCHMARKS:
            data, task_columns, metrics = get_seed_averaged_metrics(
                loader_name,
                load_cardinal_benchmark,
                get_diversity,
                get_sensitivity,
            )
            representation = f"native_loader_mean_{NUM_SEEDS}_seeds"
        else:
            # Preserve the native loader's rows and task columns exactly.
            # In particular, do not deduplicate MMLU snapshots or filter
            # MTEB's expanded score columns: the authors' notebook passes
            # these values straight to get_diversity/get_sensitivity.
            data, task_columns = load_cardinal_benchmark(loader_name)
            task_columns = list(task_columns)
            metrics = get_native_metrics(
                data, task_columns, get_diversity, get_sensitivity
            )
            representation = "native_loader"

        print_result(name, data, task_columns, metrics)
        rows.append({
            "benchmark": name,
            "representation": representation,
            "num_models": len(data),
            "num_task_columns": len(task_columns),
            **metrics,
        })

    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved BenchBench metrics: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
