from __future__ import annotations

"""BenchBench cardinal metrics for the thesis.

Computes the cardinal diversity/sensitivity measures using the actual
BenchBench implementation.

Representations
---------------
* TabArena:
    Existing thesis-normalized `dataset_score` matrix.
    `dataset_score` is higher-is-better.

* GLUE/OpenLLM/ImageNet:
    Loaded through BenchBench-native loaders.

Metrics
-------
BenchBench returns:

* diversity_W:
    BenchBench's diversity value, implemented as 1 - Kendall's W.

* diversity_MRC:
    Mean pairwise normalized maximum rank change across task rankings.

* sensitivity_tau:
    BenchBench's normalized Kendall rank distance:
        (1 - Kendall_tau) / 2

* sensitivity_MRC:
    Maximum normalized rank displacement found by the sensitivity
    optimization.

No metric formulas are reimplemented here. The actual BenchBench functions
are called directly.
"""

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"
if str(BENCHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHBENCH_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "results"

OUTPUT_ROOT = RESULTS_ROOT / "benchbench_metrics"
CACHE_FILE = OUTPUT_ROOT / "benchbench_metrics.json"
SUMMARY_FILE = OUTPUT_ROOT / "benchbench_metrics.csv"
CONFIG_FILE = OUTPUT_ROOT / "benchbench_configuration.json"


# ============================================================================
# Configuration
# ============================================================================

# True  -> recompute BenchBench metrics
# False -> load previously computed cache
RUN_BENCHBENCH = True

# OpenLLM contains duplicate model names in the native representation.
# Keep the native row-level result and additionally compute a unique-model
# mean as a robustness check.
RUN_OPENLLM_UNIQUE_MODEL_CHECK = True

BENCHMARKS = (
    "TabArena",
    "GLUE",
    "OpenLLM",
    "ImageNet",
)

# BenchBench's default sensitivity behavior:
#
# use_weighted_loss=None
#     evaluates both weighted and unweighted loss and keeps the more
#     sensitive result.
#
# We explicitly request return_weight=True because otherwise the version
# of BenchBench used here returns only the sensitivity scalar.
SENSITIVITY_KWARGS = {
    "use_weighted_loss": None,
    "return_weight": True,
}


# Explicit model columns for BenchBench-native loaders.
#
# Task columns themselves come from the native loader and are therefore not
# guessed here.
NATIVE_MODEL_COLUMNS = {
    "GLUE": "Model",
    # These are checked against the actual loader output at runtime.
    "OpenLLM": "Model",
    "ImageNet": "Model",
}


# ============================================================================
# Validation
# ============================================================================

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}"
        )


def validate_matrix(
    data: pd.DataFrame,
    cols: list[str],
    model_col: str,
    name: str,
) -> None:
    if model_col not in data.columns:
        raise ValueError(
            f"{name}: model column {model_col!r} not found"
        )

    if len(cols) < 2:
        raise ValueError(
            f"{name}: fewer than two task columns"
        )

    if data.empty:
        raise ValueError(
            f"{name}: no models"
        )

    if data[model_col].isna().any():
        raise ValueError(
            f"{name}: missing model identifiers"
        )

    values = data[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if values.isna().any().any():
        bad = values.columns[
            values.isna().any()
        ].tolist()

        raise ValueError(
            f"{name}: missing/non-numeric task scores in {bad}"
        )

    numeric_values = values.to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"{name}: non-finite task scores"
        )


def check_duplicate_models(
    data: pd.DataFrame,
    model_col: str,
    name: str,
) -> bool:
    duplicated = data[model_col].duplicated().any()

    if duplicated:
        count = int(
            data[model_col].duplicated().sum()
        )

        print(
            f"WARNING: {name} contains {count} duplicated "
            f"model-name rows."
        )

    return bool(duplicated)


# ============================================================================
# BenchBench
# ============================================================================

def benchbench_functions():
    from benchbench.measures.cardinal import (
        get_diversity,
        get_sensitivity,
    )

    return get_diversity, get_sensitivity


def get_benchbench_version() -> str | None:
    try:
        import benchbench

        return getattr(
            benchbench,
            "__version__",
            None,
        )
    except Exception:
        return None


def compute_metrics(
    data: pd.DataFrame,
    cols: list[str],
) -> dict[str, float]:

    get_diversity, get_sensitivity = benchbench_functions()

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------
    #
    # BenchBench returns:
    #
    #   (diversity_W, diversity_MRC)
    #
    diversity = get_diversity(
        data,
        cols,
    )

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------
    #
    # Important:
    #
    # In the supplied BenchBench implementation,
    #
    #   get_sensitivity(...,
    #                   use_weighted_loss=None,
    #                   return_weight=False)
    #
    # returns only the scalar tau.
    #
    # With return_weight=True it returns:
    #
    #   ((tau, MRC), weights)
    #
    # Therefore we explicitly request return_weight=True and extract
    # the rank-difference tuple.
    #
    sensitivity_result = get_sensitivity(
        data,
        cols,
        **SENSITIVITY_KWARGS,
    )

    sensitivity_rank_diff = sensitivity_result[0]

    sensitivity_tau = sensitivity_rank_diff[0]
    sensitivity_mrc = sensitivity_rank_diff[1]

    return {
        "diversity_W": float(diversity[0]),
        "diversity_MRC": float(diversity[1]),
        "sensitivity_tau": float(sensitivity_tau),
        "sensitivity_MRC": float(sensitivity_mrc),
    }


# ============================================================================
# Matrix preparation
# ============================================================================

def prepare_tabarena():
    source = RESULTS_ROOT / "score_method_dataset.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Missing TabArena score matrix: {source}"
        )

    df = pd.read_csv(source)

    require_columns(
        df,
        [
            "dataset",
            "method",
            "dataset_score",
        ],
        "TabArena",
    )

    df["dataset_score"] = pd.to_numeric(
        df["dataset_score"],
        errors="coerce",
    )

    if df["dataset_score"].isna().any():
        raise ValueError(
            "TabArena: dataset_score contains "
            "NaN/non-numeric values"
        )

    if (
        (df["dataset_score"] < 0)
        | (df["dataset_score"] > 1)
    ).any():
        raise ValueError(
            "TabArena: dataset_score is expected to be in [0, 1]"
        )

    if df.duplicated(
        ["method", "dataset"]
    ).any():
        raise ValueError(
            "TabArena: duplicate method/dataset cells"
        )

    matrix = (
        df.pivot(
            index="method",
            columns="dataset",
            values="dataset_score",
        )
        .reset_index()
        .rename(
            columns={
                "method": "model"
            }
        )
    )

    cols = [
        c
        for c in matrix.columns
        if c != "model"
    ]

    validate_matrix(
        matrix,
        cols,
        "model",
        "TabArena",
    )

    if matrix[cols].isna().any().any():
        raise ValueError(
            "TabArena: incomplete model x dataset matrix"
        )

    duplicated = check_duplicate_models(
        matrix,
        "model",
        "TabArena",
    )

    metadata = {
        "source": str(
            source.relative_to(PROJECT_ROOT)
        ),
        "score_column": "dataset_score",
        "score_orientation": "higher_is_better",
        "raw_metric_error_used": False,
        "normalization": (
            "precomputed TabArena dataset_score"
        ),
        "duplicate_model_names": duplicated,
    }

    return matrix, cols, "model", metadata


def load_native(name: str):
    """Load a benchmark through its BenchBench-native loader."""

    candidates = {
        "GLUE": [
            (
                "benchbench.data.glue",
                "load_glue",
            ),
        ],
        "OpenLLM": [
            (
                "benchbench.data.openllm",
                "load_openllm",
            ),
        ],
        "ImageNet": [
            (
                "benchbench.data.imagenet",
                "load_imagenet",
            ),
        ],
    }

    if name not in candidates:
        raise ValueError(
            f"No native loader configured for {name}"
        )

    errors = []

    for module_name, function_name in candidates[name]:
        try:
            module = importlib.import_module(
                module_name
            )

            loader = getattr(
                module,
                function_name,
            )

            result = loader()

            if (
                not isinstance(result, tuple)
                or len(result) != 2
            ):
                raise TypeError(
                    "loader did not return (data, cols)"
                )

            data, cols = result

            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    f"loader returned {type(data)}, "
                    "expected pandas.DataFrame"
                )

            return data, list(cols)

        except Exception as exc:
            errors.append(
                f"{module_name}.{function_name}: {exc}"
            )

    raise RuntimeError(
        f"Could not load {name} through the "
        f"BenchBench-native loader.\n"
        + "\n".join(errors)
    )


def prepare_native(name: str):
    data, cols = load_native(name)

    model_col = NATIVE_MODEL_COLUMNS.get(name)

    if model_col is None:
        raise ValueError(
            f"{name}: no explicit model column configured"
        )

    if model_col not in data.columns:
        raise ValueError(
            f"{name}: expected model column "
            f"{model_col!r}, but it is not present.\n"
            f"Available columns: {list(data.columns)}"
        )

    require_columns(
        data,
        cols,
        name,
    )

    validate_matrix(
        data,
        cols,
        model_col,
        name,
    )

    duplicated = check_duplicate_models(
        data,
        model_col,
        name,
    )

    metadata = {
        "loader": "BenchBench native loader",
        "score_orientation": "higher_is_better",
        "duplicate_model_names": duplicated,
        "num_columns_total": int(len(data.columns)),
    }

    return (
        data,
        cols,
        model_col,
        metadata,
    )


# ============================================================================
# OpenLLM robustness check
# ============================================================================

def prepare_openllm_unique_model_mean(
    data: pd.DataFrame,
    cols: list[str],
    model_col: str,
):
    unique = (
        data.groupby(
            model_col,
            as_index=False,
        )[cols]
        .mean()
    )

    validate_matrix(
        unique,
        cols,
        model_col,
        "OpenLLM unique-model mean",
    )

    metadata = {
        "deduplication": (
            "mean task score across rows sharing "
            "the same model name"
        ),
        "source_native_rows": int(len(data)),
        "unique_models": int(len(unique)),
    }

    return unique, metadata


# ============================================================================
# Output
# ============================================================================

def add_record(
    records,
    name,
    representation,
    data,
    cols,
    model_col,
    metadata,
):
    metrics = compute_metrics(
        data,
        cols,
    )

    record = {
        "benchmark": name,
        "representation": representation,
        "num_models": int(len(data)),
        "num_tasks": int(len(cols)),
        "model_column": model_col,
        **metrics,
        "metadata": metadata,
    }

    print(
        f"{name:10s} "
        f"{representation:20s} "
        f"models={len(data):4d} "
        f"tasks={len(cols):4d} "
        f"W={metrics['diversity_W']:.6f} "
        f"tau={metrics['sensitivity_tau']:.6f} "
        f"sens_MRC={metrics['sensitivity_MRC']:.6f}"
    )

    records.append(record)


def run_fresh():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    # ------------------------------------------------------------------
    # TabArena
    # ------------------------------------------------------------------

    (
        data,
        cols,
        model_col,
        metadata,
    ) = prepare_tabarena()

    add_record(
        records,
        "TabArena",
        "score_matrix",
        data,
        cols,
        model_col,
        metadata,
    )

    # ------------------------------------------------------------------
    # Native BenchBench benchmarks
    # ------------------------------------------------------------------

    for name in (
        "GLUE",
        "OpenLLM",
        "ImageNet",
    ):
        (
            data,
            cols,
            model_col,
            metadata,
        ) = prepare_native(name)

        add_record(
            records,
            name,
            "native",
            data,
            cols,
            model_col,
            metadata,
        )

        # --------------------------------------------------------------
        # OpenLLM unique-model robustness check
        # --------------------------------------------------------------

        if (
            name == "OpenLLM"
            and RUN_OPENLLM_UNIQUE_MODEL_CHECK
            and data[model_col].duplicated().any()
        ):
            (
                unique,
                unique_metadata,
            ) = prepare_openllm_unique_model_mean(
                data,
                cols,
                model_col,
            )

            add_record(
                records,
                "OpenLLM",
                "unique_model_mean",
                unique,
                cols,
                model_col,
                {
                    **metadata,
                    **unique_metadata,
                },
            )

    payload = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "benchmarks": list(BENCHMARKS),
        "run_benchbench": True,
        "openllm_unique_model_check": (
            RUN_OPENLLM_UNIQUE_MODEL_CHECK
        ),
        "sensitivity_kwargs": SENSITIVITY_KWARGS,

        "methodology": {
            "interface": (
                "BenchBench cardinal measures"
            ),
            "score_orientation": (
                "higher_is_better"
            ),
            "tabarena_source": (
                "score_method_dataset.csv"
            ),
            "tabarena_score": (
                "dataset_score"
            ),
            "tabarena_raw_metric_error_used": False,

            "diversity_W_definition": (
                "BenchBench complement of Kendall's W"
            ),

            "diversity_MRC_definition": (
                "mean pairwise normalized maximum "
                "rank change across task rankings"
            ),

            "sensitivity_tau_definition": (
                "BenchBench normalized Kendall "
                "rank distance: (1 - Kendall tau) / 2"
            ),

            "sensitivity_MRC_definition": (
                "maximum normalized rank displacement "
                "found by BenchBench sensitivity optimization"
            ),

            "native_loaders_used": True,
            "hardcoded_metric_values_used": False,
            "synthetic_reference_values_used": False,
        },

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "benchbench_version": (
                get_benchbench_version()
            ),
        },

        "records": records,
    }

    CACHE_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = [
        {
            k: v
            for k, v in record.items()
            if k != "metadata"
        }
        for record in records
    ]

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_FILE,
        index=False,
    )

    CONFIG_FILE.write_text(
        json.dumps(
            {
                "run_benchbench": True,
                "benchmarks": list(BENCHMARKS),
                "openllm_unique_model_check": (
                    RUN_OPENLLM_UNIQUE_MODEL_CHECK
                ),
                "sensitivity_kwargs": (
                    SENSITIVITY_KWARGS
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return payload


def load_cache():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"No cache found at {CACHE_FILE}. "
            "Set RUN_BENCHBENCH=True first."
        )

    return json.loads(
        CACHE_FILE.read_text(
            encoding="utf-8"
        )
    )


def print_summary(payload):
    rows = [
        {
            k: v
            for k, v in record.items()
            if k != "metadata"
        }
        for record in payload["records"]
    ]

    df = pd.DataFrame(rows)

    print("\nBenchBench results")

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        f"\nCache: {CACHE_FILE}"
    )


def main():
    if RUN_BENCHBENCH:
        payload = run_fresh()
    else:
        payload = load_cache()

    print_summary(payload)


if __name__ == "__main__":
    main()