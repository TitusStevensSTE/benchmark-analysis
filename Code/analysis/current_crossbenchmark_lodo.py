from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr

from bencheval.elo_utils import (
    EloHelper,
    elo_solver_tol,
    use_legacy_elo_solver_tol,
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"

if str(BENCHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHBENCH_ROOT))

from benchbench.data.glue import load_glue
from benchbench.data.imagenet import load_imagenet
from benchbench.data.openllm import load_openllm


# ============================================================
# PATHS
# ============================================================

TABARENA_INPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
    / "phase1_crossbenchmark"
)


# ============================================================
# SETTINGS
# ============================================================

TAU_BREAK_THRESHOLD = 0.90
TOP_K = 5

# ------------------------------------------------------------
# TabArena columns
# ------------------------------------------------------------

TABARENA_DATASET_COL = "dataset"
TABARENA_FOLD_COL = "fold"
TABARENA_METHOD_COL = "method"
TABARENA_ERROR_COL = "metric_error"

EXPECTED_TABARENA_DATASETS = 51
EXPECTED_TABARENA_METHODS = 44

# ------------------------------------------------------------
# TabArena official MLE ELO configuration
#
# Explicitly passed for reproducibility. These values match
# your trusted current ELO scripts.
# ------------------------------------------------------------

TABARENA_ELO_CONFIG = {
    "SCALE": 400,
    "BASE": 10,
    "INIT_RATING": 1000,
    "use_pair_aggregation": True,
    "force_iterative_elo": False,
    "max_iter": 1000,
}


# ============================================================
# REPRODUCIBILITY HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of one input file."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_git_commit(path: Path) -> str | None:
    """Return Git commit hash if the given path is a Git repository."""

    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    except Exception:
        return None


def save_metadata(
    output_dir: Path,
    benchmark: str,
    representation: str,
    input_files: list[Path] | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """Save machine-readable metadata for reproducibility."""

    input_hashes = {}

    if input_files is not None:
        for path in input_files:
            if path.exists():
                input_hashes[str(path)] = sha256_file(path)

    metadata = {
        "benchmark": benchmark,
        "representation": representation,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "tau_break_threshold": TAU_BREAK_THRESHOLD,
        "top_k": TOP_K,
        "project_git_commit": get_git_commit(PROJECT_ROOT),
        "benchbench_git_commit": get_git_commit(
            BENCHBENCH_ROOT
        ),
        "input_file_sha256": input_hashes,
    }

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# ============================================================
# GENERIC RANKING HELPERS
# ============================================================

def strict_ranking(
    metric: pd.Series,
    higher_is_better: bool,
) -> pd.Series:
    """
    Convert one scalar value per entity to strict integer ranks.

    Rank 1 is best.

    Exact ties are resolved deterministically through stable sorting
    and the existing Series order. This permits clear rank-change
    calculations and reproducible leaderboards.
    """

    if metric.index.has_duplicates:
        raise ValueError(
            "Metric index contains duplicate entity IDs."
        )

    if metric.isna().any():
        examples = metric[
            metric.isna()
        ].index.tolist()[:10]

        raise ValueError(
            "Cannot rank missing metric values. "
            f"Examples: {examples}"
        )

    values = metric.to_numpy()

    if higher_is_better:
        order = np.argsort(
            -values,
            kind="stable",
        )
    else:
        order = np.argsort(
            values,
            kind="stable",
        )

    ranks = pd.Series(
        np.empty(len(metric), dtype=int),
        index=metric.index,
        name="rank",
    )

    ranks.iloc[order] = np.arange(
        1,
        len(metric) + 1,
    )

    return ranks


def winner_from_ranking(
    ranking: pd.Series,
) -> str:
    """Return the internal entity ID with rank 1."""

    return str(ranking.idxmin())


def normalized_mrc(
    full_ranking: pd.Series,
    lodo_ranking: pd.Series,
) -> float:
    r"""
    Maximum rank change normalized to the interval $[0, 1]$.

    
$$
\mathrm{MRC}
    =
    \frac{
        \max_i
        |r_i^{\mathrm{LODO}} - r_i^{\mathrm{full}}|
    }{n - 1}.
$$

    """

    n = len(full_ranking)

    if n <= 1:
        return 0.0

    absolute_changes = np.abs(
        lodo_ranking.to_numpy()
        - full_ranking.to_numpy()
    )

    return float(
        absolute_changes.max() / (n - 1)
    )


def top_k_overlap(
    full_ranking: pd.Series,
    lodo_ranking: pd.Series,
    k: int = TOP_K,
) -> float:
    """Calculate the fraction of overlapping top-k entities."""

    effective_k = min(
        k,
        len(full_ranking),
    )

    if effective_k == 0:
        return np.nan

    full_top = set(
        full_ranking
        .sort_values()
        .index[:effective_k]
    )

    lodo_top = set(
        lodo_ranking
        .sort_values()
        .index[:effective_k]
    )

    return float(
        len(full_top & lodo_top)
        / effective_k
    )


def safe_correlation(
    function: Callable,
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    """Calculate one correlation only if data permit it."""

    valid = pd.concat(
        [x, y],
        axis=1,
    ).dropna()

    if len(valid) < 3:
        return np.nan, np.nan

    if valid.iloc[:, 0].nunique() < 2:
        return np.nan, np.nan

    if valid.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan

    result = function(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy(),
    )

    return float(result.statistic), float(result.pvalue)


def format_task_preview(
    tasks: list[str],
    max_items: int = 12,
) -> str:
    """Create concise console output for a task list."""

    if len(tasks) <= max_items:
        return str(tasks)

    head = ", ".join(
        map(str, tasks[:6])
    )

    tail = ", ".join(
        map(str, tasks[-3:])
    )

    return (
        f"[{head}, ..., {tail}] "
        f"({len(tasks)} tasks)"
    )


def require_same_entities(
    full_metric: pd.Series,
    lodo_metric: pd.Series,
    removed_task: str,
) -> pd.Series:
    """
    Require identical entity sets in full and LODO leaderboards.

    This prevents invalid rank comparisons caused by missing methods
    or models after task removal.
    """

    full_entities = set(full_metric.index)
    lodo_entities = set(lodo_metric.index)

    if full_entities != lodo_entities:
        missing = sorted(
            full_entities - lodo_entities
        )

        added = sorted(
            lodo_entities - full_entities
        )

        raise ValueError(
            f"Entity universe changed after removing "
            f"'{removed_task}'.\n"
            f"Missing entities: {missing[:10]}\n"
            f"Added entities: {added[:10]}"
        )

    return lodo_metric.reindex(
        full_metric.index
    )


# ============================================================
# GENERIC LODO ANALYSIS
# ============================================================

def run_lodo_analysis(
    benchmark: str,
    representation: str,
    tasks: list[str],
    metric_builder: Callable[[list[str]], pd.Series],
    entity_labels: pd.Series,
    higher_is_better: bool,
    output_dir: Path,
    task_gap_builder: Callable[[str], float] | None = None,
    input_files: list[Path] | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Run a leave-one-task/dataset-out ranking stability analysis.

    `metric_builder(active_tasks)` must return one scalar metric per
    entity. The ranking is then recomputed after each omitted task.
    """

    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — {representation}")
    print("=" * 70)

    # --------------------------------------------------------
    # Full leaderboard
    # --------------------------------------------------------

    full_metric = metric_builder(tasks).copy()

    entity_labels = entity_labels.reindex(
        full_metric.index
    )

    if entity_labels.isna().any():
        raise ValueError(
            "Entity labels do not align with metric IDs."
        )

    full_ranking = strict_ranking(
        metric=full_metric,
        higher_is_better=higher_is_better,
    )

    full_winner_entity = winner_from_ranking(
        full_ranking
    )

    full_winner_label = str(
        entity_labels.loc[
            full_winner_entity
        ]
    )

    full_table = pd.DataFrame(
        {
            "entity_id": full_metric.index,
            "model": entity_labels.to_numpy(),
            "full_metric": full_metric.to_numpy(),
            "full_rank": full_ranking.to_numpy(),
        }
    ).sort_values(
        "full_rank",
        kind="stable",
    )

    print(f"Number of entities: {len(full_metric)}")
    print(f"Number of tasks: {len(tasks)}")
    print("Tasks:", format_task_preview(tasks))
    print("Winner:", full_winner_label)

    print()
    print("FULL LEADERBOARD")
    print()

    print(
        full_table
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # LODO calculations
    # --------------------------------------------------------

    rows = []

    for position, removed_task in enumerate(
        tasks,
        start=1,
    ):

        print(
            f"[{position:04d}/{len(tasks):04d}] "
            f"Removing: {removed_task}"
        )

        active_tasks = [
            task
            for task in tasks
            if task != removed_task
        ]

        lodo_metric = metric_builder(
            active_tasks
        ).copy()

        lodo_metric = require_same_entities(
            full_metric=full_metric,
            lodo_metric=lodo_metric,
            removed_task=removed_task,
        )

        lodo_ranking = strict_ranking(
            metric=lodo_metric,
            higher_is_better=higher_is_better,
        )

        tau, tau_p = kendalltau(
            full_ranking.to_numpy(),
            lodo_ranking.to_numpy(),
        )

        rho, rho_p = spearmanr(
            full_ranking.to_numpy(),
            lodo_ranking.to_numpy(),
        )

        tau = float(tau)
        tau_p = float(tau_p)
        rho = float(rho)
        rho_p = float(rho_p)

        rank_changes = np.abs(
            lodo_ranking.to_numpy()
            - full_ranking.to_numpy()
        )

        lodo_winner_entity = winner_from_ranking(
            lodo_ranking
        )

        lodo_winner_label = str(
            entity_labels.loc[
                lodo_winner_entity
            ]
        )

        rows.append(
            {
                "benchmark": benchmark,
                "representation": representation,
                "removed_task": removed_task,
                "num_remaining_tasks": len(active_tasks),
                "kendall_tau": tau,
                "kendall_p_value": tau_p,
                "normalized_kendall_distance": (
                    float((1 - tau) / 2)
                ),
                "ranking_instability_1_minus_tau": (
                    float(1 - tau)
                ),
                "spearman_rho": rho,
                "spearman_p_value": rho_p,
                "mrc": normalized_mrc(
                    full_ranking,
                    lodo_ranking,
                ),
                "max_rank_change": int(
                    rank_changes.max()
                ),
                "mean_rank_change": float(
                    rank_changes.mean()
                ),
                "num_rank_changes": int(
                    (rank_changes > 0).sum()
                ),
                "top5_overlap": top_k_overlap(
                    full_ranking,
                    lodo_ranking,
                ),
                "full_winner": full_winner_label,
                "lodo_winner": lodo_winner_label,
                "winner_changed": (
                    lodo_winner_entity
                    != full_winner_entity
                ),
                "tau_break": (
                    tau < TAU_BREAK_THRESHOLD
                ),
                "original_winner_lodo_rank": int(
                    lodo_ranking.loc[
                        full_winner_entity
                    ]
                ),
                "task_performance_gap": (
                    task_gap_builder(removed_task)
                    if task_gap_builder is not None
                    else np.nan
                ),
            }
        )

    results_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Gap correlations
    # --------------------------------------------------------

    instability = results_df[
        "ranking_instability_1_minus_tau"
    ]

    pearson_r, pearson_p = safe_correlation(
        pearsonr,
        results_df["task_performance_gap"],
        instability,
    )

    spearman_gap_rho, spearman_gap_p = safe_correlation(
        spearmanr,
        results_df["task_performance_gap"],
        instability,
    )

    kendall_gap_tau, kendall_gap_p = safe_correlation(
        kendalltau,
        results_df["task_performance_gap"],
        instability,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "benchmark": benchmark,
        "representation": representation,
        "n_models": len(full_metric),
        "n_tasks": len(tasks),
        "full_winner": full_winner_label,
        "mean_kendall_tau": float(
            results_df["kendall_tau"].mean()
        ),
        "min_kendall_tau": float(
            results_df["kendall_tau"].min()
        ),
        "mean_normalized_kendall_distance": float(
            results_df[
                "normalized_kendall_distance"
            ].mean()
        ),
        "mean_lodo_instability": float(
            results_df[
                "ranking_instability_1_minus_tau"
            ].mean()
        ),
        "max_lodo_instability": float(
            results_df[
                "ranking_instability_1_minus_tau"
            ].max()
        ),
        "winner_change_frequency": float(
            results_df["winner_changed"].mean()
        ),
        "winner_change_count": int(
            results_df["winner_changed"].sum()
        ),
        "tau_break_frequency": float(
            results_df["tau_break"].mean()
        ),
        "tau_break_count": int(
            results_df["tau_break"].sum()
        ),
        "tau_break_threshold": TAU_BREAK_THRESHOLD,
        "mean_max_rank_change": float(
            results_df["max_rank_change"].mean()
        ),
        "largest_max_rank_change": int(
            results_df["max_rank_change"].max()
        ),
        "mean_num_rank_changes": float(
            results_df["num_rank_changes"].mean()
        ),
        "largest_num_rank_changes": int(
            results_df["num_rank_changes"].max()
        ),
        "mean_top5_overlap": float(
            results_df["top5_overlap"].mean()
        ),
        "most_disruptive_task": str(
            results_df.loc[
                results_df[
                    "kendall_tau"
                ].idxmin(),
                "removed_task",
            ]
        ),
        "gap_pearson_r": pearson_r,
        "gap_pearson_p": pearson_p,
        "gap_spearman_rho": spearman_gap_rho,
        "gap_spearman_p": spearman_gap_p,
        "gap_kendall_tau": kendall_gap_tau,
        "gap_kendall_p": kendall_gap_p,
    }

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_table.to_csv(
        output_dir / "full_ranking.csv",
        index=False,
    )

    results_df.to_csv(
        output_dir / "lodo_results.csv",
        index=False,
    )

    results_df.sort_values(
        [
            "kendall_tau",
            "max_rank_change",
        ],
        ascending=[
            True,
            False,
        ],
    ).head(20).to_csv(
        output_dir
        / "most_disruptive_tasks_top20.csv",
        index=False,
    )

    results_df[
        results_df["winner_changed"]
    ].to_csv(
        output_dir / "winner_changes.csv",
        index=False,
    )

    results_df[
        [
            "removed_task",
            "task_performance_gap",
            "kendall_tau",
            "normalized_kendall_distance",
            "ranking_instability_1_minus_tau",
            "spearman_rho",
            "mrc",
            "max_rank_change",
            "mean_rank_change",
            "num_rank_changes",
            "top5_overlap",
            "winner_changed",
            "tau_break",
        ]
    ].to_csv(
        output_dir / "task_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "representation": representation,
                "n": int(
                    results_df.dropna(
                        subset=[
                            "task_performance_gap",
                            "ranking_instability_1_minus_tau",
                        ]
                    ).shape[0]
                ),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_gap_rho,
                "spearman_p": spearman_gap_p,
                "kendall_tau": kendall_gap_tau,
                "kendall_p": kendall_gap_p,
            }
        ]
    ).to_csv(
        output_dir
        / "performance_gap_correlations.csv",
        index=False,
    )

    pd.DataFrame([summary]).to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    save_metadata(
        output_dir=output_dir,
        benchmark=benchmark,
        representation=representation,
        input_files=input_files,
        extra_metadata=extra_metadata,
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print()
    print("SUMMARY")
    print()

    print(
        f"Mean Kendall tau:    "
        f"{summary['mean_kendall_tau']:.6f}"
    )

    print(
        f"Minimum Kendall tau: "
        f"{summary['min_kendall_tau']:.6f}"
    )

    print(
        f"Mean instability:    "
        f"{summary['mean_lodo_instability']:.6f}"
    )

    print(
        f"Winner changes:      "
        f"{summary['winner_change_count']}"
        f"/{len(results_df)} "
        f"({summary['winner_change_frequency']:.3f})"
    )

    print(
        f"Tau breaks (<{TAU_BREAK_THRESHOLD}): "
        f"{summary['tau_break_count']}"
        f"/{len(results_df)} "
        f"({summary['tau_break_frequency']:.3f})"
    )

    print()
    print("Saved to:")
    print(output_dir)

    return summary


# ============================================================
# TABARENA
# ============================================================

def validate_tabarena_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the expected TabArena data structure.

    Every method must occur exactly once per dataset-fold unit.
    """

    required_columns = {
        TABARENA_DATASET_COL,
        TABARENA_FOLD_COL,
        TABARENA_METHOD_COL,
        TABARENA_ERROR_COL,
    }

    missing = required_columns.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Missing TabArena columns: "
            f"{sorted(missing)}"
        )

    data = data.copy()

    data[TABARENA_ERROR_COL] = pd.to_numeric(
        data[TABARENA_ERROR_COL],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            TABARENA_DATASET_COL,
            TABARENA_FOLD_COL,
            TABARENA_METHOD_COL,
            TABARENA_ERROR_COL,
        ]
    ).copy()

    duplicate_mask = data.duplicated(
        subset=[
            TABARENA_DATASET_COL,
            TABARENA_FOLD_COL,
            TABARENA_METHOD_COL,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate (dataset, fold, method) "
            "rows found in TabArena input."
        )

    datasets = data[
        TABARENA_DATASET_COL
    ].nunique()

    methods = data[
        TABARENA_METHOD_COL
    ].nunique()

    if datasets != EXPECTED_TABARENA_DATASETS:
        raise ValueError(
            f"Expected {EXPECTED_TABARENA_DATASETS} "
            f"datasets, found {datasets}."
        )

    if methods != EXPECTED_TABARENA_METHODS:
        raise ValueError(
            f"Expected {EXPECTED_TABARENA_METHODS} "
            f"methods, found {methods}."
        )

    methods_per_unit = (
        data.groupby(
            [
                TABARENA_DATASET_COL,
                TABARENA_FOLD_COL,
            ]
        )[TABARENA_METHOD_COL]
        .nunique()
    )

    if (
        methods_per_unit
        != EXPECTED_TABARENA_METHODS
    ).any():
        raise ValueError(
            "Not every TabArena dataset-fold unit "
            "contains all methods."
        )

    return data


def calculate_tabarena_score_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    r"""
    Compute the TabArena score representation.

    First, average error across folds:

    
$$
\bar e_{m,d}
    =
    \frac{1}{F_d}
    \sum_f e_{m,d,f}.
$$


    Then normalize separately per dataset:

    
$$
S_{m,d}
    =
    \max
    \left(
        0,
        \frac{
            \operatorname{median}(e_{\cdot,d}) - e_{m,d}
        }{
            \operatorname{median}(e_{\cdot,d})
            - \min(e_{\cdot,d})
        }
    \right).
$$


    The final benchmark score is the equally weighted mean of
    $S_{m,d}$ across datasets.
    """

    method_dataset = (
        data.groupby(
            [
                TABARENA_DATASET_COL,
                TABARENA_METHOD_COL,
            ],
            as_index=False,
            sort=True,
        )[TABARENA_ERROR_COL]
        .mean()
    )

    expected_datasets = data[
        TABARENA_DATASET_COL
    ].nunique()

    coverage = (
        method_dataset.groupby(
            TABARENA_METHOD_COL
        )[TABARENA_DATASET_COL]
        .nunique()
    )

    if (coverage != expected_datasets).any():
        raise ValueError(
            "Incomplete TabArena method-dataset coverage."
        )

    parts = []

    for _, group in method_dataset.groupby(
        TABARENA_DATASET_COL,
        sort=True,
    ):
        group = group.copy()

        best_error = float(
            group[TABARENA_ERROR_COL].min()
        )

        median_error = float(
            group[TABARENA_ERROR_COL].median()
        )

        denominator = (
            median_error - best_error
        )

        if np.isclose(denominator, 0.0):
            group["dataset_score"] = 1.0
        else:
            group["dataset_score"] = (
                (
                    median_error
                    - group[TABARENA_ERROR_COL]
                )
                / denominator
            ).clip(lower=0.0)

        group["best_error"] = best_error
        group["median_error"] = median_error

        parts.append(group)

    score_table = pd.concat(
        parts,
        ignore_index=True,
    )

    if score_table["dataset_score"].min() < -1e-10:
        raise RuntimeError(
            "Negative TabArena score detected."
        )

    if score_table["dataset_score"].max() > 1.0 + 1e-10:
        raise RuntimeError(
            "TabArena score above 1 detected."
        )

    return score_table


def make_tabarena_gap_builder(
    score_table: pd.DataFrame,
) -> Callable[[str], float]:
    """Return dataset winner-runner-up normalized score gaps."""

    def task_gap(dataset: str) -> float:
        subset = score_table[
            score_table[TABARENA_DATASET_COL]
            == dataset
        ].sort_values(
            [
                "dataset_score",
                TABARENA_METHOD_COL,
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        if len(subset) < 2:
            return np.nan

        return float(
            subset.iloc[0]["dataset_score"]
            - subset.iloc[1]["dataset_score"]
        )

    return task_gap


def run_tabarena() -> list[dict]:
    """Run TabArena official Score and official MLE ELO LODO."""

    print()
    print("=" * 70)
    print("LOAD TABARENA")
    print("=" * 70)

    if not TABARENA_INPUT_FILE.exists():
        raise FileNotFoundError(
            "TabArena input file not found:\n"
            f"{TABARENA_INPUT_FILE}"
        )

    data = pd.read_csv(TABARENA_INPUT_FILE)
    data = validate_tabarena_data(data)

    datasets = sorted(
        data[TABARENA_DATASET_COL]
        .unique()
        .tolist()
    )

    methods = sorted(
        data[TABARENA_METHOD_COL]
        .unique()
        .tolist()
    )

    print(f"Rows: {len(data):,}")
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: {len(methods)}")

    labels = pd.Series(
        methods,
        index=methods,
        name="model",
    )

    # --------------------------------------------------------
    # TabArena official Score
    # --------------------------------------------------------

    score_table = calculate_tabarena_score_table(
        data
    )

    def build_score(
        active_datasets: list[str],
    ) -> pd.Series:
        metric = (
            score_table[
                score_table[TABARENA_DATASET_COL].isin(
                    active_datasets
                )
            ]
            .groupby(
                TABARENA_METHOD_COL,
                sort=False,
            )["dataset_score"]
            .mean()
            .reindex(methods)
        )

        metric.name = "official_score"

        return metric

    score_dir = (
        OUTPUT_ROOT
        / "tabarena"
        / "official_score"
    )

    score_summary = run_lodo_analysis(
        benchmark="TabArena",
        representation="official_score",
        tasks=datasets,
        metric_builder=build_score,
        entity_labels=labels,
        higher_is_better=True,
        output_dir=score_dir,
        task_gap_builder=make_tabarena_gap_builder(
            score_table
        ),
        input_files=[TABARENA_INPUT_FILE],
        extra_metadata={
            "score_definition": (
                "Mean metric_error across folds per "
                "(dataset, method), best-to-median "
                "normalization per dataset, and equal "
                "mean across datasets."
            ),
            "expected_datasets": EXPECTED_TABARENA_DATASETS,
            "expected_methods": EXPECTED_TABARENA_METHODS,
        },
    )

    score_table.to_csv(
        score_dir
        / "tabarena_method_dataset_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # TabArena official MLE ELO
    # --------------------------------------------------------

    elo_helper = EloHelper(
        method_col=TABARENA_METHOD_COL,
        task_col=TABARENA_DATASET_COL,
        error_col=TABARENA_ERROR_COL,
        split_col=TABARENA_FOLD_COL,
    )

    full_battles = elo_helper.convert_results_to_battles(
        results_df=data
    )

    if full_battles.empty:
        raise RuntimeError(
            "TabArena battle construction produced no battles."
        )

    def build_elo(
        active_datasets: list[str],
    ) -> pd.Series:
        battles = full_battles[
            full_battles[TABARENA_DATASET_COL].isin(
                set(active_datasets)
            )
        ]

        if battles.empty:
            raise RuntimeError(
                "No TabArena battles remain."
            )

        metric = elo_helper.compute_mle_elo(
            battles=battles,
            **TABARENA_ELO_CONFIG,
        ).reindex(methods)

        metric.name = "official_mle_elo"

        return metric

    elo_dir = (
        OUTPUT_ROOT
        / "tabarena"
        / "official_mle_elo"
    )

    elo_summary = run_lodo_analysis(
        benchmark="TabArena",
        representation="official_mle_elo",
        tasks=datasets,
        metric_builder=build_elo,
        entity_labels=labels,
        higher_is_better=True,
        output_dir=elo_dir,
        task_gap_builder=make_tabarena_gap_builder(
            score_table
        ),
        input_files=[TABARENA_INPUT_FILE],
        extra_metadata={
            "elo_algorithm": (
                "EloHelper.compute_mle_elo"
            ),
            "elo_configuration": TABARENA_ELO_CONFIG,
            "elo_solver_tolerance": elo_solver_tol(),
            "legacy_solver_tolerance_enabled": (
                use_legacy_elo_solver_tol()
            ),
            "battle_definition": (
                "All method pairs within each "
                "(dataset, fold); lower metric_error wins; "
                "each dataset receives equal total weight."
            ),
            "lodo_implementation": (
                "Filter full battle table by removed dataset "
                "and recompute MLE ELO."
            ),
        },
    )

    return [
        score_summary,
        elo_summary,
    ]


# ============================================================
# BENCHBENCH HELPERS
# ============================================================

def validate_tasks(
    benchmark: str,
    returned_columns: list[str],
    expected_columns: list[str],
) -> list[str]:
    """Validate task identity while preserving loader order."""

    returned = list(returned_columns)
    expected = list(expected_columns)

    if set(returned) != set(expected):
        raise ValueError(
            f"Unexpected {benchmark} task columns.\n"
            f"Expected: {expected}\n"
            f"Received: {returned}"
        )

    return returned


def build_mean_score_metric(
    data: pd.DataFrame,
    entity_ids: pd.Index,
) -> Callable[[list[str]], pd.Series]:
    """
    Build an equal-weight arithmetic mean score metric.

    Missing values are rejected to avoid unequal implicit task
    weighting across rows/models.
    """

    def metric_builder(
        active_tasks: list[str],
    ) -> pd.Series:
        values = data[active_tasks]

        if values.isna().any().any():
            raise ValueError(
                "Missing benchmark task values detected."
            )

        metric = values.mean(axis=1)
        metric.index = entity_ids
        metric.name = "mean_score"

        return metric

    return metric_builder


def make_score_gap_builder(
    data: pd.DataFrame,
) -> Callable[[str], float]:
    """Return task winner-runner-up raw score gaps."""

    def task_gap(task: str) -> float:
        values = (
            pd.to_numeric(
                data[task],
                errors="coerce",
            )
            .dropna()
            .sort_values(ascending=False)
        )

        if len(values) < 2:
            return np.nan

        return float(
            values.iloc[0]
            - values.iloc[1]
        )

    return task_gap


def make_row_entities(
    data: pd.DataFrame,
    label_column: str,
    prefix: str,
) -> tuple[pd.Index, pd.Series]:
    """
    Create unique internal IDs while preserving potentially duplicate
    model names as display labels.
    """

    entity_ids = pd.Index(
        [
            f"{prefix}_{i:04d}"
            for i in range(len(data))
        ],
        name="entity_id",
    )

    labels = pd.Series(
        data[label_column].astype(str).to_numpy(),
        index=entity_ids,
        name="model",
    )

    return entity_ids, labels


# ============================================================
# GLUE
# ============================================================

def run_glue() -> dict:

    glue_tasks = [
        "CoLA",
        "SST-2",
        "MRPC",
        "STS-B",
        "QQP",
        "MNLI",
        "QNLI",
        "RTE",
        "WNLI",
    ]

    print()
    print("=" * 70)
    print("LOAD GLUE")
    print("=" * 70)

    data, returned_columns = load_glue()

    tasks = validate_tasks(
        benchmark="GLUE",
        returned_columns=returned_columns,
        expected_columns=glue_tasks,
    )

    data = data[
        ["Model", *tasks]
    ].copy()

    print(f"GLUE input rows: {len(data)}")
    print(
        f"GLUE unique model labels: "
        f"{data['Model'].nunique()}"
    )

    # Duplicate model labels are allowed in BenchBench GLUE.
    # Therefore we preserve rows and use unique internal IDs.
    entity_ids, labels = make_row_entities(
        data=data,
        label_column="Model",
        prefix="glue_row",
    )

    return run_lodo_analysis(
        benchmark="GLUE",
        representation="row_level",
        tasks=tasks,
        metric_builder=build_mean_score_metric(
            data=data,
            entity_ids=entity_ids,
        ),
        entity_labels=labels,
        higher_is_better=True,
        output_dir=(
            OUTPUT_ROOT
            / "glue"
            / "row_level"
        ),
        task_gap_builder=make_score_gap_builder(
            data
        ),
    )


# ============================================================
# OPENLLM
# ============================================================

def run_openllm() -> list[dict]:

    openllm_tasks = [
        "ARC",
        "HellaSwag",
        "MMLU",
        "TruthfulQA",
        "Winogrande",
        "GSM8K",
    ]

    print()
    print("=" * 70)
    print("LOAD OPENLLM")
    print("=" * 70)

    data, returned_columns = load_openllm()

    tasks = validate_tasks(
        benchmark="OpenLLM",
        returned_columns=returned_columns,
        expected_columns=openllm_tasks,
    )

    data = data[
        ["Model", *tasks]
    ].copy()

    print(f"OpenLLM input rows: {len(data)}")
    print(
        f"OpenLLM unique models: "
        f"{data['Model'].nunique()}"
    )

    summaries = []

    # --------------------------------------------------------
    # Original row-level representation
    # --------------------------------------------------------

    row_ids, row_labels = make_row_entities(
        data=data,
        label_column="Model",
        prefix="openllm_row",
    )

    summaries.append(
        run_lodo_analysis(
            benchmark="OpenLLM",
            representation="row_level",
            tasks=tasks,
            metric_builder=build_mean_score_metric(
                data=data,
                entity_ids=row_ids,
            ),
            entity_labels=row_labels,
            higher_is_better=True,
            output_dir=(
                OUTPUT_ROOT
                / "openllm"
                / "row_level"
            ),
            task_gap_builder=make_score_gap_builder(
                data
            ),
        )
    )

    # --------------------------------------------------------
    # Unique model mean representation
    # --------------------------------------------------------

    unique_data = (
        data.groupby(
            "Model",
            sort=False,
        )[tasks]
        .mean()
        .reset_index()
    )

    unique_ids = pd.Index(
        unique_data["Model"].astype(str),
        name="entity_id",
    )

    unique_labels = pd.Series(
        unique_data["Model"].astype(str).to_numpy(),
        index=unique_ids,
        name="model",
    )

    summaries.append(
        run_lodo_analysis(
            benchmark="OpenLLM",
            representation="unique_model_mean",
            tasks=tasks,
            metric_builder=build_mean_score_metric(
                data=unique_data,
                entity_ids=unique_ids,
            ),
            entity_labels=unique_labels,
            higher_is_better=True,
            output_dir=(
                OUTPUT_ROOT
                / "openllm"
                / "unique_model_mean"
            ),
            task_gap_builder=make_score_gap_builder(
                unique_data
            ),
        )
    )

    return summaries


# ============================================================
# IMAGENET
# ============================================================

def run_imagenet() -> dict:

    imagenet_tasks = [
        f"acc_{i}"
        for i in range(1000)
    ]

    print()
    print("=" * 70)
    print("LOAD IMAGENET")
    print("=" * 70)

    data, returned_columns = load_imagenet(
        num_task=1000
    )

    tasks = validate_tasks(
        benchmark="ImageNet",
        returned_columns=returned_columns,
        expected_columns=imagenet_tasks,
    )

    data = data[
        ["Weight", *tasks]
    ].copy()

    # Row IDs are used even if names are unique. This gives the
    # same robust handling as GLUE and OpenLLM.
    entity_ids, labels = make_row_entities(
        data=data,
        label_column="Weight",
        prefix="imagenet_row",
    )

    return run_lodo_analysis(
        benchmark="ImageNet",
        representation="mean_class_accuracy",
        tasks=tasks,
        metric_builder=build_mean_score_metric(
            data=data,
            entity_ids=entity_ids,
        ),
        entity_labels=labels,
        higher_is_better=True,
        output_dir=(
            OUTPUT_ROOT
            / "imagenet"
            / "mean_class_accuracy"
        ),
        task_gap_builder=make_score_gap_builder(
            data
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHASE 1 CROSS-BENCHMARK LODO ANALYSIS")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Output root:")
    print(OUTPUT_ROOT)

    print()
    print(
        "Kendall tau break threshold:",
        TAU_BREAK_THRESHOLD,
    )

    if use_legacy_elo_solver_tol():
        print()
        print(
            "WARNING: TabArena legacy ELO solver "
            "tolerance is enabled."
        )

    summaries = []

    # Main benchmark: TabArena Score and TabArena ELO.
    summaries.extend(
        run_tabarena()
    )

    # Comparison benchmarks.
    summaries.append(
        run_glue()
    )

    summaries.extend(
        run_openllm()
    )

    summaries.append(
        run_imagenet()
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(summaries)

    summary_file = (
        OUTPUT_ROOT
        / "crossbenchmark_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("CROSS-BENCHMARK SUMMARY")
    print("=" * 70)
    print()

    display_columns = [
        "benchmark",
        "representation",
        "n_models",
        "n_tasks",
        "full_winner",
        "min_kendall_tau",
        "mean_lodo_instability",
        "winner_change_frequency",
        "tau_break_frequency",
        "largest_max_rank_change",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(summary_file)

    print()
    print("Done.")


if __name__ == "__main__":
    main()