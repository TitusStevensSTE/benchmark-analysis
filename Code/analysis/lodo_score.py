from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)
# ============================================================
# TABARENA SCORE LODO ANALYSIS
# ============================================================
#
# Reproduces the TabArena complementary "Score" aggregation:
#
#   1. Compute one mean metric_error per method × dataset
#   2. For each dataset:
#        best error   -> Score = 1
#        median error -> Score = 0
#        worse values -> Score < 0 -> clipped to 0
#   3. Average normalized dataset scores across datasets
#   4. Rank methods by this aggregate Score
#
# LODO:
#   Remove one dataset at a time
#   Recompute the complete Score ranking
#   Compare LODO ranking against full-benchmark ranking
#
# Output:
#   results/lodo_score_results.csv
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr


# ============================================================
# PATHS
# ============================================================

# Script:
#
# benchmark-analysis/
# └── Code/
#     └── analysis/
#         ├── data/
#         │   └── df_results_tabarena_2026_08_13.csv
#         ├── results/
#         │   └── lodo_score_results.csv
#         └── lodo_score.py


DATA_FILE = (
    TABARENA_DATA_FILE
)

RESULTS_DIR = (
    RESULTS_DIR
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "lodo_score_results.csv"
)

FULL_RANKING_FILE = (
    RESULTS_DIR
    / "score_full_ranking.csv"
)

TASK_SCORE_FILE = (
    RESULTS_DIR
    / "score_method_dataset.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_COL = "dataset"
METHOD_COL = "method"
ERROR_COL = "metric_error"

EXPECTED_DATASETS = 51


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TABARENA SCORE — LEAVE-ONE-DATASET-OUT ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

print("\nChecking paths...")

print(f"Project root:")
print(f"  {PROJECT_ROOT}")

print(f"\nInput data:")
print(f"  {DATA_FILE}")

print(f"\nOutput directory:")
print(f"  {RESULTS_DIR}")

print(f"\nLODO output:")
print(f"  {OUTPUT_FILE}")


if not DATA_FILE.exists():
    raise FileNotFoundError(
        "TabArena data file does not exist:\n"
        f"  {DATA_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading TabArena data...")

df = pd.read_csv(DATA_FILE)

print(
    f"Loaded data: {df.shape}"
)

print(
    f"Datasets: "
    f"{df[DATASET_COL].nunique()}"
)

print(
    f"Methods: "
    f"{df[METHOD_COL].nunique()}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = {
    DATASET_COL,
    METHOD_COL,
    ERROR_COL,
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        f"  {sorted(missing_columns)}"
    )


# ============================================================
# CLEAN ERROR COLUMN
# ============================================================

df[ERROR_COL] = pd.to_numeric(
    df[ERROR_COL],
    errors="coerce",
)

before_rows = len(df)

df = df.dropna(
    subset=[
        DATASET_COL,
        METHOD_COL,
        ERROR_COL,
    ]
)

after_rows = len(df)

if before_rows != after_rows:
    print(
        f"\nDropped "
        f"{before_rows - after_rows} rows "
        "with missing metric_error."
    )


# ============================================================
# DATASET/METHOD STRUCTURE
# ============================================================

datasets = sorted(
    df[DATASET_COL]
    .unique()
)

methods = sorted(
    df[METHOD_COL]
    .unique()
)

print(
    f"\nFinal dataset count: "
    f"{len(datasets)}"
)

print(
    f"Final method count: "
    f"{len(methods)}"
)


if len(datasets) != EXPECTED_DATASETS:
    print(
        "\nWARNING:"
        f" expected {EXPECTED_DATASETS} datasets "
        f"but found {len(datasets)}."
    )


# ============================================================
# STEP 1:
# METHOD × DATASET ERROR
# ============================================================

print(
    "\nAggregating metric_error to "
    "method × dataset level..."
)

method_dataset = (
    df.groupby(
        [
            DATASET_COL,
            METHOD_COL,
        ],
        as_index=False,
    )[ERROR_COL]
    .mean()
)

print(
    "Method × dataset table:",
    method_dataset.shape,
)


# ============================================================
# COVERAGE CHECK
# ============================================================

coverage = (
    method_dataset
    .groupby(METHOD_COL)[DATASET_COL]
    .nunique()
)

print(
    "\nMethod coverage:"
)

print(
    coverage.describe()
)

incomplete_methods = coverage[
    coverage < len(datasets)
]

if len(incomplete_methods) > 0:

    print(
        "\nWARNING:"
        f" {len(incomplete_methods)} methods "
        "do not have results on all datasets."
    )

    print(
        incomplete_methods
        .sort_values()
        .to_string()
    )


# ============================================================
# TABARENA SCORE FUNCTION
# ============================================================

def calculate_tabarena_dataset_scores(
    method_dataset_subset,
):
    """
    Calculate TabArena's normalized Score per dataset.

    For one dataset:

        best error   -> Score = 1
        median error -> Score = 0

    Linear interpolation:

        score =
            (median_error - method_error)
            /
            (median_error - best_error)

    Negative values are clipped to zero.

    Higher Score = better.

    The returned table contains:
        dataset
        method
        metric_error
        dataset_score
    """

    score_rows = []

    for dataset, group in (
        method_dataset_subset
        .groupby(DATASET_COL)
    ):

        group = group.copy()

        errors = group[ERROR_COL]

        # ----------------------------------------------------
        # Best error
        # ----------------------------------------------------

        best_error = errors.min()

        # ----------------------------------------------------
        # Median error
        # ----------------------------------------------------

        median_error = errors.median()

        # ----------------------------------------------------
        # Degenerate case:
        # best == median
        #
        # This happens when at least half of the methods
        # have the same best error.
        #
        # The linear denominator becomes zero.
        # In this case all methods should receive the same
        # normalized value because there is no meaningful
        # separation between best and median.
        # ----------------------------------------------------

        denominator = (
            median_error
            - best_error
        )

        if np.isclose(
            denominator,
            0.0,
        ):

            group["dataset_score"] = 1.0

        else:

            group["dataset_score"] = (
                (
                    median_error
                    - group[ERROR_COL]
                )
                / denominator
            )

            # TabArena clips negative scores to zero.
            group["dataset_score"] = (
                group["dataset_score"]
                .clip(lower=0.0)
            )

        # ----------------------------------------------------
        # Store metadata useful for validation
        # ----------------------------------------------------

        group["best_error"] = best_error

        group["median_error"] = median_error

        score_rows.append(
            group[
                [
                    DATASET_COL,
                    METHOD_COL,
                    ERROR_COL,
                    "dataset_score",
                    "best_error",
                    "median_error",
                ]
            ]
        )

    if not score_rows:
        return pd.DataFrame(
            columns=[
                DATASET_COL,
                METHOD_COL,
                ERROR_COL,
                "dataset_score",
                "best_error",
                "median_error",
            ]
        )

    return pd.concat(
        score_rows,
        ignore_index=True,
    )


# ============================================================
# STEP 2:
# CALCULATE DATASET SCORES
# ============================================================

print(
    "\nCalculating TabArena normalized "
    "Score per dataset..."
)

score_table = (
    calculate_tabarena_dataset_scores(
        method_dataset
    )
)

print(
    "Score table:",
    score_table.shape,
)


# ============================================================
# SANITY CHECK SCORE RANGE
# ============================================================

minimum_score = (
    score_table["dataset_score"]
    .min()
)

maximum_score = (
    score_table["dataset_score"]
    .max()
)

print(
    "\nDataset Score range:"
)

print(
    f"  Minimum: {minimum_score:.8f}"
)

print(
    f"  Maximum: {maximum_score:.8f}"
)


if minimum_score < -1e-10:
    raise RuntimeError(
        "Negative Score found after clipping."
    )


if maximum_score > 1.0 + 1e-10:
    raise RuntimeError(
        "Score greater than 1 found."
    )


# ============================================================
# SAVE TASK-LEVEL SCORE TABLE
# ============================================================

score_table.to_csv(
    TASK_SCORE_FILE,
    index=False,
)

print(
    "\nSaved task-level Score table:"
)

print(
    f"  {TASK_SCORE_FILE}"
)


# ============================================================
# AGGREGATE SCORE ACROSS DATASETS
# ============================================================

def aggregate_score(
    score_table_subset,
):
    """
    Aggregate dataset Scores across datasets.

    TabArena Score:
        arithmetic mean of per-dataset normalized scores.
    """

    aggregate = (
        score_table_subset
        .groupby(
            METHOD_COL,
            as_index=False,
        )["dataset_score"]
        .mean()
        .rename(
            columns={
                "dataset_score":
                    "score"
            }
        )
    )

    aggregate = aggregate.sort_values(
        [
            "score",
            METHOD_COL,
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    aggregate["rank"] = aggregate["score"].rank(
        ascending=False,
        method="average",
    )

    return aggregate


# ============================================================
# FULL BENCHMARK SCORE RANKING
# ============================================================

print(
    "\nCalculating full-benchmark Score ranking..."
)

full_ranking = aggregate_score(
    score_table
)

print(
    "\nFull benchmark ranking:"
)

print(
    full_ranking.head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE FULL RANKING
# ============================================================

full_ranking.to_csv(
    FULL_RANKING_FILE,
    index=False,
)

print(
    "\nSaved full Score ranking:"
)

print(
    f"  {FULL_RANKING_FILE}"
)


# ============================================================
# FULL RANKING DICTIONARY
# ============================================================

full_rank_dict = dict(
    zip(
        full_ranking[METHOD_COL],
        full_ranking["rank"],
    )
)

full_score_dict = dict(
    zip(
        full_ranking[METHOD_COL],
        full_ranking["score"],
    )
)


full_winner = (
    full_ranking.iloc[0][METHOD_COL]
)

full_winner_score = (
    full_ranking.iloc[0]["score"]
)


print(
    "\nFull-benchmark winner:"
)

print(
    f"  {full_winner}"
)

print(
    f"  Score = "
    f"{full_winner_score:.8f}"
)


# ============================================================
# RANK COMPARISON FUNCTION
# ============================================================

def compare_rankings(
    full_ranking,
    lodo_ranking,
):
    """
    Compare full and LODO rankings.

    Returns:
        Kendall tau
        Spearman rho
        number of rank changes
        maximum rank change
        full winner
        LODO winner
    """

    # --------------------------------------------------------
    # Align methods
    # --------------------------------------------------------

    full = (
        full_ranking
        [[METHOD_COL, "rank"]]
        .rename(
            columns={
                "rank": "rank_full"
            }
        )
    )

    lodo = (
        lodo_ranking
        [[METHOD_COL, "rank"]]
        .rename(
            columns={
                "rank": "rank_lodo"
            }
        )
    )

    merged = full.merge(
        lodo,
        on=METHOD_COL,
        how="inner",
    )

    if len(merged) < 2:
        raise RuntimeError(
            "Not enough methods to compare rankings."
        )

    # --------------------------------------------------------
    # Rank arrays
    # --------------------------------------------------------

    ranks_full = (
        merged["rank_full"]
        .to_numpy()
    )

    ranks_lodo = (
        merged["rank_lodo"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Kendall tau
    # --------------------------------------------------------

    kendall_result = kendalltau(
        ranks_full,
        ranks_lodo,
    )

    kendall_tau_value = (
        float(kendall_result.statistic)
        if hasattr(
            kendall_result,
            "statistic",
        )
        else float(kendall_result[0])
    )

    # --------------------------------------------------------
    # Spearman rho
    # --------------------------------------------------------

    spearman_result = spearmanr(
        ranks_full,
        ranks_lodo,
    )

    spearman_rho_value = (
        float(spearman_result.statistic)
        if hasattr(
            spearman_result,
            "statistic",
        )
        else float(spearman_result[0])
    )

    # --------------------------------------------------------
    # Rank movement
    # --------------------------------------------------------

    rank_changes = (
        merged["rank_lodo"]
        - merged["rank_full"]
    )

    absolute_rank_changes = (
        rank_changes.abs()
    )

    num_rank_changes = int(
        (
            absolute_rank_changes
            > 0
        )
        .sum()
    )

    max_rank_change = int(
        absolute_rank_changes.max()
    )

    # --------------------------------------------------------
    # Winner information
    # --------------------------------------------------------

    full_winner_method = (
        full_ranking.iloc[0][METHOD_COL]
    )

    lodo_winner_method = (
        lodo_ranking.iloc[0][METHOD_COL]
    )

    # --------------------------------------------------------
    # Winner rank in LODO
    # --------------------------------------------------------

    full_winner_lodo_row = (
        lodo_ranking[
            lodo_ranking[METHOD_COL]
            == full_winner_method
        ]
    )

    if len(full_winner_lodo_row) > 0:

        full_winner_lodo_rank = int(
            full_winner_lodo_row.iloc[0]["rank"]
        )

    else:

        full_winner_lodo_rank = np.nan

    return {
        "kendall_tau":
            kendall_tau_value,

        "spearman_rho":
            spearman_rho_value,

        "max_rank_change":
            max_rank_change,

        "num_rank_changes":
            num_rank_changes,

        "full_winner":
            full_winner_method,

        "lodo_winner":
            lodo_winner_method,

        "full_winner_lodo_rank":
            full_winner_lodo_rank,
    }


# ============================================================
# LODO FUNCTION
# ============================================================

def run_lodo(
    score_table,
    removed_dataset,
):
    """
    Remove one dataset and recompute the complete
    TabArena Score ranking.
    """

    remaining = score_table[
        score_table[DATASET_COL]
        != removed_dataset
    ].copy()

    if (
        remaining[DATASET_COL]
        .nunique()
        < 1
    ):

        raise RuntimeError(
            "LODO would leave zero datasets."
        )

    # --------------------------------------------------------
    # Aggregate Score
    # --------------------------------------------------------

    lodo_ranking = aggregate_score(
        remaining
    )

    # --------------------------------------------------------
    # Compare against full ranking
    # --------------------------------------------------------

    comparison = compare_rankings(
        full_ranking,
        lodo_ranking,
    )

    # --------------------------------------------------------
    # Removed dataset Score of full winner
    # --------------------------------------------------------

    winner_task_score_rows = (
        score_table[
            (
                score_table[DATASET_COL]
                == removed_dataset
            )
            &
            (
                score_table[METHOD_COL]
                == full_winner
            )
        ]
    )

    if len(winner_task_score_rows) > 0:

        full_winner_removed_score = float(
            winner_task_score_rows.iloc[0][
                "dataset_score"
            ]
        )

    else:

        full_winner_removed_score = np.nan

    # --------------------------------------------------------
    # Dataset winner
    # --------------------------------------------------------

    removed_dataset_scores = (
        score_table[
            score_table[DATASET_COL]
            == removed_dataset
        ]
        .sort_values(
            "dataset_score",
            ascending=False,
        )
    )

    if len(removed_dataset_scores) > 0:

        dataset_winner = (
            removed_dataset_scores.iloc[0][
                METHOD_COL
            ]
        )

        dataset_winner_score = float(
            removed_dataset_scores.iloc[0][
                "dataset_score"
            ]
        )

    else:

        dataset_winner = np.nan
        dataset_winner_score = np.nan

    # --------------------------------------------------------
    # Runner-up on removed dataset
    # --------------------------------------------------------

    if len(removed_dataset_scores) >= 2:

        dataset_runnerup = (
            removed_dataset_scores.iloc[1][
                METHOD_COL
            ]
        )

        dataset_runnerup_score = float(
            removed_dataset_scores.iloc[1][
                "dataset_score"
            ]
        )

    else:

        dataset_runnerup = np.nan
        dataset_runnerup_score = np.nan

    # --------------------------------------------------------
    # Winner-runner-up gap
    # --------------------------------------------------------

    if (
        pd.notna(dataset_winner_score)
        and pd.notna(dataset_runnerup_score)
    ):

        dataset_gap = (
            dataset_winner_score
            - dataset_runnerup_score
        )

    else:

        dataset_gap = np.nan

    return {
        "dataset_removed":
            removed_dataset,

        "num_datasets_remaining":
            remaining[DATASET_COL].nunique(),

        "kendall_tau":
            comparison["kendall_tau"],

        "spearman_rho":
            comparison["spearman_rho"],

        "max_rank_change":
            comparison["max_rank_change"],

        "num_rank_changes":
            comparison["num_rank_changes"],

        "full_winner":
            comparison["full_winner"],

        "lodo_winner":
            comparison["lodo_winner"],

        "full_winner_lodo_rank":
            comparison[
                "full_winner_lodo_rank"
            ],

        "dataset_winner":
            dataset_winner,

        "dataset_winner_score":
            dataset_winner_score,

        "dataset_runnerup":
            dataset_runnerup,

        "dataset_runnerup_score":
            dataset_runnerup_score,

        "winner_runnerup_score_gap":
            dataset_gap,

        "full_winner_removed_dataset_score":
            full_winner_removed_score,
    }


# ============================================================
# RUN ALL LODO EXPERIMENTS
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING SCORE LODO")
print("=" * 70)

lodo_rows = []

for i, dataset in enumerate(
    datasets,
    start=1,
):

    print(
        f"[{i:02d}/{len(datasets):02d}] "
        f"Removing: {dataset}"
    )

    result = run_lodo(
        score_table,
        dataset,
    )

    lodo_rows.append(
        result
    )


# ============================================================
# BUILD RESULT TABLE
# ============================================================

lodo_results = pd.DataFrame(
    lodo_rows
)


# ============================================================
# ORDER RESULTS BY INSTABILITY
# ============================================================

lodo_results = (
    lodo_results
    .sort_values(
        "kendall_tau",
        ascending=True,
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE LODO RESULTS
# ============================================================

lodo_results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    "\nSaved LODO results:"
)

print(
    f"  {OUTPUT_FILE}"
)


# ============================================================
# PRINT MOST INFLUENTIAL DATASETS
# ============================================================

print("\n")
print("=" * 70)
print("MOST INFLUENTIAL DATASETS")
print("=" * 70)


display_columns = [
    "dataset_removed",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
    "full_winner",
    "lodo_winner",
    "winner_runnerup_score_gap",
]


print(
    lodo_results[
        display_columns
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# OVERALL LODO SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("LODO SUMMARY")
print("=" * 70)


print(
    "\nKendall tau:"
)

print(
    f"  Minimum: "
    f"{lodo_results['kendall_tau'].min():.6f}"
)

print(
    f"  Maximum: "
    f"{lodo_results['kendall_tau'].max():.6f}"
)

print(
    f"  Mean: "
    f"{lodo_results['kendall_tau'].mean():.6f}"
)

print(
    f"  Median: "
    f"{lodo_results['kendall_tau'].median():.6f}"
)


print(
    "\nSpearman rho:"
)

print(
    f"  Minimum: "
    f"{lodo_results['spearman_rho'].min():.6f}"
)

print(
    f"  Maximum: "
    f"{lodo_results['spearman_rho'].max():.6f}"
)

print(
    f"  Mean: "
    f"{lodo_results['spearman_rho'].mean():.6f}"
)

print(
    f"  Median: "
    f"{lodo_results['spearman_rho'].median():.6f}"
)


print(
    "\nMaximum rank change:"
)

print(
    f"  Maximum observed: "
    f"{lodo_results['max_rank_change'].max()}"
)

print(
    f"  Mean: "
    f"{lodo_results['max_rank_change'].mean():.4f}"
)


print(
    "\nNumber of rank changes:"
)

print(
    f"  Maximum observed: "
    f"{lodo_results['num_rank_changes'].max()}"
)

print(
    f"  Mean: "
    f"{lodo_results['num_rank_changes'].mean():.4f}"
)


# ============================================================
# FULL-WINNER SURVIVAL
# ============================================================

full_winner_survives = (
    lodo_results["lodo_winner"]
    == full_winner
).sum()

print(
    "\nFull-benchmark winner remains "
    "LODO winner:"
)

print(
    f"  {full_winner_survives} / "
    f"{len(lodo_results)} datasets"
)

print(
    f"  Fraction: "
    f"{full_winner_survives / len(lodo_results):.4f}"
)


# ============================================================
# VERIFY RESULT STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION")
print("=" * 70)


expected_rows = len(datasets)

actual_rows = len(lodo_results)

print(
    f"\nExpected LODO rows: "
    f"{expected_rows}"
)

print(
    f"Actual LODO rows:   "
    f"{actual_rows}"
)


if actual_rows != expected_rows:

    raise RuntimeError(
        "LODO result does not contain exactly "
        "one row per dataset."
    )


duplicate_datasets = (
    lodo_results[
        "dataset_removed"
    ]
    .duplicated()
    .sum()
)

print(
    f"Duplicate removed datasets: "
    f"{duplicate_datasets}"
)


if duplicate_datasets != 0:

    raise RuntimeError(
        "Duplicate dataset entries found "
        "in LODO results."
    )


required_output_columns = [
    "dataset_removed",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
    "full_winner",
    "lodo_winner",
]


missing_output_columns = (
    set(required_output_columns)
    - set(lodo_results.columns)
)


if missing_output_columns:

    raise RuntimeError(
        "Missing output columns:\n"
        f"  {sorted(missing_output_columns)}"
    )


print(
    "\nValidation successful."
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("SCORE LODO ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles generated:"
)

print(
    f"  {OUTPUT_FILE}"
)

print(
    f"  {FULL_RANKING_FILE}"
)

print(
    f"  {TASK_SCORE_FILE}"
)

print("\nDone.")