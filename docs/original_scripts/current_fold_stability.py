from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ============================================================
# TABARENA FOLD STABILITY ANALYSIS
# ============================================================
#
# Purpose
# -------
# Analyze sensitivity of TabArena method rankings to evaluation
# splits ("folds").
#
# Main analyses
# -------------
# 1. Benchmark-level fold-specific rankings
#    - Common folds 0-8
#    - All 51 datasets
#    - 44 methods
#
# 2. Pairwise fold ranking stability
#    - Kendall tau
#    - Spearman rho
#
# 3. Method-level rank stability across folds
#
# 4. Fold LODO
#    - Remove one common fold at a time
#    - Compare resulting ranking with the full 9-fold ranking
#
# 5. Dataset-specific fold stability
#    - Uses ALL folds available for each dataset
#    - 34 datasets with 9 folds
#    - 17 datasets with 30 folds
#
# Important
# ---------
# The TabArena Score is calculated separately for every
# (dataset, fold) group:
#
#     best error   -> Score = 1
#     median error -> Score = 0
#     worse than median -> Score clipped to 0
#
# This is intentionally different from the full benchmark Score,
# which first averages metric_error over folds and then normalizes.
# Here the goal is to measure sensitivity to the evaluation split.
#
# No random operations are used.
#
# ============================================================


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
    / "fold_stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_COL = "dataset"
FOLD_COL = "fold"
METHOD_COL = "method"
ERROR_COL = "metric_error"

COMMON_FOLDS = tuple(range(9))

EXPECTED_DATASETS = 51
EXPECTED_METHODS = 44


# ============================================================
# HELPER: TABARENA SCORE
# ============================================================

def calculate_tabarena_scores(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the TabArena normalized Score for one
    (dataset, fold) group.

    Formula:

        score =
            (median_error - method_error)
            /
            (median_error - best_error)

    Negative values are clipped to zero.

    If best_error == median_error, all methods receive 1.0,
    matching the existing lodo_score.py implementation.
    """

    errors = group[ERROR_COL]

    best_error = float(errors.min())
    median_error = float(errors.median())

    denominator = median_error - best_error

    result = group[
        [
            DATASET_COL,
            FOLD_COL,
            METHOD_COL,
            ERROR_COL,
        ]
    ].copy()

    if np.isclose(denominator, 0.0):
        result["dataset_fold_score"] = 1.0
    else:
        result["dataset_fold_score"] = (
            (
                median_error
                - result[ERROR_COL]
            )
            / denominator
        ).clip(lower=0.0)

    result["best_error"] = best_error
    result["median_error"] = median_error

    return result


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TABARENA FOLD STABILITY ANALYSIS")
print("=" * 70)

print()
print(f"Project root:\n{PROJECT_ROOT}")
print(f"\nInput file:\n{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput shape: {df.shape}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    DATASET_COL,
    FOLD_COL,
    METHOD_COL,
    ERROR_COL,
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\nValidating input...")

if df[
    [DATASET_COL, FOLD_COL, METHOD_COL]
].duplicated().any():

    duplicates = df.loc[
        df[
            [DATASET_COL, FOLD_COL, METHOD_COL]
        ].duplicated(keep=False),
        [
            DATASET_COL,
            FOLD_COL,
            METHOD_COL,
        ],
    ]

    raise ValueError(
        "Duplicate (dataset, fold, method) rows found.\n"
        f"Examples:\n{duplicates.head(10)}"
    )


if df[ERROR_COL].isna().any():
    missing_errors = int(
        df[ERROR_COL].isna().sum()
    )

    raise ValueError(
        f"Found {missing_errors} missing "
        f"{ERROR_COL} values."
    )


num_datasets = df[DATASET_COL].nunique()
num_methods = df[METHOD_COL].nunique()

if num_datasets != EXPECTED_DATASETS:
    raise ValueError(
        f"Expected {EXPECTED_DATASETS} datasets, "
        f"found {num_datasets}."
    )

if num_methods != EXPECTED_METHODS:
    raise ValueError(
        f"Expected {EXPECTED_METHODS} methods, "
        f"found {num_methods}."
    )


# ============================================================
# DATASET/FOLD COVERAGE
# ============================================================

datasets_per_fold = (
    df.groupby(FOLD_COL)[DATASET_COL]
    .nunique()
    .sort_index()
)

print("\nDatasets per fold:")
print(datasets_per_fold.to_string())

methods_per_dataset_fold = (
    df.groupby(
        [
            DATASET_COL,
            FOLD_COL,
        ]
    )[METHOD_COL]
    .nunique()
)

if (
    methods_per_dataset_fold
    .nunique()
    != 1
):
    raise ValueError(
        "The number of methods is not constant "
        "across dataset/fold groups."
    )

observed_methods_per_group = int(
    methods_per_dataset_fold.iloc[0]
)

if observed_methods_per_group != EXPECTED_METHODS:
    raise ValueError(
        f"Expected {EXPECTED_METHODS} methods per "
        f"dataset/fold, found {observed_methods_per_group}."
    )

print(
    f"\nMethods per dataset/fold: "
    f"{observed_methods_per_group}"
)


# ============================================================
# CALCULATE SCORE FOR EVERY DATASET/FOLD
# ============================================================

print(
    "\nCalculating TabArena Score for every "
    "(dataset, fold) group..."
)

score_parts: list[pd.DataFrame] = []

for (
    dataset,
    fold,
), group in df.groupby(
    [DATASET_COL, FOLD_COL],
    sort=True,
):

    score_parts.append(
        calculate_tabarena_scores(group)
    )

dataset_fold_scores = pd.concat(
    score_parts,
    ignore_index=True,
)

dataset_fold_scores = (
    dataset_fold_scores
    .sort_values(
        [
            DATASET_COL,
            FOLD_COL,
            METHOD_COL,
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# SCORE VALIDATION
# ============================================================

score_min = float(
    dataset_fold_scores["dataset_fold_score"].min()
)

score_max = float(
    dataset_fold_scores["dataset_fold_score"].max()
)

print(
    f"\nDataset-fold Score range: "
    f"{score_min:.6f} to {score_max:.6f}"
)

if (
    score_min < -1e-8
    or score_max > 1.0 + 1e-8
):
    raise ValueError(
        "Dataset-fold scores outside expected [0, 1] range."
    )


expected_score_rows = (
    df[
        [
            DATASET_COL,
            FOLD_COL,
        ]
    ]
    .drop_duplicates()
    .shape[0]
    * EXPECTED_METHODS
)

if len(dataset_fold_scores) != expected_score_rows:
    raise ValueError(
        f"Unexpected score-table size: "
        f"{len(dataset_fold_scores)}; "
        f"expected {expected_score_rows}."
    )

print(
    f"Validated {len(dataset_fold_scores):,} "
    "dataset-fold-method score rows."
)


# ============================================================
# SAVE ALL DATASET-FOLD SCORES
# ============================================================

dataset_fold_score_file = (
    OUTPUT_DIR
    / "dataset_fold_scores.csv"
)

dataset_fold_scores.to_csv(
    dataset_fold_score_file,
    index=False,
)


# ============================================================
# COMMON 9-FOLD ANALYSIS
# ============================================================

common_df = dataset_fold_scores[
    dataset_fold_scores[FOLD_COL].isin(
        COMMON_FOLDS
    )
].copy()

common_fold_coverage = (
    common_df
    .groupby(DATASET_COL)[FOLD_COL]
    .nunique()
)

if not (
    common_fold_coverage
    == len(COMMON_FOLDS)
).all():

    problematic = common_fold_coverage[
        common_fold_coverage
        != len(COMMON_FOLDS)
    ]

    raise ValueError(
        "Not every dataset contains all common folds 0-8.\n"
        f"Problematic datasets:\n{problematic}"
    )


common_datasets = sorted(
    common_df[DATASET_COL].unique()
)

print(
    "\nCommon analysis: "
    f"{len(common_datasets)} datasets × "
    f"{len(COMMON_FOLDS)} folds × "
    f"{EXPECTED_METHODS} methods"
)


# ============================================================
# BENCHMARK SCORE PER COMMON FOLD / METHOD
# ============================================================

fold_method_scores = (
    common_df
    .groupby(
        [
            FOLD_COL,
            METHOD_COL,
        ],
        as_index=False,
    )["dataset_fold_score"]
    .mean()
    .rename(
        columns={
            "dataset_fold_score":
                "benchmark_score"
        }
    )
)


# Deterministic sorting.
fold_method_scores = (
    fold_method_scores
    .sort_values(
        [
            FOLD_COL,
            "benchmark_score",
            METHOD_COL,
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


# Rank using competition ranks.
fold_method_scores["rank"] = (
    fold_method_scores
    .groupby(FOLD_COL)["benchmark_score"]
    .rank(
        ascending=False,
        method="min",
    )
)


fold_method_scores = (
    fold_method_scores
    .sort_values(
        [
            FOLD_COL,
            "rank",
            METHOD_COL,
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT FOLD WINNERS
# ============================================================

print("\nFold winners:")
print("-" * 70)

for fold in COMMON_FOLDS:

    subset = fold_method_scores[
        fold_method_scores[FOLD_COL] == fold
    ].sort_values(
        [
            "rank",
            METHOD_COL,
        ]
    )

    winner = subset.iloc[0]

    print(
        f"Fold {fold}: "
        f"{winner[METHOD_COL]} "
        f"(score={winner['benchmark_score']:.6f})"
    )


# ============================================================
# SAVE FOLD RANKINGS
# ============================================================

fold_ranking_file = (
    OUTPUT_DIR
    / "fold_method_rankings.csv"
)

fold_method_scores.to_csv(
    fold_ranking_file,
    index=False,
)


# ============================================================
# BENCHMARK-LEVEL PAIRWISE FOLD STABILITY
# ============================================================

ranking_matrix = (
    fold_method_scores
    .pivot(
        index=METHOD_COL,
        columns=FOLD_COL,
        values="rank",
    )
    .sort_index()
)

pairwise_rows: list[dict] = []

for fold_a, fold_b in combinations(
    COMMON_FOLDS,
    2,
):

    ranks_a = ranking_matrix[
        fold_a
    ]

    ranks_b = ranking_matrix[
        fold_b
    ]

    # Explicitly align on identical method order.
    aligned = pd.concat(
        [
            ranks_a,
            ranks_b,
        ],
        axis=1,
        join="inner",
    ).sort_index()

    kendall_tau, kendall_p = (
        kendalltau(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
        )
    )

    spearman_rho, spearman_p = (
        spearmanr(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
        )
    )

    pairwise_rows.append(
        {
            "fold_a": fold_a,
            "fold_b": fold_b,
            "kendall_tau": kendall_tau,
            "kendall_p": kendall_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        }
    )


fold_pairwise_df = pd.DataFrame(
    pairwise_rows
)


fold_pairwise_file = (
    OUTPUT_DIR
    / "fold_pairwise_correlations.csv"
)

fold_pairwise_df.to_csv(
    fold_pairwise_file,
    index=False,
)


print("\n" + "=" * 70)
print("PAIRWISE FOLD RANKING STABILITY")
print("=" * 70)

print(
    f"\nNumber of fold pairs: "
    f"{len(fold_pairwise_df)}"
)

print(
    f"\nKendall tau:"
    f"\n  Mean: "
    f"{fold_pairwise_df['kendall_tau'].mean():.6f}"
    f"\n  Std:  "
    f"{fold_pairwise_df['kendall_tau'].std():.6f}"
    f"\n  Min:  "
    f"{fold_pairwise_df['kendall_tau'].min():.6f}"
    f"\n  Max:  "
    f"{fold_pairwise_df['kendall_tau'].max():.6f}"
)

print(
    f"\nSpearman rho:"
    f"\n  Mean: "
    f"{fold_pairwise_df['spearman_rho'].mean():.6f}"
    f"\n  Std:  "
    f"{fold_pairwise_df['spearman_rho'].std():.6f}"
)


# ============================================================
# METHOD-LEVEL RANK STABILITY
# ============================================================

method_rank_stability = (
    ranking_matrix
    .copy()
    .reset_index()
)

method_rank_stability["mean_rank"] = (
    method_rank_stability[
        list(COMMON_FOLDS)
    ].mean(axis=1)
)

method_rank_stability["rank_std"] = (
    method_rank_stability[
        list(COMMON_FOLDS)
    ].std(axis=1)
)

method_rank_stability["rank_min"] = (
    method_rank_stability[
        list(COMMON_FOLDS)
    ].min(axis=1)
)

method_rank_stability["rank_max"] = (
    method_rank_stability[
        list(COMMON_FOLDS)
    ].max(axis=1)
)

method_rank_stability["rank_range"] = (
    method_rank_stability["rank_max"]
    - method_rank_stability["rank_min"]
)

method_rank_stability = (
    method_rank_stability
    .sort_values(
        [
            "rank_std",
            "mean_rank",
            METHOD_COL,
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .reset_index(drop=True)
)


method_rank_file = (
    OUTPUT_DIR
    / "method_rank_stability.csv"
)

method_rank_stability.to_csv(
    method_rank_file,
    index=False,
)


print("\n" + "=" * 70)
print("METHOD-LEVEL RANK STABILITY")
print("=" * 70)

print(
    "\nMost variable methods by rank SD:"
)

print(
    method_rank_stability[
        [
            METHOD_COL,
            "mean_rank",
            "rank_std",
            "rank_min",
            "rank_max",
            "rank_range",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# FOLD LODO
# ============================================================

# Full 9-fold benchmark score.
full_method_scores = (
    common_df
    .groupby(
        METHOD_COL,
        as_index=False,
    )["dataset_fold_score"]
    .mean()
    .rename(
        columns={
            "dataset_fold_score":
                "benchmark_score"
        }
    )
)

full_method_scores = (
    full_method_scores
    .sort_values(
        [
            "benchmark_score",
            METHOD_COL,
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

full_method_scores["rank"] = (
    full_method_scores["benchmark_score"]
    .rank(
        ascending=False,
        method="min",
    )
)

full_winner = (
    full_method_scores.iloc[0][METHOD_COL]
)

# Fixed method ordering for EVERY rank comparison.
method_order = sorted(
    full_method_scores[METHOD_COL]
)

full_ranking = (
    full_method_scores
    .set_index(METHOD_COL)["rank"]
    .reindex(method_order)
)


fold_lodo_rows: list[dict] = []

for removed_fold in COMMON_FOLDS:

    remaining = common_df[
        common_df[FOLD_COL]
        != removed_fold
    ]

    scores_without_fold = (
        remaining
        .groupby(
            METHOD_COL,
            as_index=False,
        )["dataset_fold_score"]
        .mean()
        .rename(
            columns={
                "dataset_fold_score":
                    "benchmark_score"
            }
        )
    )

    scores_without_fold = (
        scores_without_fold
        .sort_values(
            [
                "benchmark_score",
                METHOD_COL,
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    scores_without_fold["rank"] = (
        scores_without_fold["benchmark_score"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    removed_ranking = (
        scores_without_fold
        .set_index(METHOD_COL)["rank"]
        .reindex(method_order)
    )

    # IMPORTANT:
    # Both ranking vectors are explicitly aligned to the same
    # method order before calculating correlations.
    kendall_tau, kendall_p = (
        kendalltau(
            full_ranking,
            removed_ranking,
        )
    )

    spearman_rho, spearman_p = (
        spearmanr(
            full_ranking,
            removed_ranking,
        )
    )

    rank_change = (
        removed_ranking
        - full_ranking
    )

    winner_without_fold = (
        scores_without_fold.iloc[0][METHOD_COL]
    )

    winner_runnerup_gap = (
        scores_without_fold.iloc[0]["benchmark_score"]
        - scores_without_fold.iloc[1]["benchmark_score"]
    )

    fold_lodo_rows.append(
        {
            "removed_fold": removed_fold,
            "kendall_tau": kendall_tau,
            "kendall_p": kendall_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "full_winner": full_winner,
            "winner_without_fold":
                winner_without_fold,
            "winner_changed":
                winner_without_fold != full_winner,
            "full_winner_rank":
                int(
                    removed_ranking.loc[
                        full_winner
                    ]
                ),
            "changed_rank_count":
                int(
                    (
                        rank_change != 0
                    ).sum()
                ),
            "max_absolute_rank_change":
                float(
                    rank_change.abs().max()
                ),
            "mean_absolute_rank_change":
                float(
                    rank_change.abs().mean()
                ),
            "winner_runnerup_gap":
                float(
                    winner_runnerup_gap
                ),
        }
    )


fold_lodo_df = pd.DataFrame(
    fold_lodo_rows
)


fold_lodo_file = (
    OUTPUT_DIR
    / "fold_lodo_results.csv"
)

fold_lodo_df.to_csv(
    fold_lodo_file,
    index=False,
)


print("\n" + "=" * 70)
print("FOLD LODO RANKING STABILITY")
print("=" * 70)

print(
    fold_lodo_df[
        [
            "removed_fold",
            "kendall_tau",
            "spearman_rho",
            "winner_without_fold",
            "winner_changed",
            "full_winner_rank",
            "changed_rank_count",
            "max_absolute_rank_change",
        ]
    ].to_string(index=False)
)

print("\nSummary:")
print(
    f"Mean Kendall tau: "
    f"{fold_lodo_df['kendall_tau'].mean():.6f}"
)

print(
    f"Minimum Kendall tau: "
    f"{fold_lodo_df['kendall_tau'].min():.6f}"
)

print(
    f"Mean absolute rank change: "
    f"{fold_lodo_df['mean_absolute_rank_change'].mean():.6f}"
)

print(
    f"Maximum absolute rank change observed: "
    f"{fold_lodo_df['max_absolute_rank_change'].max():.0f}"
)

print(
    f"Winner changes: "
    f"{int(fold_lodo_df['winner_changed'].sum())}"
    f" / {len(fold_lodo_df)} folds"
)


# ============================================================
# DATASET-SPECIFIC FOLD STABILITY
# ============================================================

dataset_fold_rows: list[dict] = []

for dataset, group in df.groupby(
    DATASET_COL,
    sort=True,
):

    available_folds = sorted(
        group[FOLD_COL].unique()
    )

    ranking_by_fold: dict[
        int,
        pd.Series,
    ] = {}

    for fold in available_folds:

        fold_group = group[
            group[FOLD_COL] == fold
        ].copy()

        scored = calculate_tabarena_scores(
            fold_group
        )

        ranking = (
            scored
            .set_index(METHOD_COL)[
                "dataset_fold_score"
            ]
            .rank(
                ascending=False,
                method="average",
            )
        )

        ranking_by_fold[fold] = ranking

    pairwise_tau = []

    for fold_a, fold_b in combinations(
        available_folds,
        2,
    ):

        ranks_a = ranking_by_fold[
            fold_a
        ]

        ranks_b = ranking_by_fold[
            fold_b
        ]

        aligned = pd.concat(
            [
                ranks_a,
                ranks_b,
            ],
            axis=1,
            join="inner",
        ).sort_index()

        tau, _ = kendalltau(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
        )

        if not np.isnan(tau):
            pairwise_tau.append(
                float(tau)
            )

    if pairwise_tau:
        mean_tau = float(
            np.mean(pairwise_tau)
        )

        std_tau = float(
            np.std(
                pairwise_tau,
                ddof=1,
            )
        )

        min_tau = float(
            np.min(pairwise_tau)
        )

        max_tau = float(
            np.max(pairwise_tau)
        )

    else:
        mean_tau = np.nan
        std_tau = np.nan
        min_tau = np.nan
        max_tau = np.nan

    dataset_fold_rows.append(
        {
            DATASET_COL: dataset,
            "n_folds": len(available_folds),
            "n_fold_pairs": len(
                pairwise_tau
            ),
            "mean_kendall_tau": mean_tau,
            "std_kendall_tau": std_tau,
            "min_kendall_tau": min_tau,
            "max_kendall_tau": max_tau,
        }
    )


dataset_fold_stability = (
    pd.DataFrame(dataset_fold_rows)
    .sort_values(
        [
            "mean_kendall_tau",
            DATASET_COL,
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(drop=True)
)


dataset_fold_file = (
    OUTPUT_DIR
    / "dataset_fold_stability.csv"
)

dataset_fold_stability.to_csv(
    dataset_fold_file,
    index=False,
)


print("\n" + "=" * 70)
print("DATASET-SPECIFIC FOLD STABILITY")
print("=" * 70)

print(
    "\nLeast stable datasets by mean Kendall tau:"
)

print(
    dataset_fold_stability[
        [
            DATASET_COL,
            "n_folds",
            "mean_kendall_tau",
            "std_kendall_tau",
            "min_kendall_tau",
            "max_kendall_tau",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print(
    f"\nOverall mean dataset fold stability: "
    f"{dataset_fold_stability['mean_kendall_tau'].mean():.6f}"
)

print(
    f"Minimum dataset fold stability: "
    f"{dataset_fold_stability['mean_kendall_tau'].min():.6f}"
)

print(
    f"Maximum dataset fold stability: "
    f"{dataset_fold_stability['mean_kendall_tau'].max():.6f}"
)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

for output_file in [
    dataset_fold_score_file,
    fold_ranking_file,
    fold_pairwise_file,
    method_rank_file,
    fold_lodo_file,
    dataset_fold_file,
]:

    print(f"  {output_file}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)