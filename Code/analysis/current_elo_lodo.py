from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr

from bencheval.elo_utils import EloHelper


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "current_lodo_elo_results.csv"
)


# ============================================================
# LOAD TABARENA DATA
# ============================================================

print("=" * 70)
print("LOADING TABARENA DATA")
print("=" * 70)

tabarena_df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Loaded dataframe: "
    f"{tabarena_df.shape}"
)

print(
    f"Number of datasets: "
    f"{tabarena_df['dataset'].nunique()}"
)

print(
    f"Number of methods: "
    f"{tabarena_df['method'].nunique()}"
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = {
    "dataset",
    "fold",
    "method",
    "metric_error",
}

missing_columns = (
    required_columns
    - set(tabarena_df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# Ensure metric_error is numeric
tabarena_df["metric_error"] = (
    pd.to_numeric(
        tabarena_df["metric_error"],
        errors="coerce",
    )
)


# Remove invalid rows
tabarena_df = (
    tabarena_df
    .dropna(
        subset=[
            "dataset",
            "fold",
            "method",
            "metric_error",
        ]
    )
    .copy()
)


# ============================================================
# DATASETS
# ============================================================

datasets = sorted(
    tabarena_df["dataset"]
    .dropna()
    .unique()
)

print(
    f"\nDatasets ({len(datasets)}):"
)

for i, dataset in enumerate(
    datasets,
    start=1,
):
    print(
        f"{i:2d}: {dataset}"
    )


if len(datasets) != 51:
    raise ValueError(
        f"Expected 51 datasets, "
        f"found {len(datasets)}."
    )


# ============================================================
# CREATE TABARENA ELO HELPER
# ============================================================

elo_helper = EloHelper(
    method_col="method",
    task_col="dataset",
    error_col="metric_error",
    split_col="fold",
)


# ============================================================
# ELO CALCULATION FUNCTION
# ============================================================

def calculate_tabarena_elo(
    results_df,
):
    """
    Calculate TabArena Elo using exactly the same
    EloHelper configuration as current_elo_1_51.py.
    """

    # --------------------------------------------------------
    # Convert results -> battles
    # --------------------------------------------------------

    battles = (
        elo_helper
        .convert_results_to_battles(
            results_df=results_df
        )
    )

    if battles.empty:
        raise ValueError(
            "No battles were generated."
        )


    # --------------------------------------------------------
    # Calculate MLE Elo
    # --------------------------------------------------------

    elo_ranking = (
        elo_helper
        .compute_mle_elo(
            battles=battles,
            SCALE=400,
            BASE=10,
            INIT_RATING=1000,
            use_pair_aggregation=True,
            force_iterative_elo=False,
            max_iter=1000,
        )
    )

    return (
        elo_ranking,
        battles,
    )


# ============================================================
# FULL BENCHMARK ELO
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING FULL BENCHMARK ELO")
print("=" * 70)

full_elo, full_battles = (
    calculate_tabarena_elo(
        tabarena_df
    )
)

full_elo = (
    full_elo
    .sort_values(
        ascending=False
    )
)


# ============================================================
# FULL RANKING
# ============================================================

full_ranking = pd.DataFrame(
    {
        "method": full_elo.index,
        "elo_full": full_elo.values,
    }
)

full_ranking["rank_full"] = (
    full_ranking["elo_full"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


full_winner = (
    full_ranking
    .iloc[0]["method"]
)

print(
    f"\nFull benchmark winner: "
    f"{full_winner}"
)

print(
    f"Number of methods in full ranking: "
    f"{len(full_ranking)}"
)

print(
    f"Number of full benchmark battles: "
    f"{len(full_battles):,}"
)


# ============================================================
# LODO ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STARTING LODO ANALYSIS")
print("=" * 70)

lodo_results = []


for i, dataset in enumerate(
    datasets,
    start=1,
):

    print(
        f"[{i}/{len(datasets)}] "
        f"Leaving out: {dataset}"
    )


    # ========================================================
    # REMOVE ONE DATASET
    # ========================================================

    df_lodo = (
        tabarena_df[
            tabarena_df["dataset"] != dataset
        ]
        .copy()
    )


    # ========================================================
    # CALCULATE LODO ELO
    # ========================================================

    elo_lodo, battles_lodo = (
        calculate_tabarena_elo(
            df_lodo
        )
    )

    elo_lodo = (
        elo_lodo
        .sort_values(
            ascending=False
        )
    )


    # ========================================================
    # CREATE LODO RANKING
    # ========================================================

    lodo_ranking = pd.DataFrame(
        {
            "method": elo_lodo.index,
            "elo_lodo": elo_lodo.values,
        }
    )

    lodo_ranking["rank_lodo"] = (
        lodo_ranking["elo_lodo"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    # ========================================================
    # LODO WINNER
    # ========================================================

    lodo_winner = (
        lodo_ranking
        .iloc[0]["method"]
    )

    winner_changed = (
        lodo_winner
        != full_winner
    )


    # ========================================================
    # MERGE FULL + LODO RANKINGS
    # ========================================================

    comparison = (
        full_ranking
        .merge(
            lodo_ranking,
            on="method",
            how="inner",
        )
    )


    # ========================================================
    # RANK DIFFERENCES
    # ========================================================

    comparison["rank_change"] = (
        comparison["rank_lodo"]
        - comparison["rank_full"]
    )

    comparison["abs_rank_change"] = (
        comparison["rank_change"]
        .abs()
    )


    # ========================================================
    # KENDALL TAU
    # ========================================================

    kendall_tau, kendall_p = (
        kendalltau(
            comparison["rank_full"],
            comparison["rank_lodo"],
        )
    )


    # ========================================================
    # SPEARMAN RHO
    # ========================================================

    spearman_rho, spearman_p = (
        spearmanr(
            comparison["rank_full"],
            comparison["rank_lodo"],
        )
    )


    # ========================================================
    # MAXIMUM RANK MOVEMENT
    # ========================================================

    max_rank_change = (
        comparison["abs_rank_change"]
        .max()
    )


    # ========================================================
    # NUMBER OF METHODS WHOSE RANK CHANGED
    # ========================================================

    num_rank_changes = (
        comparison["rank_change"] != 0
    ).sum()


    # ========================================================
    # NUMBER OF METHODS
    # ========================================================

    num_methods = (
        len(comparison)
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    lodo_results.append(
        {
            "dataset_removed": dataset,

            # Winner information
            "full_winner": full_winner,
            "lodo_winner": lodo_winner,
            "winner_changed": winner_changed,

            # Ranking stability
            "kendall_tau": kendall_tau,
            "kendall_p": kendall_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,

            # Rank movement
            "max_rank_change": max_rank_change,
            "num_rank_changes": num_rank_changes,

            # Additional information
            "num_methods": num_methods,
            "num_battles_lodo": len(
                battles_lodo
            ),
        }
    )


# ============================================================
# CONVERT RESULTS TO DATAFRAME
# ============================================================

lodo_results = pd.DataFrame(
    lodo_results
)


# ============================================================
# SORT BY KENDALL TAU
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
# SAVE RESULTS
# ============================================================

lodo_results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LODO RESULTS")
print("=" * 70)

print(
    lodo_results.to_string(
        index=False
    )
)


# ============================================================
# TOP MOST UNSTABLE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("MOST INFLUENTIAL DATASETS")
print("=" * 70)

print(
    lodo_results[
        [
            "dataset_removed",
            "full_winner",
            "lodo_winner",
            "winner_changed",
            "kendall_tau",
            "spearman_rho",
            "max_rank_change",
            "num_rank_changes",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# WINNER CHANGES
# ============================================================

winner_change_count = (
    lodo_results["winner_changed"]
    .sum()
)

print("\n" + "=" * 70)
print("WINNER CHANGE SUMMARY")
print("=" * 70)

print(
    f"Full benchmark winner: "
    f"{full_winner}"
)

print(
    f"LODO cases with winner change: "
    f"{winner_change_count}/{len(datasets)}"
)

print(
    f"LODO cases with unchanged winner: "
    f"{len(datasets) - winner_change_count}"
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)