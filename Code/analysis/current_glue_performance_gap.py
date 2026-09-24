from analysis_config import (BENCHBENCH_ROOT, PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# PROJECT SETUP
# ============================================================
# Use the original GLUE loader from benchbench
from benchbench.data.glue import load_glue


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = (
    RESULTS_DIR
    / "phase1_crossbenchmark"
    / "glue"
    / "performance_gap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PAIRWISE_OUTPUT = (
    OUTPUT_DIR
    / "glue_pairwise_performance_gap_lodo.csv"
)

BINNED_OUTPUT = (
    OUTPUT_DIR
    / "glue_performance_gap_bins.csv"
)


# ============================================================
# RANKING FUNCTIONS
# ============================================================

def compute_ranking(data, cols):
    """
    Compute leaderboard ranking using the arithmetic mean
    across the selected tasks.

    Higher score = better.

    Returns
    -------
    ranking : pd.Series
        Rank 1 = best.
    aggregate_score : pd.Series
        Mean score across selected tasks.
    """
    aggregate_score = data[cols].mean(axis=1)

    order = np.argsort(
        -aggregate_score.to_numpy(),
        kind="stable"
    )

    ranking = pd.Series(
        np.empty(len(data), dtype=int),
        index=data.index
    )

    ranking.iloc[order] = np.arange(
        1,
        len(data) + 1
    )

    return ranking, aggregate_score


# ============================================================
# LOAD ORIGINAL GLUE DATA
# ============================================================

print("=" * 80)
print("GLUE PERFORMANCE-GAP VS. LODO RANKING REVERSAL ANALYSIS")
print("=" * 80)
print()

print("Benchbench root:")
print(BENCHBENCH_ROOT)
print()

data, cols = load_glue()

print("Number of models:", len(data))
print("Number of tasks:", len(cols))
print("Tasks:", cols)
print()


# ============================================================
# FULL GLUE RANKING
# ============================================================

full_ranking, full_score = compute_ranking(
    data,
    cols
)

data = data.copy()

data["full_score"] = full_score
data["full_rank"] = full_ranking

# Sort for display
full_sorted = (
    data[
        [
            "Model",
            "full_score",
            "full_rank"
        ]
    ]
    .sort_values("full_rank")
)

print("=" * 80)
print("FULL GLUE RANKING")
print("=" * 80)
print()

print(
    full_sorted.head(15).to_string(index=False)
)

print()

full_winner_idx = data["full_score"].idxmax()
full_winner = data.loc[
    full_winner_idx,
    "Model"
]

print("Full winner:", full_winner)

# Top-2 gap
top2 = full_sorted.head(2)

top1_score = top2.iloc[0]["full_score"]
top2_score = top2.iloc[1]["full_score"]

print(
    f"Top-1 vs Top-2 gap: "
    f"{top1_score - top2_score:.6f}"
)

print()


# ============================================================
# PAIRWISE PERFORMANCE GAPS
# ============================================================

print("=" * 80)
print("PAIRWISE LODO ANALYSIS")
print("=" * 80)
print()

pairwise_results = []

n_models = len(data)

# ------------------------------------------------------------
# Iterate over every unique model pair
# ------------------------------------------------------------

for i in range(n_models):

    model_i_idx = data.index[i]
    model_i_name = data.loc[
        model_i_idx,
        "Model"
    ]

    model_i_score = data.loc[
        model_i_idx,
        "full_score"
    ]

    model_i_rank = data.loc[
        model_i_idx,
        "full_rank"
    ]

    for j in range(i + 1, n_models):

        model_j_idx = data.index[j]
        model_j_name = data.loc[
            model_j_idx,
            "Model"
        ]

        model_j_score = data.loc[
            model_j_idx,
            "full_score"
        ]

        model_j_rank = data.loc[
            model_j_idx,
            "full_rank"
        ]

        # ----------------------------------------------------
        # Full-ranking ordering
        # ----------------------------------------------------

        full_score_difference = (
            model_i_score - model_j_score
        )

        performance_gap = abs(
            full_score_difference
        )

        # Which model is better in the full ranking?
        if full_score_difference > 0:
            full_better_model = model_i_name
            full_better_idx = model_i_idx
            full_worse_model = model_j_name
            full_worse_idx = model_j_idx

        elif full_score_difference < 0:
            full_better_model = model_j_name
            full_better_idx = model_j_idx
            full_worse_model = model_i_name
            full_worse_idx = model_i_idx

        else:
            # Exact tie
            full_better_model = None
            full_better_idx = None
            full_worse_model = None
            full_worse_idx = None

        # Skip exact aggregate-score ties because there is
        # no unique ordering to reverse.
        if full_better_idx is None:
            continue

        # ----------------------------------------------------
        # LODO reversals
        # ----------------------------------------------------

        reversal_count = 0
        comparison_count = 0

        reversal_tasks = []

        # Store rank changes between the pair
        pair_rank_changes = []

        for removed_task in cols:

            remaining_tasks = [
                task
                for task in cols
                if task != removed_task
            ]

            _, lodo_score = compute_ranking(
                data,
                remaining_tasks
            )

            better_score = lodo_score.loc[
                full_better_idx
            ]

            worse_score = lodo_score.loc[
                full_worse_idx
            ]

            # Difference from the perspective of the
            # model that was better in the full benchmark.
            lodo_difference = (
                better_score - worse_score
            )

            # Count only non-tied comparisons.
            if lodo_difference == 0:
                continue

            comparison_count += 1

            # A reversal occurs when the original worse
            # model becomes better after removing the task.
            if lodo_difference < 0:
                reversal_count += 1
                reversal_tasks.append(
                    removed_task
                )

            pair_rank_changes.append(
                {
                    "removed_task": removed_task,
                    "lodo_difference": lodo_difference
                }
            )

        if comparison_count == 0:
            reversal_rate = np.nan
        else:
            reversal_rate = (
                reversal_count
                / comparison_count
            )

        # ----------------------------------------------------
        # Pairwise result
        # ----------------------------------------------------

        pairwise_results.append(
            {
                "model_a": model_i_name,
                "model_b": model_j_name,

                "model_a_full_score": model_i_score,
                "model_b_full_score": model_j_score,

                "model_a_full_rank": model_i_rank,
                "model_b_full_rank": model_j_rank,

                "full_score_difference":
                    full_score_difference,

                "absolute_performance_gap":
                    performance_gap,

                "full_better_model":
                    full_better_model,

                "full_worse_model":
                    full_worse_model,

                "lodo_comparisons":
                    comparison_count,

                "lodo_reversal_count":
                    reversal_count,

                "lodo_reversal_rate":
                    reversal_rate,

                "reversal_tasks":
                    "; ".join(reversal_tasks)
            }
        )


# ============================================================
# PAIRWISE RESULTS DATAFRAME
# ============================================================

pairwise_df = pd.DataFrame(
    pairwise_results
)

pairwise_df = (
    pairwise_df
    .sort_values(
        [
            "lodo_reversal_rate",
            "absolute_performance_gap"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


print("Number of model pairs analysed:")
print(len(pairwise_df))

print()


# ============================================================
# EXAMPLE REVERSALS
# ============================================================

print("=" * 80)
print("MODEL PAIRS WITH THE MOST LODO REVERSALS")
print("=" * 80)
print()

display_columns = [
    "model_a",
    "model_b",
    "absolute_performance_gap",
    "lodo_reversal_count",
    "lodo_reversal_rate",
    "reversal_tasks"
]

print(
    pairwise_df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)

print()


# ============================================================
# SMALLEST PERFORMANCE GAPS
# ============================================================

print("=" * 80)
print("CLOSEST MODEL PAIRS")
print("=" * 80)
print()

closest_pairs = (
    pairwise_df
    .sort_values(
        "absolute_performance_gap",
        ascending=True
    )
    .head(20)
)

print(
    closest_pairs[
        display_columns
    ].to_string(index=False)
)

print()


# ============================================================
# OVERALL REVERSAL STATISTICS
# ============================================================

print("=" * 80)
print("OVERALL REVERSAL STATISTICS")
print("=" * 80)
print()

pairs_with_reversal = pairwise_df[
    pairwise_df["lodo_reversal_count"] > 0
]

pairs_without_reversal = pairwise_df[
    pairwise_df["lodo_reversal_count"] == 0
]

print(
    "Pairs with at least one reversal:",
    len(pairs_with_reversal)
)

print(
    "Pairs without a reversal:",
    len(pairs_without_reversal)
)

print(
    "Percentage of pairs with reversal:",
    100
    * len(pairs_with_reversal)
    / len(pairwise_df)
)

print()

print(
    "Mean performance gap — pairs WITH reversal:",
    pairs_with_reversal[
        "absolute_performance_gap"
    ].mean()
)

print(
    "Mean performance gap — pairs WITHOUT reversal:",
    pairs_without_reversal[
        "absolute_performance_gap"
    ].mean()
)

print()


# ============================================================
# CORRELATION: PERFORMANCE GAP VS. REVERSAL RATE
# ============================================================

print("=" * 80)
print("CORRELATION: PERFORMANCE GAP VS. REVERSAL RATE")
print("=" * 80)
print()

valid = pairwise_df.dropna(
    subset=[
        "absolute_performance_gap",
        "lodo_reversal_rate"
    ]
)

# Spearman is appropriate because we care about a monotonic
# relationship and do not assume linearity.
spearman_gap_reversal, spearman_p = spearmanr(
    valid["absolute_performance_gap"],
    valid["lodo_reversal_rate"]
)

print(
    "Spearman correlation:",
    f"{spearman_gap_reversal:.6f}"
)

print(
    "p-value:",
    f"{spearman_p:.6g}"
)

print()

# Kendall tau as another non-parametric measure
kendall_gap_reversal, kendall_p = kendalltau(
    valid["absolute_performance_gap"],
    valid["lodo_reversal_rate"]
)

print(
    "Kendall tau:",
    f"{kendall_gap_reversal:.6f}"
)

print(
    "p-value:",
    f"{kendall_p:.6g}"
)

print()


# ============================================================
# GAP BINS
# ============================================================

print("=" * 80)
print("REVERSAL RATE BY PERFORMANCE-GAP BIN")
print("=" * 80)
print()

# Quantile-based bins give approximately equal numbers
# of pairs in each group.
pairwise_df["gap_quantile"] = pd.qcut(
    pairwise_df["absolute_performance_gap"],
    q=5,
    duplicates="drop"
)

gap_bins = (
    pairwise_df
    .groupby(
        "gap_quantile",
        observed=True
    )
    .agg(
        num_pairs=(
            "absolute_performance_gap",
            "size"
        ),
        mean_gap=(
            "absolute_performance_gap",
            "mean"
        ),
        median_gap=(
            "absolute_performance_gap",
            "median"
        ),
        mean_reversal_rate=(
            "lodo_reversal_rate",
            "mean"
        ),
        median_reversal_rate=(
            "lodo_reversal_rate",
            "median"
        ),
        pairs_with_reversal=(
            "lodo_reversal_count",
            lambda x: int((x > 0).sum())
        )
    )
    .reset_index()
)

print(
    gap_bins.to_string(index=False)
)

print()


# ============================================================
# TOP-OF-LEADERBOARD ANALYSIS
# ============================================================

print("=" * 80)
print("TOP-OF-LEADERBOARD PERFORMANCE GAPS")
print("=" * 80)
print()

top_n = 10

top_models = (
    full_sorted
    .head(top_n)
    .reset_index(drop=True)
)

adjacent_gaps = []

for i in range(len(top_models) - 1):

    model_1 = top_models.loc[i]
    model_2 = top_models.loc[i + 1]

    gap = (
        model_1["full_score"]
        - model_2["full_score"]
    )

    adjacent_gaps.append(
        {
            "rank_1": int(model_1["full_rank"]),
            "model_1": model_1["Model"],
            "rank_2": int(model_2["full_rank"]),
            "model_2": model_2["Model"],
            "score_gap": gap
        }
    )

adjacent_gaps_df = pd.DataFrame(
    adjacent_gaps
)

print(
    adjacent_gaps_df.to_string(
        index=False
    )
)

print()


# ============================================================
# LODO REVERSAL SUMMARY FOR TOP PAIRS
# ============================================================

print("=" * 80)
print("TOP-OF-LEADERBOARD PAIRWISE STABILITY")
print("=" * 80)
print()

top_model_names = set(
    top_models["Model"]
)

top_pair_results = pairwise_df[
    pairwise_df["model_a"].isin(top_model_names)
    &
    pairwise_df["model_b"].isin(top_model_names)
].copy()

top_pair_results = (
    top_pair_results
    .sort_values(
        "absolute_performance_gap"
    )
)

print(
    top_pair_results[
        [
            "model_a",
            "model_b",
            "absolute_performance_gap",
            "lodo_reversal_count",
            "lodo_reversal_rate",
            "reversal_tasks"
        ]
    ]
    .to_string(index=False)
)

print()


# ============================================================
# SAVE RESULTS
# ============================================================

pairwise_df.to_csv(
    PAIRWISE_OUTPUT,
    index=False
)

gap_bins.to_csv(
    BINNED_OUTPUT,
    index=False
)

print("=" * 80)
print("SAVED")
print("=" * 80)
print()

print("Pairwise results:")
print(PAIRWISE_OUTPUT)

print()

print("Gap-bin results:")
print(BINNED_OUTPUT)

print()

print("Done.")