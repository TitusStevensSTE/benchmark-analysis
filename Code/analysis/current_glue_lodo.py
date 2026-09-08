import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ============================================================
# PROJECT SETUP
# ============================================================

# Path to the cloned benchbench repository
BENCHBENCH_ROOT = Path(
    r"D:\Users\titus\Informatik-DO\benchmark-analysis\benchbench"
)

sys.path.insert(0, str(BENCHBENCH_ROOT))


# Import the ORIGINAL GLUE loader from benchbench
from benchbench.data.glue import load_glue


# ============================================================
# RANKING FUNCTIONS
# ============================================================

def compute_ranking(data, cols):
    """
    Compute the leaderboard ranking using the same basic
    aggregation used by benchbench's cardinal benchmark loader.

    Higher score = better.

    Returns:
        ranking: Series indexed by model, rank 1 = best
        aggregate_score: Series with the mean score
    """

    aggregate_score = data[cols].mean(axis=1)

    # Stable sorting makes the result deterministic in case
    # two models have exactly the same aggregate score.
    order = np.argsort(
        -aggregate_score.to_numpy(),
        kind="stable"
    )

    ranking = pd.Series(
        np.empty(len(data), dtype=int),
        index=data.index
    )

    ranking.iloc[order] = np.arange(1, len(data) + 1)

    return ranking, aggregate_score


def kendall_distance(rank_full, rank_lodo):
    """
    Benchbench's Kendall distance:

        (1 - Kendall_tau) / 2

    0 = identical ranking
    1 = completely reversed ranking
    """

    tau, _ = kendalltau(rank_full, rank_lodo)

    if np.isnan(tau):
        return np.nan

    return (1 - tau) / 2


def spearman_correlation(rank_full, rank_lodo):
    """
    Standard Spearman rank correlation.

    1 = identical ranking
    -1 = completely reversed ranking
    """

    rho, _ = spearmanr(rank_full, rank_lodo)

    return rho


def normalized_mrc(rank_full, rank_lodo):
    """
    Maximum Rank Change as implemented by benchbench:

        max(|new_rank - old_rank|) / (N - 1)
    """

    n = len(rank_full)

    if n <= 1:
        return 0.0

    return (
        np.max(np.abs(rank_lodo - rank_full))
        / (n - 1)
    )


# ============================================================
# LOAD ORIGINAL GLUE DATA
# ============================================================

print("=" * 70)
print("GLUE LODO ANALYSIS — ORIGINAL BENCHBENCH DATA")
print("=" * 70)

print()
print("Benchbench root:")
print(BENCHBENCH_ROOT)

data, cols = load_glue()

print()
print("Number of models:", len(data))
print("Number of tasks:", len(cols))
print("Tasks:", cols)

print()
print("Models:")
print(data["Model"].head(10).to_string(index=False))


# ============================================================
# FULL GLUE RANKING
# ============================================================

full_ranking, full_score = compute_ranking(data, cols)

data = data.copy()
data["full_score"] = full_score
data["full_rank"] = full_ranking

# Full winner
full_winner_idx = data["full_score"].idxmax()
full_winner = data.loc[full_winner_idx, "Model"]

print()
print("=" * 70)
print("FULL GLUE LEADERBOARD")
print("=" * 70)

print()
print("Winner:", full_winner)
print()

display_cols = ["Model", "full_score", "full_rank"]

print(
    data[display_cols]
    .sort_values("full_rank")
    .head(10)
    .to_string(index=False)
)


# ============================================================
# LODO ANALYSIS
# ============================================================

results = []

print()
print("=" * 70)
print("LODO ANALYSIS")
print("=" * 70)

for removed_task in cols:

    remaining_tasks = [
        task for task in cols
        if task != removed_task
    ]

    # --------------------------------------------------------
    # Recompute leaderboard without the removed task
    # --------------------------------------------------------

    lodo_ranking, lodo_score = compute_ranking(
        data,
        remaining_tasks
    )

    # --------------------------------------------------------
    # Align rankings
    # --------------------------------------------------------

    common_index = data.index

    full_ranks = data.loc[
        common_index, "full_rank"
    ].to_numpy()

    lodo_ranks = lodo_ranking.loc[
        common_index
    ].to_numpy()

    # --------------------------------------------------------
    # Ranking-change metrics
    # --------------------------------------------------------

    kendall = kendall_distance(
        full_ranks,
        lodo_ranks
    )

    spearman = spearman_correlation(
        full_ranks,
        lodo_ranks
    )

    mrc = normalized_mrc(
        full_ranks,
        lodo_ranks
    )

    absolute_rank_change = np.abs(
        lodo_ranks - full_ranks
    )

    num_rank_changes = int(
        np.sum(absolute_rank_change > 0)
    )

    max_rank_change = int(
        np.max(absolute_rank_change)
    )

    mean_rank_change = float(
        np.mean(absolute_rank_change)
    )

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    lodo_winner_idx = lodo_score.idxmax()
    lodo_winner = data.loc[
        lodo_winner_idx,
        "Model"
    ]

    winner_changed = (
        lodo_winner != full_winner
    )

    # --------------------------------------------------------
    # Original winner's new rank
    # --------------------------------------------------------

    original_winner_lodo_rank = int(
        lodo_ranking.loc[full_winner_idx]
    )

    original_winner_rank_change = (
        original_winner_lodo_rank
        - 1
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "removed_task": removed_task,
        "num_remaining_tasks": len(remaining_tasks),

        "kendall_distance": kendall,
        "spearman_rho": spearman,

        "mrc": mrc,
        "max_rank_change": max_rank_change,
        "mean_rank_change": mean_rank_change,
        "num_rank_changes": num_rank_changes,

        "full_winner": full_winner,
        "lodo_winner": lodo_winner,
        "winner_changed": winner_changed,

        "original_winner_lodo_rank":
            original_winner_lodo_rank,

        "original_winner_rank_change":
            original_winner_rank_change,
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("LODO RESULTS")
print("=" * 70)

print()

print(
    results_df[
        [
            "removed_task",
            "kendall_distance",
            "spearman_rho",
            "mrc",
            "max_rank_change",
            "num_rank_changes",
            "full_winner",
            "lodo_winner",
            "winner_changed",
        ]
    ].to_string(index=False)
)


# ============================================================
# MOST / LEAST DISRUPTIVE TASKS
# ============================================================

print()
print("=" * 70)
print("MOST DISRUPTIVE TASKS")
print("=" * 70)

print()
print(
    results_df
    .sort_values(
        "kendall_distance",
        ascending=False
    )[
        [
            "removed_task",
            "kendall_distance",
            "spearman_rho",
            "mrc",
            "max_rank_change",
            "num_rank_changes",
            "winner_changed",
        ]
    ]
    .to_string(index=False)
)


print()
print("=" * 70)
print("MOST DISRUPTIVE TASKS BY MRC")
print("=" * 70)

print()
print(
    results_df
    .sort_values(
        "mrc",
        ascending=False
    )[
        [
            "removed_task",
            "mrc",
            "max_rank_change",
            "num_rank_changes",
            "kendall_distance",
            "winner_changed",
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# WINNER CHANGES
# ============================================================

winner_changes = results_df[
    results_df["winner_changed"]
]

print()
print("=" * 70)
print("WINNER CHANGES")
print("=" * 70)

if len(winner_changes) == 0:
    print()
    print("The GLUE winner never changes under LODO.")
else:
    print()
    print(
        winner_changes[
            [
                "removed_task",
                "full_winner",
                "lodo_winner",
                "kendall_distance",
                "mrc",
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = (
    BENCHBENCH_ROOT.parent
    / "Code"
    / "analysis"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir
    / "glue_lodo_original_benchbench.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print()
print(output_file)

print()
print("Done.")