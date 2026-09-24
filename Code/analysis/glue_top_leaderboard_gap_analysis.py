from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT SETUP — PORTABLE
# ============================================================

from benchbench.data.glue import load_glue

PHASE1_OUTPUT_DIR = RESULTS_DIR / "phase1_crossbenchmark" / "glue" / "performance_gap"
PHASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

def compute_ranking(data, tasks):
    """
    Rank models by arithmetic mean over the selected tasks.
    Higher score = better.
    """
    scores = data[tasks].mean(axis=1)

    order = np.argsort(
        -scores.to_numpy(),
        kind="stable"
    )

    ranks = pd.Series(
        np.empty(len(data), dtype=int),
        index=data.index
    )

    ranks.iloc[order] = np.arange(
        1,
        len(data) + 1
    )

    return ranks, scores


# ============================================================
# LOAD GLUE
# ============================================================

print("=" * 80)
print("GLUE TOP-OF-LEADERBOARD PERFORMANCE-GAP ANALYSIS")
print("=" * 80)
print()

data, tasks = load_glue()

print(f"Models: {len(data)}")
print(f"Tasks:  {len(tasks)}")
print(f"Tasks:  {tasks}")
print()


# ============================================================
# FULL RANKING
# ============================================================

full_rank, full_score = compute_ranking(
    data,
    tasks
)

data = data.copy()
data["full_rank"] = full_rank
data["full_score"] = full_score

full = (
    data[
        [
            "Model",
            "full_score",
            "full_rank"
        ]
    ]
    .sort_values("full_rank")
    .reset_index()
)

# Keep original row index for later lookup
full["original_index"] = full["index"]


# ============================================================
# TOP N
# ============================================================

TOP_N = 10

top = full.head(TOP_N).copy()

print("=" * 80)
print(f"TOP {TOP_N} MODELS")
print("=" * 80)
print()

print(
    top[
        [
            "Model",
            "full_score",
            "full_rank"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# ANALYSE EVERY ADJACENT PAIR
# ============================================================

results = []

for position in range(len(top) - 1):

    model_a = top.iloc[position]
    model_b = top.iloc[position + 1]

    # --------------------------------------------------------
    # Full benchmark
    # --------------------------------------------------------

    model_a_name = model_a["Model"]
    model_b_name = model_b["Model"]

    model_a_index = model_a["original_index"]
    model_b_index = model_b["original_index"]

    model_a_score = model_a["full_score"]
    model_b_score = model_b["full_score"]

    score_gap = abs(
        model_a_score - model_b_score
    )

    # model_a is always the better-ranked model here
    full_better = model_a_name
    full_worse = model_b_name

    reversal_tasks = []
    lodo_rank_changes = []

    # --------------------------------------------------------
    # LODO
    # --------------------------------------------------------

    for removed_task in tasks:

        remaining_tasks = [
            task
            for task in tasks
            if task != removed_task
        ]

        lodo_rank, lodo_score = compute_ranking(
            data,
            remaining_tasks
        )

        a_score = lodo_score.loc[model_a_index]
        b_score = lodo_score.loc[model_b_index]

        # Difference from the original winner's perspective
        lodo_gap = a_score - b_score

        a_rank = int(lodo_rank.loc[model_a_index])
        b_rank = int(lodo_rank.loc[model_b_index])

        lodo_rank_changes.append({
            "task": removed_task,
            "a_rank": a_rank,
            "b_rank": b_rank,
            "lodo_gap": lodo_gap
        })

        # Original ordering reverses if model B becomes better
        if lodo_gap < 0:
            reversal_tasks.append(removed_task)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    reversal_count = len(reversal_tasks)
    reversal_rate = reversal_count / len(tasks)

    results.append({
        "rank_a": int(model_a["full_rank"]),
        "model_a": model_a_name,
        "rank_b": int(model_b["full_rank"]),
        "model_b": model_b_name,
        "model_a_score": model_a_score,
        "model_b_score": model_b_score,
        "score_gap": score_gap,
        "reversal_count": reversal_count,
        "reversal_rate": reversal_rate,
        "reversal_tasks": "; ".join(reversal_tasks)
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("=" * 80)
print("ADJACENT TOP-10 PAIRS")
print("=" * 80)
print()

print(
    results_df[
        [
            "rank_a",
            "model_a",
            "rank_b",
            "model_b",
            "score_gap",
            "reversal_count",
            "reversal_rate",
            "reversal_tasks"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# SORT BY PERFORMANCE GAP
# ============================================================

print("=" * 80)
print("TOP-10 PAIRS SORTED BY PERFORMANCE GAP")
print("=" * 80)
print()

sorted_by_gap = (
    results_df
    .sort_values("score_gap")
)

print(
    sorted_by_gap[
        [
            "rank_a",
            "model_a",
            "rank_b",
            "model_b",
            "score_gap",
            "reversal_count",
            "reversal_rate",
            "reversal_tasks"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(
    "Adjacent top-10 pairs:",
    len(results_df)
)

print(
    "Pairs with at least one reversal:",
    int(
        (results_df["reversal_count"] > 0).sum()
    )
)

print(
    "Pairs with no reversal:",
    int(
        (results_df["reversal_count"] == 0).sum()
    )
)

print(
    "Mean score gap:",
    f"{results_df['score_gap'].mean():.6f}"
)

print(
    "Mean reversal rate:",
    f"{results_df['reversal_rate'].mean():.6f}"
)

print()


# ============================================================
# SAVE
# ============================================================

output_file = PHASE1_OUTPUT_DIR / "glue_top10_adjacent_gap_analysis.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("=" * 80)
print("SAVED")
print("=" * 80)
print()
print(output_file)
print()
print("Done.")