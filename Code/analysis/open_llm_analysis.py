import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr
from huggingface_hub import list_repo_files, hf_hub_download


# ============================================================
# OPEN LLM LEADERBOARD V1
# LODO + WINNER ANALYSIS
# ============================================================

print("=" * 70)
print("OPEN LLM LEADERBOARD V1")
print("LODO + WINNER ANALYSIS")
print("=" * 70)


# ============================================================
# 1. SETTINGS
# ============================================================

REPO_ID = "open-llm-leaderboard-old/results"

TASKS = [
    "ARC",
    "HellaSwag",
    "MMLU",
    "TruthfulQA",
    "Winogrande",
    "GSM8K"
]


# ============================================================
# 2. LOAD JSON FILE LIST
# ============================================================

print("\nFinding Open LLM result files...")

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

json_files = [
    f for f in files
    if f.endswith(".json")
]

print(f"Found {len(json_files)} JSON files.")


# ============================================================
# 3. FUNCTION TO EXTRACT THE SIX BENCHMARK SCORES
# ============================================================

def extract_scores(result):
    """
    Extract the six Open LLM Leaderboard v1 benchmark scores
    from one model's JSON result.

    Returns:
        dict with ARC, HellaSwag, MMLU, TruthfulQA,
        Winogrande and GSM8K.
    """

    results = result["results"]

    scores = {}

    # --------------------------------------------------------
    # ARC
    # --------------------------------------------------------

    arc_key = "harness|arc:challenge|25"

    if arc_key in results:
        scores["ARC"] = results[arc_key]["acc_norm"]

    # --------------------------------------------------------
    # HellaSwag
    # --------------------------------------------------------

    hellaswag_key = "harness|hellaswag|10"

    if hellaswag_key in results:
        scores["HellaSwag"] = results[hellaswag_key]["acc_norm"]

    # --------------------------------------------------------
    # MMLU
    # --------------------------------------------------------

    mmlu_scores = []

    for key, values in results.items():

        if key.startswith(
            "harness|hendrycksTest-"
        ):

            # All MMLU subjects use acc
            if "acc" in values:

                mmlu_scores.append(
                    values["acc"]
                )

    if len(mmlu_scores) > 0:

        scores["MMLU"] = np.mean(
            mmlu_scores
        )

    # --------------------------------------------------------
    # TruthfulQA
    # --------------------------------------------------------

    truthfulqa_candidates = [
        key for key in results
        if "truthfulqa" in key.lower()
    ]

    for key in truthfulqa_candidates:

        if "mc2" in results[key]:

            scores["TruthfulQA"] = (
                results[key]["mc2"]
            )

            break

    # --------------------------------------------------------
    # Winogrande
    # --------------------------------------------------------

    winogrande_candidates = [
        key for key in results
        if "winogrande" in key.lower()
    ]

    for key in winogrande_candidates:

        if "acc" in results[key]:

            scores["Winogrande"] = (
                results[key]["acc"]
            )

            break

    # --------------------------------------------------------
    # GSM8K
    # --------------------------------------------------------

    gsm_candidates = [
        key for key in results
        if "gsm8k" in key.lower()
    ]

    for key in gsm_candidates:

        if "acc" in results[key]:

            scores["GSM8K"] = (
                results[key]["acc"]
            )

            break

    return scores


# ============================================================
# 4. DOWNLOAD AND PROCESS MODELS
# ============================================================

records = []

print("\nProcessing models...")

for i, file in enumerate(json_files):

    try:

        local_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="dataset"
        )

        with open(
            local_file,
            "r",
            encoding="utf-8"
        ) as f:

            result = json.load(f)

        model_name = result[
            "config_general"
        ].get(
            "model_name",
            None
        )

        if model_name is None:
            continue

        scores = extract_scores(result)

        # Only keep models for which all six
        # benchmark scores are available.

        if all(
            task in scores
            for task in TASKS
        ):

            record = {
                "model": model_name
            }

            record.update(scores)

            records.append(record)

    except Exception as e:

        print(
            f"Could not process {file}: {e}"
        )

    if (i + 1) % 250 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(json_files)} files..."
        )


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

data = pd.DataFrame(records)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"\nModels with all six benchmark scores: "
    f"{len(data)}"
)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 10 models:")
print(data.head(10).to_string(index=False))


# ============================================================
# 6. REMOVE DUPLICATE MODEL NAMES
# ============================================================

data = data.drop_duplicates(
    subset="model"
).reset_index(drop=True)


# ============================================================
# 7. SAVE CLEAN SIX-TASK DATASET
# ============================================================

data.to_csv(
    "openllm_six_tasks.csv",
    index=False
)

print(
    "\nSaved: openllm_six_tasks.csv"
)


# ============================================================
# 8. COMPUTE RANKING
# ============================================================

def compute_ranking(data, tasks):

    result = data[
        ["model"] + tasks
    ].copy()

    # Equal weighting of benchmark tasks.
    #
    # This reproduces the conceptual benchmark-level
    # aggregation we need for LODO.
    #
    # Each of the six benchmark tasks receives equal weight.

    result["score"] = result[
        tasks
    ].mean(axis=1)

    result = result.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    result["rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ============================================================
# 9. FULL RANKING
# ============================================================

full_ranking = compute_ranking(
    data,
    TASKS
)

print("\n" + "=" * 70)
print("FULL OPEN LLM RANKING")
print("=" * 70)

print(
    full_ranking[
        ["rank", "model", "score"]
    ]
    .head(20)
    .to_string(index=False)
)

full_ranking.to_csv(
    "openllm_full_ranking.csv",
    index=False
)


# ============================================================
# 10. RANKING COMPARISON
# ============================================================

def compare_rankings(
    full_ranking,
    subset_ranking
):

    full = (
        full_ranking
        .set_index("model")["rank"]
    )

    subset = (
        subset_ranking
        .set_index("model")["rank"]
    )

    common_models = (
        full.index
        .intersection(subset.index)
    )

    full_ranks = full.loc[
        common_models
    ]

    subset_ranks = subset.loc[
        common_models
    ]

    tau, _ = kendalltau(
        full_ranks,
        subset_ranks
    )

    rho, _ = spearmanr(
        full_ranks,
        subset_ranks
    )

    rank_changes = (
        subset_ranks - full_ranks
    ).abs()

    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        "max_rank_change": rank_changes.max(),
        "num_rank_changes": (
            rank_changes > 0
        ).sum()
    }


# ============================================================
# 11. LODO ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LODO ANALYSIS")
print("=" * 70)

lodo_records = []

for task_to_remove in TASKS:

    remaining_tasks = [
        task
        for task in TASKS
        if task != task_to_remove
    ]

    subset_ranking = compute_ranking(
        data,
        remaining_tasks
    )

    comparison = compare_rankings(
        full_ranking,
        subset_ranking
    )

    comparison[
        "left_out_task"
    ] = task_to_remove

    lodo_records.append(
        comparison
    )


lodo_results = pd.DataFrame(
    lodo_records
)

lodo_results = lodo_results[
    [
        "left_out_task",
        "kendall_tau",
        "spearman_rho",
        "max_rank_change",
        "num_rank_changes"
    ]
]

# Most unstable first
lodo_results = lodo_results.sort_values(
    "kendall_tau",
    ascending=True
).reset_index(drop=True)


print("\nLODO RESULTS:")
print(
    lodo_results.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE LODO
# ============================================================

lodo_results.to_csv(
    "openllm_lodo_results.csv",
    index=False
)

print(
    "\nSaved: openllm_lodo_results.csv"
)


# ============================================================
# 13. LODO PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    lodo_results["left_out_task"],
    lodo_results["kendall_tau"]
)

plt.ylabel("Kendall's tau")
plt.xlabel("Task left out")

plt.title(
    "Open LLM: LODO Ranking Stability"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "openllm_lodo_kendall.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. WINNER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WINNER ANALYSIS")
print("=" * 70)

winner_records = []


# Six tasks → 2^6 - 1 = 63 non-empty subsets

for subset_size in range(
    1,
    len(TASKS) + 1
):

    for subset in combinations(
        TASKS,
        subset_size
    ):

        subset = list(subset)

        ranking = compute_ranking(
            data,
            subset
        )

        winner = ranking.iloc[0]

        winner_records.append({
            "tasks": ", ".join(subset),
            "num_tasks": len(subset),
            "winner": winner["model"],
            "winning_score": winner["score"]
        })


winner_results = pd.DataFrame(
    winner_records
)


# ============================================================
# 15. WINNER COUNTS
# ============================================================

winner_counts = (
    winner_results[
        "winner"
    ]
    .value_counts()
    .rename_axis("model")
    .reset_index(
        name="wins"
    )
)

total_subsets = len(
    winner_results
)

winner_counts[
    "winner_percentage"
] = (
    winner_counts["wins"]
    / total_subsets
    * 100
)

winner_counts = winner_counts.sort_values(
    "winner_percentage",
    ascending=False
).reset_index(drop=True)


print("\nWINNER FREQUENCIES:")
print(
    winner_counts
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE WINNER RESULTS
# ============================================================

winner_results.to_csv(
    "openllm_winner_subsets.csv",
    index=False
)

winner_counts.to_csv(
    "openllm_winner_percentages.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "  openllm_winner_subsets.csv"
)

print(
    "  openllm_winner_percentages.csv"
)


# ============================================================
# 17. WINNER PERCENTAGE BAR CHART
# ============================================================

top_winners = (
    winner_counts.head(15)
)

plt.figure(figsize=(12, 7))

plt.bar(
    top_winners["model"],
    top_winners[
        "winner_percentage"
    ]
)

plt.xticks(
    rotation=75,
    ha="right"
)

plt.ylabel(
    "Winner percentage (%)"
)

plt.xlabel("Model")

plt.title(
    "Open LLM: Winner Frequency Across Task Subsets"
)

plt.tight_layout()

plt.savefig(
    "openllm_winner_percentages.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. WINNER ANALYSIS BY NUMBER OF TASKS
# ============================================================

winner_by_size = (
    winner_results
    .groupby(
        ["num_tasks", "winner"]
    )
    .size()
    .reset_index(
        name="wins"
    )
)

winner_by_size[
    "percentage"
] = (
    winner_by_size
    .groupby("num_tasks")[
        "wins"
    ]
    .transform(
        lambda x:
        x / x.sum() * 100
    )
)


print("\n" + "=" * 70)
print("WINNER FREQUENCY BY NUMBER OF TASKS")
print("=" * 70)

print(
    winner_by_size
    .to_string(index=False)
)


# ============================================================
# 19. WINNER STACKPLOT
# ============================================================

pivot = winner_by_size.pivot(
    index="num_tasks",
    columns="winner",
    values="percentage"
).fillna(0)


ax = pivot.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

ax.set_xlabel(
    "Number of tasks included"
)

ax.set_ylabel(
    "Winner percentage (%)"
)

ax.set_title(
    "Open LLM: Winner Distribution by Number of Tasks"
)

plt.xticks(rotation=0)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "openllm_winner_stackplot.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nModels analysed: {len(data)}"
)

print(
    f"Benchmark tasks: {len(TASKS)}"
)

print(
    f"Task subsets analysed: {total_subsets}"
)

print(
    "\nMost sensitive task according to "
    "Kendall's tau:"
)

print(
    lodo_results.iloc[0][
        "left_out_task"
    ]
)

print(
    f"\nKendall's tau: "
    f"{lodo_results.iloc[0]['kendall_tau']:.4f}"
)

print(
    "\nMost frequent winner:"
)

print(
    winner_counts.iloc[0]["model"]
)

print(
    f"Winner percentage: "
    f"{winner_counts.iloc[0]['winner_percentage']:.2f}%"
)

print("\nOutput files:")

print("  openllm_six_tasks.csv")
print("  openllm_full_ranking.csv")
print("  openllm_lodo_results.csv")
print("  openllm_lodo_kendall.png")
print("  openllm_winner_subsets.csv")
print("  openllm_winner_percentages.csv")
print("  openllm_winner_percentages.png")
print("  openllm_winner_stackplot.png")