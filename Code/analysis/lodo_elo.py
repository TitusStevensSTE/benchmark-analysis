import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr
#import pathlib
from bencheval.elo_utils import EloHelper


# ============================================================
# 1. LOAD TABARENA RESULTS
# ============================================================

tabarena_df = pd.read_csv("C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\df_results_tabarena_2026_08_13.csv")


print("Loaded:", tabarena_df.shape)
print("Datasets:", tabarena_df["dataset"].nunique())
print("Methods:", tabarena_df["method"].nunique())


# ============================================================
# 2. CREATE TABARENA ELO HELPER
# ============================================================

elo = EloHelper(
    method_col="method",
    task_col="dataset",
    error_col="metric_error",
    split_col="fold",
)


# ============================================================
# 3. CONVERT RESULTS TO TABARENA BATTLES
# ============================================================

battles = elo.convert_results_to_battles(tabarena_df)

print("\nBattles:")
print(battles.head())

print("\nNumber of battles:", len(battles))
print("Number of datasets:", battles["dataset"].nunique())


# ============================================================
# 4. COMPUTE FULL-DATA ELO
# ============================================================

full_elo = elo.compute_mle_elo(
    battles=battles,
    INIT_RATING=1000,
    SCALE=400,
)

full_elo = full_elo.sort_values(ascending=False)

full_ranking = pd.DataFrame({
    "method": full_elo.index,
    "elo_full": full_elo.values,
})

full_ranking["rank_full"] = (
    full_ranking["elo_full"]
    .rank(method="min", ascending=False)
    .astype(int)
)

print("\nFull leaderboard:")
print(full_ranking)


# ============================================================
# 5. LODO ANALYSIS
# ============================================================

datasets = tabarena_df["dataset"].unique()

lodo_results = []

for i, dataset in enumerate(datasets, start=1):

    print(
        f"[{i}/{len(datasets)}] "
        f"Leaving out: {dataset}"
    )

    # Remove one dataset
    df_lodo = tabarena_df[
        tabarena_df["dataset"] != dataset
    ].copy()

    # Convert the reduced benchmark to battles
    battles_lodo = elo.convert_results_to_battles(
        df_lodo
    )

    # Recompute Elo from scratch
    elo_lodo = elo.compute_mle_elo(
        battles=battles_lodo,
        INIT_RATING=1000,
        SCALE=400,
    )

    elo_lodo = elo_lodo.sort_values(
        ascending=False
    )

    lodo_ranking = pd.DataFrame({
        "method": elo_lodo.index,
        "elo_lodo": elo_lodo.values,
    })

    lodo_ranking["rank_lodo"] = (
        lodo_ranking["elo_lodo"]
        .rank(method="min", ascending=False)
        .astype(int)
    )


    # --------------------------------------------------------
    # Compare rankings
    # --------------------------------------------------------

    comparison = full_ranking.merge(
        lodo_ranking,
        on="method",
        how="inner",
    )

    # Rank differences
    comparison["rank_change"] = (
        comparison["rank_lodo"]
        - comparison["rank_full"]
    )

    comparison["abs_rank_change"] = (
        comparison["rank_change"].abs()
    )


    # Kendall Tau
    kendall_tau, kendall_p = kendalltau(
        comparison["rank_full"],
        comparison["rank_lodo"],
    )

    # Spearman Rho
    spearman_rho, spearman_p = spearmanr(
        comparison["rank_full"],
        comparison["rank_lodo"],
    )

    # Maximum rank movement
    max_rank_change = (
        comparison["abs_rank_change"].max()
    )

    # Number of methods whose rank changed
    num_rank_changes = (
        comparison["rank_change"] != 0
    ).sum()


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    lodo_results.append({
        "dataset_removed": dataset,
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "max_rank_change": max_rank_change,
        "num_rank_changes": num_rank_changes,
    })


# ============================================================
# 6. SAVE RESULTS
# ============================================================

lodo_results = pd.DataFrame(lodo_results)

lodo_results = lodo_results.sort_values(
    "kendall_tau",
    ascending=True,
)

print("\n==============================")
print("LODO RESULTS")
print("==============================")

print(lodo_results.to_string(index=False))


lodo_results.to_csv(
    "lodo_elo_results.csv",
    index=False,
)


# ============================================================
# 7. TOP MOST UNSTABLE DATASETS
# ============================================================

print("\nMost influential datasets:")
print(
    lodo_results[
        [
            "dataset_removed",
            "kendall_tau",
            "spearman_rho",
            "max_rank_change",
            "num_rank_changes",
        ]
    ].head(10).to_string(index=False)
)