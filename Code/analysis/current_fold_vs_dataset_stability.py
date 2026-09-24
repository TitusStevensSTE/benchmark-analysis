from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr


# ============================================================
# FOLD STABILITY VS. DATASET STABILITY / PERFORMANCE GAP
# ============================================================
#
# Purpose
# -------
# Connect the new fold-based stability analysis with the
# existing TabArena dataset-level analyses.
#
# Inputs
# ------
# 1. dataset_fold_stability.csv
#    Dataset-specific fold ranking stability.
#
# 2. dataset_influence_complete.csv
#    Existing dataset-LODO / influence analysis.
#
# 3. dataset_performance_gap_statistics.csv
#    Existing dataset-level performance-gap analysis.
#
# Main analyses
# -------------
# A. Merge all dataset-level results.
#
# B. Correlations:
#    - fold stability vs LODO instability
#    - fold stability vs winner-change frequency
#    - fold stability vs winner-runner-up gap
#    - fold instability vs winner-runner-up gap
#    - fold stability vs winner-third gap
#    - fold stability vs winner-fifth gap
#    - fold stability vs top-5 dispersion
#    - fold stability vs overall dispersion
#    - fold stability vs number of folds
#
# C. Stratified correlations:
#    separately for 9-fold and 30-fold datasets.
#
# D. 9-fold vs 30-fold stability comparison:
#    - descriptive statistics
#    - Mann-Whitney U
#    - rank-biserial correlation
#
# Important
# ---------
# Higher mean_kendall_tau = MORE stable.
#
# fold_instability = 1 - mean_kendall_tau
# Higher fold_instability = LESS stable.
#
# No random operations are used.
#
# Constant variables are handled explicitly. No scipy
# ConstantInputWarning is intentionally generated.
#
# ============================================================


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================


RESULTS_DIR = (
    RESULTS_DIR
)

FOLD_DIR = (
    RESULTS_DIR
    / "fold_stability"
)

FOLD_FILE = (
    FOLD_DIR
    / "dataset_fold_stability.csv"
)

INFLUENCE_FILE = (
    RESULTS_DIR
    / "dataset_influence_final"
    / "dataset_influence_complete.csv"
)

GAP_FILE = (
    RESULTS_DIR
    / "performance_gap"
    / "dataset_performance_gap_statistics.csv"
)

OUTPUT_DIR = FOLD_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_DATASETS = 51

REQUIRED_FOLD_COLUMNS = {
    "dataset",
    "n_folds",
    "n_fold_pairs",
    "mean_kendall_tau",
    "std_kendall_tau",
    "min_kendall_tau",
    "max_kendall_tau",
}

REQUIRED_INFLUENCE_COLUMNS = {
    "dataset",
    "kendall_tau",
    "lodo_instability",
    "winner_change_frequency",
    "tau_break_frequency",
}

REQUIRED_GAP_COLUMNS = {
    "dataset",
    "winner_runnerup_score_gap",
    "winner_third_score_gap",
    "winner_fifth_score_gap",
    "top5_score_std",
    "overall_score_std",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FOLD STABILITY VS. DATASET STABILITY / PERFORMANCE GAP")
print("=" * 70)

print()
print(f"Project root:\n{PROJECT_ROOT}")


# ============================================================
# HELPER: VALIDATE INPUT
# ============================================================

def validate_input(
    name: str,
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} is missing columns: "
            f"{sorted(missing)}"
        )

    if df["dataset"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate dataset rows."
        )

    if len(df) != EXPECTED_DATASETS:
        raise ValueError(
            f"{name}: expected {EXPECTED_DATASETS} "
            f"datasets, found {len(df)}."
        )


# ============================================================
# LOAD INPUTS
# ============================================================

for path in [
    FOLD_FILE,
    INFLUENCE_FILE,
    GAP_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

fold_df = pd.read_csv(FOLD_FILE)
influence_df = pd.read_csv(INFLUENCE_FILE)
gap_df = pd.read_csv(GAP_FILE)

validate_input(
    "Fold stability",
    fold_df,
    REQUIRED_FOLD_COLUMNS,
)

validate_input(
    "Dataset influence",
    influence_df,
    REQUIRED_INFLUENCE_COLUMNS,
)

validate_input(
    "Performance gap",
    gap_df,
    REQUIRED_GAP_COLUMNS,
)

print("\nInput files validated.")


# ============================================================
# MERGE
# ============================================================

merged = (
    fold_df[
        [
            "dataset",
            "n_folds",
            "n_fold_pairs",
            "mean_kendall_tau",
            "std_kendall_tau",
            "min_kendall_tau",
            "max_kendall_tau",
        ]
    ]
    .merge(
        influence_df[
            [
                "dataset",
                "kendall_tau",
                "lodo_instability",
                "winner_change_frequency",
                "tau_break_frequency",
            ]
        ],
        on="dataset",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        gap_df[
            [
                "dataset",
                "winner_runnerup_score_gap",
                "winner_third_score_gap",
                "winner_fifth_score_gap",
                "top5_score_std",
                "overall_score_std",
            ]
        ],
        on="dataset",
        how="inner",
        validate="one_to_one",
    )
)


if len(merged) != EXPECTED_DATASETS:
    raise ValueError(
        f"Merged dataset contains {len(merged)} rows; "
        f"expected {EXPECTED_DATASETS}."
    )


# ============================================================
# DERIVED FOLD INSTABILITY
# ============================================================

merged["fold_instability"] = (
    1.0
    - merged["mean_kendall_tau"]
)


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

merged_file = (
    OUTPUT_DIR
    / "dataset_fold_stability_combined.csv"
)

merged.to_csv(
    merged_file,
    index=False,
)


print(
    f"\nMerged shape: {merged.shape}"
)

print(
    f"Saved combined dataset:\n  {merged_file}"
)


# ============================================================
# CORRELATION HELPER
# ============================================================

def calculate_correlations(
    x: pd.Series,
    y: pd.Series,
) -> dict:

    valid = pd.concat(
        [x, y],
        axis=1,
    ).dropna()

    n = len(valid)

    result = {
        "n": n,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "kendall_tau": np.nan,
        "kendall_p": np.nan,
        "status": "ok",
    }

    if n < 3:
        result["status"] = "insufficient_n"
        return result

    x_valid = valid.iloc[:, 0]
    y_valid = valid.iloc[:, 1]

    # Explicit constant-input handling.
    #
    # If either variable has no variation, a correlation is
    # mathematically undefined and scipy would otherwise emit
    # ConstantInputWarning.
    if (
        x_valid.nunique(dropna=True) <= 1
        or y_valid.nunique(dropna=True) <= 1
    ):
        result["status"] = "constant_input"
        return result

    spearman_rho, spearman_p = spearmanr(
        x_valid,
        y_valid,
    )

    kendall_tau, kendall_p = kendalltau(
        x_valid,
        y_valid,
    )

    result.update(
        {
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_tau),
            "kendall_p": float(kendall_p),
        }
    )

    return result


# ============================================================
# POOLED CORRELATIONS
# ============================================================

analyses = [
    (
        "fold_stability_vs_lodo_instability",
        "mean_kendall_tau",
        "lodo_instability",
    ),
    (
        "fold_stability_vs_winner_change_frequency",
        "mean_kendall_tau",
        "winner_change_frequency",
    ),
    (
        "fold_stability_vs_performance_gap",
        "mean_kendall_tau",
        "winner_runnerup_score_gap",
    ),
    (
        "fold_instability_vs_performance_gap",
        "fold_instability",
        "winner_runnerup_score_gap",
    ),
    (
        "fold_stability_vs_third_gap",
        "mean_kendall_tau",
        "winner_third_score_gap",
    ),
    (
        "fold_stability_vs_fifth_gap",
        "mean_kendall_tau",
        "winner_fifth_score_gap",
    ),
    (
        "fold_stability_vs_top5_dispersion",
        "mean_kendall_tau",
        "top5_score_std",
    ),
    (
        "fold_stability_vs_overall_dispersion",
        "mean_kendall_tau",
        "overall_score_std",
    ),
    (
        "fold_stability_vs_number_of_folds",
        "mean_kendall_tau",
        "n_folds",
    ),
]


correlation_rows = []

for (
    analysis_name,
    x_col,
    y_col,
) in analyses:

    result = calculate_correlations(
        merged[x_col],
        merged[y_col],
    )

    correlation_rows.append(
        {
            "analysis": analysis_name,
            "x": x_col,
            "y": y_col,
            **result,
        }
    )


correlations = pd.DataFrame(
    correlation_rows
)[
    [
        "analysis",
        "x",
        "y",
        "n",
        "spearman_rho",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "status",
    ]
]


# ============================================================
# SAVE POOLED CORRELATIONS
# ============================================================

correlation_file = (
    OUTPUT_DIR
    / "fold_stability_correlations.csv"
)

correlations.to_csv(
    correlation_file,
    index=False,
)


# ============================================================
# PRINT POOLED CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATIONS")
print("=" * 70)

print(
    correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# STRATIFIED CORRELATIONS BY FOLD COUNT
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS BY FOLD COUNT")
print("=" * 70)


stratified_analyses = [
    (
        "fold_stability_vs_lodo_instability",
        "mean_kendall_tau",
        "lodo_instability",
    ),
    (
        "fold_stability_vs_winner_change_frequency",
        "mean_kendall_tau",
        "winner_change_frequency",
    ),
    (
        "fold_stability_vs_performance_gap",
        "mean_kendall_tau",
        "winner_runnerup_score_gap",
    ),
]


stratified_rows = []


for fold_count in sorted(
    merged["n_folds"].unique()
):

    subset = merged[
        merged["n_folds"] == fold_count
    ].copy()

    print()
    print(
        f"=== {fold_count}-fold datasets "
        f"(n={len(subset)}) ==="
    )

    for (
        analysis_name,
        x_col,
        y_col,
    ) in stratified_analyses:

        result = calculate_correlations(
            subset[x_col],
            subset[y_col],
        )

        stratified_rows.append(
            {
                "n_folds": int(fold_count),
                "analysis": analysis_name,
                "x": x_col,
                "y": y_col,
                **result,
            }
        )

        if result["status"] == "ok":

            print(
                f"{analysis_name}: "
                f"Spearman="
                f"{result['spearman_rho']:.4f} "
                f"(p="
                f"{result['spearman_p']:.4f}), "
                f"Kendall="
                f"{result['kendall_tau']:.4f} "
                f"(p="
                f"{result['kendall_p']:.4f})"
            )

        else:

            print(
                f"{analysis_name}: "
                f"undefined "
                f"({result['status']})"
            )


stratified_df = pd.DataFrame(
    stratified_rows
)[
    [
        "n_folds",
        "analysis",
        "x",
        "y",
        "n",
        "spearman_rho",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "status",
    ]
]


stratified_file = (
    OUTPUT_DIR
    / "fold_stability_correlations_by_fold_count.csv"
)

stratified_df.to_csv(
    stratified_file,
    index=False,
)


# ============================================================
# 9-FOLD VS. 30-FOLD STABILITY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("9-FOLD VS. 30-FOLD DATASET STABILITY")
print("=" * 70)


x9 = (
    merged.loc[
        merged["n_folds"] == 9,
        "mean_kendall_tau",
    ]
    .dropna()
)

x30 = (
    merged.loc[
        merged["n_folds"] == 30,
        "mean_kendall_tau",
    ]
    .dropna()
)


if len(x9) < 2 or len(x30) < 2:
    raise ValueError(
        "Not enough observations for 9-fold/30-fold comparison."
    )


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

mean_9 = float(x9.mean())
mean_30 = float(x30.mean())

median_9 = float(x9.median())
median_30 = float(x30.median())

std_9 = float(x9.std())
std_30 = float(x30.std())

min_9 = float(x9.min())
min_30 = float(x30.min())

max_9 = float(x9.max())
max_30 = float(x30.max())


mean_difference = (
    mean_9 - mean_30
)

median_difference = (
    median_9 - median_30
)


# ------------------------------------------------------------
# Mann-Whitney U
# ------------------------------------------------------------

u_statistic, u_p_value = (
    mannwhitneyu(
        x9,
        x30,
        alternative="two-sided",
    )
)


# ------------------------------------------------------------
# Rank-biserial correlation
# ------------------------------------------------------------

n9 = len(x9)
n30 = len(x30)

rank_biserial = (
    2.0 * float(u_statistic)
    / (n9 * n30)
    - 1.0
)


# ------------------------------------------------------------
# Save comparison
# ------------------------------------------------------------

fold_count_comparison = pd.DataFrame(
    [
        {
            "group_9fold_n": n9,
            "group_30fold_n": n30,
            "mean_stability_9fold": mean_9,
            "mean_stability_30fold": mean_30,
            "median_stability_9fold": median_9,
            "median_stability_30fold": median_30,
            "std_stability_9fold": std_9,
            "std_stability_30fold": std_30,
            "min_stability_9fold": min_9,
            "min_stability_30fold": min_30,
            "max_stability_9fold": max_9,
            "max_stability_30fold": max_30,
            "mean_difference_9minus30":
                mean_difference,
            "median_difference_9minus30":
                median_difference,
            "mann_whitney_u":
                float(u_statistic),
            "mann_whitney_p":
                float(u_p_value),
            "rank_biserial_correlation":
                rank_biserial,
        }
    ]
)


fold_count_file = (
    OUTPUT_DIR
    / "fold_count_stability_comparison.csv"
)

fold_count_comparison.to_csv(
    fold_count_file,
    index=False,
)


# ------------------------------------------------------------
# Print comparison
# ------------------------------------------------------------

print(
    f"\n9-fold datasets:"
    f"\n  n = {n9}"
    f"\n  mean = {mean_9:.6f}"
    f"\n  median = {median_9:.6f}"
    f"\n  std = {std_9:.6f}"
    f"\n  min = {min_9:.6f}"
    f"\n  max = {max_9:.6f}"
)

print(
    f"\n30-fold datasets:"
    f"\n  n = {n30}"
    f"\n  mean = {mean_30:.6f}"
    f"\n  median = {median_30:.6f}"
    f"\n  std = {std_30:.6f}"
    f"\n  min = {min_30:.6f}"
    f"\n  max = {max_30:.6f}"
)

print(
    f"\nMean difference (9 - 30): "
    f"{mean_difference:.6f}"
)

print(
    f"Median difference (9 - 30): "
    f"{median_difference:.6f}"
)

print(
    f"\nMann-Whitney U: "
    f"{float(u_statistic):.6f}"
)

print(
    f"Mann-Whitney p-value: "
    f"{float(u_p_value):.6f}"
)

print(
    f"Rank-biserial correlation: "
    f"{rank_biserial:.6f}"
)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

output_files = [
    merged_file,
    correlation_file,
    stratified_file,
    fold_count_file,
]

for output_file in output_files:
    print(f"  {output_file}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)