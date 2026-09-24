from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


# ============================================================
# 1. PATHS
# ============================================================

# Project root = folder containing Code/, data files, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "df_results_tabarena_2026_08_13.csv"
LODO_PATH = PROJECT_ROOT / "lodo_elo_results.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("PERFORMANCE GAP VS. LODO INSTABILITY ANALYSIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nLoading TabArena results...")
df = pd.read_csv(DATA_PATH)

print(f"Loaded TabArena data: {df.shape}")
print(f"Datasets: {df['dataset'].nunique()}")
print(f"Methods: {df['method'].nunique()}")

print("\nLoading LODO results...")
lodo = pd.read_csv(LODO_PATH)

print(f"Loaded LODO results: {lodo.shape}")


# ============================================================
# 3. BASIC DATA CHECKS
# ============================================================

required_columns = [
    "dataset",
    "method",
    "metric_error",
    "score",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in TabArena data: {missing_columns}"
    )

required_lodo_columns = [
    "dataset_removed",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
]

missing_lodo_columns = [
    column for column in required_lodo_columns
    if column not in lodo.columns
]

if missing_lodo_columns:
    raise ValueError(
        f"Missing required columns in LODO data: {missing_lodo_columns}"
    )


# ============================================================
# 4. VERIFY SCORE DEFINITION
# ============================================================

print("\nChecking score = 1 / metric_error ...")

# Avoid division by zero.
valid_score_rows = (
    df["metric_error"].notna()
    & df["score"].notna()
    & (df["metric_error"] != 0)
)

score_difference = (
    df.loc[valid_score_rows, "score"]
    - 1 / df.loc[valid_score_rows, "metric_error"]
).abs()

max_score_difference = score_difference.max()

print(f"Maximum absolute difference: {max_score_difference:.12g}")

if max_score_difference > 1e-8:
    print(
        "WARNING: score does not appear to be exactly 1 / metric_error "
        "for all checked rows."
    )
else:
    print("Confirmed: score = 1 / metric_error.")


# ============================================================
# 5. AGGREGATE TO METHOD × DATASET
# ============================================================

print("\nAggregating performance to method × dataset level...")

# A method may have multiple folds / observations within a dataset.
# We use the mean performance score across observations.
method_dataset = (
    df.groupby(["dataset", "method"], as_index=False)
      .agg(
          performance_score=("score", "mean"),
          metric_error_mean=("metric_error", "mean"),
          num_observations=("score", "count"),
      )
)

print(
    "Method × dataset table:",
    method_dataset.shape
)


# ============================================================
# 6. CALCULATE PERFORMANCE GAP FOR EACH DATASET
# ============================================================

print("\nCalculating dataset-level performance gaps...")

gap_results = []

for dataset, group in method_dataset.groupby("dataset"):

    # Remove missing values.
    group = group.dropna(subset=["performance_score"]).copy()

    if len(group) < 2:
        print(
            f"WARNING: {dataset} has fewer than 2 valid methods. "
            "Skipping."
        )
        continue

    # Higher score = better performance.
    group = group.sort_values(
        "performance_score",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Winner and runner-up
    # --------------------------------------------------------

    winner_score = group.loc[0, "performance_score"]
    runnerup_score = group.loc[1, "performance_score"]

    winner_method = group.loc[0, "method"]
    runnerup_method = group.loc[1, "method"]

    # Relative winner-runner-up gap:
    #
    # (winner - runner-up) / winner
    #
    # Smaller value = winner and runner-up are closer.
    if winner_score != 0:
        winner_runnerup_gap = (
            winner_score - runnerup_score
        ) / winner_score
    else:
        winner_runnerup_gap = np.nan

    # --------------------------------------------------------
    # Top-5 dispersion
    # --------------------------------------------------------

    top5 = group.head(5)["performance_score"]

    top5_std = top5.std(ddof=1)

    # A scale-free version of top-5 dispersion.
    top5_mean = top5.mean()

    if top5_mean != 0:
        top5_relative_std = top5_std / top5_mean
    else:
        top5_relative_std = np.nan

    # --------------------------------------------------------
    # Overall dispersion
    # --------------------------------------------------------

    overall_std = group["performance_score"].std(ddof=1)
    overall_mean = group["performance_score"].mean()

    if overall_mean != 0:
        overall_relative_std = overall_std / overall_mean
    else:
        overall_relative_std = np.nan

    gap_results.append(
        {
            "dataset_removed": dataset,
            "winner_method": winner_method,
            "runnerup_method": runnerup_method,
            "winner_score": winner_score,
            "runnerup_score": runnerup_score,
            "winner_runnerup_gap": winner_runnerup_gap,
            "top5_std": top5_std,
            "top5_relative_std": top5_relative_std,
            "overall_std": overall_std,
            "overall_relative_std": overall_relative_std,
            "num_methods": len(group),
        }
    )

gap_results = pd.DataFrame(gap_results)

print(
    f"Calculated performance-gap statistics for "
    f"{len(gap_results)} datasets."
)


# ============================================================
# 7. MERGE WITH LODO RESULTS
# ============================================================

print("\nMerging performance gaps with LODO instability results...")

analysis = gap_results.merge(
    lodo,
    on="dataset_removed",
    how="inner",
)

print(
    f"Merged analysis contains {len(analysis)} datasets."
)

if len(analysis) == 0:
    raise ValueError(
        "No datasets could be matched between the performance-gap "
        "analysis and LODO results."
    )

if len(analysis) < len(gap_results):
    print(
        f"WARNING: {len(gap_results) - len(analysis)} dataset(s) "
        "from the performance-gap calculation were not found "
        "in the LODO results."
    )


# ============================================================
# 8. SAVE COMPLETE ANALYSIS TABLE
# ============================================================

output_csv = RESULTS_DIR / "performance_gap_lodo_analysis.csv"

analysis.to_csv(
    output_csv,
    index=False,
)

print(f"\nSaved merged analysis to:")
print(output_csv)


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)


def correlation_report(x, y, x_name, y_name):
    """
    Compute Spearman and Kendall correlations while dropping
    missing observations.
    """

    valid = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(valid) < 3:
        print(
            f"\n{x_name} vs {y_name}: insufficient observations."
        )
        return None

    spearman_r, spearman_p = spearmanr(
        valid["x"],
        valid["y"],
    )

    kendall_tau, kendall_p = kendalltau(
        valid["x"],
        valid["y"],
    )

    print(f"\n{x_name} vs. {y_name}")
    print(f"  n = {len(valid)}")
    print(
        f"  Spearman rho = {spearman_r:.4f}, "
        f"p = {spearman_p:.6g}"
    )
    print(
        f"  Kendall tau = {kendall_tau:.4f}, "
        f"p = {kendall_p:.6g}"
    )

    return {
        "x": x_name,
        "y": y_name,
        "n": len(valid),
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
    }


correlations = []

# ------------------------------------------------------------
# Primary measure
# ------------------------------------------------------------

result = correlation_report(
    analysis["winner_runnerup_gap"],
    analysis["kendall_tau"],
    "Winner-runner-up relative gap",
    "LODO Kendall tau",
)

if result:
    correlations.append(result)

result = correlation_report(
    analysis["winner_runnerup_gap"],
    analysis["spearman_rho"],
    "Winner-runner-up relative gap",
    "LODO Spearman rho",
)

if result:
    correlations.append(result)

result = correlation_report(
    analysis["winner_runnerup_gap"],
    analysis["max_rank_change"],
    "Winner-runner-up relative gap",
    "Maximum rank change",
)

if result:
    correlations.append(result)

result = correlation_report(
    analysis["winner_runnerup_gap"],
    analysis["num_rank_changes"],
    "Winner-runner-up relative gap",
    "Number of rank changes",
)

if result:
    correlations.append(result)


# ------------------------------------------------------------
# Robustness measure: top-5 relative dispersion
# ------------------------------------------------------------

result = correlation_report(
    analysis["top5_relative_std"],
    analysis["kendall_tau"],
    "Top-5 relative dispersion",
    "LODO Kendall tau",
)

if result:
    correlations.append(result)

result = correlation_report(
    analysis["top5_relative_std"],
    analysis["max_rank_change"],
    "Top-5 relative dispersion",
    "Maximum rank change",
)

if result:
    correlations.append(result)


# ------------------------------------------------------------
# Robustness measure: overall relative dispersion
# ------------------------------------------------------------

result = correlation_report(
    analysis["overall_relative_std"],
    analysis["kendall_tau"],
    "Overall relative dispersion",
    "LODO Kendall tau",
)

if result:
    correlations.append(result)

result = correlation_report(
    analysis["overall_relative_std"],
    analysis["max_rank_change"],
    "Overall relative dispersion",
    "Maximum rank change",
)

if result:
    correlations.append(result)


correlations_df = pd.DataFrame(correlations)

correlations_csv = RESULTS_DIR / "performance_gap_correlations.csv"

correlations_df.to_csv(
    correlations_csv,
    index=False,
)

print("\nSaved correlation results to:")
print(correlations_csv)


# ============================================================
# 10. PRINT MOST INFLUENTIAL DATASETS
# ============================================================

print("\n" + "=" * 70)
print("MOST INFLUENTIAL DATASETS")
print("=" * 70)

most_influential = analysis.sort_values(
    "kendall_tau",
    ascending=True,
)

print(
    most_influential[
        [
            "dataset_removed",
            "winner_method",
            "runnerup_method",
            "winner_runnerup_gap",
            "kendall_tau",
            "spearman_rho",
            "max_rank_change",
            "num_rank_changes",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 11. PRINT DATASETS WITH SMALLEST PERFORMANCE GAPS
# ============================================================

print("\n" + "=" * 70)
print("DATASETS WITH SMALLEST WINNER-RUNNER-UP GAPS")
print("=" * 70)

smallest_gaps = analysis.sort_values(
    "winner_runnerup_gap",
    ascending=True,
)

print(
    smallest_gaps[
        [
            "dataset_removed",
            "winner_method",
            "runnerup_method",
            "winner_runnerup_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 12. PRINT DATASETS WITH LARGEST PERFORMANCE GAPS
# ============================================================

print("\n" + "=" * 70)
print("DATASETS WITH LARGEST WINNER-RUNNER-UP GAPS")
print("=" * 70)

largest_gaps = analysis.sort_values(
    "winner_runnerup_gap",
    ascending=False,
)

print(
    largest_gaps[
        [
            "dataset_removed",
            "winner_method",
            "runnerup_method",
            "winner_runnerup_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 13. PLOT FUNCTION
# ============================================================

def make_scatter(
    x,
    y,
    xlabel,
    ylabel,
    title,
    output_path,
):
    """
    Create a simple scatter plot using valid observations only.
    """

    plot_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    plt.figure(figsize=(8, 6))

    plt.scatter(
        plot_df["x"],
        plot_df["y"],
        alpha=0.7,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {output_path}")


# ============================================================
# 14. PRIMARY PLOTS
# ============================================================

make_scatter(
    analysis["winner_runnerup_gap"],
    analysis["kendall_tau"],
    "Relative winner–runner-up performance gap",
    "LODO Kendall's tau",
    "Performance Gap vs. LODO Ranking Stability",
    RESULTS_DIR / "performance_gap_vs_kendall.png",
)

make_scatter(
    analysis["winner_runnerup_gap"],
    analysis["max_rank_change"],
    "Relative winner–runner-up performance gap",
    "Maximum rank change",
    "Performance Gap vs. Maximum Rank Change",
    RESULTS_DIR / "performance_gap_vs_max_rank_change.png",
)

make_scatter(
    analysis["winner_runnerup_gap"],
    analysis["num_rank_changes"],
    "Relative winner–runner-up performance gap",
    "Number of changed ranks",
    "Performance Gap vs. Number of Rank Changes",
    RESULTS_DIR / "performance_gap_vs_num_rank_changes.png",
)


# ============================================================
# 15. ROBUSTNESS PLOTS
# ============================================================

make_scatter(
    analysis["top5_relative_std"],
    analysis["kendall_tau"],
    "Top-5 relative performance dispersion",
    "LODO Kendall's tau",
    "Top-5 Performance Dispersion vs. LODO Stability",
    RESULTS_DIR / "top5_dispersion_vs_kendall.png",
)

make_scatter(
    analysis["overall_relative_std"],
    analysis["kendall_tau"],
    "Overall relative performance dispersion",
    "LODO Kendall's tau",
    "Overall Performance Dispersion vs. LODO Stability",
    RESULTS_DIR / "overall_dispersion_vs_kendall.png",
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

for file_path in sorted(RESULTS_DIR.iterdir()):
    if file_path.is_file():
        print(f"  {file_path.name}")

print("\nDone.")