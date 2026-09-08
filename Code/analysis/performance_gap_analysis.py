from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
)

SCORE_FILE = (
    RESULTS_DIR
    / "score_method_dataset.csv"
)

LODO_FILE = (
    RESULTS_DIR
    / "lodo_score_results.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "performance_gap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict:
    """
    Compute Pearson, Spearman and Kendall correlations
    between two numeric columns.
    """

    if x_col not in df.columns:
        raise KeyError(
            f"Column '{x_col}' not found in dataframe."
        )

    if y_col not in df.columns:
        raise KeyError(
            f"Column '{y_col}' not found in dataframe."
        )

    data = (
        df[[x_col, y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(data) < 3:
        return {
            "n": len(data),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
        }

    pearson_r, pearson_p = pearsonr(
        data[x_col],
        data[y_col],
    )

    spearman_rho, spearman_p = spearmanr(
        data[x_col],
        data[y_col],
    )

    kendall_tau, kendall_p = kendalltau(
        data[x_col],
        data[y_col],
    )

    return {
        "n": len(data),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
    }


def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    annotate: bool = True,
):
    """
    Create and save a scatter plot.
    """

    plot_df = (
        df[[x_col, y_col, "dataset"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        plot_df[x_col],
        plot_df[y_col],
        alpha=0.75,
    )

    if annotate:
        for _, row in plot_df.iterrows():
            plt.annotate(
                row["dataset"],
                (
                    row[x_col],
                    row[y_col],
                ),
                fontsize=8,
                alpha=0.8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PERFORMANCE GAP VS. LODO INSTABILITY ANALYSIS")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Score file:")
print(SCORE_FILE)

print()
print("LODO file:")
print(LODO_FILE)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not SCORE_FILE.exists():
    raise FileNotFoundError(
        f"Score file not found:\n{SCORE_FILE}\n\n"
        "Run lodo_score.py first."
    )

if not LODO_FILE.exists():
    raise FileNotFoundError(
        f"LODO file not found:\n{LODO_FILE}\n\n"
        "Run lodo_score.py first."
    )


# ============================================================
# LOAD EXACT TABARENA SCORE DATA
# ============================================================

print()
print("Loading exact TabArena Score data...")

score_df = pd.read_csv(
    SCORE_FILE
)

print(
    f"Shape: {score_df.shape}"
)

print(
    f"Columns: {list(score_df.columns)}"
)

required_score_columns = {
    "dataset",
    "method",
    "dataset_score",
}

missing_score_columns = (
    required_score_columns
    - set(score_df.columns)
)

if missing_score_columns:
    raise ValueError(
        "Missing required columns in "
        "score_method_dataset.csv: "
        f"{sorted(missing_score_columns)}"
    )


# ============================================================
# VALIDATE SCORE DATA
# ============================================================

print()
print("Validating Score data...")

num_datasets = (
    score_df["dataset"]
    .nunique()
)

num_methods = (
    score_df["method"]
    .nunique()
)

print(
    f"Number of datasets: {num_datasets}"
)

print(
    f"Number of methods: {num_methods}"
)

duplicate_pairs = (
    score_df
    .duplicated(
        subset=[
            "dataset",
            "method",
        ]
    )
    .sum()
)

if duplicate_pairs > 0:
    raise ValueError(
        f"Found {duplicate_pairs} duplicate "
        "dataset-method pairs."
    )

print(
    "No duplicate dataset-method pairs found."
)


score_min = (
    score_df["dataset_score"]
    .min()
)

score_max = (
    score_df["dataset_score"]
    .max()
)

print(
    f"Dataset Score range: "
    f"{score_min:.6f} to {score_max:.6f}"
)

if (
    score_min < -1e-8
    or score_max > 1 + 1e-8
):
    raise ValueError(
        "dataset_score contains values outside "
        "the expected [0, 1] range."
    )


# ============================================================
# COMPUTE DATASET-LEVEL PERFORMANCE GAP
# ============================================================

print()
print(
    "Computing dataset-level performance gaps..."
)

dataset_statistics = []

for dataset, group in score_df.groupby(
    "dataset"
):

    group = (
        group
        .sort_values(
            "dataset_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    winner = group.iloc[0]

    # --------------------------------------------------------
    # Runner-up
    # --------------------------------------------------------

    if len(group) >= 2:
        runnerup = group.iloc[1]
    else:
        runnerup = None

    # --------------------------------------------------------
    # Winner - runner-up gap
    # --------------------------------------------------------

    if runnerup is not None:
        winner_runnerup_gap = (
            winner["dataset_score"]
            - runnerup["dataset_score"]
        )
    else:
        winner_runnerup_gap = np.nan

    # --------------------------------------------------------
    # Winner - third place gap
    # --------------------------------------------------------

    if len(group) >= 3:
        third_place_score = (
            group.iloc[2]["dataset_score"]
        )

        winner_third_gap = (
            winner["dataset_score"]
            - third_place_score
        )
    else:
        winner_third_gap = np.nan

    # --------------------------------------------------------
    # Winner - fifth place gap
    # --------------------------------------------------------

    if len(group) >= 5:
        fifth_place_score = (
            group.iloc[4]["dataset_score"]
        )

        winner_fifth_gap = (
            winner["dataset_score"]
            - fifth_place_score
        )
    else:
        winner_fifth_gap = np.nan

    # --------------------------------------------------------
    # Top-5 dispersion
    # --------------------------------------------------------

    top5 = (
        group
        .head(5)["dataset_score"]
    )

    if len(top5) >= 2:
        top5_std = top5.std()
    else:
        top5_std = np.nan

    # --------------------------------------------------------
    # Overall dispersion
    # --------------------------------------------------------

    overall_std = (
        group["dataset_score"]
        .std()
    )

    dataset_statistics.append(
        {
            "dataset": dataset,

            "dataset_winner": (
                winner["method"]
            ),

            "dataset_winner_score": (
                winner["dataset_score"]
            ),

            "dataset_runnerup": (
                runnerup["method"]
                if runnerup is not None
                else np.nan
            ),

            "dataset_runnerup_score": (
                runnerup["dataset_score"]
                if runnerup is not None
                else np.nan
            ),

            "winner_runnerup_score_gap":
                winner_runnerup_gap,

            "winner_third_score_gap":
                winner_third_gap,

            "winner_fifth_score_gap":
                winner_fifth_gap,

            "top5_score_std":
                top5_std,

            "overall_score_std":
                overall_std,

            "num_methods":
                len(group),
        }
    )


gap_df = pd.DataFrame(
    dataset_statistics
)

print(
    f"Created dataset-level statistics for "
    f"{len(gap_df)} datasets."
)


# ============================================================
# SAVE DATASET PERFORMANCE STATISTICS
# ============================================================

gap_output = (
    OUTPUT_DIR
    / "dataset_performance_gap_statistics.csv"
)

gap_df.to_csv(
    gap_output,
    index=False,
)

print()
print("Saved:")
print(gap_output)


# ============================================================
# LOAD LODO RESULTS
# ============================================================

print()
print("Loading Score LODO results...")

lodo_df = pd.read_csv(
    LODO_FILE
)

print(
    f"LODO shape: {lodo_df.shape}"
)

print(
    f"LODO columns: {list(lodo_df.columns)}"
)

required_lodo_columns = {
    "dataset_removed",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
}

missing_lodo_columns = (
    required_lodo_columns
    - set(lodo_df.columns)
)

if missing_lodo_columns:
    raise ValueError(
        "Missing required columns in "
        "lodo_score_results.csv: "
        f"{sorted(missing_lodo_columns)}"
    )


# ============================================================
# PREPARE LODO DATA
# ============================================================

lodo_df = (
    lodo_df
    .rename(
        columns={
            "dataset_removed": "dataset"
        }
    )
)


# ============================================================
# IMPORTANT:
# DROP DUPLICATE PERFORMANCE-GAP COLUMNS FROM LODO
# ============================================================
#
# The exact performance-gap quantities are calculated above
# directly from score_method_dataset.csv.
#
# lodo_score_results.csv also contains:
#
#   dataset_winner
#   dataset_runnerup
#   dataset_winner_score
#   dataset_runnerup_score
#   winner_runnerup_score_gap
#
# Those are useful for LODO diagnostics, but merging them
# would create columns such as:
#
#   winner_runnerup_score_gap_x
#   winner_runnerup_score_gap_y
#
# Therefore we keep the Score-derived values from gap_df
# as the authoritative performance-gap measurements.
# ============================================================

duplicate_gap_columns = [
    "dataset_winner",
    "dataset_winner_score",
    "dataset_runnerup",
    "dataset_runnerup_score",
    "winner_runnerup_score_gap",
]

columns_to_drop = [
    column
    for column in duplicate_gap_columns
    if column in lodo_df.columns
]

if columns_to_drop:
    print()
    print(
        "Removing duplicate performance-gap "
        "columns from LODO data:"
    )

    for column in columns_to_drop:
        print(
            f"  - {column}"
        )

    lodo_df = lodo_df.drop(
        columns=columns_to_drop
    )


# ============================================================
# MERGE PERFORMANCE GAP + LODO
# ============================================================

print()
print(
    "Merging performance-gap and LODO results..."
)

analysis_df = pd.merge(
    gap_df,
    lodo_df,
    on="dataset",
    how="inner",
    validate="one_to_one",
)

print(
    f"Merged dataset count: "
    f"{len(analysis_df)}"
)


# ============================================================
# VALIDATE MERGE
# ============================================================

expected_columns = [
    "dataset",
    "winner_runnerup_score_gap",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
]

missing_after_merge = [
    column
    for column in expected_columns
    if column not in analysis_df.columns
]

if missing_after_merge:
    raise ValueError(
        "Expected columns are missing after merge: "
        f"{missing_after_merge}"
    )

print(
    "Merge validation successful."
)

print()
print(
    "Performance-gap column after merge:"
)

print(
    "  winner_runnerup_score_gap"
)

if any(
    column.endswith("_x")
    or column.endswith("_y")
    for column in analysis_df.columns
):
    print()
    print(
        "WARNING: suffixed duplicate columns remain:"
    )

    for column in analysis_df.columns:
        if (
            column.endswith("_x")
            or column.endswith("_y")
        ):
            print(
                f"  - {column}"
            )


# ============================================================
# SAVE MERGED ANALYSIS
# ============================================================

merged_output = (
    OUTPUT_DIR
    / "performance_gap_vs_lodo.csv"
)

analysis_df.to_csv(
    merged_output,
    index=False,
)

print()
print("Saved:")
print(merged_output)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)


relationships = [
    (
        "winner_runnerup_score_gap",
        "kendall_tau",
    ),
    (
        "winner_runnerup_score_gap",
        "spearman_rho",
    ),
    (
        "winner_runnerup_score_gap",
        "max_rank_change",
    ),
    (
        "winner_runnerup_score_gap",
        "num_rank_changes",
    ),
    (
        "winner_third_score_gap",
        "kendall_tau",
    ),
    (
        "winner_fifth_score_gap",
        "kendall_tau",
    ),
    (
        "top5_score_std",
        "kendall_tau",
    ),
    (
        "overall_score_std",
        "kendall_tau",
    ),
]


correlation_results = []

for x_col, y_col in relationships:

    result = compute_correlations(
        analysis_df,
        x_col,
        y_col,
    )

    result["x_variable"] = x_col
    result["y_variable"] = y_col

    correlation_results.append(
        result
    )

    print()
    print(
        f"{x_col} vs {y_col}"
    )

    print(
        "-" * 60
    )

    print(
        f"n            = {result['n']}"
    )

    print(
        f"Pearson r    = "
        f"{result['pearson_r']:.4f} "
        f"(p={result['pearson_p']:.4f})"
    )

    print(
        f"Spearman rho = "
        f"{result['spearman_rho']:.4f} "
        f"(p={result['spearman_p']:.4f})"
    )

    print(
        f"Kendall tau  = "
        f"{result['kendall_tau']:.4f} "
        f"(p={result['kendall_p']:.4f})"
    )


correlation_df = pd.DataFrame(
    correlation_results
)


correlation_output = (
    OUTPUT_DIR
    / "performance_gap_correlations.csv"
)

correlation_df.to_csv(
    correlation_output,
    index=False,
)

print()
print("Saved:")
print(correlation_output)


# ============================================================
# RANK DATASETS BY LODO INSTABILITY
# ============================================================

print()
print("=" * 70)
print("MOST INFLUENTIAL DATASETS")
print("=" * 70)


most_influential_kendall = (
    analysis_df
    .sort_values(
        "kendall_tau",
        ascending=True,
    )
    .reset_index(drop=True)
)


print()
print(
    "Top 10 by lowest Kendall tau:"
)

print()

print(
    most_influential_kendall[
        [
            "dataset",
            "kendall_tau",
            "spearman_rho",
            "max_rank_change",
            "num_rank_changes",
            "winner_runnerup_score_gap",
            "dataset_winner",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# TOP DATASETS BY MAXIMUM RANK CHANGE
# ============================================================

print()
print(
    "Top 10 by maximum rank change:"
)

print()

most_rank_change = (
    analysis_df
    .sort_values(
        [
            "max_rank_change",
            "num_rank_changes",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)

print(
    most_rank_change[
        [
            "dataset",
            "max_rank_change",
            "num_rank_changes",
            "kendall_tau",
            "winner_runnerup_score_gap",
            "dataset_winner",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# TOP DATASETS BY NUMBER OF CHANGED RANKS
# ============================================================

print()
print(
    "Top 10 by number of changed ranks:"
)

print()

most_changed_ranks = (
    analysis_df
    .sort_values(
        [
            "num_rank_changes",
            "max_rank_change",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)

print(
    most_changed_ranks[
        [
            "dataset",
            "num_rank_changes",
            "max_rank_change",
            "kendall_tau",
            "winner_runnerup_score_gap",
            "dataset_winner",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAVE DATASET INFLUENCE RANKING
# ============================================================

influence_output = (
    OUTPUT_DIR
    / "dataset_influence_ranking.csv"
)

influence_columns = [
    "dataset",
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
    "winner_runnerup_score_gap",
    "winner_third_score_gap",
    "winner_fifth_score_gap",
    "top5_score_std",
    "overall_score_std",
    "dataset_winner",
]

analysis_df[
    influence_columns
].sort_values(
    "kendall_tau",
    ascending=True,
).to_csv(
    influence_output,
    index=False,
)

print()
print("Saved:")
print(influence_output)


# ============================================================
# LARGEST / SMALLEST PERFORMANCE GAPS
# ============================================================

print()
print("=" * 70)
print("LARGEST / SMALLEST PERFORMANCE GAPS")
print("=" * 70)


print()
print(
    "Largest winner-runner-up gaps:"
)

print()

largest_gaps = (
    analysis_df
    .sort_values(
        "winner_runnerup_score_gap",
        ascending=False,
    )
    .head(10)
)

print(
    largest_gaps[
        [
            "dataset",
            "winner_runnerup_score_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
            "dataset_winner",
        ]
    ]
    .to_string(index=False)
)


print()
print(
    "Smallest winner-runner-up gaps:"
)

print()

smallest_gaps = (
    analysis_df
    .sort_values(
        "winner_runnerup_score_gap",
        ascending=True,
    )
    .head(10)
)

print(
    smallest_gaps[
        [
            "dataset",
            "winner_runnerup_score_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
            "dataset_winner",
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# SCATTER PLOTS
# ============================================================

print()
print("=" * 70)
print("CREATING SCATTER PLOTS")
print("=" * 70)


save_scatter_plot(
    analysis_df,
    "winner_runnerup_score_gap",
    "kendall_tau",
    "Winner–runner-up Score gap",
    "LODO Kendall tau",
    "Winner–Runner-up Score Gap vs. LODO Kendall Tau",
    OUTPUT_DIR
    / "gap_vs_kendall_tau.png",
)


save_scatter_plot(
    analysis_df,
    "winner_runnerup_score_gap",
    "spearman_rho",
    "Winner–runner-up Score gap",
    "LODO Spearman rho",
    "Winner–Runner-up Score Gap vs. LODO Spearman Rho",
    OUTPUT_DIR
    / "gap_vs_spearman_rho.png",
)


save_scatter_plot(
    analysis_df,
    "winner_runnerup_score_gap",
    "max_rank_change",
    "Winner–runner-up Score gap",
    "Maximum rank change",
    "Winner–Runner-up Score Gap vs. Maximum Rank Change",
    OUTPUT_DIR
    / "gap_vs_max_rank_change.png",
)


save_scatter_plot(
    analysis_df,
    "winner_runnerup_score_gap",
    "num_rank_changes",
    "Winner–runner-up Score gap",
    "Number of changed ranks",
    "Winner–Runner-up Score Gap vs. Number of Changed Ranks",
    OUTPUT_DIR
    / "gap_vs_num_rank_changes.png",
)


save_scatter_plot(
    analysis_df,
    "top5_score_std",
    "kendall_tau",
    "Top-5 Score standard deviation",
    "LODO Kendall tau",
    "Top-5 Score Dispersion vs. LODO Kendall Tau",
    OUTPUT_DIR
    / "top5_dispersion_vs_kendall_tau.png",
)


save_scatter_plot(
    analysis_df,
    "overall_score_std",
    "kendall_tau",
    "Overall Score standard deviation",
    "LODO Kendall tau",
    "Overall Score Dispersion vs. LODO Kendall Tau",
    OUTPUT_DIR
    / "overall_dispersion_vs_kendall_tau.png",
)


# ============================================================
# HIGH-GAP VS. LOW-GAP GROUPS
# ============================================================

print()
print("=" * 70)
print("HIGH-GAP VS. LOW-GAP DATASETS")
print("=" * 70)


median_gap = (
    analysis_df[
        "winner_runnerup_score_gap"
    ]
    .median()
)


analysis_df["gap_group"] = np.where(
    analysis_df[
        "winner_runnerup_score_gap"
    ] >= median_gap,
    "High gap",
    "Low gap",
)


group_summary = (
    analysis_df
    .groupby(
        "gap_group"
    )
    .agg(
        datasets=(
            "dataset",
            "count",
        ),
        mean_kendall=(
            "kendall_tau",
            "mean",
        ),
        median_kendall=(
            "kendall_tau",
            "median",
        ),
        mean_max_rank_change=(
            "max_rank_change",
            "mean",
        ),
        mean_num_rank_changes=(
            "num_rank_changes",
            "mean",
        ),
        mean_gap=(
            "winner_runnerup_score_gap",
            "mean",
        ),
    )
    .reset_index()
)


print()

print(
    group_summary.to_string(
        index=False
    )
)


group_output = (
    OUTPUT_DIR
    / "high_vs_low_gap_summary.csv"
)

group_summary.to_csv(
    group_output,
    index=False,
)

print()
print("Saved:")
print(group_output)


# ============================================================
# EXTREME CASES
# ============================================================

print()
print("=" * 70)
print("EXTREME CASES")
print("=" * 70)


lowest_tau = (
    analysis_df
    .nsmallest(
        5,
        "kendall_tau",
    )
)

largest_gap = (
    analysis_df
    .nlargest(
        5,
        "winner_runnerup_score_gap",
    )
)

smallest_gap = (
    analysis_df
    .nsmallest(
        5,
        "winner_runnerup_score_gap",
    )
)


print()
print(
    "Most unstable datasets:"
)

print(
    lowest_tau[
        [
            "dataset",
            "kendall_tau",
            "winner_runnerup_score_gap",
            "max_rank_change",
            "num_rank_changes",
        ]
    ].to_string(index=False)
)


print()
print(
    "Largest performance-gap datasets:"
)

print(
    largest_gap[
        [
            "dataset",
            "winner_runnerup_score_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
        ]
    ].to_string(index=False)
)


print()
print(
    "Smallest performance-gap datasets:"
)

print(
    smallest_gap[
        [
            "dataset",
            "winner_runnerup_score_gap",
            "kendall_tau",
            "max_rank_change",
            "num_rank_changes",
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)


print()

print(
    f"Datasets analysed: "
    f"{len(analysis_df)}"
)

print(
    f"Median winner-runner-up gap: "
    f"{median_gap:.6f}"
)

print(
    f"Mean Kendall tau: "
    f"{analysis_df['kendall_tau'].mean():.6f}"
)

print(
    f"Minimum Kendall tau: "
    f"{analysis_df['kendall_tau'].min():.6f}"
)

print(
    f"Maximum rank change: "
    f"{analysis_df['max_rank_change'].max():.0f}"
)

print(
    f"Mean number of changed ranks: "
    f"{analysis_df['num_rank_changes'].mean():.4f}"
)


most_influential = (
    analysis_df.loc[
        analysis_df["kendall_tau"].idxmin()
    ]
)


print()
print(
    "Most influential dataset by Kendall tau:"
)

print(
    f"  {most_influential['dataset']}"
)

print(
    f"  Kendall tau: "
    f"{most_influential['kendall_tau']:.6f}"
)

print(
    f"  Winner-runner-up gap: "
    f"{most_influential['winner_runnerup_score_gap']:.6f}"
)

print(
    f"  Max rank change: "
    f"{most_influential['max_rank_change']:.0f}"
)

print(
    f"  Changed ranks: "
    f"{most_influential['num_rank_changes']:.0f}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print(
    "Analysis completed successfully."
)

print()
print(
    "Output files:"
)

for path in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        f"  {path.name}"
    )