from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================


RESULTS_DIR = (
    RESULTS_DIR
    / "fold_stability"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

PAIRWISE_FILE = (
    RESULTS_DIR
    / "fold_pairwise_correlations.csv"
)

DATASET_STABILITY_FILE = (
    RESULTS_DIR
    / "dataset_fold_stability.csv"
)

COMBINED_FILE = (
    RESULTS_DIR
    / "dataset_fold_stability_combined.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FIGURE_1 = (
    FIGURE_DIR
    / "figure_1_fold_ranking_stability_heatmap.png"
)

FIGURE_2 = (
    FIGURE_DIR
    / "figure_2_dataset_fold_stability.png"
)

FIGURE_3 = (
    FIGURE_DIR
    / "figure_3_fold_vs_lodo_instability.png"
)

FIGURE_1_PDF = (
    FIGURE_DIR
    / "figure_1_fold_ranking_stability_heatmap.pdf"
)

FIGURE_2_PDF = (
    FIGURE_DIR
    / "figure_2_dataset_fold_stability.pdf"
)

FIGURE_3_PDF = (
    FIGURE_DIR
    / "figure_3_fold_vs_lodo_instability.pdf"
)


# ============================================================
# GENERAL PLOT SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


# ============================================================
# VALIDATION
# ============================================================

for path in [
    PAIRWISE_FILE,
    DATASET_STABILITY_FILE,
    COMBINED_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


pairwise_df = pd.read_csv(
    PAIRWISE_FILE
)

dataset_df = pd.read_csv(
    DATASET_STABILITY_FILE
)

combined_df = pd.read_csv(
    COMBINED_FILE
)


# ============================================================
# FIGURE 1
# BENCHMARK-LEVEL PAIRWISE FOLD RANKING STABILITY
#
# 9 x 9 Kendall-tau heatmap
# ============================================================

common_folds = sorted(
    set(pairwise_df["fold_a"])
    | set(pairwise_df["fold_b"])
)

if common_folds != list(range(9)):
    raise ValueError(
        "Expected common folds 0-8 for Figure 1."
    )


tau_matrix = np.eye(
    len(common_folds)
)

fold_to_index = {
    fold: i
    for i, fold in enumerate(common_folds)
}

for _, row in pairwise_df.iterrows():

    i = fold_to_index[int(row["fold_a"])]
    j = fold_to_index[int(row["fold_b"])]

    tau = float(
        row["kendall_tau"]
    )

    tau_matrix[i, j] = tau
    tau_matrix[j, i] = tau


fig, ax = plt.subplots(
    figsize=(8.2, 7.0)
)

image = ax.imshow(
    tau_matrix,
    vmin=-1,
    vmax=1,
    aspect="equal",
)

ax.set_xticks(
    range(len(common_folds))
)

ax.set_yticks(
    range(len(common_folds))
)

ax.set_xticklabels(
    [str(f) for f in common_folds]
)

ax.set_yticklabels(
    [str(f) for f in common_folds]
)

ax.set_xlabel(
    "Evaluation split"
)

ax.set_ylabel(
    "Evaluation split"
)

ax.set_title(
    "Pairwise Kendall rank correlation across evaluation splits"
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

colorbar.set_label(
    "Kendall's $\\tau$"
)


# Annotate every cell.
for i in range(len(common_folds)):
    for j in range(len(common_folds)):

        value = tau_matrix[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )


fig.tight_layout()

fig.savefig(
    FIGURE_1,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_1_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# DATASET-LEVEL FOLD STABILITY
#
# Sorted horizontal dot plot.
# Marker shape distinguishes 9- vs. 30-fold datasets.
# ============================================================

plot_df = (
    dataset_df[
        [
            "dataset",
            "n_folds",
            "mean_kendall_tau",
        ]
    ]
    .sort_values(
        "mean_kendall_tau",
        ascending=True,
    )
    .reset_index(drop=True)
)

y_positions = np.arange(
    len(plot_df)
)

fig, ax = plt.subplots(
    figsize=(10.5, 14.0)
)

for fold_count, marker in [
    (9, "o"),
    (30, "s"),
]:

    subset = plot_df[
        plot_df["n_folds"] == fold_count
    ]

    # Recover positions in the sorted full data frame.
    positions = subset.index.to_numpy()

    ax.scatter(
        subset["mean_kendall_tau"],
        positions,
        marker=marker,
        s=50,
        label=f"{fold_count} folds",
    )


ax.axvline(
    plot_df["mean_kendall_tau"].mean(),
    linestyle="--",
    linewidth=1,
    label=(
        "Overall dataset mean "
        f"($\\tau$={plot_df['mean_kendall_tau'].mean():.3f})"
    ),
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    plot_df["dataset"]
)

ax.set_xlim(
    0,
    1
)

ax.set_xlabel(
    "Mean pairwise Kendall's $\\tau$ across folds"
)

ax.set_ylabel(
    "Dataset"
)

ax.set_title(
    "Dataset-level stability of method rankings across evaluation splits"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    axis="x",
    linestyle=":",
    linewidth=0.7,
    alpha=0.5,
)

fig.tight_layout()

fig.savefig(
    FIGURE_2,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_2_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# FOLD STABILITY VS. DATASET-REMOVAL INSTABILITY
#
# Scatter plot:
# x = mean fold Kendall tau
# y = dataset LODO instability
#
# Includes ordinary least-squares trend line.
# ============================================================

x = combined_df[
    "mean_kendall_tau"
].to_numpy(
    dtype=float
)

y = combined_df[
    "lodo_instability"
].to_numpy(
    dtype=float
)

if len(x) != len(y):
    raise ValueError(
        "X and Y have different lengths."
    )

valid = (
    np.isfinite(x)
    & np.isfinite(y)
)

x_valid = x[valid]
y_valid = y[valid]

if len(x_valid) < 3:
    raise ValueError(
        "Not enough observations for Figure 3."
    )


# Linear trend.
slope, intercept = np.polyfit(
    x_valid,
    y_valid,
    deg=1,
)

x_line = np.linspace(
    x_valid.min(),
    x_valid.max(),
    200,
)

y_line = (
    slope * x_line
    + intercept
)


# Correlation values from the analysis.
spearman_row = pd.read_csv(
    RESULTS_DIR
    / "fold_stability_correlations.csv"
)

target_row = spearman_row[
    spearman_row["analysis"]
    == "fold_stability_vs_lodo_instability"
]

if len(target_row) != 1:
    raise ValueError(
        "Could not identify the fold-stability vs "
        "LODO-instability correlation result."
    )

target_row = target_row.iloc[0]

rho = float(
    target_row["spearman_rho"]
)

p_value = float(
    target_row["spearman_p"]
)


fig, ax = plt.subplots(
    figsize=(8.5, 6.5)
)

ax.scatter(
    x_valid,
    y_valid,
    s=45,
    alpha=0.8,
)

ax.plot(
    x_line,
    y_line,
    linestyle="--",
    linewidth=1.5,
)

ax.set_xlabel(
    "Mean fold-ranking stability (Kendall's $\\tau$)"
)

ax.set_ylabel(
    "Dataset-removal instability"
)

ax.set_title(
    "Evaluation-split stability vs. dataset-removal instability"
)

ax.text(
    0.03,
    0.97,
    (
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"$p$ = {p_value:.3f}\n"
        f"$n$ = {len(x_valid)}"
    ),
    transform=ax.transAxes,
    va="top",
)

ax.grid(
    linestyle=":",
    linewidth=0.7,
    alpha=0.5,
)

fig.tight_layout()

fig.savefig(
    FIGURE_3,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_3_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("FOLD STABILITY FIGURES CREATED")
print("=" * 70)

print()
print("PNG:")
print(f"  {FIGURE_1}")
print(f"  {FIGURE_2}")
print(f"  {FIGURE_3}")

print()
print("PDF:")
print(f"  {FIGURE_1_PDF}")
print(f"  {FIGURE_2_PDF}")
print(f"  {FIGURE_3_PDF}")

print()
print("Figure 1:")
print(
    f"  Mean pairwise Kendall tau = "
    f"{pairwise_df['kendall_tau'].mean():.6f}"
)

print()
print("Figure 2:")
print(
    f"  Mean dataset fold stability = "
    f"{dataset_df['mean_kendall_tau'].mean():.6f}"
)

print()
print("Figure 3:")
print(
    f"  Spearman rho = {rho:.6f}"
    f"  (p = {p_value:.6f})"
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)