import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PORTABLE PROJECT PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in [CURRENT_DIR.parent, *CURRENT_DIR.parents]:
    if (
        (parent / "benchbench" / "pyproject.toml").exists()
        and (parent / "Code").exists()
    ):
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not locate project root. "
        "Expected a 'benchbench/pyproject.toml' and 'Code/' directory."
    )

BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"

# Analysis files: CSVs and other data needed for analysis
ANALYSIS_DIR = PROJECT_ROOT / "Code" / "analysis"

# Final outputs: plots, figures, etc.
RESULTS_DIR = PROJECT_ROOT / "Code" / "analysis" / "results" / "phase1_crossbenchmark" / "glue" / "performance_gap"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Add benchbench repository to Python path if needed
sys.path.insert(0, str(BENCHBENCH_ROOT))


# ============================================================
# INPUT FILES
# ============================================================

PAIRWISE_INPUT = (
    ANALYSIS_DIR
    / "glue_pairwise_performance_gap_lodo.csv"
)

TOP10_INPUT = (
    ANALYSIS_DIR
    / "glue_top10_adjacent_gap_analysis.csv"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 80)
print("GLUE PERFORMANCE-GAP PLOTTING")
print("=" * 80)
print()

print("Project root:")
print(PROJECT_ROOT)
print()

print("Analysis directory:")
print(ANALYSIS_DIR)
print()

print("Results directory:")
print(RESULTS_DIR)
print()

if not PAIRWISE_INPUT.exists():
    raise FileNotFoundError(
        f"Could not find pairwise input file:\n{PAIRWISE_INPUT}\n\n"
        "Expected:\n"
        "Code/analysis/glue_pairwise_performance_gap_lodo.csv"
    )

if not TOP10_INPUT.exists():
    raise FileNotFoundError(
        f"Could not find Top-10 input file:\n{TOP10_INPUT}\n\n"
        "Expected:\n"
        "Code/analysis/glue_top10_adjacent_gap_analysis.csv"
    )


# ============================================================
# LOAD DATA
# ============================================================

pairwise = pd.read_csv(
    PAIRWISE_INPUT
)

top10 = pd.read_csv(
    TOP10_INPUT
)

print("Loaded pairwise rows:", len(pairwise))
print("Loaded Top-10 pairs:", len(top10))
print()


# ============================================================
# FIGURE 1
# TOP-10 ADJACENT MODEL PAIRS
# ============================================================

print("=" * 80)
print("FIGURE 1 — TOP-10 ADJACENT MODEL PAIRS")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.scatter(
    top10["score_gap"],
    top10["reversal_rate"],
    s=80
)

for _, row in top10.iterrows():

    label = (
        f'{int(row["rank_a"])}–{int(row["rank_b"])} '
        f'{row["model_a"]} / {row["model_b"]}'
    )

    ax.annotate(
        label,
        (
            row["score_gap"],
            row["reversal_rate"]
        ),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel(
    "Absolute performance gap"
)

ax.set_ylabel(
    "LODO reversal rate"
)

ax.set_title(
    "GLUE: Performance Gap vs. LODO Reversal Rate\n"
    "Adjacent model pairs in the Top 10"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

output_file = (
    RESULTS_DIR
    / "glue_top10_performance_gap_vs_lodo_reversal.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(output_file)
print()


# ============================================================
# FIGURE 2
# ALL MODEL PAIRS
# ============================================================

print("=" * 80)
print("FIGURE 2 — ALL MODEL PAIRS")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.scatter(
    pairwise["absolute_performance_gap"],
    pairwise["lodo_reversal_rate"],
    alpha=0.35,
    s=18
)

ax.set_xlabel(
    "Absolute performance gap"
)

ax.set_ylabel(
    "LODO reversal rate"
)

ax.set_title(
    "GLUE: Performance Gap vs. LODO Reversal Rate\n"
    "All model pairs"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

output_file = (
    RESULTS_DIR
    / "glue_all_pairs_performance_gap_vs_lodo_reversal.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(output_file)
print()


# ============================================================
# FIGURE 3
# ALL MODEL PAIRS — LOGARITHMIC GAP AXIS
# ============================================================

print("=" * 80)
print("FIGURE 3 — ALL MODEL PAIRS, LOG-SCALE")
print("=" * 80)

# A log scale cannot represent zero.
plot_data = pairwise[
    pairwise["absolute_performance_gap"] > 0
].copy()

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.scatter(
    plot_data["absolute_performance_gap"],
    plot_data["lodo_reversal_rate"],
    alpha=0.35,
    s=18
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Absolute performance gap (log scale)"
)

ax.set_ylabel(
    "LODO reversal rate"
)

ax.set_title(
    "GLUE: Performance Gap vs. LODO Reversal Rate\n"
    "All model pairs — logarithmic gap scale"
)

ax.grid(
    True,
    alpha=0.3,
    which="both"
)

fig.tight_layout()

output_file = (
    RESULTS_DIR
    / "glue_all_pairs_performance_gap_vs_lodo_reversal_log.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(output_file)
print()


# ============================================================
# FIGURE 4
# REVERSAL RATE BY PERFORMANCE-GAP QUINTILE
# ============================================================

print("=" * 80)
print("FIGURE 4 — REVERSAL RATE BY PERFORMANCE-GAP BIN")
print("=" * 80)

# Create five approximately equally sized gap groups.
pairwise["gap_quantile"] = pd.qcut(
    pairwise["absolute_performance_gap"],
    q=5,
    duplicates="drop"
)

binned = (
    pairwise
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

print()
print("Gap-bin summary:")
print(
    binned.to_string(
        index=False
    )
)
print()

fig, ax = plt.subplots(
    figsize=(10, 7)
)

x = np.arange(
    len(binned)
)

ax.bar(
    x,
    binned["mean_reversal_rate"]
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        str(interval)
        for interval in binned["gap_quantile"]
    ],
    rotation=25,
    ha="right"
)

ax.set_xlabel(
    "Performance-gap quintile"
)

ax.set_ylabel(
    "Mean LODO reversal rate"
)

ax.set_title(
    "GLUE: LODO Reversal Rate by Performance Gap"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3
)

fig.tight_layout()

output_file = (
    RESULTS_DIR
    / "glue_reversal_rate_by_performance_gap_bin.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:")
print(output_file)
print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("PLOTTING COMPLETE")
print("=" * 80)
print()

print("Final plots are stored in:")
print(RESULTS_DIR)
print()

print("Generated files:")
print(
    "1.",
    RESULTS_DIR
    / "glue_top10_performance_gap_vs_lodo_reversal.png"
)

print(
    "2.",
    RESULTS_DIR
    / "glue_all_pairs_performance_gap_vs_lodo_reversal.png"
)

print(
    "3.",
    RESULTS_DIR
    / "glue_all_pairs_performance_gap_vs_lodo_reversal_log.png"
)

print(
    "4.",
    RESULTS_DIR
    / "glue_reversal_rate_by_performance_gap_bin.png"
)

print()
print("Done.")