from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import RESULTS_DIR


# ============================================================
# PATHS
# ============================================================

FINAL_DIR = RESULTS_DIR / "final_figures"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

LODO_SCORE = RESULTS_DIR / "lodo_score_results.csv"

ELO_CANDIDATES = [
    RESULTS_DIR / "phase1_crossbenchmark" / "tabarena" / "official_mle_elo" / "lodo_results.csv",
    RESULTS_DIR / "lodo_elo_results.csv",
]

RANDOM_SUBSET = (
    RESULTS_DIR / "random_subset_winner_frequency" / "random_subset_repetitions.csv"
)

INFLUENCE = (
    RESULTS_DIR / "dataset_influence_final" / "dataset_influence_ranked.csv"
)

PERFORMANCE_GAP = (
    RESULTS_DIR / "performance_gap" / "performance_gap_vs_lodo.csv"
)

JOINT_REMOVAL = (
    RESULTS_DIR / "dataset_removal_search" / "score" / "depth_summary.csv"
)

WINNER_CHANGE_SUBSETS = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "score"
    / "influence"
    / "minimum_winner_change_subsets.csv"
)

TAU_BREAK_SUBSETS = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "score"
    / "influence"
    / "minimum_tau_break_subsets.csv"
)

BENCHBENCH = RESULTS_DIR / "benchbench_metrics" / "benchbench_metrics.csv"

CROSS_BENCHMARKS = {
    "TabArena": LODO_SCORE,
    "GLUE": RESULTS_DIR / "phase1_crossbenchmark" / "glue" / "lodo_results.csv",
    "OpenLLM": (
        RESULTS_DIR
        / "phase1_crossbenchmark"
        / "openllm"
        / "unique_model_mean"
        / "lodo_results.csv"
    ),
    "ImageNet": RESULTS_DIR / "phase1_crossbenchmark" / "imagenet" / "lodo_results.csv",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_file(path: Path) -> bool:
    if not path.exists():
        warnings.warn(f"Missing input: {path}")
        return False
    return True


def read_csv(path: Path) -> pd.DataFrame | None:
    if not require_file(path):
        return None
    return pd.read_csv(path)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {str(column).strip().lower(): column for column in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not find any of {candidates} in {list(df.columns)}"
        )
    return None


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = FINAL_DIR / f"{stem}.png"
    pdf_path = FINAL_DIR / f"{stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved {png_path}")
    print(f"  saved {pdf_path}")


# ============================================================
# FIGURE 1
# ============================================================

def figure_01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    stages = [
        (0.4, 2.0, 2.0, 1.5, "Benchmark data", "Models × datasets/tasks"),
        (3.0, 2.0, 2.0, 1.5, "Baseline ranking", "Aggregate task scores\nor MLE Elo"),
        (5.6, 2.0, 2.0, 1.5, "Perturbations", "LODO\nRandom subsets\nJoint removals"),
        (8.2, 2.0, 2.0, 1.5, "Stability diagnostics", "Kendall τ\nRank movement\nWinner changes"),
        (10.8, 2.0, 2.0, 1.5, "Cross-benchmark\nsynthesis", "Diversity ↔ sensitivity"),
    ]

    for x, y, w, h, title, subtitle in stages:
        patch = plt.Rectangle(
            (x, y), w, h, fill=False, linewidth=1.8
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + 0.98,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + 0.42,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
        )

    for start_x in [2.4, 5.0, 7.6, 10.2]:
        ax.annotate(
            "",
            xy=(start_x + 0.6, 2.75),
            xytext=(start_x, 2.75),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

    ax.text(
        6.5,
        4.45,
        "Benchmark ranking robustness analysis",
        ha="center",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        6.5,
        0.65,
        "Primary stability view: leave-one-dataset-out changes in the model ranking; stronger perturbations probe failure boundaries.",
        ha="center",
        fontsize=10.5,
    )

    save_figure(fig, "figure_01_pipeline")


# ============================================================
# FIGURE 2
# ============================================================

def figure_02_lodo_score_vs_elo() -> None:
    score = read_csv(LODO_SCORE)
    if score is None:
        return

    elo_path = next(
        (path for path in ELO_CANDIDATES if path.exists()),
        None,
    )
    if elo_path is None:
        warnings.warn("No Elo LODO result file found.")
        return

    elo = pd.read_csv(elo_path)

    score_task = find_column(score, ["dataset_removed", "removed_task"])
    elo_task = find_column(elo, ["dataset_removed", "removed_task"])
    score_tau = find_column(score, ["kendall_tau"])
    elo_tau = find_column(elo, ["kendall_tau"])

    # Rename before merging so the Kendall columns remain distinct.
    score_part = score[[score_task, score_tau]].rename(
        columns={
            score_task: "dataset",
            score_tau: "score_tau",
        }
    )
    elo_part = elo[[elo_task, elo_tau]].rename(
        columns={
            elo_task: "dataset",
            elo_tau: "elo_tau",
        }
    )

    merged = score_part.merge(
        elo_part,
        on="dataset",
        how="inner",
    ).sort_values("score_tau")

    if merged.empty:
        warnings.warn("Score/Elo LODO merge produced no shared datasets.")
        return

    x = np.arange(len(merged))

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(
        x,
        merged["score_tau"],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Score",
    )
    ax.plot(
        x,
        merged["elo_tau"],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Elo",
    )
    ax.axhline(
        0.90,
        linestyle="--",
        linewidth=1,
        label="τ = 0.90",
    )

    ax.set_xlabel("Datasets removed (ordered by Score Kendall's τ)")
    ax.set_ylabel("Kendall's τ")
    ax.set_ylim(
        max(
            0.0,
            merged[["score_tau", "elo_tau"]].min().min() - 0.02,
        ),
        1.005,
    )
    ax.set_title("TabArena LODO ranking stability: Score vs. Elo")
    ax.legend()

    mean_score = merged["score_tau"].mean()
    mean_elo = merged["elo_tau"].mean()
    min_score = merged["score_tau"].min()
    min_elo = merged["elo_tau"].min()

    ax.text(
        0.01,
        0.04,
        f"Mean τ: Score={mean_score:.4f}, Elo={mean_elo:.4f}   |   "
        f"Minimum τ: Score={min_score:.4f}, Elo={min_elo:.4f}",
        transform=ax.transAxes,
        fontsize=9.5,
    )

    fig.tight_layout()
    save_figure(fig, "figure_02_lodo_score_vs_elo")


# ============================================================
# FIGURE 3
# ============================================================

def figure_03_lodo_rank_movement() -> None:
    df = read_csv(LODO_SCORE)
    if df is None:
        return

    dataset_col = find_column(df, ["dataset_removed", "removed_task"])
    max_col = find_column(df, ["max_rank_change"])
    num_col = find_column(df, ["num_rank_changes"])

    top_max = df.nlargest(10, max_col).sort_values(max_col)
    top_num = df.nlargest(10, num_col).sort_values(num_col)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    axes[0].barh(top_max[dataset_col], top_max[max_col])
    axes[0].set_xlabel("Maximum absolute rank change")
    axes[0].set_ylabel("Dataset removed")
    axes[0].set_title("Largest individual rank movements")

    axes[1].barh(top_num[dataset_col], top_num[num_col])
    axes[1].set_xlabel("Methods with changed rank")
    axes[1].set_ylabel("Dataset removed")
    axes[1].set_title("Largest numbers of affected methods")

    fig.suptitle(
        "LODO can move rankings materially even when the winner remains stable",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "figure_03_lodo_rank_movement")


# ============================================================
# FIGURE 4
# ============================================================

def figure_04_random_subset_stability() -> None:
    df = read_csv(RANDOM_SUBSET)
    if df is None:
        return

    size_col = find_column(df, ["num_datasets"])
    winner_col = find_column(df, ["winner"])
    rank_col = find_column(df, ["winner_baseline_rank"])

    rank_info = df.groupby(winner_col)[rank_col].min().sort_values()
    top_methods = rank_info.head(10).index.tolist()

    counts = (
        df[df[winner_col].isin(top_methods)]
        .groupby([size_col, winner_col])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        df.groupby(size_col)
        .size()
        .rename("total")
        .reset_index()
    )
    counts = counts.merge(totals, on=size_col, how="left")
    counts["frequency"] = counts["count"] / counts["total"]

    pivot = (
        counts.pivot(
            index=size_col,
            columns=winner_col,
            values="frequency",
        )
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 6.3))

    for method in top_methods:
        if method not in pivot.columns:
            continue
        ax.plot(
            pivot.index,
            pivot[method],
            linewidth=1.5,
            label=method,
        )

    ax.set_xlabel("Number of datasets sampled")
    ax.set_ylabel("Winner frequency")
    ax.set_ylim(0, 1.02)
    ax.set_title("Random benchmark composition changes which methods win")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.text(
        0.01,
        0.03,
        "1000 random subsets were evaluated at each benchmark size.",
        transform=ax.transAxes,
        fontsize=9.5,
    )

    fig.tight_layout()
    save_figure(fig, "figure_04_random_subset_stability")


# ============================================================
# FIGURE 5
# ============================================================

def _event_depth_from_file(path: Path) -> int | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    depth_col = find_column(
        df,
        ["depth", "removal_depth", "num_removed"],
        required=False,
    )

    if depth_col is None or df.empty:
        return None

    return int(pd.to_numeric(df[depth_col]).min())


def figure_05_joint_removal_boundary() -> None:
    df = read_csv(JOINT_REMOVAL)
    if df is None:
        return

    depth_col = find_column(
        df,
        ["depth", "removal_depth", "num_removed"],
    )
    tau_col = find_column(
        df,
        [
            "best_tau",
            "best_kendall_tau",
            "min_kendall_tau",
            "minimum_kendall_tau",
        ],
    )
    winner_col = find_column(
        df,
        [
            "winner_change_count",
            "winner_changes",
            "num_winner_changes",
            "winner_change_subsets",
            "winner_change_rate",
        ],
        required=False,
    )
    break_col = find_column(
        df,
        [
            "tau_break_count",
            "tau_breaks",
            "num_tau_breaks",
            "tau_break_subsets",
            "tau_unstable_rate",
        ],
        required=False,
    )

    data = df.copy()
    data[depth_col] = pd.to_numeric(data[depth_col])
    data[tau_col] = pd.to_numeric(data[tau_col])
    data = data.sort_values(depth_col)

    winner_depth = _event_depth_from_file(WINNER_CHANGE_SUBSETS)
    tau_break_depth = _event_depth_from_file(TAU_BREAK_SUBSETS)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    axes[0].plot(
        data[depth_col],
        data[tau_col],
        marker="o",
        linewidth=1.8,
    )
    axes[0].axhline(
        0.90,
        linestyle="--",
        linewidth=1,
        label="τ = 0.90",
    )

    if tau_break_depth is not None:
        axes[0].axvline(
            tau_break_depth,
            linestyle=":",
            linewidth=1.5,
            label=f"First τ-break: depth {tau_break_depth}",
        )

    axes[0].set_xlabel("Number of removed datasets")
    axes[0].set_ylabel("Best observed Kendall's τ")
    axes[0].set_title("Global ranking stability")
    axes[0].legend(fontsize=8)

    if winner_col is not None:
        axes[1].plot(
            data[depth_col],
            pd.to_numeric(data[winner_col]),
            marker="o",
            linewidth=1.8,
        )
        axes[1].set_ylabel("Observed winner-change rate")
    elif break_col is not None:
        axes[1].plot(
            data[depth_col],
            pd.to_numeric(data[break_col]),
            marker="o",
            linewidth=1.8,
        )
        axes[1].set_ylabel("Observed τ-breaking subsets")
    else:
        axes[1].axis("off")

    axes[1].set_xlabel("Number of removed datasets")

    if winner_depth is not None:
        axes[1].axvline(
            winner_depth,
            linestyle=":",
            linewidth=1.5,
            label=f"First winner change: depth {winner_depth}",
        )
        axes[1].legend(fontsize=8)

    axes[1].set_title("Decision-level failure")

    fig.suptitle(
        "Coordinated dataset removal reveals different failure thresholds",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "figure_05_joint_removal_boundary")


# ============================================================
# FIGURE 6
# ============================================================

def figure_06_dataset_influence() -> None:
    df = read_csv(INFLUENCE)
    if df is None:
        return

    dataset_col = find_column(df, ["dataset"])
    mean_rank_col = find_column(df, ["mean_rank"])

    rank_columns = [
        find_column(df, ["rank_lodo_instability"]),
        find_column(df, ["rank_winner_change"]),
        find_column(df, ["rank_tau_break"]),
    ]

    top = df.sort_values(mean_rank_col).head(10).copy()
    matrix = top[rank_columns].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis_r",
    )

    ax.set_xticks(range(len(rank_columns)))
    ax.set_xticklabels(
        ["LODO instability", "Winner change", "τ-break"]
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[dataset_col])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Influence rank (1 = highest)")
    ax.set_title("Dataset influence depends on the instability dimension")
    ax.set_xlabel("Sensitivity dimension")
    ax.set_ylabel("Dataset")

    fig.tight_layout()
    save_figure(fig, "figure_06_dataset_influence")


# ============================================================
# FIGURE 7
# ============================================================

def figure_07_performance_gap() -> None:
    # GLUE pairwise performance-gap analysis: each model-pair reversal
    # rate is measured across LODO task removals.
    path = (
        RESULTS_DIR
        / "phase1_crossbenchmark"
        / "glue"
        / "performance_gap"
        / "glue_pairwise_performance_gap_lodo.csv"
    )

    df = read_csv(path)
    if df is None:
        return

    gap_col = find_column(
        df,
        ["absolute_performance_gap", "performance_gap"],
    )
    reversal_col = find_column(
        df,
        ["lodo_reversal_rate", "reversal_rate"],
    )

    plot_df = df[[gap_col, reversal_col]].copy()
    plot_df[gap_col] = pd.to_numeric(
        plot_df[gap_col],
        errors="coerce",
    )
    plot_df[reversal_col] = pd.to_numeric(
        plot_df[reversal_col],
        errors="coerce",
    )
    plot_df = plot_df.dropna()
    plot_df = plot_df[plot_df[gap_col] > 0]

    if plot_df.empty:
        warnings.warn("No valid GLUE performance-gap rows available.")
        return

    rho = plot_df[gap_col].corr(
        plot_df[reversal_col],
        method="spearman",
    )
    tau = plot_df[gap_col].corr(
        plot_df[reversal_col],
        method="kendall",
    )

    # Quantile bins provide a second view of the relationship.
    plot_df["gap_bin"] = pd.qcut(
        plot_df[gap_col],
        q=5,
        duplicates="drop",
    )
    bins = (
        plot_df.groupby("gap_bin", observed=False)[reversal_col]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    axes[0].scatter(
        plot_df[gap_col],
        plot_df[reversal_col],
        alpha=0.30,
        s=18,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Absolute task-level performance gap (log scale)")
    axes[0].set_ylabel("Pairwise LODO reversal rate")
    axes[0].set_title("Performance gap versus pairwise reversals")
    axes[0].text(
        0.02,
        0.96,
        f"Spearman ρ = {rho:.3f}\nKendall τ = {tau:.3f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9.5,
    )

    centers = np.arange(len(bins))
    axes[1].plot(
        centers,
        bins[reversal_col],
        marker="o",
        linewidth=1.8,
    )
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels(
        [str(value) for value in bins["gap_bin"].tolist()],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    axes[1].set_xlabel("Performance-gap quintile")
    axes[1].set_ylabel("Mean pairwise LODO reversal rate")
    axes[1].set_title("Reversal rate across gap groups")

    fig.suptitle(
        "Smaller GLUE performance gaps are associated with more reversals",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "figure_07_performance_gap")


# ============================================================
# FIGURE 8
# ============================================================

def figure_08_crossbenchmark_lodo() -> None:
    rows = []

    for benchmark, path in CROSS_BENCHMARKS.items():
        df = read_csv(path)
        if df is None:
            continue

        tau_col = find_column(df, ["kendall_tau"])
        winner_col = find_column(
            df,
            ["winner_changed"],
            required=False,
        )

        tau = pd.to_numeric(
            df[tau_col],
            errors="coerce",
        ).dropna()

        if tau.empty:
            continue

        winner_changes = (
            int(pd.to_numeric(df[winner_col]).sum())
            if winner_col is not None
            else np.nan
        )

        rows.append(
            {
                "benchmark": benchmark,
                "mean_tau": tau.mean(),
                "min_tau": tau.min(),
                "winner_changes": winner_changes,
                "n_tasks": len(tau),
            }
        )

    if not rows:
        return

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    x = np.arange(len(summary))

    axes[0].vlines(
        x,
        summary["min_tau"],
        summary["mean_tau"],
        linewidth=3,
    )
    axes[0].scatter(
        x,
        summary["min_tau"],
        s=60,
        label="Minimum τ",
    )
    axes[0].scatter(
        x,
        summary["mean_tau"],
        s=70,
        label="Mean τ",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["benchmark"])
    axes[0].set_ylabel("Kendall's τ")
    axes[0].set_ylim(
        max(0, summary["min_tau"].min() - 0.03),
        1.005,
    )
    axes[0].set_title("Global LODO stability")
    axes[0].legend()

    if summary["winner_changes"].notna().any():
        axes[1].bar(
            summary["benchmark"],
            summary["winner_changes"],
        )
        axes[1].set_ylabel("Winner changes")
        axes[1].set_title("Winner changes under LODO")
    else:
        axes[1].axis("off")

    fig.suptitle(
        "LODO stability varies substantially across benchmarks",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "figure_08_crossbenchmark_lodo")


# ============================================================
# FIGURE 9 — BENCHBENCH DIVERSITY VS. SENSITIVITY
# Two panels: Kendall tau on the left, MRC on the right
# ============================================================

def figure_09_diversity_vs_sensitivity() -> None:
    df = read_csv(BENCHBENCH)
    if df is None:
        return

    benchmark_col = find_column(df, ["benchmark", "name"])
    diversity_col = find_column(df, ["diversity_W"])
    tau_col = find_column(df, ["sensitivity_tau"])
    mrc_col = find_column(df, ["sensitivity_MRC"])
    rep_col = find_column(df, ["representation"], required=False)

    # Original BenchBench cardinal benchmarks plus TabArena.
    benchmark_order = [
        "TabArena",
        "GLUE",
        "SuperGLUE",
        "BIG-Bench-Hard",
        "MTEB",
        "OpenLLM",
        "MMLU",
        "VTAB",
        "ImageNet",
        "Random",
        "Constant",
    ]

    def canonical_name(value):
        key = (
            str(value).lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

        aliases = {
            "tabarena": "TabArena",
            "glue": "GLUE",
            "superglue": "SuperGLUE",
            "bigbenchhard": "BIG-Bench-Hard",
            "mteb": "MTEB",
            "openllm": "OpenLLM",
            "mmlu": "MMLU",
            "vtab": "VTAB",
            "imagenet": "ImageNet",
            "random": "Random",
            "constant": "Constant",
        }

        return aliases.get(key)

    df["_plot_name"] = df[benchmark_col].map(canonical_name)
    df = df[df["_plot_name"].notna()].copy()

    # If multiple representations exist, use the intended main
    # representation for each benchmark.
    preferred_representation = {
        "TabArena": "score_matrix",
        "OpenLLM": "native_loader",
        "ImageNet": "native_loader_mean_5_seeds",
        "Random": "native_loader_mean_5_seeds",
        "Constant": "native_loader_mean_5_seeds",
    }

    selected_rows = []

    for name in benchmark_order:
        candidates = df[df["_plot_name"] == name].copy()

        if candidates.empty:
            warnings.warn(f"Figure 9: missing benchmark {name}")
            continue

        preferred = preferred_representation.get(name)

        if preferred and rep_col is not None:
            match = candidates[
                candidates[rep_col].astype(str) == preferred
            ]
            if not match.empty:
                candidates = match

        selected_rows.append(candidates.iloc[0])

    if not selected_rows:
        warnings.warn("Figure 9: no benchmark data available.")
        return

    plot_df = pd.DataFrame(selected_rows)

    for col in [diversity_col, tau_col, mrc_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="raise")

    if plot_df[[diversity_col, tau_col, mrc_col]].isna().any().any():
        raise ValueError("Figure 9: missing diversity or sensitivity values.")

    # Use the same regression convention as the original notebook:
    # a linear fit through the origin, y = slope * x.
    def draw_panel(ax, sensitivity_col, ylabel, title):
        x = plot_df[diversity_col].to_numpy(dtype=float)
        y = plot_df[sensitivity_col].to_numpy(dtype=float)

        for _, row in plot_df.iterrows():
            name = row["_plot_name"]
            marker = "*" if name == "TabArena" else "o"
            size = 180 if name == "TabArena" else 65

            ax.scatter(
                row[diversity_col],
                row[sensitivity_col],
                marker=marker,
                s=size,
                zorder=3,
            )

            # Small offsets reduce label overlap.
            offset = {
                "TabArena": (7, 8),
                "OpenLLM": (-48, -12),
                "Random": (-38, 7),
                "Constant": (7, 7),
                "SuperGLUE": (7, -12),
                "BIG-Bench-Hard": (7, -12),
                "VTAB": (7, -12),
            }.get(name, (6, 6))

            ax.annotate(
                name,
                (row[diversity_col], row[sensitivity_col]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )

        denominator = float(x @ x)

        if len(x) >= 2 and denominator > 0:
            slope = float(x @ y) / denominator
            line_x = np.linspace(0, 1, 200)
            line_y = slope * line_x

            valid = line_y <= 1.0

            ax.plot(
                line_x[valid],
                line_y[valid],
                "--",
                color="green",
                linewidth=1.5,
                label=f"Linear fit through origin (slope={slope:.3f})",
            )
            ax.legend(loc="best", fontsize=8)

        ax.set_xlabel("BenchBench diversity (W)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    draw_panel(
        axes[0],
        tau_col,
        r"BenchBench sensitivity (Kendall $\tau$)",
        r"(a) Diversity vs. sensitivity $\tau$",
    )

    draw_panel(
        axes[1],
        mrc_col,
        "BenchBench sensitivity (MRC)",
        "(b) Diversity vs. sensitivity MRC",
    )
    axes[0].set_ylim(0, 0.6) # Limit the y-axis for Kendall tau for readability
    
    fig.suptitle(
        "Benchmark diversity and sensitivity: TabArena added to BenchBench",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        "figure_09_benchbench_diversity_vs_sensitivity_two_panel",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("FINAL THESIS FIGURES")
    print("=" * 70)
    print(f"Results root: {RESULTS_DIR}")
    print(f"Output root:  {FINAL_DIR}")
    print()

    figure_01_pipeline()
    figure_02_lodo_score_vs_elo()
    figure_03_lodo_rank_movement()
    figure_04_random_subset_stability()
    figure_05_joint_removal_boundary()
    figure_06_dataset_influence()
    figure_07_performance_gap()
    figure_08_crossbenchmark_lodo()
    figure_09_diversity_vs_sensitivity()

    print()
    print("Generated final thesis figures:")

    for path in sorted(FINAL_DIR.glob("figure_*")):
        print(f"  {path.name}")

    manifest = {
        "output_directory": str(FINAL_DIR),
        "figures": sorted(
            path.name for path in FINAL_DIR.glob("figure_*")
        ),
    }

    with (
        FINAL_DIR / "figure_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print()
    print(f"Manifest: {FINAL_DIR / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()