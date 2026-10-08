from __future__ import annotations

"""Create the main supporting thesis figures from existing CSV results.

This script is deliberately a reporting layer. It does not recompute any
benchmark scores, rankings, LODO evaluations, or removal searches.

Generated figures:
1. Performance gap vs. LODO instability.
2. Performance-gap quintiles vs. median LODO instability.
3. Dataset sensitivity profile across independent diagnostics.
4. Coordinated-removal depth and observed failure events.

All paths are repository-relative through analysis_config.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import RESULTS_DIR


PERFORMANCE_GAP_DIR = RESULTS_DIR / "performance_gap"
INFLUENCE_DIR = RESULTS_DIR / "dataset_influence_final"
REMOVAL_DIR = RESULTS_DIR / "dataset_removal_search" / "score"

FIGURE_DIR = RESULTS_DIR / "thesis_figures" / "supporting"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TAU_BREAK_THRESHOLD = 0.90


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )


def _add_panel_note(
    ax: plt.Axes,
    text: str,
    *,
    loc: str = "upper right",
) -> None:
    loc_map = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }
    x, y, ha, va = loc_map[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.85},
    )


def plot_performance_gap_vs_instability() -> Path | None:
    """Show whether narrow winner-runner-up gaps coincide with instability."""

    input_file = PERFORMANCE_GAP_DIR / "performance_gap_vs_lodo.csv"
    correlation_file = PERFORMANCE_GAP_DIR / "performance_gap_correlations.csv"

    if not input_file.exists():
        print(f"Skipping performance-gap figure: {input_file} not found.")
        return None

    df = pd.read_csv(input_file)
    _require_columns(
        df,
        {
            "dataset",
            "winner_runnerup_score_gap",
            "kendall_tau",
        },
        "performance_gap_vs_lodo.csv",
    )

    plot_df = df[
        ["dataset", "winner_runnerup_score_gap", "kendall_tau"]
    ].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df["lodo_instability"] = 1.0 - plot_df["kendall_tau"]

    if len(plot_df) < 3:
        print("Skipping performance-gap figure: fewer than 3 valid datasets.")
        return None

    rho = np.nan
    p_value = np.nan
    if correlation_file.exists():
        corr = pd.read_csv(correlation_file)
        mask = (
            (corr["x_variable"] == "winner_runnerup_score_gap")
            & (corr["y_variable"] == "kendall_tau")
        )
        match = corr.loc[mask]
        if len(match) == 1:
            # The correlation is gap vs tau; for instability the sign changes.
            rho = -float(match.iloc[0]["spearman_rho"])
            p_value = float(match.iloc[0]["spearman_p"])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    ax.scatter(
        plot_df["winner_runnerup_score_gap"],
        plot_df["lodo_instability"],
        s=42,
        alpha=0.8,
    )

    # Annotate only the most informative cases. Labeling all 51 datasets makes
    # the original exploratory scatter unreadable.
    selected = pd.concat(
        [
            plot_df.nsmallest(5, "lodo_instability"),
            plot_df.nlargest(5, "lodo_instability"),
            plot_df.nsmallest(3, "winner_runnerup_score_gap"),
        ]
    ).drop_duplicates(subset=["dataset"])

    for _, row in selected.iterrows():
        ax.annotate(
            str(row["dataset"]),
            (
                row["winner_runnerup_score_gap"],
                row["lodo_instability"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
            alpha=0.9,
        )

    ax.set_xlabel("Winner–runner-up performance gap")
    ax.set_ylabel("LODO instability (1 − Kendall's τ)")
    ax.set_title("Performance gap vs. ranking instability after dataset removal")
    ax.grid(axis="both", alpha=0.25)

    if np.isfinite(rho):
        _add_panel_note(
            ax,
            f"Spearman ρ = {rho:.3f}\n"
            f"p = {p_value:.3g}\n"
            f"n = {len(plot_df)}",
        )
    else:
        _add_panel_note(
            ax,
            f"n = {len(plot_df)}",
        )

    fig.tight_layout()
    output = FIGURE_DIR / "performance_gap_vs_lodo_instability.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_performance_gap_quintiles() -> Path | None:
    """Aggregate the same relation into five approximately equal-size bins."""

    input_file = PERFORMANCE_GAP_DIR / "performance_gap_vs_lodo.csv"

    if not input_file.exists():
        print(f"Skipping performance-gap quintile figure: {input_file} not found.")
        return None

    df = pd.read_csv(input_file)
    _require_columns(
        df,
        {
            "winner_runnerup_score_gap",
            "kendall_tau",
        },
        "performance_gap_vs_lodo.csv",
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["winner_runnerup_score_gap", "kendall_tau"]
    ).copy()
    df["lodo_instability"] = 1.0 - df["kendall_tau"]

    if len(df) < 5:
        print("Skipping performance-gap quintile figure: fewer than 5 valid datasets.")
        return None

    df["gap_bin"] = pd.qcut(
        df["winner_runnerup_score_gap"],
        q=5,
        duplicates="drop",
    )

    grouped = (
        df.groupby("gap_bin", observed=True)
        .agg(
            n=("lodo_instability", "size"),
            median_gap=("winner_runnerup_score_gap", "median"),
            q25=("lodo_instability", lambda x: x.quantile(0.25)),
            median_instability=("lodo_instability", "median"),
            q75=("lodo_instability", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    x = np.arange(len(grouped))
    y = grouped["median_instability"].to_numpy()
    lower = y - grouped["q25"].to_numpy()
    upper = grouped["q75"].to_numpy() - y

    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o-",
        capsize=4,
        linewidth=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Q{i + 1}\n"
            f"median gap={gap:.3f}\n"
            f"n={n}"
            for i, (gap, n) in enumerate(
                zip(grouped["median_gap"], grouped["n"])
            )
        ]
    )
    ax.set_xlabel(
        "Performance-gap quintile (Q1 = smallest gap)"
    )
    ax.set_ylabel("Median LODO instability (1 − Kendall's τ)")
    ax.set_title("Ranking instability across performance-gap groups")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output = FIGURE_DIR / "performance_gap_quintiles_vs_instability.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _rank_percentile(series: pd.Series, higher_is_more_sensitive: bool = True) -> pd.Series:
    """Return [0, 1] where 1 means most sensitive across all datasets."""
    rank = series.rank(method="average", ascending=not higher_is_more_sensitive)
    n = rank.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=series.index)
    return (n - rank) / (n - 1)


def plot_dataset_sensitivity_profile() -> Path | None:
    """Show the independent sensitivity dimensions for the most affected datasets.

    The figure does not define or visualize a causal 'influence score'.
    Rows are selected using the descriptive mean-rank already computed by the
    analysis, while the heatmap shows the three underlying diagnostics.
    """

    input_file = INFLUENCE_DIR / "dataset_influence_ranked.csv"

    if not input_file.exists():
        print(f"Skipping dataset-sensitivity figure: {input_file} not found.")
        return None

    df = pd.read_csv(input_file)
    required = {
        "dataset",
        "lodo_instability",
        "winner_change_frequency",
        "tau_break_frequency",
        "mean_rank",
    }
    _require_columns(df, required, "dataset_influence_ranked.csv")

    # Display the strongest 15 datasets under the descriptive mean-rank
    # ordering. The underlying dimensions remain visible separately.
    signals = [
        ("lodo_instability", "LODO instability"),
        ("winner_change_frequency", "Winner-change frequency"),
        ("tau_break_frequency", "Tau-break frequency"),
    ]

    # Recompute percentile values on the original frame so indices stay aligned.
    selected_source = (
        df.sort_values(
            ["mean_rank", "lodo_instability"],
            ascending=[True, False],
        )
        .head(15)
        .copy()
    )
    selected_source = selected_source.sort_values(
        "mean_rank", ascending=False
    )

    heatmap = np.column_stack(
        [
            _rank_percentile(
                df[col],
                higher_is_more_sensitive=True,
            ).loc[selected_source.index].to_numpy()
            for col, _ in signals
        ]
    )

    labels = selected_source["dataset"].astype(str).tolist()

    fig, ax = plt.subplots(
        figsize=(9.4, max(6.5, 0.43 * len(labels) + 1.6))
    )

    image = ax.imshow(
        heatmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(signals)))
    ax.set_xticklabels([label for _, label in signals])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Sensitivity dimension")
    ax.set_ylabel("Dataset")
    ax.set_title(
        "Dataset sensitivity across independent stability diagnostics"
    )

    # Annotate with raw values, since the heatmap itself uses percentile ranks.
    for row_idx, (_, row) in enumerate(selected_source.iterrows()):
        raw_values = [
            float(row["lodo_instability"]),
            float(row["winner_change_frequency"]),
            float(row["tau_break_frequency"]),
        ]
        for col_idx, value in enumerate(raw_values):
            if col_idx == 0:
                text = f"{value:.3f}"
            else:
                text = f"{value:.2f}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Relative sensitivity rank (0 = low, 1 = high)")

    fig.text(
        0.5,
        0.01,
        "Rows selected by descriptive mean rank. "
        "Cell colour = rank percentile within all datasets. "
        "Cell text = raw diagnostic value.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output = FIGURE_DIR / "dataset_sensitivity_profile.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_joint_removal_failure() -> Path | None:
    """Show ranking degradation and observed failure events by removal depth."""

    input_file = REMOVAL_DIR / "depth_summary.csv"
    config_file = REMOVAL_DIR / "config.json"

    if not input_file.exists():
        print(f"Skipping joint-removal figure: {input_file} not found.")
        return None

    df = pd.read_csv(input_file)
    _require_columns(
        df,
        {
            "num_removed",
            "evaluated",
            "best_tau",
            "winner_change_rate",
            "tau_unstable_rate",
        },
        "depth_summary.csv",
    )

    df = df.sort_values("num_removed").copy()

    exact_depth = int(df["num_removed"].max())
    if config_file.exists():
        try:
            config = json.loads(
                config_file.read_text(encoding="utf-8")
            )
            exact_depth = int(config.get("exact_max_removals", exact_depth))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    # Top: best observed ranking degradation.
    ax1.plot(
        df["num_removed"],
        1.0 - df["best_tau"],
        marker="o",
        linewidth=1.5,
    )
    ax1.axhline(
        1.0 - TAU_BREAK_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label=f"Kendall's τ = {TAU_BREAK_THRESHOLD:.2f}",
    )

    if exact_depth < int(df["num_removed"].max()):
        ax1.axvline(
            exact_depth + 0.5,
            linestyle=":",
            linewidth=1.2,
        )

    ax1.set_ylabel("Best observed instability (1 − Kendall's τ)")
    ax1.set_title(
        "Coordinated dataset removal: ranking degradation and failure events"
    )
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left")

    # Bottom: event rates. For beam-search depths this is the observed rate
    # within evaluated candidates, not the population rate over all subsets.
    x = df["num_removed"].to_numpy()
    width = 0.34
    ax2.bar(
        x - width / 2,
        df["winner_change_rate"],
        width=width,
        label="Winner change",
    )
    ax2.bar(
        x + width / 2,
        df["tau_unstable_rate"],
        width=width,
        label=f"τ < {TAU_BREAK_THRESHOLD:.2f}",
    )

    ax2.set_xlabel("Number of removed datasets")
    ax2.set_ylabel("Observed event rate among evaluated subsets")
    ax2.set_ylim(0, max(1.0, float(
        np.nanmax(
            [
                df["winner_change_rate"].max(),
                df["tau_unstable_rate"].max(),
                0.05,
            ]
        )
    ) * 1.15))
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left")

    if exact_depth < int(df["num_removed"].max()):
        for ax in (ax1, ax2):
            ax.axvline(
                exact_depth + 0.5,
                linestyle=":",
                linewidth=1.2,
            )
        ax2.text(
            exact_depth + 0.55,
            ax2.get_ylim()[1] * 0.88,
            "Exact → beam",
            fontsize=9,
            va="top",
        )

    # Show exact evaluated-subset counts so the heuristic boundary is clear.
    for _, row in df.iterrows():
        ax2.text(
            row["num_removed"],
            max(
                row["winner_change_rate"],
                row["tau_unstable_rate"],
                0.0,
            ) + 0.01,
            f"n={int(row['evaluated']):,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout()
    output = FIGURE_DIR / "joint_removal_failure_profile.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    print("=" * 78)
    print("THESIS SUPPORTING FIGURES")
    print("=" * 78)
    print(f"Results root: {RESULTS_DIR}")
    print(f"Figure root:  {FIGURE_DIR}")
    print()

    outputs = [
        plot_performance_gap_vs_instability(),
        plot_performance_gap_quintiles(),
        plot_dataset_sensitivity_profile(),
        plot_joint_removal_failure(),
    ]

    print("\nGenerated:")
    generated = [path for path in outputs if path is not None]
    if generated:
        for path in generated:
            print(f"  {path}")
    else:
        print("  No figures were generated because required result files are missing.")

    print("\nDone.")


if __name__ == "__main__":
    main()
