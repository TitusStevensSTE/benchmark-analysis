from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "analysis" / "results"
OUTPUT_DIR = RESULTS_DIR / "stability_dimensions"
PLOTS_DIR = OUTPUT_DIR / "plots"

SCORE_LODO = RESULTS_DIR / "lodo_score_results.csv"
ELO_LODO = RESULTS_DIR / "lodo_elo_results.csv"
REMOVAL_DIR = RESULTS_DIR / "dataset_removal_search" / "score"
REMOVAL_COMPACT = REMOVAL_DIR / "most_destabilizing_subsets.csv"
REMOVAL_EVALUATIONS = REMOVAL_DIR / "evaluations.csv"
REMOVAL_DEPTH_SUMMARY = REMOVAL_DIR / "depth_summary.csv"
REMOVAL_BEAM_STATES = REMOVAL_DIR / "beam_states.csv"
REMOVAL_WINNER = REMOVAL_DIR / "influence" / "minimum_winner_change_subsets.csv"
REMOVAL_TAU = REMOVAL_DIR / "influence" / "minimum_tau_break_subsets.csv"

# ============================================================
# CONCEPT
# ============================================================
# We explicitly separate three observable levels of ranking stability:
#
# 1. GLOBAL RANKING STABILITY
#    How much does the complete ordering change?
#    -> 1 - Kendall tau
#
# 2. RANK-POSITION STABILITY
#    How far can one method move?
#    -> maximum absolute rank change
#    -> number of methods whose ranks change
#
# 3. WINNER STABILITY
#    Does the identity of the top-ranked method change?
#    -> winner_changed
#
# These are deliberately treated as different levels of ranking stability,
# not as independent or orthogonal dimensions. The script analyzes both
# single-dataset LODO and joint-removal perturbations.
#
# Reproducibility note for joint removal:
# - depths 1--5 were evaluated exhaustively in the completed search;
# - depths 6--8 are partially evaluated search/beam results.
# Therefore, depth >= 6 values are reported as observed search results,
# not as exhaustive worst cases.


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_dataset_list(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    # Fallback for simple comma-separated encodings.
    return [x.strip() for x in text.split(",") if x.strip()]


def load_lodo(path: Path, aggregation: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "dataset" not in df.columns:
        for candidate in ("removed_task", "dataset_removed"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "dataset"})
                break

    required = {"dataset", "kendall_tau", "max_rank_change", "num_rank_changes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["aggregation"] = aggregation
    df["perturbation"] = "single_dataset_lodo"
    df["global_rank_instability"] = 1.0 - pd.to_numeric(df["kendall_tau"], errors="coerce")
    df["max_rank_change"] = pd.to_numeric(df["max_rank_change"], errors="coerce")
    df["num_rank_changes"] = pd.to_numeric(df["num_rank_changes"], errors="coerce")

    if "winner_changed" in df.columns:
        df["winner_changed"] = df["winner_changed"].map(parse_bool)
    else:
        df["winner_changed"] = False

    return df


def load_removal(max_rows_per_depth: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load joint-removal results without requiring the large evaluations.csv.

    On the desktop, the removal search produced the full evaluations cache, but
    that cache is intentionally not part of the laptop working copy. This
    function therefore uses the small depth_summary.csv for exact per-depth
    search statistics and most_destabilizing_subsets.csv for representative
    perturbations used in plots.
    """
    if not REMOVAL_DEPTH_SUMMARY.exists():
        raise FileNotFoundError(
            "The dataset-removal depth summary was not found:\n"
            f"  {REMOVAL_DEPTH_SUMMARY}\n"
            "Transfer depth_summary.csv from the completed removal search."
        )

    summary = pd.read_csv(REMOVAL_DEPTH_SUMMARY)
    required_summary = {"num_removed", "best_tau"}
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(
            f"{REMOVAL_DEPTH_SUMMARY} is missing columns: {sorted(missing)}"
        )

    summary = summary.copy()
    summary["num_removed"] = pd.to_numeric(summary["num_removed"], errors="coerce")
    summary["best_kendall_tau"] = pd.to_numeric(summary["best_tau"], errors="coerce")
    if "evaluated" in summary.columns:
        summary["evaluated_subsets"] = pd.to_numeric(summary["evaluated"], errors="coerce")
    else:
        summary["evaluated_subsets"] = np.nan

    # A depth is exhaustive exactly when the number of evaluated subsets equals
    # the total number of combinations from 51 datasets. This is an independent
    # coverage check and avoids labelling partial beam-search depths as worst cases.
    n_datasets = 51
    summary["total_possible_subsets"] = summary["num_removed"].apply(
        lambda d: math.comb(n_datasets, int(d)) if pd.notna(d) else np.nan
    )
    summary["exhaustive"] = (
        summary["evaluated_subsets"].notna()
        & (summary["evaluated_subsets"] == summary["total_possible_subsets"])
    )

    # depth_summary stores rates, not absolute event counts. We retain both the
    # exact search rate and an integer count when an evaluated-subset count is
    # available.
    if "winner_change_rate" in summary.columns:
        summary["winner_change_rate"] = pd.to_numeric(
            summary["winner_change_rate"], errors="coerce"
        )
    else:
        summary["winner_change_rate"] = 0.0
    if "tau_unstable_rate" in summary.columns:
        summary["tau_unstable_rate"] = pd.to_numeric(
            summary["tau_unstable_rate"], errors="coerce"
        )
    else:
        summary["tau_unstable_rate"] = (summary["best_kendall_tau"] < 0.90).astype(float)

    summary["winner_changes"] = np.where(
        summary["winner_change_rate"].notna() & summary["evaluated_subsets"].notna(),
        np.rint(summary["winner_change_rate"] * summary["evaluated_subsets"]).astype(int),
        np.nan,
    )
    summary["tau_breaks"] = np.where(
        summary["tau_unstable_rate"].notna() & summary["evaluated_subsets"].notna(),
        np.rint(summary["tau_unstable_rate"] * summary["evaluated_subsets"]).astype(int),
        np.nan,
    )
    summary["winner_change_observed"] = summary["winner_change_rate"] > 0
    summary["tau_break_observed"] = summary["tau_unstable_rate"] > 0
    summary["global_rank_instability"] = 1.0 - summary["best_kendall_tau"]
    summary["max_rank_change_observed"] = np.nan
    summary = summary.sort_values("num_removed").reset_index(drop=True)
    summary = summary[
        [
            "num_removed",
            "evaluated_subsets",
            "best_kendall_tau",
            "winner_changes",
            "tau_breaks",
            "max_rank_change_observed",
            "winner_change_observed",
            "tau_break_observed",
            "global_rank_instability",
            "winner_change_rate",
            "tau_unstable_rate",
            "total_possible_subsets",
            "exhaustive",
        ]
    ]

    # Compact representative subsets are sufficient for the global/local
    # stability-space plot. They are NOT treated as an exhaustive sample.
    if REMOVAL_COMPACT.exists():
        removal_plot = pd.read_csv(REMOVAL_COMPACT)
        if "depth" in removal_plot.columns and "num_removed" not in removal_plot.columns:
            removal_plot = removal_plot.rename(columns={"depth": "num_removed"})
    elif REMOVAL_BEAM_STATES.exists():
        removal_plot = pd.read_csv(REMOVAL_BEAM_STATES)
        if "depth" in removal_plot.columns and "num_removed" not in removal_plot.columns:
            removal_plot = removal_plot.rename(columns={"depth": "num_removed"})
    else:
        removal_plot = pd.DataFrame()

    if not removal_plot.empty:
        for col in ["num_removed", "kendall_tau", "max_rank_change", "num_rank_changes"]:
            if col in removal_plot.columns:
                removal_plot[col] = pd.to_numeric(removal_plot[col], errors="coerce")
        if "winner_changed" in removal_plot.columns:
            removal_plot["winner_changed"] = removal_plot["winner_changed"].map(parse_bool)
        else:
            removal_plot["winner_changed"] = False
        removal_plot["global_rank_instability"] = 1.0 - removal_plot["kendall_tau"]
        removal_plot["aggregation"] = "score"
        removal_plot["perturbation"] = "joint_removal_representative"

        # Keep the plotting file compact even if a larger representative file
        # is present.
        if "num_removed" in removal_plot.columns:
            removal_plot = (
                removal_plot
                .sort_values(["num_removed", "winner_changed", "kendall_tau"], ascending=[True, False, True])
                .groupby("num_removed", group_keys=False)
                .head(max_rows_per_depth)
                .reset_index(drop=True)
            )

    print(f"Using depth summary: {REMOVAL_DEPTH_SUMMARY}")
    depths = summary['num_removed'].astype(int).tolist()
    exact_flags = summary['exhaustive'].tolist() if 'exhaustive' in summary.columns else []
    exact_depths = [d for d, is_exact in zip(depths, exact_flags) if bool(is_exact)]
    partial_depths = [d for d, is_exact in zip(depths, exact_flags) if not bool(is_exact)]
    if exact_depths:
        print(f"Exhaustive search summaries: depths {exact_depths}.")
    if partial_depths:
        print(f"Partial/beam search summaries: depths {partial_depths}.")
    if not removal_plot.empty:
        print(f"Loaded {len(removal_plot):,} representative joint-removal subsets for plotting from {REMOVAL_COMPACT if REMOVAL_COMPACT.exists() else REMOVAL_BEAM_STATES}.")
    else:
        print("No compact joint-removal subset file available; depth summary plots will still be generated.")

    return removal_plot, summary


def load_event_depths(path: Path, event_name: str) -> list[int]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "num_removed" not in df.columns:
        for candidate in ("depth", "removal_depth"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "num_removed"})
                break
    if "num_removed" not in df.columns:
        return []
    values = pd.to_numeric(df["num_removed"], errors="coerce").dropna().astype(int)
    return sorted(values.unique().tolist())


def save_correlation(x: pd.Series, y: pd.Series, x_name: str, y_name: str) -> dict:
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < 3:
        return {
            "x": x_name,
            "y": y_name,
            "n": len(sub),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
        }
    rho, rho_p = spearmanr(sub["x"], sub["y"])
    tau, tau_p = kendalltau(sub["x"], sub["y"])
    return {
        "x": x_name,
        "y": y_name,
        "n": len(sub),
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "kendall_tau": tau,
        "kendall_p": tau_p,
    }


def make_lodo_plot(score: pd.DataFrame, elo: pd.DataFrame | None = None) -> None:
    # Core analysis uses score LODO only. Elo is an optional secondary comparison
    # because the project currently stores Elo diagnostics under score_vs_elo/
    # rather than as a single lodo_elo_results.csv export.
    merged = score[["dataset", "global_rank_instability", "max_rank_change", "num_rank_changes"]].copy()

    # Figure A: global vs local stability, one point per dataset.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].scatter(merged["global_rank_instability"], merged["max_rank_change"])
    axes[0].set_xlabel("Global instability (1 − Kendall τ)")
    axes[0].set_ylabel("Maximum rank change")
    axes[0].set_title("TabArena LODO: global vs. local instability")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(merged["global_rank_instability"], merged["num_rank_changes"])
    axes[1].set_xlabel("Global instability (1 − Kendall τ)")
    axes[1].set_ylabel("Number of changed method ranks")
    axes[1].set_title("TabArena LODO: global vs. breadth of change")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lodo_multidimensional_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Optional Figure B: score vs Elo global instability, only if a full Elo LODO
    # table is available.
    if elo is not None:
        merged_elo = score[["dataset", "global_rank_instability"]].merge(
            elo[["dataset", "global_rank_instability"]],
            on="dataset",
            suffixes=("_score", "_elo"),
        )
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(merged_elo["global_rank_instability_score"], merged_elo["global_rank_instability_elo"])
        lo = min(merged_elo["global_rank_instability_score"].min(), merged_elo["global_rank_instability_elo"].min())
        hi = max(merged_elo["global_rank_instability_score"].max(), merged_elo["global_rank_instability_elo"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel("Score aggregation: 1 − Kendall τ")
        ax.set_ylabel("Elo aggregation: 1 − Kendall τ")
        ax.set_title("LODO global instability: score vs. Elo")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "lodo_score_vs_elo_instability.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_joint_removal_plot(removal: pd.DataFrame, event_summary: pd.DataFrame) -> None:
    # Figure C: representative joint-removal perturbations in global/local space.
    # This uses only the compact files that are safe to keep on both machines.
    plot_df = removal.dropna(subset=["global_rank_instability"]).copy()
    if not plot_df.empty and "max_rank_change" in plot_df.columns:
        plot_df = plot_df.dropna(subset=["max_rank_change"])

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        changed = plot_df["winner_changed"].astype(bool)
        ax.scatter(
            plot_df.loc[~changed, "global_rank_instability"],
            plot_df.loc[~changed, "max_rank_change"],
            s=18,
            alpha=0.35,
            label="Winner unchanged",
        )
        ax.scatter(
            plot_df.loc[changed, "global_rank_instability"],
            plot_df.loc[changed, "max_rank_change"],
            s=35,
            alpha=0.9,
            label="Winner changed",
        )
        ax.set_xlabel("Global ranking instability (1 − Kendall τ)")
        ax.set_ylabel("Maximum rank change")
        ax.set_title("Representative joint-removal perturbations")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "joint_removal_stability_space.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Figure D: observed/global instability by perturbation depth.
    # Depths 1--5 are exhaustive; later depths are partial search results.
    depth_summary = event_summary.copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        depth_summary["num_removed"],
        depth_summary["global_rank_instability"],
        marker="o",
        label="Observed global instability (1 − Kendall τ)",
    )
    winner_depths = depth_summary.loc[
        depth_summary["winner_change_observed"], "num_removed"
    ]
    for depth in winner_depths:
        ax.axvline(depth, linestyle="--", alpha=0.5)
    ax.axhline(0.10, linestyle=":", alpha=0.6, label="Kendall τ = 0.90")
    ax.set_xlabel("Number of jointly removed datasets")
    ax.set_ylabel("Global instability (1 − Kendall τ)")
    ax.set_title("Observed ranking instability under increasing task removal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stability_dimensions_by_removal_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze different levels of ranking stability on TabArena.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    global RESULTS_DIR, OUTPUT_DIR, PLOTS_DIR, SCORE_LODO, ELO_LODO, REMOVAL_DIR, REMOVAL_EVALUATIONS, REMOVAL_DEPTH_SUMMARY, REMOVAL_BEAM_STATES, REMOVAL_WINNER, REMOVAL_TAU
    if args.project_root != PROJECT_ROOT:
        RESULTS_DIR = args.project_root / "Code" / "analysis" / "results"
        OUTPUT_DIR = RESULTS_DIR / "stability_dimensions"
        PLOTS_DIR = OUTPUT_DIR / "plots"
        SCORE_LODO = RESULTS_DIR / "lodo_score_results.csv"
        ELO_LODO = RESULTS_DIR / "lodo_elo_results.csv"
        REMOVAL_DIR = RESULTS_DIR / "dataset_removal_search" / "score"
        REMOVAL_EVALUATIONS = REMOVAL_DIR / "evaluations.csv"
        REMOVAL_DEPTH_SUMMARY = REMOVAL_DIR / "depth_summary.csv"
        REMOVAL_BEAM_STATES = REMOVAL_DIR / "beam_states.csv"
        REMOVAL_WINNER = REMOVAL_DIR / "influence" / "minimum_winner_change_subsets.csv"
        REMOVAL_TAU = REMOVAL_DIR / "influence" / "minimum_tau_break_subsets.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    score = load_lodo(SCORE_LODO, "score")
    elo = None
    if ELO_LODO.exists():
        try:
            elo = load_lodo(ELO_LODO, "elo")
        except Exception as exc:
            print(f"Warning: ignoring Elo LODO file {ELO_LODO}: {exc}")
    else:
        print(f"Note: no standalone Elo LODO CSV found at {ELO_LODO}; continuing with score LODO only.")
    removal, event_summary = load_removal()

    # --------------------------------------------------------
    # Core TabArena LODO table
    # --------------------------------------------------------
    lodo = score.copy()
    lodo = lodo.rename(columns={
        "global_rank_instability": "global_instability_score",
        "max_rank_change": "max_rank_change_score",
        "num_rank_changes": "num_rank_changes_score",
        "winner_changed": "winner_changed_score",
    })

    if elo is not None:
        elo_small = elo[[
            "dataset", "kendall_tau", "spearman_rho", "max_rank_change", "num_rank_changes", "winner_changed",
        ]].rename(columns={
            "kendall_tau": "kendall_tau_elo",
            "spearman_rho": "spearman_rho_elo",
            "max_rank_change": "max_rank_change_elo",
            "num_rank_changes": "num_rank_changes_elo",
            "winner_changed": "winner_changed_elo",
            "global_rank_instability": "global_instability_elo",
        })
        lodo = lodo.merge(elo_small, on="dataset", how="left")
        lodo["global_instability_elo"] = 1.0 - pd.to_numeric(lodo["kendall_tau_elo"], errors="coerce")

    # Rank summaries: which datasets are extreme under each stability definition?
    lodo["global_instability_rank"] = lodo["global_instability_score"].rank(ascending=False, method="min")
    lodo["max_rank_change_rank"] = lodo["max_rank_change_score"].rank(ascending=False, method="min")
    lodo["num_rank_changes_rank"] = lodo["num_rank_changes_score"].rank(ascending=False, method="min")
    lodo["dimension_rank_disagreement"] = (
        (lodo["global_instability_rank"] - lodo["max_rank_change_rank"]).abs()
        + (lodo["global_instability_rank"] - lodo["num_rank_changes_rank"]).abs()
    ) / 2.0
    lodo.to_csv(OUTPUT_DIR / "lodo_stability_dimensions.csv", index=False)

    # Correlations among rank-based stability measures.
    correlations = [
        save_correlation(lodo["global_instability_score"], lodo["max_rank_change_score"], "global_instability", "max_rank_change"),
        save_correlation(lodo["global_instability_score"], lodo["num_rank_changes_score"], "global_instability", "num_rank_changes"),
        save_correlation(lodo["max_rank_change_score"], lodo["num_rank_changes_score"], "max_rank_change", "num_rank_changes"),
        save_correlation(lodo["global_instability_score"], lodo["global_instability_elo"], "score_global_instability", "elo_global_instability") if "global_instability_elo" in lodo.columns else None,
    ]
    correlations = [row for row in correlations if row is not None]
    pd.DataFrame(correlations).to_csv(OUTPUT_DIR / "stability_dimension_correlations.csv", index=False)

    # Top datasets by each definition.
    top_rows = []
    for metric, label in [
        ("global_instability_score", "global_ranking_instability"),
        ("max_rank_change_score", "maximum_rank_change"),
        ("num_rank_changes_score", "number_of_changed_ranks"),
    ]:
        for _, row in lodo.nlargest(10, metric).iterrows():
            top_rows.append({"dimension": label, "dataset": row["dataset"], "value": row[metric]})
    pd.DataFrame(top_rows).to_csv(OUTPUT_DIR / "top_datasets_by_stability_dimension.csv", index=False)

    # Joint-removal summary from the stored depth-level search results.
    # Depths marked exhaustive can support exhaustive claims; later depths are
    # only observed search results.
    event_summary.to_csv(OUTPUT_DIR / "joint_removal_stability_by_depth.csv", index=False)

    # Small machine-readable summary for use in thesis writing.
    winner_depths = event_summary.loc[event_summary["winner_change_observed"], "num_removed"]
    tau_depths = event_summary.loc[event_summary["tau_break_observed"], "num_removed"]
    min_winner_depth = int(winner_depths.min()) if not winner_depths.empty else np.nan
    min_tau_depth = int(tau_depths.min()) if not tau_depths.empty else np.nan
    summary = pd.DataFrame([
        {
            "n_datasets": len(score),
            "n_methods": "44 (from final score matrix)",
            "min_removals_winner_change": min_winner_depth,
            "min_removals_tau_below_0_90": min_tau_depth,
            "score_lodo_winner_changes": int(score["winner_changed"].sum()),
            "score_lodo_min_kendall_tau": float(score["kendall_tau"].min()),
        }
    ])
    summary.to_csv(OUTPUT_DIR / "stability_dimensions_summary.csv", index=False)

    make_lodo_plot(score, elo)
    make_joint_removal_plot(removal, event_summary)

    print("=" * 78)
    print("TABARENA STABILITY LEVELS ANALYSIS")
    print("=" * 78)
    print(f"Project root: {args.project_root}")
    print(f"Output:       {OUTPUT_DIR}")
    print()
    print("Different levels of ranking stability:")
    print("  1. Global ranking: 1 - Kendall tau")
    print("  2. Rank-position stability: maximum rank change + number of changed ranks")
    print("  3. Winner stability: winner_changed")
    print()
    print(f"Score LODO winner changes: {int(score['winner_changed'].sum())}")
    print(f"Standalone Elo LODO loaded: {elo is not None}")
    print(f"Score LODO minimum Kendall tau: {score['kendall_tau'].min():.6f}")
    print(f"First observed removal depth with winner change: {min_winner_depth}")
    print(f"First observed removal depth with tau < 0.90: {min_tau_depth}")
    print()
    print("Saved:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        if path.is_file():
            print(f"  {path.relative_to(RESULTS_DIR)}")
    print(f"  plots/")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    {path.name}")


if __name__ == "__main__":
    main()
