"""Plot additional rank diagnostics from the saved TabArena random-subset runs.

The script reconstructs every method's rank from the exact sampled datasets
stored in random_subset_repetitions.csv. It does not rerun the random-subset
experiment. Before plotting, it verifies that the reconstructed winner matches
the winner stored for every repetition.

Outputs:
    results/final_figures/figure_10_random_subset_top10_rank_distributions.{png,pdf}
    results/final_figures/figure_11_random_subset_global_stability.{png,pdf}
    results/random_subset_winner_frequency/random_subset_rank_diagnostics_repetitions.csv
    results/random_subset_winner_frequency/random_subset_rank_diagnostics_summary.csv

Run from the repository root:
    python Code/analysis/plot_random_subset_rank_diagnostics.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

from analysis_config import RESULTS_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SUBSET_DIR = RESULTS_DIR / "random_subset_winner_frequency"
SCORE_FILE = RESULTS_DIR / "score_method_dataset.csv"
REPETITIONS_FILE = RANDOM_SUBSET_DIR / "random_subset_repetitions.csv"
BASELINE_FILE = RANDOM_SUBSET_DIR / "baseline_ranking.csv"

FINAL_DIR = RESULTS_DIR / "final_figures"
TOP_N = 10
# Sizes selected to show the transition from a small subset to a large one.
# The full 51-dataset case is omitted from the violins because every sampled
# subset contains the full benchmark, so its ranks are fixed by construction.
VIOLIN_SUBSET_SIZES = (5, 10, 20, 30, 40)
DPI = 300


# ---------------------------------------------------------------------------
# Input handling and validation
# ---------------------------------------------------------------------------
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n  {path}\n"
            "Run the existing random-subset analysis first, or check the "
            "repository-relative paths."
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the dataset-level scores, saved subsets and baseline ranking."""
    for path in (SCORE_FILE, REPETITIONS_FILE, BASELINE_FILE):
        require_file(path)

    scores_df = pd.read_csv(SCORE_FILE)
    required_score_columns = {"dataset", "method", "dataset_score"}
    missing = required_score_columns - set(scores_df.columns)
    if missing:
        raise ValueError(
            f"{SCORE_FILE.name} is missing columns: {sorted(missing)}"
        )

    scores_df["dataset"] = scores_df["dataset"].astype(str)
    scores_df["method"] = scores_df["method"].astype(str)
    scores_df["dataset_score"] = pd.to_numeric(
        scores_df["dataset_score"], errors="raise"
    )

    duplicate_mask = scores_df.duplicated(
        subset=["method", "dataset"], keep=False
    )
    if duplicate_mask.any():
        examples = scores_df.loc[
            duplicate_mask, ["method", "dataset"]
        ].head(10)
        raise ValueError(
            "The score matrix contains duplicate (method, dataset) rows.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    matrix = scores_df.pivot(
        index="method", columns="dataset", values="dataset_score"
    ).sort_index().sort_index(axis=1)

    if matrix.empty or matrix.isna().any().any():
        missing_cells = int(matrix.isna().sum().sum())
        raise ValueError(
            "The method-by-dataset score matrix is empty or incomplete. "
            f"Missing cells: {missing_cells}"
        )
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("The score matrix contains non-finite values.")

    repetitions = pd.read_csv(REPETITIONS_FILE)
    required_repetition_columns = {
        "num_datasets", "repetition", "winner", "sampled_datasets"
    }
    missing = required_repetition_columns - set(repetitions.columns)
    if missing:
        raise ValueError(
            f"{REPETITIONS_FILE.name} is missing columns: {sorted(missing)}"
        )
    if repetitions.empty:
        raise ValueError(f"{REPETITIONS_FILE.name} contains no rows.")

    repetitions["num_datasets"] = pd.to_numeric(
        repetitions["num_datasets"], errors="raise"
    ).astype(int)
    repetitions["repetition"] = pd.to_numeric(
        repetitions["repetition"], errors="raise"
    ).astype(int)
    repetitions["winner"] = repetitions["winner"].astype(str)

    baseline = pd.read_csv(BASELINE_FILE)
    required_baseline_columns = {"method", "baseline_rank"}
    missing = required_baseline_columns - set(baseline.columns)
    if missing:
        raise ValueError(
            f"{BASELINE_FILE.name} is missing columns: {sorted(missing)}"
        )

    baseline["method"] = baseline["method"].astype(str)
    baseline["baseline_rank"] = pd.to_numeric(
        baseline["baseline_rank"], errors="raise"
    ).astype(int)
    if baseline["method"].duplicated().any():
        raise ValueError("The saved baseline ranking contains duplicate methods.")
    if set(baseline["method"]) != set(matrix.index.astype(str)):
        raise ValueError(
            "The methods in the saved baseline ranking do not match those in "
            "score_method_dataset.csv. The artifacts may come from different runs."
        )

    # Validate the entire saved baseline order, not only the rank-1 winner.
    # The original analysis sorts method names alphabetically before applying a
    # stable descending score sort, which is reproduced by the sorted matrix.
    expected_rank_values = list(range(1, len(matrix) + 1))
    if sorted(baseline["baseline_rank"].tolist()) != expected_rank_values:
        raise ValueError(
            "The saved baseline ranking must contain each integer rank from "
            f"1 to {len(matrix)} exactly once."
        )

    expected_order = (
        matrix.mean(axis=1)
        .sort_values(ascending=False, kind="mergesort")
        .index.astype(str)
        .tolist()
    )
    saved_order = (
        baseline.sort_values("baseline_rank", kind="mergesort")["method"]
        .astype(str)
        .tolist()
    )
    if saved_order != expected_order:
        first_differences = []
        for rank, (saved_method, expected_method) in enumerate(
            zip(saved_order, expected_order), start=1
        ):
            if saved_method != expected_method:
                first_differences.append(
                    f"  rank {rank}: saved={saved_method!r}, "
                    f"expected={expected_method!r}"
                )
            if len(first_differences) == 10:
                break
        raise ValueError(
            "The full saved baseline ranking does not match the ranking "
            "recomputed from score_method_dataset.csv. The input files may "
            "come from different runs.\n"
            + "\n".join(first_differences)
        )

    print("Baseline ranking consistency check passed for all methods.")

    dataset_names = set(matrix.columns.astype(str))
    if repetitions["sampled_datasets"].isna().any():
        raise ValueError("Some repetitions have no sampled_datasets value.")

    print("Input validation")
    print("----------------")
    print(f"Methods:             {matrix.shape[0]}")
    print(f"Datasets:            {matrix.shape[1]}")
    print(f"Saved repetitions:   {len(repetitions):,}")
    print(f"Subset sizes:        {repetitions['num_datasets'].nunique()}")
    print()

    # Check the sampled-dataset strings before doing the main computation.
    for row in repetitions[["num_datasets", "sampled_datasets"]].itertuples(
        index=False, name=None
    ):
        subset_size, sampled_string = row
        sampled = str(sampled_string).split("|")
        if len(sampled) != subset_size:
            raise ValueError(
                "The number of dataset names in sampled_datasets does not "
                f"match num_datasets={subset_size}: {sampled_string!r}"
            )
        if len(set(sampled)) != len(sampled):
            raise ValueError(
                f"A sampled subset contains duplicate dataset names: {sampled_string!r}"
            )
        unknown = set(sampled) - dataset_names
        if unknown:
            raise ValueError(
                "Saved repetitions refer to datasets that are absent from the "
                f"score matrix: {sorted(unknown)[:10]}"
            )

    return matrix, repetitions, baseline


# ---------------------------------------------------------------------------
# Reconstruct ranks for each saved subset
# ---------------------------------------------------------------------------
def reconstruct_diagnostics(
    matrix: pd.DataFrame,
    repetitions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Recover all method ranks and validate against the saved winner labels."""
    # The original experiment breaks equal-score ties alphabetically by method.
    method_names = matrix.index.to_numpy(dtype=str)
    dataset_names = matrix.columns.to_numpy(dtype=str)
    raw_scores = matrix.to_numpy(dtype=float)
    method_order = np.argsort(method_names, kind="mergesort")
    methods = method_names[method_order]
    scores = raw_scores[method_order, :]

    dataset_to_index = {name: i for i, name in enumerate(dataset_names)}
    method_to_index = {name: i for i, name in enumerate(methods)}

    baseline_rank_map = dict(
        zip(baseline["method"], baseline["baseline_rank"].astype(int))
    )
    full_ranks = np.asarray(
        [baseline_rank_map[method] for method in methods], dtype=int
    )
    top_methods = (
        baseline.sort_values("baseline_rank", kind="mergesort")["method"]
        .head(min(TOP_N, len(baseline)))
        .tolist()
    )
    full_top5 = set(top_methods[: min(5, len(top_methods))])

    rows: list[dict[str, object]] = []
    mismatches: list[tuple[int, int, str, str]] = []
    scores_by_method = {name: i for i, name in enumerate(methods)}

    print("Reconstructing saved subset rankings...")
    total = len(repetitions)
    for processed, row in enumerate(
        repetitions[
            ["num_datasets", "repetition", "winner", "sampled_datasets"]
        ].itertuples(index=False, name=None),
        start=1,
    ):
        subset_size, repetition_id, saved_winner, sampled_string = row
        sampled_names = str(sampled_string).split("|")
        sampled_indices = [dataset_to_index[name] for name in sampled_names]

        # Match the original experiment's NumPy mean and stable, descending sort.
        subset_scores = scores[:, sampled_indices].mean(axis=1)
        score_order = np.argsort(-subset_scores, kind="mergesort")
        reconstructed_winner = str(methods[score_order[0]])

        if reconstructed_winner != str(saved_winner):
            if len(mismatches) < 10:
                mismatches.append(
                    (
                        int(subset_size),
                        int(repetition_id),
                        str(saved_winner),
                        reconstructed_winner,
                    )
                )

        # When all datasets are included, the benchmark composition is exactly
        # the full benchmark. Use the saved baseline order for the diagnostic
        # ranks so floating-point summation order cannot create artificial
        # differences between mathematically identical full-dataset averages.
        if int(subset_size) == len(dataset_names):
            ranks = full_ranks.copy()
            diagnostic_order = np.argsort(full_ranks, kind="mergesort")
        else:
            ranks = np.empty(len(methods), dtype=np.int16)
            ranks[score_order] = np.arange(1, len(methods) + 1, dtype=np.int16)
            diagnostic_order = score_order

        tau_result = kendalltau(full_ranks, ranks)
        tau_value = float(tau_result.statistic)
        subset_top5 = set(methods[diagnostic_order[: min(5, len(methods))]])
        top5_overlap = len(full_top5.intersection(subset_top5)) / max(
            1, len(full_top5)
        )

        result: dict[str, object] = {
            "num_datasets": int(subset_size),
            "repetition": int(repetition_id),
            "sampled_datasets": str(sampled_string),
            "kendall_tau": tau_value,
            "top5_overlap": float(top5_overlap),
            "reconstructed_winner": reconstructed_winner,
            "saved_winner": str(saved_winner),
        }
        for method in top_methods:
            result[f"rank__{method}"] = int(ranks[scores_by_method[method]])
        rows.append(result)

        if processed % 10000 == 0 or processed == total:
            print(f"  Processed {processed:,}/{total:,} repetitions")

    if mismatches:
        sample_text = "\n".join(
            f"  k={k}, repetition={rep}: saved={saved!r}, "
            f"reconstructed={recomputed!r}"
            for k, rep, saved, recomputed in mismatches
        )
        raise ValueError(
            "Reconstructed winners do not exactly match the saved winners. "
            "The score matrix and repetition file may come from different runs, "
            "so plotting has been stopped to avoid producing misleading figures.\n"
            f"First mismatches:\n{sample_text}"
        )

    print("Winner consistency check passed for every saved repetition.\n")
    diagnostics = pd.DataFrame(rows)
    return diagnostics, top_methods, baseline_rank_map


# ---------------------------------------------------------------------------
# Summaries and shared output helpers
# ---------------------------------------------------------------------------
def make_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_size, group in diagnostics.groupby("num_datasets", sort=True):
        tau = group["kendall_tau"].to_numpy(dtype=float)
        overlap = group["top5_overlap"].to_numpy(dtype=float)
        rows.append(
            {
                "num_datasets": int(subset_size),
                "n_repetitions": int(len(group)),
                "kendall_tau_median": float(np.median(tau)),
                "kendall_tau_p10": float(np.quantile(tau, 0.10)),
                "kendall_tau_p90": float(np.quantile(tau, 0.90)),
                "top5_overlap_median": float(np.median(overlap)),
                "top5_overlap_p10": float(np.quantile(overlap, 0.10)),
                "top5_overlap_p90": float(np.quantile(overlap, 0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("num_datasets").reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FINAL_DIR / f"{stem}.png"
    pdf_path = FINAL_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png_path}")
    print(f"  saved {pdf_path}")


# ---------------------------------------------------------------------------
# Figure 10: distributions of the top-10 methods' ranks
# ---------------------------------------------------------------------------
def plot_top10_rank_distributions(
    diagnostics: pd.DataFrame,
    top_methods: list[str],
    baseline_rank_map: dict[str, int],
    n_methods: int,
) -> None:
    available_sizes = set(diagnostics["num_datasets"].astype(int).unique())
    plot_sizes = [size for size in VIOLIN_SUBSET_SIZES if size in available_sizes]
    if len(plot_sizes) < 3:
        raise ValueError(
            "Fewer than three configured subset sizes are present in the saved "
            f"repetitions. Requested {VIOLIN_SUBSET_SIZES}; found "
            f"{sorted(available_sizes)}."
        )
    if len(plot_sizes) != len(VIOLIN_SUBSET_SIZES):
        warnings.warn(
            "Some configured violin subset sizes are absent; plotting the "
            f"available sizes only: {plot_sizes}"
        )

    fig = plt.figure(figsize=(15.5, 9.5))
    grid = fig.add_gridspec(
        2, 6, left=0.12, right=0.99, bottom=0.15, top=0.88,
        wspace=0.48, hspace=0.38,
    )
    placements = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]
    palette = plt.get_cmap("tab10")
    positions = np.arange(1, len(top_methods) + 1)

    for subset_size, (grid_row, grid_col) in zip(plot_sizes, placements):
        ax = fig.add_subplot(grid[grid_row, grid_col : grid_col + 2])
        subset = diagnostics.loc[
            diagnostics["num_datasets"] == subset_size
        ]
        values_by_method = [
            subset[f"rank__{method}"].to_numpy(dtype=float)
            for method in top_methods
        ]

        # Matplotlib's KDE-based violin fails for a constant distribution.
        # Plot non-degenerate distributions as violins and constant ranks as dots.
        variable_values = []
        variable_positions = []
        variable_method_indices = []
        for i, values in enumerate(values_by_method):
            if len(np.unique(values)) > 1:
                variable_values.append(values)
                variable_positions.append(positions[i])
                variable_method_indices.append(i)
            else:
                ax.scatter(
                    values[0], positions[i],
                    s=24, color=palette(i), edgecolor="black",
                    linewidth=0.45, zorder=4,
                )

        if variable_values:
            violin = ax.violinplot(
                variable_values,
                positions=variable_positions,
                orientation="horizontal",
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
                bw_method=0.25,
            )
            for body, method_index in zip(
                violin["bodies"], variable_method_indices
            ):
                body.set_facecolor(palette(method_index))
                body.set_edgecolor(palette(method_index))
                body.set_alpha(0.38)
            if "cmedians" in violin:
                violin["cmedians"].set_color("black")
                violin["cmedians"].set_linewidth(1.0)

        # White diamonds mark each method's rank in the complete benchmark.
        full_ranks = [baseline_rank_map[method] for method in top_methods]
        ax.scatter(
            full_ranks,
            positions,
            marker="D",
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        ax.set_title(
            f"{subset_size} datasets  (n = {len(subset):,} subsets)",
            fontsize=10.5,
            pad=7,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(top_methods, fontsize=7.6)
        ax.set_ylim(len(top_methods) + 0.5, 0.5)
        ax.set_xlim(0.5, n_methods + 0.5)
        ticks = [tick for tick in (1, 5, 10, 20, 30, 40, n_methods) if tick <= n_methods]
        ax.set_xticks(sorted(set(ticks)))
        ax.set_xlabel(f"Rank among {n_methods} methods (1 = best)", fontsize=8.5)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Rank distributions of the full-benchmark top 10 under random subsampling",
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5, 0.925,
        "Each violin summarizes method ranks across random dataset subsets; lower ranks are better.",
        ha="center", va="center", fontsize=9.5,
    )
    legend_handle = Line2D(
        [0], [0], marker="D", linestyle="None", color="black",
        markerfacecolor="white", markeredgecolor="black", markersize=5.5,
        label="Rank in the complete benchmark",
    )
    fig.legend(
        handles=[legend_handle], loc="lower center", bbox_to_anchor=(0.5, 0.075),
        frameon=False, fontsize=8.5,
    )
    fig.text(
        0.5, 0.035,
        "Violin width shows smoothed density over observed integer ranks; the diamond marks the full 51-dataset rank.",
        ha="center", va="center", fontsize=8,
    )
    save_figure(fig, "figure_10_random_subset_top10_rank_distributions")


# ---------------------------------------------------------------------------
# Figure 11: global rank agreement and top-five overlap by subset size
# ---------------------------------------------------------------------------
def plot_global_stability(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    x = summary["num_datasets"].to_numpy(dtype=int)

    # Panel A: whole-ranking agreement.
    ax = axes[0]
    median = summary["kendall_tau_median"].to_numpy(dtype=float)
    p10 = summary["kendall_tau_p10"].to_numpy(dtype=float)
    p90 = summary["kendall_tau_p90"].to_numpy(dtype=float)
    ax.fill_between(
        x, p10, p90, alpha=0.20, color="tab:blue",
        label="10th–90th percentile",
    )
    ax.plot(
        x, median, color="tab:blue", linewidth=2.0,
        label="Median Kendall's τ",
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.75)
    y_min = min(0.0, float(np.nanmin(p10)) - 0.04)
    ax.set_ylim(y_min, 1.03)
    ax.set_xlabel("Number of datasets in the subset")
    ax.set_ylabel("Kendall's τ versus the full ranking")
    ax.set_title("(a) Agreement across all methods")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.22, linewidth=0.7)

    # Panel B: retention of the full benchmark's top five.
    ax = axes[1]
    median = 100.0 * summary["top5_overlap_median"].to_numpy(dtype=float)
    p10 = 100.0 * summary["top5_overlap_p10"].to_numpy(dtype=float)
    p90 = 100.0 * summary["top5_overlap_p90"].to_numpy(dtype=float)
    ax.fill_between(
        x, p10, p90, alpha=0.20, color="tab:orange",
        label="10th–90th percentile",
    )
    ax.plot(
        x, median, color="tab:orange", linewidth=2.0,
        label="Median top-five overlap",
    )
    ax.axhline(100.0, color="black", linewidth=0.8, linestyle=":", alpha=0.75)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Number of datasets in the subset")
    ax.set_ylabel("Full-ranking top-five methods retained (%)")
    ax.set_title("(b) Agreement within the top five")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.22, linewidth=0.7)

    for ax in axes:
        ax.set_xlim(int(x.min()), int(x.max()))
        ticks = sorted(set([int(x.min()), 10, 20, 30, 40, int(x.max())]))
        ticks = [tick for tick in ticks if int(x.min()) <= tick <= int(x.max())]
        ax.set_xticks(ticks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "How subset size affects ranking stability",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.915,
        "Lines show medians across random subsets; shaded bands show the 10th–90th percentiles, not confidence intervals.",
        ha="center", va="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    save_figure(fig, "figure_11_random_subset_global_stability")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("RANDOM-SUBSET RANK DIAGNOSTICS")
    print("=" * 72)
    print(f"Results root: {RESULTS_DIR}")
    print()

    matrix, repetitions, baseline = load_inputs()
    diagnostics, top_methods, baseline_rank_map = reconstruct_diagnostics(
        matrix, repetitions, baseline
    )
    summary = make_summary(diagnostics)

    RANDOM_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics_path = (
        RANDOM_SUBSET_DIR / "random_subset_rank_diagnostics_repetitions.csv"
    )
    summary_path = (
        RANDOM_SUBSET_DIR / "random_subset_rank_diagnostics_summary.csv"
    )
    diagnostics.to_csv(diagnostics_path, index=False, float_format="%.8f")
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    print(f"Saved reconstructed diagnostics: {diagnostics_path}")
    print(f"Saved size-level summary:       {summary_path}")
    print()

    plot_top10_rank_distributions(
        diagnostics=diagnostics,
        top_methods=top_methods,
        baseline_rank_map=baseline_rank_map,
        n_methods=matrix.shape[0],
    )
    plot_global_stability(summary)

    print("\nSelected summary rows:")
    print(
        summary.loc[
            summary["num_datasets"].isin([5, 10, 20, 30, 40, matrix.shape[1]]),
            [
                "num_datasets", "n_repetitions", "kendall_tau_median",
                "kendall_tau_p10", "kendall_tau_p90", "top5_overlap_median",
            ],
        ].to_string(index=False)
    )
    print("\nAdditional random-subset rank plots complete.")


if __name__ == "__main__":
    main()
