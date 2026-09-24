import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"

sys.path.insert(0, str(BENCHBENCH_ROOT))


# Import the ORIGINAL BenchBench loaders
from benchbench.data.glue import load_glue
from benchbench.data.openllm import load_openllm
from benchbench.data.imagenet import load_imagenet


# ============================================================
# SETTINGS
# ============================================================

TAU_BREAK_THRESHOLD = 0.90
TOP_K = 5

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
    / "phase1_crossbenchmark"
)


# ============================================================
# RANKING FUNCTIONS
# ============================================================

def compute_ranking(data, cols):
    """
    Compute the leaderboard ranking using the mean score across tasks.

    Higher score = better.
    """

    aggregate_score = data[cols].mean(axis=1)

    order = np.argsort(
        -aggregate_score.to_numpy(),
        kind="stable"
    )

    ranking = pd.Series(
        np.empty(len(data), dtype=int),
        index=data.index
    )

    ranking.iloc[order] = np.arange(1, len(data) + 1)

    return ranking, aggregate_score


def kendall_tau(rank_full, rank_lodo):
    tau, _ = kendalltau(rank_full, rank_lodo)
    return float(tau)


def kendall_distance(rank_full, rank_lodo):
    """BenchBench-style Kendall distance: (1 - tau) / 2."""

    tau = kendall_tau(rank_full, rank_lodo)

    if np.isnan(tau):
        return np.nan

    return float((1 - tau) / 2)


def spearman_correlation(rank_full, rank_lodo):
    rho, _ = spearmanr(rank_full, rank_lodo)
    return float(rho)


def normalized_mrc(rank_full, rank_lodo):
    """Maximum absolute rank change divided by N - 1."""

    n = len(rank_full)

    if n <= 1:
        return 0.0

    return float(
        np.max(np.abs(rank_lodo - rank_full))
        / (n - 1)
    )


def performance_gap(data, task):
    """Winner-runner-up score gap for one task."""

    values = (
        pd.to_numeric(data[task], errors="coerce")
        .dropna()
        .sort_values(ascending=False)
    )

    if len(values) < 2:
        return np.nan

    return float(values.iloc[0] - values.iloc[1])


def top_k_overlap(rank_full, rank_lodo, k=TOP_K):
    full_top = set(rank_full.sort_values().index[:k])
    lodo_top = set(rank_lodo.sort_values().index[:k])

    return len(full_top & lodo_top) / k


def safe_correlation(function, x, y):
    valid = pd.concat(
        [pd.Series(x), pd.Series(y)],
        axis=1
    ).dropna()

    if len(valid) < 3:
        return np.nan, np.nan

    if valid.iloc[:, 0].nunique() < 2:
        return np.nan, np.nan

    if valid.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan

    result = function(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy()
    )

    return float(result.statistic), float(result.pvalue)


# ============================================================
# TASK VALIDATION
# ============================================================

def validate_tasks(benchmark, returned_cols, expected_cols):
    """
    Check that the loader returned the expected tasks.

    The order returned by BenchBench is allowed to differ; the analysis
    then uses the explicit order in expected_cols.
    """

    returned = list(returned_cols)
    expected = list(expected_cols)

    if set(returned) != set(expected):
        raise ValueError(
            f"Unexpected {benchmark} task columns.\n"
            f"Expected set: {expected}\n"
            f"Received: {returned}"
        )

    # IMPORTANT: preserve BenchBench's original loader order.
    # The mean is mathematically order-invariant, but preserving the exact
    # input order also preserves floating-point accumulation and therefore
    # deterministic tie-breaking in the stable ranking.
    return returned


# ============================================================
# LODO ANALYSIS
# ============================================================

def run_lodo_analysis(
    benchmark,
    data,
    model_col,
    cols,
    output_dir,
    representation,
):
    """Run the same LODO analysis for one benchmark representation."""

    data = data.copy().reset_index(drop=True)

    # ========================================================
    # FULL RANKING
    # ========================================================

    full_ranking, full_score = compute_ranking(data, cols)

    data["full_score"] = full_score
    data["full_rank"] = full_ranking

    full_winner_idx = data["full_score"].idxmax()
    full_winner = data.loc[full_winner_idx, model_col]

    print()
    print("=" * 70)
    print(f"{benchmark.upper()} — {representation}")
    print("=" * 70)

    print()
    print("Number of models:", len(data))
    print("Number of tasks:", len(cols))
    print("Tasks:", cols)
    print("Winner:", full_winner)

    print()
    print("FULL LEADERBOARD")
    print()
    print(
        data[
            [model_col, "full_score", "full_rank"]
        ]
        .sort_values("full_rank")
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # PERFORMANCE GAPS
    # ========================================================

    gaps = {
        task: performance_gap(data, task)
        for task in cols
    }

    # ========================================================
    # LODO
    # ========================================================

    results = []

    for removed_task in cols:

        remaining_tasks = [
            task for task in cols
            if task != removed_task
        ]

        lodo_ranking, lodo_score = compute_ranking(
            data,
            remaining_tasks
        )

        full_ranks = data["full_rank"].to_numpy()
        lodo_ranks = lodo_ranking.to_numpy()

        tau = kendall_tau(
            full_ranks,
            lodo_ranks
        )

        kd = kendall_distance(
            full_ranks,
            lodo_ranks
        )

        rho = spearman_correlation(
            full_ranks,
            lodo_ranks
        )

        instability = 1 - tau

        mrc = normalized_mrc(
            full_ranks,
            lodo_ranks
        )

        absolute_rank_change = np.abs(
            lodo_ranks - full_ranks
        )

        num_rank_changes = int(
            np.sum(absolute_rank_change > 0)
        )

        max_rank_change = int(
            np.max(absolute_rank_change)
        )

        mean_rank_change = float(
            np.mean(absolute_rank_change)
        )

        lodo_winner_idx = lodo_score.idxmax()
        lodo_winner = data.loc[
            lodo_winner_idx,
            model_col
        ]

        winner_changed = (
            lodo_winner != full_winner
        )

        original_winner_lodo_rank = int(
            lodo_ranking.loc[full_winner_idx]
        )

        results.append({
            "benchmark": benchmark,
            "representation": representation,
            "removed_task": removed_task,
            "num_remaining_tasks": len(remaining_tasks),
            "kendall_tau": tau,
            "kendall_distance": kd,
            "lodo_instability": instability,
            "spearman_rho": rho,
            "mrc": mrc,
            "max_rank_change": max_rank_change,
            "mean_rank_change": mean_rank_change,
            "num_rank_changes": num_rank_changes,
            "top5_overlap": top_k_overlap(
                data["full_rank"],
                lodo_ranking
            ),
            "full_winner": full_winner,
            "lodo_winner": lodo_winner,
            "winner_changed": winner_changed,
            "tau_break": tau < TAU_BREAK_THRESHOLD,
            "original_winner_lodo_rank": original_winner_lodo_rank,
            "task_performance_gap": gaps[removed_task],
        })

    results_df = pd.DataFrame(results)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    display_columns = [
        "removed_task",
        "kendall_tau",
        "kendall_distance",
        "lodo_instability",
        "spearman_rho",
        "mrc",
        "max_rank_change",
        "num_rank_changes",
        "winner_changed",
        "tau_break",
    ]

    print()
    print("LODO RESULTS")
    print()

    # Small benchmarks: print every task.
    # Large benchmarks such as ImageNet: print only the 20 most disruptive.
    if len(results_df) <= 20:
        print(
            results_df[display_columns]
            .to_string(index=False)
        )
    else:
        print(
            f"{len(results_df)} task removals evaluated; "
            "showing the 20 most disruptive below. "
            "All results are saved to lodo_results.csv."
        )

    print()
    print("MOST DISRUPTIVE TASKS — TOP 20 BY KENDALL TAU")
    print()

    most_disruptive = (
        results_df
        .sort_values(
            ["kendall_tau", "max_rank_change"],
            ascending=[True, False]
        )
        .head(20)
    )

    print(
        most_disruptive[
            [
                "removed_task",
                "kendall_tau",
                "lodo_instability",
                "mrc",
                "max_rank_change",
                "num_rank_changes",
                "winner_changed",
                "tau_break",
            ]
        ].to_string(index=False)
    )

    print()
    print("MOST DISRUPTIVE TASKS — TOP 20 BY MRC")
    print()

    print(
        results_df
        .sort_values(
            ["mrc", "max_rank_change"],
            ascending=[False, False]
        )
        .head(20)[
            [
                "removed_task",
                "mrc",
                "max_rank_change",
                "num_rank_changes",
                "kendall_tau",
                "lodo_instability",
                "winner_changed",
            ]
        ].to_string(index=False)
    )

    print()
    print("WINNER CHANGES")
    print()

    winner_changes = results_df[
        results_df["winner_changed"]
    ]

    if len(winner_changes) == 0:
        print("The winner never changes under LODO.")
    else:
        print(
            winner_changes[
                [
                    "removed_task",
                    "full_winner",
                    "lodo_winner",
                    "kendall_tau",
                    "lodo_instability",
                ]
            ].to_string(index=False)
        )

    # ========================================================
    # PERFORMANCE GAP CORRELATIONS
    # ========================================================

    x = results_df["task_performance_gap"]
    y = results_df["lodo_instability"]

    pearson_r, pearson_p = safe_correlation(
        pearsonr,
        x,
        y
    )

    spearman_rho, spearman_p = safe_correlation(
        spearmanr,
        x,
        y
    )

    kendall_gap_tau, kendall_gap_p = safe_correlation(
        kendalltau,
        x,
        y
    )

    print()
    print("PERFORMANCE GAP VS LODO INSTABILITY")
    print()
    print(
        f"Pearson:  r={pearson_r:.6f}, p={pearson_p:.6g}"
    )
    print(
        f"Spearman: rho={spearman_rho:.6f}, p={spearman_p:.6g}"
    )
    print(
        f"Kendall:  tau={kendall_gap_tau:.6f}, p={kendall_gap_p:.6g}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "benchmark": benchmark,
        "representation": representation,
        "n_models": len(data),
        "n_tasks": len(cols),
        "full_winner": full_winner,
        "mean_kendall_tau": results_df["kendall_tau"].mean(),
        "min_kendall_tau": results_df["kendall_tau"].min(),
        "mean_kendall_distance": results_df["kendall_distance"].mean(),
        "mean_lodo_instability": results_df["lodo_instability"].mean(),
        "max_lodo_instability": results_df["lodo_instability"].max(),
        "winner_change_frequency": results_df["winner_changed"].mean(),
        "winner_change_count": int(results_df["winner_changed"].sum()),
        "tau_break_frequency": results_df["tau_break"].mean(),
        "tau_break_count": int(results_df["tau_break"].sum()),
        "tau_break_threshold": TAU_BREAK_THRESHOLD,
        "mean_max_rank_change": results_df["max_rank_change"].mean(),
        "largest_max_rank_change": results_df["max_rank_change"].max(),
        "mean_num_rank_changes": results_df["num_rank_changes"].mean(),
        "largest_num_rank_changes": results_df["num_rank_changes"].max(),
        "mean_top5_overlap": results_df["top5_overlap"].mean(),
        "most_disruptive_task": results_df.loc[
            results_df["kendall_tau"].idxmin(),
            "removed_task"
        ],
        "gap_pearson_r": pearson_r,
        "gap_pearson_p": pearson_p,
        "gap_spearman_rho": spearman_rho,
        "gap_spearman_p": spearman_p,
        "gap_kendall_tau": kendall_gap_tau,
        "gap_kendall_p": kendall_gap_p,
    }

    print()
    print("SUMMARY")
    print()
    print(
        f"Mean Kendall tau:    {summary['mean_kendall_tau']:.6f}"
    )
    print(
        f"Minimum Kendall tau: {summary['min_kendall_tau']:.6f}"
    )
    print(
        f"Mean instability:    {summary['mean_lodo_instability']:.6f}"
    )
    print(
        f"Winner changes:      "
        f"{summary['winner_change_count']}/{len(results_df)} "
        f"({summary['winner_change_frequency']:.3f})"
    )
    print(
        f"Tau breaks (<{TAU_BREAK_THRESHOLD}): "
        f"{summary['tau_break_count']}/{len(results_df)} "
        f"({summary['tau_break_frequency']:.3f})"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    full_ranking = (
        data[
            [model_col, "full_score", "full_rank"]
        ]
        .sort_values("full_rank")
    )

    full_ranking.to_csv(
        output_dir / "full_ranking.csv",
        index=False
    )

    results_df.to_csv(
        output_dir / "lodo_results.csv",
        index=False
    )

    # Save the two most useful compact task-level views separately.
    most_disruptive.to_csv(
        output_dir / "most_disruptive_tasks_top20.csv",
        index=False
    )

    results_df[results_df["winner_changed"]].to_csv(
        output_dir / "winner_changes.csv",
        index=False
    )

    pd.DataFrame([
        {
            "benchmark": benchmark,
            "representation": representation,
            "n": len(results_df.dropna(
                subset=[
                    "task_performance_gap",
                    "lodo_instability",
                ]
            )),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "kendall_tau": kendall_gap_tau,
            "kendall_p": kendall_gap_p,
        }
    ]).to_csv(
        output_dir / "performance_gap_correlations.csv",
        index=False
    )

    results_df[
        [
            "removed_task",
            "task_performance_gap",
            "kendall_tau",
            "kendall_distance",
            "lodo_instability",
            "spearman_rho",
            "mrc",
            "max_rank_change",
            "num_rank_changes",
            "top5_overlap",
            "winner_changed",
            "tau_break",
        ]
    ].to_csv(
        output_dir / "task_summary.csv",
        index=False
    )

    print()
    print("Saved to:")
    print(output_dir)

    return summary


# ============================================================
# GLUE
# ============================================================

def run_glue():

    GLUE_TASKS = [
        "CoLA",
        "SST-2",
        "MRPC",
        "STS-B",
        "QQP",
        "MNLI",
        "QNLI",
        "RTE",
        "WNLI",
    ]

    print()
    print("=" * 70)
    print("LOAD GLUE")
    print("=" * 70)

    data, returned_cols = load_glue()

    cols = validate_tasks(
        "GLUE",
        returned_cols,
        GLUE_TASKS
    )

    data = data[["Model", *cols]].copy()

    return run_lodo_analysis(
        benchmark="GLUE",
        data=data,
        model_col="Model",
        cols=cols,
        output_dir=OUTPUT_ROOT / "glue",
        representation="score",
    )


# ============================================================
# OPENLLM
# ============================================================

def run_openllm():

    OPENLLM_TASKS = [
        "ARC",
        "HellaSwag",
        "MMLU",
        "TruthfulQA",
        "Winogrande",
        "GSM8K",
    ]

    print()
    print("=" * 70)
    print("LOAD OPENLLM")
    print("=" * 70)

    data, returned_cols = load_openllm()

    cols = validate_tasks(
        "OpenLLM",
        returned_cols,
        OPENLLM_TASKS
    )

    data = data[["Model", *cols]].copy()

    print()
    print("OpenLLM input rows:", len(data))
    print("OpenLLM unique models:", data["Model"].nunique())

    # --------------------------------------------------------
    # Row-level representation
    # --------------------------------------------------------

    row_summary = run_lodo_analysis(
        benchmark="OpenLLM",
        data=data,
        model_col="Model",
        cols=cols,
        output_dir=OUTPUT_ROOT / "openllm" / "row_level",
        representation="row_level",
    )

    # --------------------------------------------------------
    # Unique-model representation
    # --------------------------------------------------------

    unique_data = (
        data
        .groupby("Model", sort=False)[cols]
        .mean()
        .reset_index()
    )

    unique_summary = run_lodo_analysis(
        benchmark="OpenLLM",
        data=unique_data,
        model_col="Model",
        cols=cols,
        output_dir=OUTPUT_ROOT / "openllm" / "unique_model_mean",
        representation="unique_model_mean",
    )

    return [
        row_summary,
        unique_summary,
    ]


# ============================================================
# IMAGENET
# ============================================================

def run_imagenet():

    IMAGENET_TASKS = [
        f"acc_{i}" for i in range(1000)
    ]

    print()
    print("=" * 70)
    print("LOAD IMAGENET")
    print("=" * 70)

    data, returned_cols = load_imagenet(
        num_task=1000
    )

    cols = validate_tasks(
        "ImageNet",
        returned_cols,
        IMAGENET_TASKS
    )

    data = data[["Weight", *cols]].copy()

    return run_lodo_analysis(
        benchmark="ImageNet",
        data=data,
        model_col="Weight",
        cols=cols,
        output_dir=OUTPUT_ROOT / "imagenet",
        representation="score",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 1 CROSS-BENCHMARK LODO ANALYSIS")
    print("=" * 70)
    print()
    print("BenchBench root:")
    print(BENCHBENCH_ROOT)
    print()
    print("Output root:")
    print(OUTPUT_ROOT)
    print()
    print("Tau-break threshold:", TAU_BREAK_THRESHOLD)

    summaries = []

    # GLUE
    summaries.append(
        run_glue()
    )

    # OpenLLM
    summaries.extend(
        run_openllm()
    )

    # ImageNet
    summaries.append(
        run_imagenet()
    )

    # --------------------------------------------------------
    # CROSS-BENCHMARK SUMMARY
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df = pd.DataFrame(summaries)

    summary_file = (
        OUTPUT_ROOT
        / "crossbenchmark_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print()
    print("=" * 70)
    print("CROSS-BENCHMARK SUMMARY")
    print("=" * 70)
    print()

    print(
        summary_df[
            [
                "benchmark",
                "representation",
                "n_models",
                "n_tasks",
                "full_winner",
                "min_kendall_tau",
                "mean_lodo_instability",
                "winner_change_frequency",
                "tau_break_frequency",
                "largest_max_rank_change",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(summary_file)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
