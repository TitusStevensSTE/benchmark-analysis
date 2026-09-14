from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================
# Repository-relative: this script should live in Code/analysis/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPENLLM_FILE = (
    PROJECT_ROOT
    / "benchbench"
    / "benchbench"
    / "data"
    / "openllm"
    / "leaderboard.tsv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots" / "openllm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LODO_OUTPUT = RESULTS_DIR / "openllm_lodo_results.csv"
FULL_OUTPUT = RESULTS_DIR / "openllm_full_ranking.csv"
RANK_MATRIX_OUTPUT = RESULTS_DIR / "openllm_lodo_rank_matrix.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "openllm_lodo_summary.txt"

UNIQUE_LODO_OUTPUT = RESULTS_DIR / "openllm_unique_model_lodo_results.csv"
UNIQUE_FULL_OUTPUT = RESULTS_DIR / "openllm_unique_model_full_ranking.csv"
UNIQUE_RANK_MATRIX_OUTPUT = RESULTS_DIR / "openllm_unique_model_lodo_rank_matrix.csv"
UNIQUE_SUMMARY_OUTPUT = RESULTS_DIR / "openllm_unique_model_lodo_summary.txt"


# ============================================================
# SETTINGS
# ============================================================
MODEL_COL = "Model"
AVERAGE_COL = "Average"

# The TSV is expected to contain:
# T | Model | Average | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K


# ============================================================
# HELPERS
# ============================================================
def rank_series(scores: pd.Series) -> pd.Series:
    """Rank high scores as rank 1; average ties."""
    return scores.rank(method="average", ascending=False)


def ranking_metrics(full_rank: pd.Series, lodo_rank: pd.Series) -> dict:
    """Calculate the ranking-comparison statistics used in the thesis."""
    tau, tau_p = kendalltau(full_rank, lodo_rank)
    rho, rho_p = spearmanr(full_rank, lodo_rank)

    rank_difference = (lodo_rank - full_rank).abs()

    return {
        "kendall_tau": tau,
        "kendall_p_value": tau_p,
        "spearman_rho": rho,
        "spearman_p_value": rho_p,
        "max_rank_change": rank_difference.max(),
        "num_rank_changes": int((rank_difference > 0).sum()),
        "mean_rank_change": rank_difference.mean(),
        "median_rank_change": rank_difference.median(),
    }


def save_bar_plot(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: Path,
    rotation: int = 35,
):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LOAD
# ============================================================
print("=" * 70)
print("OPENLLM LODO ANALYSIS")
print("=" * 70)
print(f"\nInput file:\n{OPENLLM_FILE}")

if not OPENLLM_FILE.exists():
    raise FileNotFoundError(
        f"OpenLLM TSV not found:\n{OPENLLM_FILE}\n\n"
        "Check that this script is located in Code/analysis/ and that "
        "the BenchBench repository is available under the project root."
    )

df = pd.read_csv(OPENLLM_FILE, sep="\t")

print("\nShape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# IDENTIFY TASKS
# ============================================================
expected_prefix = ["T", MODEL_COL, AVERAGE_COL]
missing_prefix = [col for col in expected_prefix if col not in df.columns]
if missing_prefix:
    raise ValueError(f"Missing expected columns: {missing_prefix}")

TASK_COLS = [col for col in df.columns if col not in expected_prefix]

if len(TASK_COLS) < 2:
    raise ValueError(f"Expected multiple task columns, found: {TASK_COLS}")

print("\nTask columns:")
for task in TASK_COLS:
    print(f"  - {task}")

print("\nNumber of tasks:", len(TASK_COLS))
print("Number of rows:", len(df))
print("Number of unique model names:", df[MODEL_COL].nunique())

if df[MODEL_COL].duplicated().any():
    duplicate_count = int(df[MODEL_COL].duplicated(keep=False).sum())
    print(
        f"WARNING: {duplicate_count} rows belong to duplicated model names. "
        "Rows are retained exactly as supplied by BenchBench."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================
for col in TASK_COLS + [AVERAGE_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[TASK_COLS].isna().any().any():
    bad_cells = int(df[TASK_COLS].isna().sum().sum())
    raise ValueError(f"Found {bad_cells} missing/non-numeric task values.")


# ============================================================
# CHECK STORED AVERAGE
# ============================================================
df["computed_average"] = df[TASK_COLS].mean(axis=1)

average_difference = (df[AVERAGE_COL] - df["computed_average"]).abs()
max_average_difference = average_difference.max()
mean_average_difference = average_difference.mean()

print("\nAverage consistency check")
print("-------------------------")
print("Maximum |stored Average - computed mean|:", max_average_difference)
print("Mean |stored Average - computed mean|:", mean_average_difference)

if max_average_difference > 1e-6:
    print(
        "NOTE: Stored Average differs slightly from the exact arithmetic mean. "
        "LODO uses the raw six task columns and recomputes the mean consistently."
    )
else:
    print("OK: Stored Average matches the task mean.")


# ============================================================
# FULL BENCHMARK RANKING
# ============================================================
df["full_score"] = df[TASK_COLS].mean(axis=1)
df["full_rank"] = rank_series(df["full_score"])

# Stable display order: score descending, then original row order.
df["_row_order"] = np.arange(len(df))
full_ranking = (
    df[[MODEL_COL, "full_score", "full_rank", "_row_order"]]
    .sort_values(["full_score", "_row_order"], ascending=[False, True])
    .drop(columns="_row_order")
    .reset_index(drop=True)
)
full_ranking["rank_position"] = np.arange(1, len(full_ranking) + 1)
full_ranking = full_ranking[["rank_position", MODEL_COL, "full_score", "full_rank"]]
full_ranking.to_csv(FULL_OUTPUT, index=False)

full_winner_idx = df["full_score"].idxmax()
full_winner = df.loc[full_winner_idx, MODEL_COL]
full_winner_score = df.loc[full_winner_idx, "full_score"]

print("\nFull benchmark")
print("--------------")
print("Winner:", full_winner)
print("Winner score:", full_winner_score)


# ============================================================
# GENERIC LODO ANALYSIS
# ============================================================
def run_lodo_analysis(
    data: pd.DataFrame,
    task_cols: list[str],
    label: str,
    lodo_output: Path,
    full_output: Path,
    rank_matrix_output: Path,
):
    """Run the same LODO procedure on a given set of model rows."""
    work = data[[MODEL_COL, *task_cols]].copy()
    work["full_score"] = work[task_cols].mean(axis=1)
    work["full_rank"] = rank_series(work["full_score"])

    work["_row_order"] = np.arange(len(work))
    full_ranking = (
        work[[MODEL_COL, "full_score", "full_rank", "_row_order"]]
        .sort_values(["full_score", "_row_order"], ascending=[False, True])
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    full_ranking["rank_position"] = np.arange(1, len(full_ranking) + 1)
    full_ranking = full_ranking[
        ["rank_position", MODEL_COL, "full_score", "full_rank"]
    ]
    full_ranking.to_csv(full_output, index=False)

    full_winner_idx = work["full_score"].idxmax()
    full_winner = work.loc[full_winner_idx, MODEL_COL]
    full_winner_score = work.loc[full_winner_idx, "full_score"]

    lodo_results = []
    rank_columns = {"full_rank": work["full_rank"]}

    for removed_task in task_cols:
        remaining_tasks = [task for task in task_cols if task != removed_task]

        lodo_score = work[remaining_tasks].mean(axis=1)
        lodo_rank = rank_series(lodo_score)
        rank_columns[f"without_{removed_task}"] = lodo_rank

        metrics = ranking_metrics(work["full_rank"], lodo_rank)

        lodo_winner_idx = lodo_score.idxmax()
        lodo_winner = work.loc[lodo_winner_idx, MODEL_COL]

        full_winner_lodo_rank = lodo_rank.loc[full_winner_idx]

        full_top_5 = set(work.nsmallest(5, "full_rank").index)
        lodo_top_5 = set(lodo_rank.nsmallest(5).index)
        top_5_overlap = len(full_top_5 & lodo_top_5)

        lodo_results.append(
            {
                "removed_task": removed_task,
                "num_remaining_tasks": len(remaining_tasks),
                **metrics,
                "full_winner": full_winner,
                "lodo_winner": lodo_winner,
                "winner_changed": lodo_winner != full_winner,
                "full_winner_lodo_rank": full_winner_lodo_rank,
                "top_5_overlap": top_5_overlap,
            }
        )

    results = pd.DataFrame(lodo_results)
    results["task_order"] = pd.Categorical(
        results["removed_task"], categories=task_cols, ordered=True
    )
    results = (
        results.sort_values("task_order")
        .drop(columns="task_order")
        .reset_index(drop=True)
    )
    results.to_csv(lodo_output, index=False)

    rank_matrix = pd.DataFrame(rank_columns)
    rank_matrix.insert(0, MODEL_COL, work[MODEL_COL].values)
    rank_matrix.to_csv(rank_matrix_output, index=False)

    return results, full_ranking, full_winner, full_winner_score


# ============================================================
# ANALYSIS A: EXACT BENCHBENCH REPRODUCTION
# ============================================================
row_results, row_full_ranking, row_full_winner, row_full_winner_score = (
    run_lodo_analysis(
        df,
        TASK_COLS,
        "BenchBench rows",
        LODO_OUTPUT,
        FULL_OUTPUT,
        RANK_MATRIX_OUTPUT,
    )
)


# ============================================================
# ANALYSIS B: UNIQUE-MODEL ROBUSTNESS CHECK
# ============================================================
# Duplicate model names occur in the BenchBench TSV.
# For this robustness analysis, all rows belonging to the same model name
# are collapsed by taking the arithmetic mean of their task scores.
# This preserves every model name exactly once and avoids one model receiving
# multiple weights in the ranking.
unique_df = (
    df.groupby(MODEL_COL, as_index=False, sort=False)[TASK_COLS]
    .mean()
)

print("\nUnique-model robustness check")
print("----------------------------")
print("Rows after collapsing duplicate model names:", len(unique_df))
print("Rows removed by deduplication:", len(df) - len(unique_df))

unique_results, unique_full_ranking, unique_full_winner, unique_full_winner_score = (
    run_lodo_analysis(
        unique_df,
        TASK_COLS,
        "Unique models",
        UNIQUE_LODO_OUTPUT,
        UNIQUE_FULL_OUTPUT,
        UNIQUE_RANK_MATRIX_OUTPUT,
    )
)


def print_results(title: str, results: pd.DataFrame):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(
        results[
            [
                "removed_task",
                "kendall_tau",
                "spearman_rho",
                "max_rank_change",
                "num_rank_changes",
                "mean_rank_change",
                "median_rank_change",
                "top_5_overlap",
                "full_winner",
                "lodo_winner",
                "winner_changed",
            ]
        ].to_string(index=False)
    )


def key_findings(results: pd.DataFrame, n_items: int):
    most_sensitive_kendall = results.loc[results["kendall_tau"].idxmin()]
    most_sensitive_spearman = results.loc[results["spearman_rho"].idxmin()]
    largest_max_change = results.loc[results["max_rank_change"].idxmax()]
    most_changed_models = results.loc[results["num_rank_changes"].idxmax()]
    winner_change_tasks = results.loc[results["winner_changed"], "removed_task"].tolist()

    print("\n" + "-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    print(
        "Lowest Kendall tau:",
        most_sensitive_kendall["removed_task"],
        f"({most_sensitive_kendall['kendall_tau']:.6f})",
    )
    print(
        "Lowest Spearman rho:",
        most_sensitive_spearman["removed_task"],
        f"({most_sensitive_spearman['spearman_rho']:.6f})",
    )
    print(
        "Largest single rank movement:",
        largest_max_change["removed_task"],
        f"({largest_max_change['max_rank_change']:.0f} positions)",
    )
    print(
        "Most models changing rank:",
        most_changed_models["removed_task"],
        f"({most_changed_models['num_rank_changes']} / {n_items})",
    )
    print(
        "Winner-changing removals:",
        ", ".join(winner_change_tasks) if winner_change_tasks else "None",
    )


print_results("LODO RESULTS — EXACT BENCHBENCH ROWS", row_results)
key_findings(row_results, len(df))

print_results("LODO RESULTS — UNIQUE-MODEL ROBUSTNESS", unique_results)
key_findings(unique_results, len(unique_df))


# ============================================================
# COMPARISON TABLE: ROW-LEVEL VS UNIQUE-MODEL
# ============================================================
comparison_rows = []
for analysis_name, results, n_items in [
    ("BenchBench rows", row_results, len(df)),
    ("Unique models", unique_results, len(unique_df)),
]:
    for _, r in results.iterrows():
        comparison_rows.append(
            {
                "analysis": analysis_name,
                "n_models_or_rows": n_items,
                "removed_task": r["removed_task"],
                "kendall_tau": r["kendall_tau"],
                "spearman_rho": r["spearman_rho"],
                "max_rank_change": r["max_rank_change"],
                "num_rank_changes": r["num_rank_changes"],
                "mean_rank_change": r["mean_rank_change"],
                "median_rank_change": r["median_rank_change"],
                "top_5_overlap": r["top_5_overlap"],
                "winner_changed": r["winner_changed"],
                "full_winner": r["full_winner"],
                "lodo_winner": r["lodo_winner"],
            }
        )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(
    RESULTS_DIR / "openllm_row_vs_unique_lodo_comparison.csv", index=False
)


# ============================================================
# SUMMARY HELPERS
# ============================================================
def summary_lines_for(
    title: str,
    results: pd.DataFrame,
    n_items: int,
    winner: str,
    winner_score: float,
):
    most_k = results.loc[results["kendall_tau"].idxmin()]
    largest = results.loc[results["max_rank_change"].idxmax()]
    winners = results.loc[results["winner_changed"], "removed_task"].tolist()

    return [
        title,
        "=" * len(title),
        f"Rows/models: {n_items}",
        f"Tasks: {', '.join(TASK_COLS)}",
        f"Full benchmark winner: {winner}",
        f"Full benchmark score: {winner_score:.6f}",
        "",
        "Most influential task by lowest Kendall tau:",
        f"  {most_k['removed_task']}",
        f"  Kendall tau = {most_k['kendall_tau']:.6f}",
        f"  Spearman rho = {most_k['spearman_rho']:.6f}",
        f"  Mean rank change = {most_k['mean_rank_change']:.2f}",
        f"  Max rank change = {most_k['max_rank_change']:.0f}",
        f"  Num rank changes = {most_k['num_rank_changes']}",
        f"  LODO winner = {most_k['lodo_winner']}",
        "",
        "Largest maximum rank change:",
        f"  {largest['removed_task']}",
        f"  Max rank change = {largest['max_rank_change']:.0f}",
        f"  LODO winner = {largest['lodo_winner']}",
        "",
        "Winner-changing removals:",
        f"  {', '.join(winners) if winners else 'None'}",
    ]


SUMMARY_OUTPUT.write_text(
    "\n".join(
        summary_lines_for(
            "OpenLLM LODO summary — exact BenchBench rows",
            row_results,
            len(df),
            row_full_winner,
            row_full_winner_score,
        )
    ),
    encoding="utf-8",
)

UNIQUE_SUMMARY_OUTPUT.write_text(
    "\n".join(
        summary_lines_for(
            "OpenLLM LODO summary — unique models",
            unique_results,
            len(unique_df),
            unique_full_winner,
            unique_full_winner_score,
        )
    ),
    encoding="utf-8",
)

# ============================================================
# PLOTS
# ============================================================
def save_lodo_plots(results: pd.DataFrame, prefix: str):
    kendall_plot = results.sort_values("kendall_tau")
    save_bar_plot(
        kendall_plot["removed_task"],
        kendall_plot["kendall_tau"],
        xlabel="Removed task",
        ylabel="Kendall tau",
        title=f"OpenLLM LODO ({prefix}): Kendall tau vs. full ranking",
        filename=PLOTS_DIR / f"openllm_{prefix.lower().replace(' ', '_')}_lodo_kendall_tau.png",
    )

    spearman_plot = results.sort_values("spearman_rho")
    save_bar_plot(
        spearman_plot["removed_task"],
        spearman_plot["spearman_rho"],
        xlabel="Removed task",
        ylabel="Spearman rho",
        title=f"OpenLLM LODO ({prefix}): Spearman rho vs. full ranking",
        filename=PLOTS_DIR / f"openllm_{prefix.lower().replace(' ', '_')}_lodo_spearman_rho.png",
    )

    rank_change_plot = results.sort_values("mean_rank_change", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(rank_change_plot))
    width = 0.38
    ax.bar(
        x - width / 2,
        rank_change_plot["mean_rank_change"],
        width,
        label="Mean absolute rank change",
    )
    ax.bar(
        x + width / 2,
        rank_change_plot["max_rank_change"],
        width,
        label="Maximum rank change",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rank_change_plot["removed_task"], rotation=35, ha="right")
    ax.set_xlabel("Removed task")
    ax.set_ylabel("Rank change")
    ax.set_title(f"OpenLLM LODO ({prefix}): rank movement")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / f"openllm_{prefix.lower().replace(' ', '_')}_lodo_rank_change.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    changed_plot = results.sort_values("num_rank_changes", ascending=False)
    save_bar_plot(
        changed_plot["removed_task"],
        changed_plot["num_rank_changes"],
        xlabel="Removed task",
        ylabel="Number of models with changed rank",
        title=f"OpenLLM LODO ({prefix}): changed ranks",
        filename=PLOTS_DIR / f"openllm_{prefix.lower().replace(' ', '_')}_lodo_num_rank_changes.png",
    )


save_lodo_plots(row_results, "BenchBench rows")
save_lodo_plots(unique_results, "Unique models")


# ============================================================
# FINISHED
# ============================================================
print("\nSaved results to:")
for path in [
    LODO_OUTPUT,
    FULL_OUTPUT,
    RANK_MATRIX_OUTPUT,
    SUMMARY_OUTPUT,
    UNIQUE_LODO_OUTPUT,
    UNIQUE_FULL_OUTPUT,
    UNIQUE_RANK_MATRIX_OUTPUT,
    UNIQUE_SUMMARY_OUTPUT,
    RESULTS_DIR / "openllm_row_vs_unique_lodo_comparison.csv",
]:
    print("  ", path)

print("\nSaved plots to:")
print("  ", PLOTS_DIR)
