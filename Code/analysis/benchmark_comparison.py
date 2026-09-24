from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PHASE1_DIR = RESULTS_DIR / "phase1_crossbenchmark"

TABARENA_LODO_FILE = RESULTS_DIR / "lodo_score_results.csv"
TABARENA_SCORE_FILE = RESULTS_DIR / "score_method_dataset.csv"
CROSSBENCHMARK_SUMMARY = PHASE1_DIR / "crossbenchmark_summary.csv"
OUTPUT_CSV = RESULTS_DIR / "benchmark_comparison.csv"
OUTPUT_MD = RESULTS_DIR / "benchmark_comparison.md"

# ============================================================
# HELPERS
# ============================================================

REQUIRED_LODO_COLUMNS = {
    "kendall_tau",
    "spearman_rho",
    "max_rank_change",
    "num_rank_changes",
    "winner_changed",
}


def load_lodo_file(path: Path, benchmark_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{benchmark_name}: result file not found:\n{path}"
        )

    df = pd.read_csv(path)
    missing = REQUIRED_LODO_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{benchmark_name}: missing required columns "
            f"{sorted(missing)} in:\n{path}"
        )
    return df


def summarize_lodo(
    benchmark: str,
    df: pd.DataFrame,
    tasks: int,
    competitors: int | None,
    removed_task_column: str = "removed_task",
) -> dict:
    worst_tau = df.loc[df["kendall_tau"].idxmin()]
    worst_spearman = df.loc[df["spearman_rho"].idxmin()]
    worst_mrc = df.loc[df["max_rank_change"].idxmax()]
    worst_num = df.loc[df["num_rank_changes"].idxmax()]

    winner_changes = df["winner_changed"].astype(str).str.lower().eq("true")

    return {
        "benchmark": benchmark,
        "tasks": tasks,
        "competitors": competitors,
        "worst_kendall_tau": float(worst_tau["kendall_tau"]),
        "worst_kendall_removed_task": str(worst_tau.get(removed_task_column, "")),
        "worst_spearman_rho": float(worst_spearman["spearman_rho"]),
        "worst_spearman_removed_task": str(worst_spearman.get(removed_task_column, "")),
        "max_rank_change": float(worst_mrc["max_rank_change"]),
        "max_rank_change_removed_task": str(worst_mrc.get(removed_task_column, "")),
        "max_num_rank_changes": int(worst_num["num_rank_changes"]),
        "max_num_rank_changes_removed_task": str(worst_num.get(removed_task_column, "")),
        "winner_changed_removals": int(winner_changes.sum()),
        "winner_changed": bool(winner_changes.any()),
        "lodo_cases": len(df),
        "mean_kendall_tau": float(df["kendall_tau"].mean()),
        "mean_lodo_instability": float((1.0 - df["kendall_tau"]).mean()),
        "tau_break_frequency": float((df["kendall_tau"] < 0.90).mean()),
    }


def summary_from_phase1_row(row: pd.Series) -> dict:
    return {
        "benchmark": f"{row['benchmark']} ({row['representation']})",
        "tasks": int(row["n_tasks"]),
        "competitors": int(row["n_models"]),
        "worst_kendall_tau": float(row["min_kendall_tau"]),
        "worst_kendall_removed_task": row.get("most_disruptive_task", ""),
        "worst_spearman_rho": float(row["min_spearman_rho"]) if "min_spearman_rho" in row and pd.notna(row["min_spearman_rho"]) else pd.NA,
        "worst_spearman_removed_task": row.get("worst_spearman_task", ""),
        "max_rank_change": float(row["largest_max_rank_change"]),
        "max_rank_change_removed_task": row.get("max_rank_change_task", ""),
        "max_num_rank_changes": int(row["largest_num_rank_changes"]),
        "max_num_rank_changes_removed_task": row.get("max_num_rank_changes_task", ""),
        "winner_changed_removals": int(row["winner_change_count"]),
        "winner_changed": bool(row["winner_change_count"] > 0),
        "lodo_cases": int(row["n_tasks"]),
        "mean_kendall_tau": float(row["mean_kendall_tau"]),
        "mean_lodo_instability": float(row["mean_lodo_instability"]),
        "tau_break_frequency": float(row["tau_break_frequency"]),
    }


def format_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    print("=" * 80)
    print("CROSS-BENCHMARK LODO COMPARISON")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")

    rows: list[dict] = []

    # TabArena uses the existing TabArena-specific pipeline.
    if TABARENA_LODO_FILE.exists():
        tabarena = load_lodo_file(TABARENA_LODO_FILE, "TabArena")
        if TABARENA_SCORE_FILE.exists():
            score_df = pd.read_csv(TABARENA_SCORE_FILE)
            competitors = int(score_df["method"].nunique()) if "method" in score_df.columns else None
        else:
            competitors = None

        rows.append(
            summarize_lodo(
                "TabArena (Score)",
                tabarena,
                tasks=tabarena["dataset_removed"].nunique() if "dataset_removed" in tabarena.columns else len(tabarena),
                competitors=competitors,
                removed_task_column="dataset_removed",
            )
        )
    else:
        print(f"\nWARNING: TabArena file not found: {TABARENA_LODO_FILE}")

    # Cross-benchmark analysis is the canonical source for GLUE/OpenLLM/ImageNet.
    if CROSSBENCHMARK_SUMMARY.exists():
        phase1 = pd.read_csv(CROSSBENCHMARK_SUMMARY)
        for _, row in phase1.iterrows():
            rows.append(summary_from_phase1_row(row))
    else:
        print(f"\nWARNING: Cross-benchmark summary not found: {CROSSBENCHMARK_SUMMARY}")
        print("Run current_simple_crossbenchmark.py first.")

    if not rows:
        raise RuntimeError("No benchmark results were found. Run the relevant analyses first.")

    comparison = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    md_columns = [
        "benchmark",
        "tasks",
        "competitors",
        "worst_kendall_tau",
        "mean_kendall_tau",
        "mean_lodo_instability",
        "max_rank_change",
        "max_num_rank_changes",
        "winner_changed_removals",
        "tau_break_frequency",
        "lodo_cases",
    ]

    md = [
        "# Cross-Benchmark LODO Comparison",
        "",
        "Canonical summary assembled from the current TabArena Score LODO and the Phase 1 BenchBench-loader analysis.",
        "",
        "| " + " | ".join(md_columns) + " |",
        "| " + " | ".join(["---"] * len(md_columns)) + " |",
    ]
    for _, row in comparison.iterrows():
        md.append("| " + " | ".join(format_value(row[c]) for c in md_columns) + " |")

    OUTPUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print("\nComparison:")
    print(comparison[md_columns].to_string(index=False))


if __name__ == "__main__":
    main()
