from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PHASE1_DIR = RESULTS_DIR / "phase1_crossbenchmark"
FIGURE_DIR = PHASE1_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


BENCHMARK_FILES = {
    "glue": PHASE1_DIR / "glue" / "lodo_results.csv",
    "openllm_row_level": PHASE1_DIR / "openllm" / "row_level" / "lodo_results.csv",
    "openllm_unique_model_mean": PHASE1_DIR / "openllm" / "unique_model_mean" / "lodo_results.csv",
    "imagenet": PHASE1_DIR / "imagenet" / "lodo_results.csv",
}


def plot_lodo(results: pd.DataFrame, benchmark_name: str, suffix: str = "") -> None:
    ordered = results.sort_values(["kendall_tau", "max_rank_change"], ascending=[True, False]).reset_index(drop=True)
    x = range(len(ordered))
    safe = benchmark_name.lower().replace(" ", "_") + suffix

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(list(x), ordered["kendall_tau"], linewidth=1)
    ax.axhline(0.90, linestyle="--", linewidth=1, label="Kendall tau = 0.90")
    ax.set_xlabel("Task removals (sorted by Kendall tau)")
    ax.set_ylabel("Kendall tau")
    ax.set_title(f"{benchmark_name}: LODO ranking stability")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{safe}_kendall_tau.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ordered = results.sort_values("max_rank_change", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(ordered)), ordered["mean_rank_change"], label="Mean absolute rank change")
    ax.plot(range(len(ordered)), ordered["max_rank_change"], label="Maximum rank change")
    ax.set_xlabel("Task removals (sorted by rank movement)")
    ax.set_ylabel("Rank change")
    ax.set_title(f"{benchmark_name}: LODO rank movement")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{safe}_rank_movement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ordered = results.sort_values("num_rank_changes", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(ordered)), ordered["num_rank_changes"], linewidth=1)
    ax.set_xlabel("Task removals (sorted by changed ranks)")
    ax.set_ylabel("Competitors with changed rank")
    ax.set_title(f"{benchmark_name}: LODO rank changes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{safe}_num_rank_changes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for name, path in BENCHMARK_FILES.items():
        if not path.exists():
            print(f"Skipping {name}: {path}")
            continue
        results = pd.read_csv(path)
        plot_lodo(results, name)
    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
