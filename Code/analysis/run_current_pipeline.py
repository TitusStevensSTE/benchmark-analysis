from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "Code" / "analysis"

STAGES = {
    "tabarena_score": "lodo_score.py",
    "dataset_search": "current-dataset_removal_search.py",
    "influence_extract": "current_extract_removal_influence.py",
    "dataset_influence": "current_dataset_influence.py",
    "performance_gap": "performance_gap_analysis.py",
    "fold_stability": "current_fold_stability.py",
    "fold_vs_dataset": "current_fold_vs_dataset_stability.py",
    "elo_search": "current_dataset_removal_search_elo.py",
    "score_elo_same_subsets": "compare_score_elo_same_subsets.py",
    "score_elo_removal_comparison": "compare_score_elo_removal.py",
    "crossbenchmark": "current_simple_crossbenchmark.py",
    "glue_gap": "current_glue_performance_gap.py",
    "glue_top_gap": "glue_top_leaderboard_gap_analysis.py",
    "glue_gap_plots": "glue_performance_gap_plots.py",
    "crossbenchmark_plots": "plot_crossbenchmark_lodo.py",
    "fold_plots": "plot_fold_stability_figures.py",
    "score_elo_plots": "plot_score_elo_removal.py",
    "comparison": "benchmark_comparison.py",
}

DEFAULT_ORDER = [
    "tabarena_score",
    "dataset_search",
    "influence_extract",
    "dataset_influence",
    "performance_gap",
    "fold_stability",
    "fold_vs_dataset",
    "elo_search",
    "score_elo_same_subsets",
    "score_elo_removal_comparison",
    "crossbenchmark",
    "glue_gap",
    "glue_top_gap",
    "glue_gap_plots",
    "crossbenchmark_plots",
    "fold_plots",
    "score_elo_plots",
    "comparison",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current thesis analysis pipeline in a reproducible order.")
    parser.add_argument("--skip", action="append", default=[], choices=sorted(STAGES), help="Skip a named stage; repeat for multiple stages.")
    parser.add_argument("--from-stage", choices=sorted(STAGES), help="Start at this stage.")
    args = parser.parse_args()

    start = 0 if args.from_stage is None else DEFAULT_ORDER.index(args.from_stage)
    for name in DEFAULT_ORDER[start:]:
        if name in args.skip:
            print(f"\n>>> SKIP {name}")
            continue
        script = ANALYSIS_DIR / STAGES[name]
        print(f"\n>>> RUN {name}: {script.name}")
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
