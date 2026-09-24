from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis_config import ANALYSIS_DIR, PROJECT_ROOT


STAGES = {
    # Reporting / plotting stages: read existing result files and do not start
    # any benchmark-level recomputation.
    "crossbenchmark_plots": ("plot_crossbenchmark_lodo.py", "report"),
    "fold_plots": ("plot_fold_stability_figures.py", "report"),
    "score_elo_plots": ("plot_score_elo_removal.py", "report"),
    "glue_gap_plots": ("glue_performance_gap_plots.py", "report"),
    "comparison": ("benchmark_comparison.py", "report"),
    # Refresh stages: recompute analyses, but are not the known long-running
    # exhaustive searches. They still only run when explicitly requested.
    "crossbenchmark": ("current_simple_crossbenchmark.py", "refresh"),
    "glue_gap": ("current_glue_performance_gap.py", "refresh"),
    "glue_top_gap": ("glue_top_leaderboard_gap_analysis.py", "refresh"),
    "influence_extract": ("current_extract_removal_influence.py", "refresh"),
    "dataset_influence": ("current_dataset_influence.py", "refresh"),
    "performance_gap": ("performance_gap_analysis.py", "refresh"),
    "fold_vs_dataset": ("current_fold_vs_dataset_stability.py", "refresh"),
    # Expensive stages: explicit --expensive is required.
    "tabarena_score": ("lodo_score.py", "expensive"),
    "dataset_search": ("current-dataset_removal_search.py", "expensive"),
    "fold_stability": ("current_fold_stability.py", "expensive"),
    "elo_search": ("current_dataset_removal_search_elo.py", "expensive"),
    "score_elo_same_subsets": ("compare_score_elo_same_subsets.py", "expensive"),
    "score_elo_removal_comparison": ("compare_score_elo_removal.py", "expensive"),
}

REPORT_ORDER = [
    "crossbenchmark_plots",
    "fold_plots",
    "score_elo_plots",
    "glue_gap_plots",
    "comparison",
]

REFRESH_ORDER = [
    "crossbenchmark",
    "glue_gap",
    "glue_top_gap",
    "influence_extract",
    "dataset_influence",
    "performance_gap",
    "fold_vs_dataset",
    *REPORT_ORDER,
]

FULL_ORDER = [
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
    *REFRESH_ORDER,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis benchmark pipeline safely. With no flags, only "
            "plots and summaries are generated from existing result files. "
            "Recomputation requires explicit --refresh or --expensive."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run non-exhaustive analysis stages before generating reports.",
    )
    parser.add_argument(
        "--expensive",
        action="store_true",
        help="Allow long-running exhaustive stages when explicitly selected or via --full.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete configured order. Requires --expensive.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGES),
        help="Run one or more specific stages. May be repeated.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=sorted(STAGES),
        help="Skip a named stage; may be repeated.",
    )
    return parser.parse_args()


def validate_permissions(stage_name: str, allow_expensive: bool) -> None:
    _, category = STAGES[stage_name]
    if category == "expensive" and not allow_expensive:
        raise RuntimeError(
            f"Refusing expensive stage '{stage_name}'. "
            "Use --expensive explicitly when you really intend to recompute it."
        )


def run_stage(name: str, allow_expensive: bool) -> None:
    validate_permissions(name, allow_expensive)
    script_name, category = STAGES[name]
    script = ANALYSIS_DIR / script_name

    if not script.exists():
        raise FileNotFoundError(f"Stage script not found: {script}")

    print(f"\n>>> RUN [{category.upper()}] {name}: {script.name}")
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if args.stage:
        order = args.stage
    elif args.full:
        if not args.expensive:
            raise SystemExit(
                "Refusing --full without --expensive. "
                "This prevents accidental long-running recomputation."
            )
        order = FULL_ORDER
    elif args.refresh:
        order = REFRESH_ORDER
    else:
        order = REPORT_ORDER

    order = [name for name in order if name not in args.skip]

    print("=" * 72)
    print("CURRENT THESIS PIPELINE")
    print("=" * 72)
    print(f"Repository: {PROJECT_ROOT}")
    print(f"Mode:       {'EXPENSIVE ALLOWED' if args.expensive else 'SAFE / EXISTING RESULTS'}")
    print(f"Stages:     {', '.join(order) if order else '(none)'}")

    for name in order:
        run_stage(name, allow_expensive=args.expensive)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
