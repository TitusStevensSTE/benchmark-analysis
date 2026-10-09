from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from analysis_config import ANALYSIS_DIR, PROJECT_ROOT, RESULTS_DIR


# ============================================================
# PIPELINE STAGES
# ============================================================

STAGES = {
    # Report-only stages. These consume existing saved results.
    "final_figures": ("plot_final_thesis_figures.py", "report"),
    "crossbenchmark_plots": ("plot_crossbenchmark_lodo.py", "report"),
    "fold_plots": ("plot_fold_stability_figures.py", "report"),
    "score_elo_plots": ("plot_score_elo_removal.py", "report"),
    "glue_gap_plots": ("glue_performance_gap_plots.py", "report"),
    "comparison": ("benchmark_comparison.py", "report"),

    # Refresh stages. These recompute derived analyses but are not the
    # known long-running exhaustive searches.
    "crossbenchmark": ("current_simple_crossbenchmark.py", "refresh"),
    "glue_gap": ("current_glue_performance_gap.py", "refresh"),
    "glue_top_gap": ("glue_top_leaderboard_gap_analysis.py", "refresh"),
    "influence_extract": ("current_extract_removal_influence.py", "refresh"),
    "dataset_influence": ("current_dataset_influence.py", "refresh"),
    "performance_gap": ("performance_gap_analysis.py", "refresh"),
    "fold_vs_dataset": ("current_fold_vs_dataset_stability.py", "refresh"),

    # Expensive stages. These require explicit --expensive.
    "tabarena_score": ("lodo_score.py", "expensive"),
    "random_subsets": (
        "current_random_subset_winner_frequency.py",
        "expensive",
    ),
    "benchbench": ("current_benchbench_metrics.py", "expensive"),
    "dataset_search": ("current-dataset_removal_search.py", "expensive"),
    "fold_stability": ("current_fold_stability.py", "expensive"),
    "elo_search": ("current_dataset_removal_search_elo.py", "expensive"),
    "score_elo_same_subsets": (
        "compare_score_elo_same_subsets.py",
        "expensive",
    ),
    "score_elo_removal_comparison": (
        "compare_score_elo_removal.py",
        "expensive",
    ),
}


REPORT_ORDER = []


PLOT_ORDER = [
    "final_figures",
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
    *PLOT_ORDER,
]


FULL_ORDER = [
    "tabarena_score",
    "random_subsets",
    "benchbench",
    "dataset_search",
    "fold_stability",
    "elo_search",
    "score_elo_same_subsets",
    "score_elo_removal_comparison",
    *REFRESH_ORDER,
]


# ============================================================
# REPORT VALIDATION
# ============================================================

REQUIRED_REPORT_FILES = [
    RESULTS_DIR / "lodo_score_results.csv",
    RESULTS_DIR / "benchbench_metrics" / "benchbench_metrics.csv",
    (
        RESULTS_DIR
        / "random_subset_winner_frequency"
        / "random_subset_repetitions.csv"
    ),
    (
        RESULTS_DIR
        / "dataset_influence_final"
        / "dataset_influence_ranked.csv"
    ),
    RESULTS_DIR / "performance_gap" / "performance_gap_vs_lodo.csv",
    (
        RESULTS_DIR
        / "phase1_crossbenchmark"
        / "glue"
        / "lodo_results.csv"
    ),
    (
        RESULTS_DIR
        / "phase1_crossbenchmark"
        / "openllm"
        / "unique_model_mean"
        / "lodo_results.csv"
    ),
    (
        RESULTS_DIR
        / "phase1_crossbenchmark"
        / "imagenet"
        / "lodo_results.csv"
    ),
    (
        RESULTS_DIR
        / "dataset_removal_search"
        / "score"
        / "depth_summary.csv"
    ),
]


FINAL_FIGURE_STEMS = [
    "figure_01_pipeline",
    "figure_02_lodo_score_vs_elo",
    "figure_03_lodo_rank_movement",
    "figure_04_random_subset_stability",
    "figure_05_joint_removal_boundary",
    "figure_06_dataset_influence",
    "figure_07_performance_gap",
    "figure_08_crossbenchmark_lodo",
    "figure_09_benchbench_diversity_vs_sensitivity_two_panel",
]


def run_report_validation() -> None:
    """Validate saved outputs without triggering any analysis computation."""
    print("\n>>> REPORT VALIDATION")

    missing = []

    for path in REQUIRED_REPORT_FILES:
        relative = path.relative_to(PROJECT_ROOT)
        if path.exists():
            print(f"  OK      {relative}")
        else:
            print(f"  MISSING {relative}")
            missing.append(path)

    final_dir = RESULTS_DIR / "final_figures"

    print("\n  Final thesis figures:")
    for stem in FINAL_FIGURE_STEMS:
        png = final_dir / f"{stem}.png"
        pdf = final_dir / f"{stem}.pdf"

        if png.exists() and pdf.exists():
            print(f"  OK      {stem}")
        else:
            print(f"  MISSING {stem}")
            if not png.exists():
                missing.append(png)
            if not pdf.exists():
                missing.append(pdf)

    manifest = final_dir / "figure_manifest.json"

    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            count = len(data.get("figures", []))
            print(f"\n  Figure manifest: OK ({count} files listed)")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Figure manifest: unreadable ({exc})")
            missing.append(manifest)
    else:
        print("  Figure manifest: MISSING")
        missing.append(manifest)

    if missing:
        print(
            f"\nReport validation found {len(missing)} "
            "missing/unreadable item(s)."
        )
    else:
        print("\nReport validation passed: required saved outputs are present.")


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis benchmark pipeline safely. With no flag or "
            "--report, only saved-result reporting/plotting stages run. "
            "Recomputation requires explicit --refresh or --expensive."
        )
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Generate report-only plots from existing saved results and "
            "validate the saved result package. No benchmark recomputation."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run non-exhaustive analysis stages before generating reports.",
    )
    parser.add_argument(
        "--expensive",
        action="store_true",
        help=(
            "Allow long-running recomputation stages when explicitly "
            "selected or via --full."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete configured pipeline. Requires --expensive.",
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


# ============================================================
# EXECUTION
# ============================================================


def validate_permissions(
    stage_name: str,
    allow_expensive: bool,
) -> None:
    _, category = STAGES[stage_name]

    if category == "expensive" and not allow_expensive:
        raise RuntimeError(
            f"Refusing expensive stage '{stage_name}'. "
            "Use --expensive explicitly when you really intend to "
            "recompute it."
        )


def run_stage(
    name: str,
    allow_expensive: bool,
) -> None:
    validate_permissions(name, allow_expensive)

    script_name, category = STAGES[name]
    script = ANALYSIS_DIR / script_name

    if not script.exists():
        raise FileNotFoundError(f"Stage script not found: {script}")

    print(f"\n>>> RUN [{category.upper()}] {name}: {script.name}")

    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()

    selected_modes = [
        args.report,
        args.refresh,
        args.full,
        bool(args.stage),
    ]

    if sum(selected_modes) > 1:
        raise SystemExit(
            "Choose exactly one of --report, --refresh, --full, or --stage."
        )

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
        # No flags behaves like --report for backwards compatibility.
        order = REPORT_ORDER

    order = [name for name in order if name not in args.skip]

    print("=" * 72)
    print("CURRENT THESIS PIPELINE")
    print("=" * 72)
    print(f"Repository: {PROJECT_ROOT}")

    if args.full:
        mode = "FULL / EXPENSIVE ALLOWED"
    elif args.refresh:
        mode = "REFRESH"
    else:
        mode = "REPORT / EXISTING RESULTS"

    print(f"Mode:       {mode}")
    print(f"Stages:     {', '.join(order) if order else '(none)'}")

    for name in order:
        run_stage(name, allow_expensive=args.expensive)

    # Run validation for the default report and explicit --report.
    if (args.report or not any([args.refresh, args.full, args.stage])) and not args.skip:
        run_report_validation()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
