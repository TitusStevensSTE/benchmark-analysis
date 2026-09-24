"""Compatibility entry point for the canonical GLUE Phase 1 analysis.

The canonical implementation lives in current_simple_crossbenchmark.py and uses
the BenchBench-native GLUE loader.
"""

from pathlib import Path
import shutil

from current_simple_crossbenchmark import OUTPUT_ROOT, run_glue


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "analysis" / "results"


def main() -> None:
    summaries = run_glue()
    source = OUTPUT_ROOT / "glue"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Backward-compatible aliases for older downstream plotting/notes.
    for filename in ["lodo_results.csv", "full_ranking.csv", "task_summary.csv", "winner_changes.csv"]:
        src = source / filename
        if src.exists():
            shutil.copy2(src, RESULTS_DIR / f"glue_{filename}")

    print("\nCanonical GLUE outputs:")
    print(source)
    return summaries


if __name__ == "__main__":
    main()
