"""Compatibility entry point for the canonical ImageNet Phase 1 analysis.

The canonical implementation lives in current_simple_crossbenchmark.py and uses
the BenchBench-native ImageNet loader.
"""

from pathlib import Path
import shutil

from current_simple_crossbenchmark import OUTPUT_ROOT, run_imagenet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "analysis" / "results"


def main() -> None:
    summary = run_imagenet()
    source = OUTPUT_ROOT / "imagenet"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    aliases = {
        source / "lodo_results.csv": RESULTS_DIR / "imagenet_lodo_results.csv",
        source / "full_ranking.csv": RESULTS_DIR / "imagenet_full_ranking.csv",
    }
    for src, dst in aliases.items():
        if src.exists():
            shutil.copy2(src, dst)

    print("\nCanonical ImageNet outputs:")
    print(source)
    return summary


if __name__ == "__main__":
    main()
