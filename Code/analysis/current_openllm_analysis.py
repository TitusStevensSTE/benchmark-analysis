"""Compatibility entry point for the canonical OpenLLM Phase 1 analysis.

The canonical implementation lives in current_simple_crossbenchmark.py and uses
the BenchBench-native OpenLLM loader.
"""

from pathlib import Path
import shutil

from current_simple_crossbenchmark import OUTPUT_ROOT, run_openllm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Code" / "analysis" / "results"


def main() -> None:
    summaries = run_openllm()
    row_source = OUTPUT_ROOT / "openllm" / "row_level"
    unique_source = OUTPUT_ROOT / "openllm" / "unique_model_mean"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    aliases = {
        row_source / "lodo_results.csv": RESULTS_DIR / "openllm_lodo_results.csv",
        row_source / "full_ranking.csv": RESULTS_DIR / "openllm_full_ranking.csv",
        unique_source / "lodo_results.csv": RESULTS_DIR / "openllm_unique_model_lodo_results.csv",
        unique_source / "full_ranking.csv": RESULTS_DIR / "openllm_unique_model_full_ranking.csv",
    }
    for src, dst in aliases.items():
        if src.exists():
            shutil.copy2(src, dst)

    print("\nCanonical OpenLLM outputs:")
    print(OUTPUT_ROOT / "openllm")
    return summaries


if __name__ == "__main__":
    main()
