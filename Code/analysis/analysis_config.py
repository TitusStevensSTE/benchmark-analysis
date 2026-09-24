"""Shared paths for the thesis benchmark-analysis pipeline.

All active analysis scripts import their repository paths from this module so
that data, code, and generated results do not drift into different locations.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Repository layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "Code"
ANALYSIS_DIR = CODE_DIR / "analysis"
DATA_DIR = ANALYSIS_DIR / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHBENCH_ROOT = PROJECT_ROOT / "benchbench"

# Make the local BenchBench checkout importable for all active scripts.
benchbench_path = str(BENCHBENCH_ROOT)
if benchbench_path not in sys.path:
    sys.path.insert(0, benchbench_path)


TABARENA_DATA_FILE = DATA_DIR / "df_results_tabarena_2026_08_13.csv"
PHASE1_DIR = RESULTS_DIR / "phase1_crossbenchmark"
TABARENA_REMOVAL_DIR = RESULTS_DIR / "dataset_removal_search"
TABARENA_INFLUENCE_DIR = RESULTS_DIR / "dataset_influence"
TABARENA_FOLD_DIR = RESULTS_DIR / "fold_stability"
TABARENA_PERFORMANCE_GAP_DIR = RESULTS_DIR / "performance_gap"
TABARENA_STABILITY_DIMENSIONS_DIR = RESULTS_DIR / "stability_dimensions"
SCORE_ELO_DIR = RESULTS_DIR / "score_vs_elo"


def ensure_results_dir(path: Path) -> Path:
    """Create a result directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
