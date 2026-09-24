from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORE_FILE = (
    PROJECT_ROOT
    / "results"
    / "score_method_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "random_subset_winner_frequency"
)

SCORE_COLUMN = "dataset_score"

SEED = 42
N_REPETITIONS = 1000

# Evaluate every possible subset size:
# 1 dataset, 2 datasets, ..., all datasets.
MIN_SUBSET_SIZE = 1
MAX_SUBSET_SIZE = None

# Number of methods shown in the thesis plot.
PLOT_TOP_N = 10


# ============================================================
# Input validation
# ============================================================

def load_score_matrix() -> pd.DataFrame:
    """Load and validate the current dataset-level score matrix."""

    if not SCORE_FILE.exists():
        raise FileNotFoundError(
            f"Score file not found:\n{SCORE_FILE}"
        )

    df = pd.read_csv(SCORE_FILE)

    required_columns = {
        "dataset",
        "method",
        SCORE_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["dataset", "method"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[duplicate_mask, ["dataset", "method"]]
            .drop_duplicates()
            .head(20)
        )

        raise ValueError(
            "The score file contains duplicate "
            "(dataset, method) combinations.\n"
            f"Examples:\n{duplicates.to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if df[SCORE_COLUMN].isna().any():
        missing_count = int(df[SCORE_COLUMN].isna().sum())

        raise ValueError(
            f"{SCORE_COLUMN} contains {missing_count} missing values."
        )

    # --------------------------------------------------------
    # Build method × dataset matrix
    # --------------------------------------------------------

    matrix = df.pivot(
        index="method",
        columns="dataset",
        values=SCORE_COLUMN,
    )

    if matrix.isna().any().any():
        missing_cells = int(matrix.isna().sum().sum())

        raise ValueError(
            "The method × dataset matrix is incomplete. "
            f"Missing cells: {missing_cells}"
        )

    # --------------------------------------------------------
    # Score sanity checks
    # --------------------------------------------------------

    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(
            f"{SCORE_COLUMN} contains non-finite values."
        )

    score_min = float(matrix.min().min())
    score_max = float(matrix.max().max())

    print("Input validation")
    print("----------------")
    print(f"Methods:   {matrix.shape[0]}")
    print(f"Datasets:  {matrix.shape[1]}")
    print(f"Score:     {SCORE_COLUMN}")
    print(f"Range:     [{score_min:.6f}, {score_max:.6f}]")
    print()

    return matrix


# ============================================================
# Ranking
# ============================================================

def compute_ranking(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the benchmark ranking.

    Higher dataset_score = better performance.
    """

    mean_scores = matrix.mean(axis=1)

    ranking = (
        mean_scores
        .sort_values(
            ascending=False,
            kind="mergesort",
        )
        .rename("mean_score")
        .reset_index()
    )

    ranking["baseline_rank"] = (
        np.arange(len(ranking)) + 1
    )

    return ranking


# ============================================================
# Random subset experiment
# ============================================================

def run_random_subset_experiment(
    matrix: pd.DataFrame,
    baseline_ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly sample benchmark datasets and determine the
    rank-1 method for every sampled subset.

    For each subset size k:
        - sample k distinct datasets
        - average each method's score over those datasets
        - rank methods
        - record the rank-1 method

    Returns:
        repetitions_df:
            One row per random subset.

        frequency_df:
            Aggregated winner frequencies.
    """

    methods = matrix.index.to_numpy()
    datasets = matrix.columns.to_numpy()
    scores = matrix.to_numpy(dtype=float)

    n_methods = len(methods)
    n_datasets = len(datasets)

    max_subset_size = (
        n_datasets
        if MAX_SUBSET_SIZE is None
        else min(MAX_SUBSET_SIZE, n_datasets)
    )

    if MIN_SUBSET_SIZE < 1:
        raise ValueError(
            "MIN_SUBSET_SIZE must be >= 1."
        )

    if MIN_SUBSET_SIZE > max_subset_size:
        raise ValueError(
            "MIN_SUBSET_SIZE cannot exceed MAX_SUBSET_SIZE."
        )

    rng = np.random.default_rng(SEED)

    # Baseline rank lookup.
    baseline_rank = {
        row["method"]: int(row["baseline_rank"])
        for _, row in baseline_ranking.iterrows()
    }

    rows = []

    print("Random subset experiment")
    print("------------------------")
    print(f"Subset sizes: {MIN_SUBSET_SIZE}–{max_subset_size}")
    print(f"Repetitions:  {N_REPETITIONS}")
    print(f"Total runs:   {(max_subset_size - MIN_SUBSET_SIZE + 1) * N_REPETITIONS:,}")
    print(f"Random seed:  {SEED}")
    print()

    for subset_size in range(
        MIN_SUBSET_SIZE,
        max_subset_size + 1,
    ):

        print(
            f"Running subset size "
            f"{subset_size}/{max_subset_size}..."
        )

        for repetition in range(
            1,
            N_REPETITIONS + 1,
        ):

            sampled_indices = rng.choice(
                n_datasets,
                size=subset_size,
                replace=False,
            )

            sampled_dataset_names = datasets[
                sampled_indices
            ]

            # Average scores across sampled datasets.
            subset_scores = (
                scores[:, sampled_indices]
                .mean(axis=1)
            )

            # Stable deterministic ranking:
            # methods are first sorted alphabetically,
            # then score sorting preserves that order for ties.
            method_order = np.argsort(methods)

            ordered_methods = methods[method_order]
            ordered_scores = subset_scores[method_order]

            score_order = np.argsort(
                -ordered_scores,
                kind="mergesort",
            )

            winner_idx = score_order[0]

            winner = ordered_methods[winner_idx]
            winner_score = float(
                ordered_scores[winner_idx]
            )

            rows.append(
                {
                    "num_datasets": subset_size,
                    "repetition": repetition,
                    "winner": winner,
                    "winner_score": winner_score,
                    "winner_baseline_rank": baseline_rank[winner],
                    "sampled_datasets": "|".join(
                        map(str, sampled_dataset_names)
                    ),
                }
            )

    repetitions_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Aggregate winner frequencies
    # --------------------------------------------------------

    counts = (
        repetitions_df
        .groupby(
            ["num_datasets", "winner"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "wins"})
    )

    all_subset_sizes = np.arange(
        MIN_SUBSET_SIZE,
        max_subset_size + 1,
    )

    # Complete Cartesian product:
    # every subset size × every method.
    full_index = pd.MultiIndex.from_product(
        [
            all_subset_sizes,
            methods,
        ],
        names=[
            "num_datasets",
            "method",
        ],
    )

    frequency_df = (
        counts
        .rename(columns={"winner": "method"})
        .set_index(
            ["num_datasets", "method"]
        )
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    frequency_df["repetitions"] = N_REPETITIONS

    frequency_df["win_frequency"] = (
        frequency_df["wins"]
        / frequency_df["repetitions"]
    )

    frequency_df["win_percentage"] = (
        frequency_df["win_frequency"]
        * 100.0
    )

    frequency_df["baseline_rank"] = (
        frequency_df["method"]
        .map(baseline_rank)
    )

    frequency_df = frequency_df[
        [
            "num_datasets",
            "method",
            "baseline_rank",
            "wins",
            "repetitions",
            "win_frequency",
            "win_percentage",
        ]
    ]

    frequency_df = frequency_df.sort_values(
        [
            "num_datasets",
            "baseline_rank",
        ]
    ).reset_index(drop=True)

    return repetitions_df, frequency_df


# ============================================================
# Plot
# ============================================================

def create_top10_plot(
    frequency_df: pd.DataFrame,
    baseline_ranking: pd.DataFrame,
) -> Path:
    """Create a plot of winner frequency for the baseline top 10."""

    top_methods = (
        baseline_ranking
        .head(PLOT_TOP_N)["method"]
        .tolist()
    )

    plot_df = frequency_df[
        frequency_df["method"].isin(top_methods)
    ].copy()

    pivot = plot_df.pivot(
        index="num_datasets",
        columns="method",
        values="win_percentage",
    )

    # Preserve baseline ranking order.
    pivot = pivot[
        [
            method
            for method in top_methods
            if method in pivot.columns
        ]
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for method in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[method],
            label=method,
            linewidth=2,
        )

    ax.set_xlabel(
        "Number of sampled datasets"
    )

    ax.set_ylabel(
        "Probability of being ranked #1 (%)"
    )

    ax.set_title(
        "Random-subset winner frequency"
    )

    ax.set_xlim(
        MIN_SUBSET_SIZE,
        pivot.index.max(),
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Baseline top-10 methods",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "random_subset_winner_frequency.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# Metadata
# ============================================================

def save_metadata(
    matrix: pd.DataFrame,
) -> Path:
    """Save reproducibility metadata."""

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "script": Path(__file__).name,

        "score_file": str(
            SCORE_FILE.relative_to(PROJECT_ROOT)
        ),

        "score_column": SCORE_COLUMN,

        "score_orientation": "higher_is_better",

        "seed": SEED,

        "repetitions_per_subset_size": N_REPETITIONS,

        "min_subset_size": MIN_SUBSET_SIZE,

        "max_subset_size": (
            matrix.shape[1]
            if MAX_SUBSET_SIZE is None
            else MAX_SUBSET_SIZE
        ),

        "n_methods": int(matrix.shape[0]),

        "n_datasets": int(matrix.shape[1]),

        "numpy_version": np.__version__,

        "pandas_version": pd.__version__,

        "python_version": platform.python_version(),

        "experiment_definition": (
            "For each subset size k, sample k distinct "
            "datasets without replacement. Average each "
            "method's dataset_score across the sampled "
            "datasets and record the method ranked first."
        ),

        "historical_note": (
            "This reproduces the historical Colab random-"
            "subset winner-frequency experiment while "
            "using the current normalized dataset_score "
            "representation and a fixed random seed."
        ),
    }

    output_path = (
        OUTPUT_DIR
        / "random_subset_config.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    return output_path


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Random-Subset Winner-Frequency Analysis")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load current score representation
    # --------------------------------------------------------

    matrix = load_score_matrix()

    # --------------------------------------------------------
    # Full benchmark baseline
    # --------------------------------------------------------

    baseline_ranking = compute_ranking(
        matrix
    )

    baseline_path = (
        OUTPUT_DIR
        / "baseline_ranking.csv"
    )

    baseline_ranking.to_csv(
        baseline_path,
        index=False,
    )

    print("Full benchmark baseline")
    print("-----------------------")
    print(
        baseline_ranking.head(10).to_string(
            index=False
        )
    )
    print()

    # --------------------------------------------------------
    # Random subset experiment
    # --------------------------------------------------------

    repetitions_df, frequency_df = (
        run_random_subset_experiment(
            matrix,
            baseline_ranking,
        )
    )

    # --------------------------------------------------------
    # Save detailed repetition results
    # --------------------------------------------------------

    repetitions_path = (
        OUTPUT_DIR
        / "random_subset_repetitions.csv"
    )

    repetitions_df.to_csv(
        repetitions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save aggregated frequencies
    # --------------------------------------------------------

    frequency_path = (
        OUTPUT_DIR
        / "random_subset_winner_frequency.csv"
    )

    frequency_df.to_csv(
        frequency_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_path = create_top10_plot(
        frequency_df,
        baseline_ranking,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = save_metadata(
        matrix
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Analysis complete")
    print("=" * 60)

    print()
    print("Output files:")
    print(f"  Baseline ranking:")
    print(f"    {baseline_path}")

    print(f"  Detailed repetitions:")
    print(f"    {repetitions_path}")

    print(f"  Winner frequencies:")
    print(f"    {frequency_path}")

    print(f"  Plot:")
    print(f"    {plot_path}")

    print(f"  Configuration:")
    print(f"    {metadata_path}")

    print()
    print("Final full-benchmark winner:")
    print(
        f"  {baseline_ranking.iloc[0]['method']}"
    )

    print()
    print("Winner-frequency check:")
    final_k = frequency_df[
        frequency_df["num_datasets"]
        == frequency_df["num_datasets"].max()
    ]

    final_k = final_k[
        final_k["wins"] > 0
    ].sort_values(
        "win_percentage",
        ascending=False,
    )

    print(
        final_k[
            [
                "method",
                "wins",
                "win_percentage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()