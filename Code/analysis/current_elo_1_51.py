from pathlib import Path
import os
import time
import pandas as pd
import numpy as np

from bencheval.elo_utils import EloHelper


# ============================================================
# SETTINGS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "current_elo_1_51_results.csv"
)


N_REPETITIONS = 1000

SUBSET_SIZES = range(1, 52)

RANDOM_SEED = 42


# ============================================================
# LOAD TABARENA DATA
# ============================================================

print("=" * 70)
print("LOADING TABARENA DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded dataframe: {df.shape}")
print(f"Number of datasets: {df['dataset'].nunique()}")
print(f"Number of methods: {df['method'].nunique()}")

datasets = sorted(
    df["dataset"]
    .dropna()
    .unique()
)

print(f"\nDatasets ({len(datasets)}):")

for i, dataset in enumerate(datasets, 1):
    print(f"{i:2d}: {dataset}")


if len(datasets) != 51:
    raise ValueError(
        f"Expected 51 datasets, found {len(datasets)}."
    )


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = {
    "dataset",
    "fold",
    "method",
    "metric_error",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df["metric_error"] = pd.to_numeric(
    df["metric_error"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "dataset",
        "fold",
        "method",
        "metric_error",
    ]
).copy()


# ============================================================
# CREATE TABARENA ELO HELPER
# ============================================================

elo_helper = EloHelper(
    method_col="method",
    task_col="dataset",
    error_col="metric_error",
    split_col="fold",
)


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FOR PREVIOUS RESULTS")
print("=" * 70)

if os.path.exists(OUTPUT_FILE):

    results_df = pd.read_csv(
        OUTPUT_FILE
    )

    print("Existing result file found:")
    print(f"  {OUTPUT_FILE}")
    print(
        f"Rows already saved: "
        f"{len(results_df):,}"
    )

else:

    results_df = pd.DataFrame()

    print("No previous result file found.")
    print("Starting from scratch.")


# ============================================================
# CHECK WHICH SUBSETS ARE REALLY COMPLETE
# ============================================================

def get_completed_sizes(
    results_df,
    n_repetitions,
):

    if results_df.empty:
        return set()

    if "subset_size" not in results_df.columns:
        return set()

    counts = (
        results_df
        .groupby("subset_size")
        .size()
    )

    completed = set(
        counts[
            counts >= n_repetitions
        ]
        .index
        .astype(int)
    )

    return completed


completed_sizes = get_completed_sizes(
    results_df,
    N_REPETITIONS,
)

print(
    f"Completed subset sizes: "
    f"{sorted(completed_sizes)}"
)


# ============================================================
# CALCULATE TABARENA ELO
# ============================================================

def calculate_tabarena_elo(
    subset_df,
):
    """
    Convert TabArena results to TabArena battles
    and calculate MLE Elo using TabArena's own
    EloHelper implementation.
    """

    # --------------------------------------------------------
    # Convert results -> battles
    # --------------------------------------------------------

    battles = (
        elo_helper
        .convert_results_to_battles(
            results_df=subset_df
        )
    )

    if battles.empty:
        raise ValueError(
            "No battles were generated."
        )

    # --------------------------------------------------------
    # Calculate MLE Elo
    # --------------------------------------------------------

    elo_ranking = (
        elo_helper
        .compute_mle_elo(
            battles=battles,
            SCALE=400,
            BASE=10,
            INIT_RATING=1000,
            use_pair_aggregation=True,
            force_iterative_elo=False,
            max_iter=1000,
        )
    )

    return elo_ranking, battles


# ============================================================
# MAIN 1–51 RANDOM SUBSET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STARTING RANDOM ELO 1–51 ANALYSIS")
print("=" * 70)

overall_start = time.time()


for subset_size in SUBSET_SIZES:

    # ========================================================
    # RESUME CHECK
    # ========================================================

    if subset_size in completed_sizes:

        print(
            f"\n[{subset_size}/51] "
            f"Already has "
            f"{N_REPETITIONS} repetitions "
            f"→ SKIPPING"
        )

        continue


    print("\n" + "=" * 70)

    print(
        f"STARTING SUBSET SIZE "
        f"{subset_size}/51"
    )

    print("=" * 70)


    subset_start = time.time()


    # ========================================================
    # RANDOM NUMBER GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED + subset_size
    )


    # ========================================================
    # REPETITIONS
    # ========================================================

    current_results = []


    for repetition in range(
        N_REPETITIONS
    ):

        # ----------------------------------------------------
        # RANDOMLY SELECT DATASETS
        # ----------------------------------------------------

        selected_datasets = list(
            rng.choice(
                datasets,
                size=subset_size,
                replace=False,
            )
        )

        selected_datasets = sorted(
            selected_datasets
        )


        # ----------------------------------------------------
        # FILTER TABARENA RESULTS
        # ----------------------------------------------------

        subset_df = df[
            df["dataset"].isin(
                selected_datasets
            )
        ].copy()


        # ----------------------------------------------------
        # CALCULATE TABARENA ELO
        # ----------------------------------------------------

        elo_ranking, battles = (
            calculate_tabarena_elo(
                subset_df
            )
        )


        # ----------------------------------------------------
        # WINNER
        # ----------------------------------------------------

        winner = (
            elo_ranking.index[0]
        )

        winner_elo = float(
            elo_ranking.iloc[0]
        )


        # ----------------------------------------------------
        # NUMBER OF METHODS
        # ----------------------------------------------------

        num_methods = (
            len(elo_ranking)
        )


        # ----------------------------------------------------
        # NUMBER OF BATTLES
        # ----------------------------------------------------

        num_battles = len(
            battles
        )


        # ----------------------------------------------------
        # SAVE RESULT IN MEMORY
        # ----------------------------------------------------

        current_results.append(
            {
                "subset_size": subset_size,
                "repetition": repetition + 1,
                "winner": winner,
                "winner_elo": winner_elo,
                "num_methods": num_methods,
                "num_battles": num_battles,
                "datasets_used": ",".join(
                    selected_datasets
                ),
            }
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (repetition + 1) % 100 == 0
            or repetition == 0
        ):

            elapsed = (
                time.time()
                - subset_start
            )

            print(
                f"  repetition "
                f"{repetition + 1:4d}/"
                f"{N_REPETITIONS} "
                f"| elapsed "
                f"{elapsed / 60:.2f} min"
            )


    # ========================================================
    # CONVERT RESULTS
    # ========================================================

    current_results = pd.DataFrame(
        current_results
    )


    # ========================================================
    # APPEND TO EXISTING RESULTS
    # ========================================================

    if results_df.empty:

        results_df = (
            current_results
            .copy()
        )

    else:

        results_df = pd.concat(
            [
                results_df,
                current_results,
            ],
            ignore_index=True,
        )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    results_df = (
        results_df
        .drop_duplicates(
            subset=[
                "subset_size",
                "repetition",
            ],
            keep="last",
        )
        .sort_values(
            [
                "subset_size",
                "repetition",
            ]
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # SAVE IMMEDIATELY
    # ========================================================

    temp_file = (
        OUTPUT_FILE + ".tmp"
    )

    results_df.to_csv(
        temp_file,
        index=False,
    )

    os.replace(
        temp_file,
        OUTPUT_FILE,
    )


    # ========================================================
    # MARK SUBSET COMPLETE
    # ========================================================

    completed_sizes.add(
        subset_size
    )


    # ========================================================
    # TIMING
    # ========================================================

    elapsed = (
        time.time()
        - subset_start
    )

    total_elapsed = (
        time.time()
        - overall_start
    )


    print("\n" + "-" * 70)

    print(
        f"FINISHED "
        f"{subset_size}/51"
    )

    print(
        f"Time for this subset: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Total runtime this session: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(
        f"Total rows saved: "
        f"{len(results_df):,}"
    )

    print(
        f"Output file:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print("-" * 70)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Completed subset sizes: "
    f"{sorted(completed_sizes)}"
)

print(
    f"Total result rows: "
    f"{len(results_df):,}"
)

print(
    f"Expected rows if fully complete: "
    f"{51 * N_REPETITIONS:,}"
)

print(
    f"Saved file:"
)

print(
    OUTPUT_FILE
)

print(
    f"Total runtime: "
    f"{(time.time() - overall_start) / 60:.2f} minutes"
)

print("=" * 70)