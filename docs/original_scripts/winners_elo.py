import os
import time
import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\df_results_tabarena_2026_08_13.csv"
OUTPUT_FILE = "C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\elo_1_51_results.csv"

N_REPETITIONS = 1000

# Analyze subsets from 1 to 51
SUBSET_SIZES = range(1, 52)

# Reproducibility
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

datasets = sorted(df["dataset"].unique())

print(f"\nDatasets ({len(datasets)}):")
for i, dataset in enumerate(datasets, 1):
    print(f"{i:2d}: {dataset}")


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FOR PREVIOUS RESULTS")
print("=" * 70)

if os.path.exists(OUTPUT_FILE):

    results_df = pd.read_csv(OUTPUT_FILE)

    if "subset_size" in results_df.columns:
        completed_sizes = set(
            results_df["subset_size"]
            .dropna()
            .astype(int)
            .unique()
        )
    else:
        completed_sizes = set()

    print(f"Existing result file found:")
    print(f"  {OUTPUT_FILE}")
    print(f"Rows already saved: {len(results_df)}")
    print(f"Completed subset sizes: {sorted(completed_sizes)}")

else:

    results_df = pd.DataFrame()
    completed_sizes = set()

    print("No previous result file found.")
    print("Starting from scratch.")


# ============================================================
# ELO FUNCTION
# ============================================================

def calculate_elo_for_subset(
    df,
    selected_datasets,
    n_repetitions=1000,
    seed=42
):
    """
    Run the Elo winner analysis for a particular subset of datasets.

    IMPORTANT:
    Replace the calculation inside this function with your
    existing Elo calculation if your current implementation
    uses a specific Elo function.
    """

    rng = np.random.default_rng(seed)

    subset_df = df[
        df["dataset"].isin(selected_datasets)
    ].copy()

    # --------------------------------------------------------
    # Aggregate performance per method and dataset
    # --------------------------------------------------------

    # TabArena metric_error:
    # lower = better
    #
    # Convert into a score where higher = better.
    #
    # If your existing code already performs a different
    # normalization, replace this section with that code.

    aggregated = (
        subset_df
        .groupby(["method", "dataset"])["metric_error"]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert metric_error to normalized score
    # --------------------------------------------------------

    # Rank methods separately on every dataset.
    #
    # rank 1 = best
    # higher score = better

    aggregated["rank"] = (
        aggregated
        .groupby("dataset")["metric_error"]
        .rank(
            method="average",
            ascending=True
        )
    )

    # --------------------------------------------------------
    # Pivot
    # --------------------------------------------------------

    ranking_matrix = aggregated.pivot(
        index="method",
        columns="dataset",
        values="rank"
    )

    # Only methods that have results on all selected datasets
    ranking_matrix = ranking_matrix.dropna(
        how="any"
    )

    if len(ranking_matrix) == 0:
        raise ValueError(
            "No methods have results for all selected datasets."
        )

    # --------------------------------------------------------
    # Repeated subset / winner analysis
    # --------------------------------------------------------

    repetition_results = []

    for repetition in range(n_repetitions):

        # Randomly shuffle the selected datasets
        shuffled = list(selected_datasets)
        rng.shuffle(shuffled)

        # Use the complete selected subset
        sampled_datasets = shuffled

        sampled_matrix = ranking_matrix[
            sampled_datasets
        ]

        # ----------------------------------------------------
        # Simple Elo-style pairwise calculation
        # ----------------------------------------------------

        methods = sampled_matrix.index.tolist()

        elo = {
            method: 1000.0
            for method in methods
        }

        K = 32

        for dataset in sampled_datasets:

            rankings = sampled_matrix[dataset].dropna()

            for i in range(len(rankings)):

                method_i = rankings.index[i]
                rank_i = rankings.iloc[i]

                for j in range(i + 1, len(rankings)):

                    method_j = rankings.index[j]
                    rank_j = rankings.iloc[j]

                    # Better rank wins
                    if rank_i < rank_j:
                        actual_i = 1
                    elif rank_i > rank_j:
                        actual_i = 0
                    else:
                        actual_i = 0.5

                    actual_j = 1 - actual_i

                    expected_i = (
                        1 /
                        (
                            1 +
                            10 ** (
                                (elo[method_j] - elo[method_i])
                                / 400
                            )
                        )
                    )

                    expected_j = 1 - expected_i

                    elo[method_i] += K * (
                        actual_i - expected_i
                    )

                    elo[method_j] += K * (
                        actual_j - expected_j
                    )

        # ----------------------------------------------------
        # Sort final Elo ranking
        # ----------------------------------------------------

        elo_ranking = (
            pd.Series(elo)
            .sort_values(
                ascending=False
            )
        )

        winner = elo_ranking.index[0]

        # Store result
        repetition_results.append({
            "subset_size": len(selected_datasets),
            "repetition": repetition + 1,
            "winner": winner,
            "winner_elo": elo_ranking.iloc[0],
            "num_methods": len(methods)
        })

    return pd.DataFrame(repetition_results)


# ============================================================
# MAIN 1–51 ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STARTING ELO 1–51 ANALYSIS")
print("=" * 70)

overall_start = time.time()


for subset_size in SUBSET_SIZES:

    # --------------------------------------------------------
    # SKIP ALREADY COMPLETED SUBSETS
    # --------------------------------------------------------

    if subset_size in completed_sizes:

        print(
            f"\n[{subset_size}/51] "
            f"Already completed → SKIPPING"
        )

        continue


    print("\n" + "=" * 70)
    print(
        f"STARTING SUBSET {subset_size}/51"
    )
    print("=" * 70)

    subset_start = time.time()


    # --------------------------------------------------------
    # SELECT DATASETS
    # --------------------------------------------------------

    # IMPORTANT:
    # This uses the first N datasets.
    #
    # If your previous analysis uses random subsets instead,
    # replace this with the exact dataset-selection logic
    # from your existing code.

    selected_datasets = datasets[:subset_size]

    print(
        f"Using {len(selected_datasets)} datasets:"
    )

    print(
        selected_datasets
    )


    # --------------------------------------------------------
    # RUN ELO ANALYSIS
    # --------------------------------------------------------

    current_results = calculate_elo_for_subset(
        df=df,
        selected_datasets=selected_datasets,
        n_repetitions=N_REPETITIONS,
        seed=RANDOM_SEED + subset_size
    )


    # --------------------------------------------------------
    # ADD DATASET INFORMATION
    # --------------------------------------------------------

    current_results["datasets_used"] = (
        ",".join(selected_datasets)
    )


    # --------------------------------------------------------
    # APPEND TO MASTER RESULTS
    # --------------------------------------------------------

    if results_df.empty:

        results_df = current_results.copy()

    else:

        results_df = pd.concat(
            [
                results_df,
                current_results
            ],
            ignore_index=True
        )


    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    temp_file = OUTPUT_FILE + ".tmp"

    results_df.to_csv(
        temp_file,
        index=False
    )

    # Atomic replacement
    os.replace(
        temp_file,
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # UPDATE COMPLETED SET
    # --------------------------------------------------------

    completed_sizes.add(
        subset_size
    )


    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

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
        f"FINISHED {subset_size}/51"
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
        f"Results saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"Total rows saved: "
        f"{len(results_df):,}"
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
    f"Saved file: "
    f"{OUTPUT_FILE}"
)

print(
    f"Total runtime: "
    f"{(time.time() - overall_start) / 60:.2f} minutes"
)

print("=" * 70)