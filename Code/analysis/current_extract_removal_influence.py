from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

# Script is in: Code/analysis/
# Therefore parents[2] = repository root

REMOVAL_DIR = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "score"
)

INPUT_FILE = REMOVAL_DIR / "evaluations.csv"

OUTPUT_DIR = REMOVAL_DIR / "influence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 200_000

# Number of strongest subsets retained for each removal depth.
TOP_N_PER_DEPTH = 100

# Ranking-instability threshold used by the removal search.
TAU_THRESHOLD = 0.90


# ============================================================
# HELPERS
# ============================================================

def parse_removed_datasets(value) -> list[str]:
    """
    Convert the removed_datasets representation into a list.

    Expected format is usually:
        "dataset_a|dataset_b|dataset_c"

    Empty / missing values become [].
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [x.strip() for x in value.split("|") if x.strip()]


def sort_best_subsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort subsets by:
      1. lowest Kendall tau
      2. highest maximum rank change
      3. highest mean rank change
    """
    if df.empty:
        return df

    return df.sort_values(
        [
            "kendall_tau",
            "max_rank_change",
            "mean_rank_change",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


# ============================================================
# CHECK INPUT
# ============================================================

print("=" * 70)
print("EXTRACTING DATASET INFLUENCE FROM REMOVAL SEARCH")
print("=" * 70)
print()

print(f"Project root : {PROJECT_ROOT}")
print(f"Input file   : {INPUT_FILE}")
print(f"Output dir   : {OUTPUT_DIR}")
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find removal-search evaluations file:\n{INPUT_FILE}"
    )


# ============================================================
# OUTPUT CONTAINERS
# ============================================================

# For each removal depth, keep only the strongest TOP_N_PER_DEPTH
# subsets according to ranking instability.
top_subsets_by_depth: dict[int, list[pd.DataFrame]] = {}

# Instead of storing every winner-changing row in memory,
# only keep rows belonging to the minimum winner-change depth.
winner_candidates_by_depth: dict[int, list[pd.DataFrame]] = {}

# Same idea for tau < threshold.
tau_candidates_by_depth: dict[int, list[pd.DataFrame]] = {}

num_chunks = 0
num_rows = 0
num_winner_changes = 0
num_tau_breaks = 0

all_depths: set[int] = set()


# ============================================================
# READ CSV IN CHUNKS
# ============================================================

usecols = [
    "removed_mask",
    "removed_datasets",
    "num_removed",
    "kendall_tau",
    "max_rank_change",
    "mean_rank_change",
    "winner",
    "winner_changed",
    "tau_unstable",
]

print("Reading evaluations.csv in chunks...")
print()

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
    low_memory=False,
):
    num_chunks += 1
    num_rows += len(chunk)

    # --------------------------------------------------------
    # Normalize column types
    # --------------------------------------------------------

    chunk["num_removed"] = pd.to_numeric(
        chunk["num_removed"],
        errors="coerce",
    ).astype("Int64")

    chunk["kendall_tau"] = pd.to_numeric(
        chunk["kendall_tau"],
        errors="coerce",
    )

    chunk["max_rank_change"] = pd.to_numeric(
        chunk["max_rank_change"],
        errors="coerce",
    )

    chunk["mean_rank_change"] = pd.to_numeric(
        chunk["mean_rank_change"],
        errors="coerce",
    )

    # Convert boolean-like columns robustly
    for col in ["winner_changed", "tau_unstable"]:
        if chunk[col].dtype != bool:
            chunk[col] = (
                chunk[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "1": True,
                        "yes": True,
                        "false": False,
                        "0": False,
                        "no": False,
                    }
                )
                .fillna(False)
            )

    # --------------------------------------------------------
    # Track depths
    # --------------------------------------------------------

    valid_depths = chunk["num_removed"].dropna().astype(int).unique()

    for depth in valid_depths:
        all_depths.add(int(depth))

    # --------------------------------------------------------
    # Keep strongest TOP_N subsets for each depth
    # --------------------------------------------------------

    for depth in valid_depths:
        depth = int(depth)

        depth_df = chunk[
            chunk["num_removed"].eq(depth)
        ].copy()

        if depth_df.empty:
            continue

        top_subsets_by_depth.setdefault(depth, []).append(
            depth_df
        )

    # --------------------------------------------------------
    # Winner-change candidates
    # --------------------------------------------------------

    winner_df = chunk[
        chunk["winner_changed"].eq(True)
    ].copy()

    if not winner_df.empty:
        num_winner_changes += len(winner_df)

        winner_depths = (
            winner_df["num_removed"]
            .dropna()
            .astype(int)
            .unique()
        )

        for depth in winner_depths:
            depth = int(depth)

            depth_df = winner_df[
                winner_df["num_removed"].eq(depth)
            ].copy()

            if depth_df.empty:
                continue

            winner_candidates_by_depth.setdefault(depth, []).append(
                depth_df
            )

    # --------------------------------------------------------
    # Tau-break candidates
    # --------------------------------------------------------

    tau_df = chunk[
        chunk["kendall_tau"].lt(TAU_THRESHOLD)
    ].copy()

    if not tau_df.empty:
        num_tau_breaks += len(tau_df)

        tau_depths = (
            tau_df["num_removed"]
            .dropna()
            .astype(int)
            .unique()
        )

        for depth in tau_depths:
            depth = int(depth)

            depth_df = tau_df[
                tau_df["num_removed"].eq(depth)
            ].copy()

            if depth_df.empty:
                continue

            tau_candidates_by_depth.setdefault(depth, []).append(
                depth_df
            )

    if num_chunks % 10 == 0:
        print(
            f"Processed chunks: {num_chunks:4d} | "
            f"rows: {num_rows:12,d} | "
            f"winner changes: {num_winner_changes:8,d} | "
            f"tau breaks: {num_tau_breaks:8,d}"
        )


print()
print("Finished reading evaluations.csv.")
print(f"Chunks processed : {num_chunks}")
print(f"Rows processed   : {num_rows:,}")
print(f"Winner changes   : {num_winner_changes:,}")
print(f"Tau < {TAU_THRESHOLD:.2f} events : {num_tau_breaks:,}")
print()


# ============================================================
# 1. TOP 100 SUBSETS PER DEPTH
# ============================================================

print("Creating top-subset table...")

top_frames = []

for depth in sorted(top_subsets_by_depth):
    combined = pd.concat(
        top_subsets_by_depth[depth],
        ignore_index=True,
    )

    combined = sort_best_subsets(combined)
    combined = combined.head(TOP_N_PER_DEPTH).copy()

    combined["removed_datasets_list"] = combined[
        "removed_datasets"
    ].apply(parse_removed_datasets)

    top_frames.append(combined)

if top_frames:
    top_subsets = pd.concat(
        top_frames,
        ignore_index=True,
    )
else:
    top_subsets = pd.DataFrame()

top_subsets_file = (
    OUTPUT_DIR / "adversarial_top100_by_depth.csv"
)

top_subsets.to_csv(
    top_subsets_file,
    index=False,
)

print(f"Saved: {top_subsets_file}")
print()


# ============================================================
# 2. MINIMUM WINNER-CHANGE SUBSETS
# ============================================================

print("Extracting minimum winner-change subsets...")

if winner_candidates_by_depth:

    min_winner_depth = min(
        winner_candidates_by_depth
    )

    winner_min = pd.concat(
        winner_candidates_by_depth[min_winner_depth],
        ignore_index=True,
    )

    winner_min = sort_best_subsets(
        winner_min
    )

    winner_min["removed_datasets_list"] = winner_min[
        "removed_datasets"
    ].apply(parse_removed_datasets)

else:

    min_winner_depth = None
    winner_min = pd.DataFrame(
        columns=usecols + ["removed_datasets_list"]
    )

winner_file = (
    OUTPUT_DIR / "minimum_winner_change_subsets.csv"
)

winner_min.to_csv(
    winner_file,
    index=False,
)

print(f"Minimum winner-change depth: {min_winner_depth}")
print(f"Saved: {winner_file}")
print()


# ============================================================
# 3. MINIMUM TAU-BREAK SUBSETS
# ============================================================

print(
    f"Extracting minimum subsets with Kendall tau < "
    f"{TAU_THRESHOLD:.2f}..."
)

if tau_candidates_by_depth:

    min_tau_depth = min(
        tau_candidates_by_depth
    )

    tau_min = pd.concat(
        tau_candidates_by_depth[min_tau_depth],
        ignore_index=True,
    )

    tau_min = sort_best_subsets(
        tau_min
    )

    tau_min["removed_datasets_list"] = tau_min[
        "removed_datasets"
    ].apply(parse_removed_datasets)

else:

    min_tau_depth = None
    tau_min = pd.DataFrame(
        columns=usecols + ["removed_datasets_list"]
    )

tau_file = (
    OUTPUT_DIR / "minimum_tau_break_subsets.csv"
)

tau_min.to_csv(
    tau_file,
    index=False,
)

print(f"Minimum tau-break depth: {min_tau_depth}")
print(f"Saved: {tau_file}")
print()


# ============================================================
# 4. DATASET FREQUENCY IN MINIMUM WINNER-CHANGE SETS
# ============================================================

print("Computing dataset frequency in minimum winner-change sets...")

winner_frequency_rows = []

if not winner_min.empty:

    for datasets in winner_min["removed_datasets_list"]:

        for dataset in datasets:
            winner_frequency_rows.append(
                {
                    "dataset": dataset,
                }
            )

if winner_frequency_rows:

    winner_frequency = pd.DataFrame(
        winner_frequency_rows
    )

    winner_frequency = (
        winner_frequency["dataset"]
        .value_counts()
        .rename_axis("dataset")
        .reset_index(name="count")
    )

    winner_frequency[
        "frequency"
    ] = (
        winner_frequency["count"]
        / len(winner_min)
    )

else:

    winner_frequency = pd.DataFrame(
        columns=[
            "dataset",
            "count",
            "frequency",
        ]
    )

winner_frequency_file = (
    OUTPUT_DIR
    / "dataset_winner_change_frequency.csv"
)

winner_frequency.to_csv(
    winner_frequency_file,
    index=False,
)

print(f"Saved: {winner_frequency_file}")
print()


# ============================================================
# 5. DATASET FREQUENCY IN MINIMUM TAU-BREAK SETS
# ============================================================

print(
    f"Computing dataset frequency in minimum tau < "
    f"{TAU_THRESHOLD:.2f} sets..."
)

tau_frequency_rows = []

if not tau_min.empty:

    for datasets in tau_min["removed_datasets_list"]:

        for dataset in datasets:
            tau_frequency_rows.append(
                {
                    "dataset": dataset,
                }
            )

if tau_frequency_rows:

    tau_frequency = pd.DataFrame(
        tau_frequency_rows
    )

    tau_frequency = (
        tau_frequency["dataset"]
        .value_counts()
        .rename_axis("dataset")
        .reset_index(name="count")
    )

    tau_frequency[
        "frequency"
    ] = (
        tau_frequency["count"]
        / len(tau_min)
    )

else:

    tau_frequency = pd.DataFrame(
        columns=[
            "dataset",
            "count",
            "frequency",
        ]
    )

tau_frequency_file = (
    OUTPUT_DIR
    / "dataset_tau_break_frequency.csv"
)

tau_frequency.to_csv(
    tau_frequency_file,
    index=False,
)

print(f"Saved: {tau_frequency_file}")
print()


# ============================================================
# 6. DATASET FREQUENCY ACROSS TOP 100 SUBSETS / DEPTH
# ============================================================

print("Computing broader adversarial dataset frequency...")

adversarial_rows = []

if not top_subsets.empty:

    for _, row in top_subsets.iterrows():

        datasets = parse_removed_datasets(
            row["removed_datasets"]
        )

        depth = int(row["num_removed"])

        for dataset in datasets:
            adversarial_rows.append(
                {
                    "dataset": dataset,
                    "num_removed": depth,
                }
            )

if adversarial_rows:

    adversarial_long = pd.DataFrame(
        adversarial_rows
    )

    adversarial_frequency = (
        adversarial_long
        .groupby(["num_removed", "dataset"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        adversarial_long
        .groupby("num_removed")
        .size()
        .rename("total_dataset_slots")
        .reset_index()
    )

    adversarial_frequency = adversarial_frequency.merge(
        totals,
        on="num_removed",
        how="left",
    )

    adversarial_frequency[
        "frequency"
    ] = (
        adversarial_frequency["count"]
        / adversarial_frequency["total_dataset_slots"]
    )

    adversarial_frequency = adversarial_frequency.sort_values(
        [
            "num_removed",
            "frequency",
            "count",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

else:

    adversarial_frequency = pd.DataFrame(
        columns=[
            "num_removed",
            "dataset",
            "count",
            "total_dataset_slots",
            "frequency",
        ]
    )

adversarial_frequency_file = (
    OUTPUT_DIR
    / "dataset_adversarial_frequency.csv"
)

adversarial_frequency.to_csv(
    adversarial_frequency_file,
    index=False,
)

print(f"Saved: {adversarial_frequency_file}")
print()


# ============================================================
# 7. COMPACT SUMMARY
# ============================================================

summary = {
    "input_file": str(INPUT_FILE),
    "rows_processed": int(num_rows),
    "chunks_processed": int(num_chunks),
    "tau_threshold": float(TAU_THRESHOLD),
    "top_n_per_depth": int(TOP_N_PER_DEPTH),
    "depths_found": sorted(int(x) for x in all_depths),
    "minimum_winner_change_depth": (
        int(min_winner_depth)
        if min_winner_depth is not None
        else None
    ),
    "minimum_tau_break_depth": (
        int(min_tau_depth)
        if min_tau_depth is not None
        else None
    ),
    "winner_change_rows": int(num_winner_changes),
    "tau_break_rows": int(num_tau_breaks),
    "top_subsets_output": str(top_subsets_file),
    "winner_change_output": str(winner_file),
    "tau_break_output": str(tau_file),
    "winner_frequency_output": str(winner_frequency_file),
    "tau_frequency_output": str(tau_frequency_file),
    "adversarial_frequency_output": str(
        adversarial_frequency_file
    ),
}

summary_json = (
    OUTPUT_DIR / "extraction_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print(f"Saved: {summary_json}")
print()


# ============================================================
# 8. HUMAN-READABLE SUMMARY
# ============================================================

summary_txt = (
    OUTPUT_DIR / "summary.txt"
)

with open(
    summary_txt,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "DATASET REMOVAL INFLUENCE EXTRACTION\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Rows processed: {num_rows:,}\n"
    )

    f.write(
        f"Minimum winner-change depth: "
        f"{min_winner_depth}\n"
    )

    f.write(
        f"Minimum tau-break depth "
        f"(tau < {TAU_THRESHOLD:.2f}): "
        f"{min_tau_depth}\n"
    )

    f.write(
        f"Winner-change evaluations: "
        f"{num_winner_changes:,}\n"
    )

    f.write(
        f"Tau-break evaluations: "
        f"{num_tau_breaks:,}\n\n"
    )

    f.write(
        "OUTPUT FILES\n"
    )
    f.write("-" * 60 + "\n")

    for path in [
        top_subsets_file,
        winner_file,
        tau_file,
        winner_frequency_file,
        tau_frequency_file,
        adversarial_frequency_file,
        summary_json,
    ]:
        f.write(
            f"{path.name}\n"
        )

print(f"Saved: {summary_txt}")
print()

# ============================================================
# 9. PRINT IMPORTANT RESULTS
# ============================================================

print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print()

if not winner_min.empty:
    print(
        f"Minimum winner-change depth: "
        f"{min_winner_depth}"
    )

    print(
        "\nStrongest minimum winner-change subsets:"
    )

    preview_cols = [
        "removed_datasets",
        "kendall_tau",
        "max_rank_change",
        "mean_rank_change",
        "winner",
    ]

    print(
        winner_min[preview_cols]
        .head(10)
        .to_string(index=False)
    )
else:
    print("No winner-changing subset found.")

print()

if not tau_min.empty:
    print(
        f"Minimum tau-break depth "
        f"(tau < {TAU_THRESHOLD:.2f}): "
        f"{min_tau_depth}"
    )

    print(
        "\nStrongest minimum tau-break subsets:"
    )

    preview_cols = [
        "removed_datasets",
        "kendall_tau",
        "max_rank_change",
        "mean_rank_change",
        "winner",
    ]

    print(
        tau_min[preview_cols]
        .head(10)
        .to_string(index=False)
    )
else:
    print(
        f"No subset with tau < "
        f"{TAU_THRESHOLD:.2f} was found."
    )

print()

if not winner_frequency.empty:
    print(
        "Most frequent datasets in minimum "
        "winner-change subsets:"
    )

    print(
        winner_frequency
        .head(15)
        .to_string(index=False)
    )

print()

if not tau_frequency.empty:
    print(
        f"Most frequent datasets in minimum "
        f"tau < {TAU_THRESHOLD:.2f} subsets:"
    )

    print(
        tau_frequency
        .head(15)
        .to_string(index=False)
    )

print()
print("Done.")
