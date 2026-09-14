from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from bencheval.elo_utils import EloHelper


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
    / "dataset_removal_search"
    / "elo"
)


# ============================================================
# TABARENA COLUMNS
# ============================================================

METHOD_COL = "method"
DATASET_COL = "dataset"
ERROR_COL = "metric_error"
SPLIT_COL = "fold"


# ============================================================
# ELO SETTINGS
# ============================================================

INIT_RATING = 1000
SCALE = 400


# ============================================================
# SEARCH SETTINGS
# ============================================================

DEFAULT_EXACT_MAX_REMOVALS = 3
DEFAULT_BEAM_WIDTH = 50
DEFAULT_MAX_REMOVALS = 10
DEFAULT_TAU_THRESHOLD = 0.90
DEFAULT_SEED = 42


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid exact + beam search for finding small "
            "dataset-removal subsets that destabilize "
            "a TabArena Elo ranking."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--exact-max-removals",
        type=int,
        default=DEFAULT_EXACT_MAX_REMOVALS,
        help=(
            "Exhaustively evaluate all subsets up to "
            "this number of removed datasets."
        ),
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help="Number of partial states retained in beam search.",
    )

    parser.add_argument(
        "--max-removals",
        type=int,
        default=DEFAULT_MAX_REMOVALS,
        help="Maximum removal depth.",
    )

    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=DEFAULT_TAU_THRESHOLD,
        help=(
            "Benchmark-breaking threshold. "
            "Candidate is unstable when Kendall tau < threshold."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--no-beam",
        action="store_true",
        help="Run exact search only.",
    )

    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Delete existing checkpoints/results and start again.",
    )

    return parser.parse_args()


# ============================================================
# BITMASK HELPERS
# ============================================================

def bitmask_from_indices(
    indices: Sequence[int],
) -> int:
    mask = 0

    for i in indices:
        mask |= 1 << int(i)

    return mask


def mask_hex(mask: int) -> str:
    return f"0x{mask:x}"


# ============================================================
# RANKING METRICS
# ============================================================

def ranking_metrics(
    full_ranks: pd.Series,
    candidate_ranks: pd.Series,
) -> dict:

    common = full_ranks.index.intersection(
        candidate_ranks.index
    )

    full = full_ranks.loc[common]
    candidate = candidate_ranks.loc[common]

    tau, tau_p = kendalltau(
        full,
        candidate,
    )

    rho, rho_p = spearmanr(
        full,
        candidate,
    )

    rank_diff = (
        candidate - full
    ).abs()

    return {
        "kendall_tau": float(tau),
        "kendall_p_value": float(tau_p),
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "max_rank_change": float(
            rank_diff.max()
        ),
        "mean_rank_change": float(
            rank_diff.mean()
        ),
        "median_rank_change": float(
            rank_diff.median()
        ),
        "num_rank_changes": int(
            (rank_diff > 1e-12).sum()
        ),
    }


# ============================================================
# ELO
# ============================================================

def compute_elo_from_battles(
    battles: pd.DataFrame,
) -> pd.Series:

    helper = EloHelper(
        method_col="method",
        task_col="dataset",
        error_col="weight",
        split_col="fold",
    )

    ratings = helper.compute_mle_elo(
        battles=battles,
        SCALE=SCALE,
        INIT_RATING=INIT_RATING,
    )

    return ratings.sort_values(
        ascending=False
    )


def prepare_battles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        METHOD_COL,
        DATASET_COL,
        ERROR_COL,
        SPLIT_COL,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    helper = EloHelper(
        method_col=METHOD_COL,
        task_col=DATASET_COL,
        error_col=ERROR_COL,
        split_col=SPLIT_COL,
    )

    battles = helper.convert_results_to_battles(
        df
    )

    return battles


# ============================================================
# DATA LOADING
# ============================================================

def load_input(
    path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    df = pd.read_csv(path)

    datasets = sorted(
        df[DATASET_COL]
        .dropna()
        .unique()
        .tolist()
    )

    methods = sorted(
        df[METHOD_COL]
        .dropna()
        .unique()
        .tolist()
    )

    battles = prepare_battles(df)

    return (
        df,
        battles,
        methods,
        datasets,
    )


# ============================================================
# SUBSET EVALUATION
# ============================================================

def evaluate_subset(
    battles: pd.DataFrame,
    methods: Sequence[str],
    datasets: Sequence[str],
    full_ranks: pd.Series,
    full_winner: str,
    removed_indices: Sequence[int],
    tau_threshold: float,
) -> dict:

    removed_indices = tuple(
        sorted(removed_indices)
    )

    removed_datasets = tuple(
        datasets[i]
        for i in removed_indices
    )

    removed_set = set(
        removed_datasets
    )

    # --------------------------------------------------------
    # Remove all pairwise battles originating from removed
    # datasets.
    # --------------------------------------------------------

    candidate_battles = battles[
        ~battles["dataset"].isin(
            removed_set
        )
    ]

    # --------------------------------------------------------
    # Recompute Elo
    # --------------------------------------------------------

    candidate_ratings = (
        compute_elo_from_battles(
            candidate_battles
        )
    )

    candidate_ranks = (
        candidate_ratings.rank(
            ascending=False,
            method="average",
        )
    )

    metrics = ranking_metrics(
        full_ranks,
        candidate_ranks,
    )

    candidate_winner = (
        candidate_ratings.index[0]
    )

    winner_changed = (
        candidate_winner != full_winner
    )

    tau_unstable = (
        metrics["kendall_tau"]
        < tau_threshold
    )

    # --------------------------------------------------------
    # Top-5 overlap
    # --------------------------------------------------------

    full_top5 = set(
        full_ranks.nsmallest(5).index
    )

    candidate_top5 = set(
        candidate_ranks.nsmallest(5).index
    )

    # --------------------------------------------------------
    # Elo movement
    # --------------------------------------------------------

    full_winner_elo_after = float(
        candidate_ratings.loc[full_winner]
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    metrics.update(
        {
            "removed_mask": mask_hex(
                bitmask_from_indices(
                    removed_indices
                )
            ),

            "removed_indices": ",".join(
                map(
                    str,
                    removed_indices,
                )
            ),

            "removed_datasets": "|".join(
                removed_datasets
            ),

            "num_removed": len(
                removed_indices
            ),

            "num_remaining": (
                len(datasets)
                - len(removed_indices)
            ),

            "winner": candidate_winner,

            "full_winner": full_winner,

            "winner_changed": winner_changed,

            "tau_unstable": tau_unstable,

            "top5_overlap": len(
                full_top5
                & candidate_top5
            ),

            "instability_score": float(
                1.0
                - metrics["kendall_tau"]
            ),

            "full_winner_elo": float(
                full_ratings.loc[
                    full_winner
                ]
            ),

            "full_winner_elo_after": (
                full_winner_elo_after
            ),

            "full_winner_elo_delta": (
                full_winner_elo_after
                - float(
                    full_ratings.loc[
                        full_winner
                    ]
                )
            ),

            "winner_elo": float(
                candidate_ratings.iloc[0]
            ),

            "n_battles_remaining": int(
                len(candidate_battles)
            ),
        }
    )

    return metrics


# ============================================================
# OUTPUT HELPERS
# ============================================================

def append_rows(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        return

    frame = pd.DataFrame(
        rows
    )

    write_header = not path.exists()

    frame.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


def save_json(
    path: Path,
    payload: dict,
) -> None:

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_existing_masks(
    path: Path,
) -> set[int]:

    if not path.exists():
        return set()

    masks = set()

    for chunk in pd.read_csv(
        path,
        usecols=["removed_mask"],
        chunksize=100_000,
    ):

        for value in (
            chunk["removed_mask"]
            .dropna()
            .tolist()
        ):

            raw = str(value)

            if raw.startswith("0x"):
                masks.add(
                    int(raw, 16)
                )
            else:
                masks.add(
                    int(raw)
                )

    return masks


def load_evaluation_cache(
    path: Path,
) -> dict[int, dict]:

    if not path.exists():
        return {}

    cache = {}

    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
    ):

        for row in chunk.to_dict(
            orient="records"
        ):

            raw = str(
                row["removed_mask"]
            )

            mask = (
                int(raw, 16)
                if raw.startswith("0x")
                else int(raw)
            )

            cache[mask] = row

    return cache


# ============================================================
# EXACT SEARCH
# ============================================================

def run_exact_search(
    battles: pd.DataFrame,
    methods: list[str],
    datasets: list[str],
    full_ranks: pd.Series,
    full_winner: str,
    output_dir: Path,
    exact_max_removals: int,
    tau_threshold: float,
    already_evaluated: set[int],
) -> None:

    evaluations_file = (
        output_dir
        / "evaluations.csv"
    )

    n = len(datasets)

    for depth in range(
        1,
        exact_max_removals + 1,
    ):

        total = math.comb(
            n,
            depth,
        )

        print(
            f"\nEXACT SEARCH — remove "
            f"{depth} dataset(s): "
            f"{total:,} combinations"
        )

        depth_start = (
            time.perf_counter()
        )

        rows = []

        best_row = None
        winner_change_row = None
        tau_row = None

        best_tau = float("inf")
        evaluated_this_depth = 0

        for combo in itertools.combinations(
            range(n),
            depth,
        ):

            mask = bitmask_from_indices(
                combo
            )

            if mask in already_evaluated:
                continue

            row = evaluate_subset(
                battles,
                methods,
                datasets,
                full_ranks,
                full_winner,
                combo,
                tau_threshold,
            )

            rows.append(row)

            already_evaluated.add(mask)

            evaluated_this_depth += 1

            if (
                row["kendall_tau"]
                < best_tau
            ):

                best_tau = (
                    row["kendall_tau"]
                )

                best_row = row

            if (
                row["winner_changed"]
                and winner_change_row is None
            ):

                winner_change_row = row

            if (
                row["tau_unstable"]
                and tau_row is None
            ):

                tau_row = row

            # Flush regularly.
            if len(rows) >= 250:
                append_rows(
                    evaluations_file,
                    rows,
                )
                rows.clear()

        append_rows(
            evaluations_file,
            rows,
        )

        elapsed = (
            time.perf_counter()
            - depth_start
        )

        depth_summary = {
            "depth": depth,
            "total_combinations": total,
            "evaluated_new": evaluated_this_depth,
            "elapsed_seconds": elapsed,
            "best_kendall_tau": (
                None
                if best_row is None
                else best_row[
                    "kendall_tau"
                ]
            ),
            "best_removed_datasets": (
                None
                if best_row is None
                else best_row[
                    "removed_datasets"
                ]
            ),
            "winner_change_found": (
                winner_change_row
                is not None
            ),
            "winner_change_datasets": (
                None
                if winner_change_row is None
                else winner_change_row[
                    "removed_datasets"
                ]
            ),
            "tau_threshold_found": (
                tau_row is not None
            ),
            "tau_threshold_datasets": (
                None
                if tau_row is None
                else tau_row[
                    "removed_datasets"
                ]
            ),
        }

        save_json(
            output_dir
            / f"exact_depth_{depth}_summary.json",
            depth_summary,
        )

        print(
            f"  evaluated: "
            f"{evaluated_this_depth:,} "
            f"new subsets in "
            f"{elapsed / 60:.2f} min"
        )

        print(
            f"  best tau: "
            f"{depth_summary['best_kendall_tau']}"
        )

        print(
            f"  best set: "
            f"{depth_summary['best_removed_datasets']}"
        )

        print(
            f"  winner change: "
            f"{depth_summary['winner_change_found']}"
        )

        print(
            f"  tau < {tau_threshold}: "
            f"{depth_summary['tau_threshold_found']}"
        )


# ============================================================
# BEAM SEARCH
# ============================================================

def run_beam_search(
    battles: pd.DataFrame,
    methods: list[str],
    datasets: list[str],
    full_ranks: pd.Series,
    full_winner: str,
    output_dir: Path,
    start_depth: int,
    max_removals: int,
    beam_width: int,
    tau_threshold: float,
    already_evaluated: set[int],
    seed: int,
) -> None:

    rng = np.random.default_rng(
        seed
    )

    n = len(datasets)

    states_file = (
        output_dir
        / "beam_states.csv"
    )

    evaluations_file = (
        output_dir
        / "evaluations.csv"
    )

    evaluation_cache = (
        load_evaluation_cache(
            evaluations_file
        )
    )

    checkpoint_file = (
        output_dir
        / "beam_checkpoint.json"
    )

    beam = [tuple()]

    actual_start_depth = (
        start_depth
    )

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    if checkpoint_file.exists():

        checkpoint = json.loads(
            checkpoint_file.read_text(
                encoding="utf-8"
            )
        )

        saved_depth = int(
            checkpoint.get(
                "depth",
                0,
            )
        )

        saved_states = (
            checkpoint.get(
                "beam_states",
                [],
            )
        )

        if (
            saved_depth >= start_depth
            and saved_states
        ):

            beam = [
                tuple(
                    int(i)
                    for i in state
                )
                for state in saved_states
            ]

            actual_start_depth = (
                saved_depth + 1
            )

            print(
                "Resuming beam from "
                f"depth {saved_depth}; "
                f"next depth is "
                f"{actual_start_depth}."
            )

    # --------------------------------------------------------
    # Beam iterations
    # --------------------------------------------------------

    for depth in range(
        actual_start_depth,
        max_removals + 1,
    ):

        candidate_states = set()

        for state in beam:

            state_set = set(
                state
            )

            for i in range(n):

                if i not in state_set:

                    candidate_states.add(
                        tuple(
                            sorted(
                                (
                                    *state,
                                    i,
                                )
                            )
                        )
                    )

        print(
            f"\nBEAM SEARCH — depth "
            f"{depth}: "
            f"{len(candidate_states):,} "
            f"candidate states"
        )

        rows = []
        state_candidates = []

        for combo in candidate_states:

            mask = bitmask_from_indices(
                combo
            )

            if mask in evaluation_cache:

                row = dict(
                    evaluation_cache[
                        mask
                    ]
                )

                # CSV stores types as strings.
                row["winner_changed"] = (
                    str(
                        row[
                            "winner_changed"
                        ]
                    ).lower()
                    == "true"
                )

                row["tau_unstable"] = (
                    str(
                        row[
                            "tau_unstable"
                        ]
                    ).lower()
                    == "true"
                )

                for key in [
                    "kendall_tau",
                    "max_rank_change",
                    "mean_rank_change",
                ]:

                    row[key] = float(
                        row[key]
                    )

            else:

                row = evaluate_subset(
                    battles,
                    methods,
                    datasets,
                    full_ranks,
                    full_winner,
                    combo,
                    tau_threshold,
                )

                rows.append(row)

                already_evaluated.add(
                    mask
                )

                evaluation_cache[
                    mask
                ] = row

            tie_break = (
                float(rng.random())
                * 1e-12
            )

            ranking_key = (
                int(
                    row[
                        "winner_changed"
                    ]
                ),
                -row["kendall_tau"],
                -row["max_rank_change"],
                -row["mean_rank_change"],
                tie_break,
            )

            state_candidates.append(
                (
                    ranking_key,
                    combo,
                    row,
                )
            )

        append_rows(
            evaluations_file,
            rows,
        )

        if not state_candidates:
            print(
                "No new states available."
            )
            break

        state_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        selected = (
            state_candidates[
                :beam_width
            ]
        )

        beam = [
            combo
            for _, combo, _ in selected
        ]

        state_rows = []

        for rank, (
            _,
            combo,
            row,
        ) in enumerate(
            selected,
            start=1,
        ):

            state_rows.append(
                {
                    "depth": depth,
                    "beam_rank": rank,
                    "removed_mask": row[
                        "removed_mask"
                    ],
                    "removed_datasets": row[
                        "removed_datasets"
                    ],
                    "kendall_tau": row[
                        "kendall_tau"
                    ],
                    "spearman_rho": row[
                        "spearman_rho"
                    ],
                    "max_rank_change": row[
                        "max_rank_change"
                    ],
                    "mean_rank_change": row[
                        "mean_rank_change"
                    ],
                    "num_rank_changes": row[
                        "num_rank_changes"
                    ],
                    "top5_overlap": row[
                        "top5_overlap"
                    ],
                    "winner": row[
                        "winner"
                    ],
                    "winner_changed": row[
                        "winner_changed"
                    ],
                    "tau_unstable": row[
                        "tau_unstable"
                    ],
                }
            )

        append_rows(
            states_file,
            state_rows,
        )

        best = selected[0][2]

        print(
            f"  best tau: "
            f"{best['kendall_tau']:.6f}; "
            f"winner={best['winner']}; "
            f"removed="
            f"{best['removed_datasets']}"
        )

        winner_rows = [
            row
            for _, _, row in selected
            if row["winner_changed"]
        ]

        tau_rows = [
            row
            for _, _, row in selected
            if row["tau_unstable"]
        ]

        if winner_rows:

            print(
                "  => beam found a "
                "winner-changing subset"
            )

        if tau_rows:

            print(
                f"  => beam found "
                f"tau < {tau_threshold}"
            )

        checkpoint = {
            "phase": "beam",
            "depth": depth,
            "beam_width": beam_width,
            "max_removals": max_removals,
            "beam_states": [
                list(x)
                for x in beam
            ],
            "beam_states_datasets": [
                [
                    datasets[i]
                    for i in x
                ]
                for x in beam
            ],
        }

        save_json(
            checkpoint_file,
            checkpoint,
        )

        if (
            winner_rows
            and tau_rows
        ):

            print(
                "  Both target criteria "
                "reached; stopping."
            )

            break


# ============================================================
# FINAL SUMMARY
# ============================================================

def make_final_summary(
    output_dir: Path,
    tau_threshold: float,
) -> None:

    evaluations_file = (
        output_dir
        / "evaluations.csv"
    )

    if not evaluations_file.exists():
        return

    df = pd.read_csv(
        evaluations_file
    )

    df["num_removed"] = pd.to_numeric(
        df["num_removed"],
        errors="coerce",
    )

    df["kendall_tau"] = pd.to_numeric(
        df["kendall_tau"],
        errors="coerce",
    )

    df["winner_changed"] = (
        df["winner_changed"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["tau_unstable"] = (
        df["tau_unstable"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    lines = [
        "TabArena dataset-removal search — Elo aggregation",
        "=" * 60,
        f"Evaluated subsets: {len(df):,}",
        f"Tau threshold: {tau_threshold}",
        "",
    ]

    for label, mask in [
        (
            "First winner-changing subset",
            df["winner_changed"],
        ),
        (
            f"First subset with Kendall tau < {tau_threshold}",
            df["tau_unstable"],
        ),
    ]:

        found = df.loc[mask]

        lines.append(
            label + ":"
        )

        if found.empty:

            lines.append(
                "  Not found in evaluated subsets."
            )

        else:

            min_depth = int(
                found[
                    "num_removed"
                ].min()
            )

            candidates = (
                found[
                    found["num_removed"]
                    == min_depth
                ]
                .sort_values(
                    "kendall_tau"
                )
            )

            best = candidates.iloc[0]

            lines.extend(
                [
                    f"  minimum removals = {min_depth}",
                    f"  removed datasets = {best['removed_datasets']}",
                    f"  Kendall tau = {best['kendall_tau']:.6f}",
                    f"  winner = {best['winner']}",
                ]
            )

        lines.append("")

    by_depth = (
        df.groupby(
            "num_removed"
        )
        .agg(
            evaluated=(
                "removed_mask",
                "size",
            ),
            best_tau=(
                "kendall_tau",
                "min",
            ),
            winner_change_rate=(
                "winner_changed",
                "mean",
            ),
            tau_unstable_rate=(
                "tau_unstable",
                "mean",
            ),
        )
        .reset_index()
    )

    by_depth.to_csv(
        output_dir
        / "depth_summary.csv",
        index=False,
    )

    best_global = (
        df.sort_values(
            [
                "num_removed",
                "kendall_tau",
            ]
        )
        .head(20)
    )

    best_global.to_csv(
        output_dir
        / "most_destabilizing_subsets.csv",
        index=False,
    )

    (
        output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    if args.exact_max_removals < 0:
        raise ValueError(
            "--exact-max-removals must be >= 0"
        )

    if args.beam_width < 1:
        raise ValueError(
            "--beam-width must be >= 1"
        )

    if args.max_removals < 1:
        raise ValueError(
            "--max-removals must be >= 1"
        )

    if args.max_removals > 50:
        raise ValueError(
            "Refusing max_removals > 50 "
            "for a 51-dataset benchmark."
        )

    if not (
        0.0
        < args.tau_threshold
        <= 1.0
    ):
        raise ValueError(
            "--tau-threshold must be in (0, 1]."
        )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if args.force_new:

        filenames = [
            "evaluations.csv",
            "beam_states.csv",
            "beam_checkpoint.json",
            "config.json",
            "summary.txt",
            "depth_summary.csv",
            "most_destabilizing_subsets.csv",
        ]

        for filename in filenames:

            path = (
                output_dir
                / filename
            )

            if path.exists():
                path.unlink()

        for path in output_dir.glob(
            "exact_depth_*_summary.json"
        ):

            path.unlink()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "TABARENA DATASET-REMOVAL SEARCH "
        "— ELO AGGREGATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Input:        {args.input}"
    )

    print(
        f"Output:       {output_dir}"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df, battles, methods, datasets = (
        load_input(args.input)
    )

    print(
        f"Rows:         {len(df):,}"
    )

    print(
        f"Methods:      {len(methods)}"
    )

    print(
        f"Datasets:     {len(datasets)}"
    )

    print(
        f"Battles:      {len(battles):,}"
    )

    # --------------------------------------------------------
    # FULL ELO
    # --------------------------------------------------------

    global full_ratings

    print(
        "\nComputing full benchmark Elo..."
    )

    full_ratings = (
        compute_elo_from_battles(
            battles
        )
    )

    full_ranks = (
        full_ratings.rank(
            ascending=False,
            method="average",
        )
    )

    full_winner = (
        full_ratings.index[0]
    )

    print(
        f"Full benchmark winner: "
        f"{full_winner}"
    )

    print(
        f"Full winner Elo: "
        f"{full_ratings.iloc[0]:.6f}"
    )

    print()

    print("Top 10:")

    top10 = pd.DataFrame(
        {
            "rank": np.arange(
                1,
                min(10, len(full_ratings))
                + 1,
            ),
            "method": (
                full_ratings
                .head(10)
                .index
            ),
            "elo": (
                full_ratings
                .head(10)
                .values
            ),
        }
    )

    print(
        top10.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    save_json(
        output_dir
        / "config.json",
        {
            "input_file": str(
                args.input.resolve()
            ),
            "project_root": str(
                PROJECT_ROOT
            ),
            "aggregation": "Elo",
            "elo_algorithm": "bencheval.compute_mle_elo",
            "init_rating": INIT_RATING,
            "scale": SCALE,
            "n_methods": len(methods),
            "n_datasets": len(datasets),
            "n_battles": len(battles),
            "methods": methods,
            "datasets": datasets,
            "exact_max_removals": (
                args.exact_max_removals
            ),
            "beam_width": args.beam_width,
            "max_removals": (
                args.max_removals
            ),
            "tau_threshold": (
                args.tau_threshold
            ),
            "seed": args.seed,
            "full_winner": full_winner,
            "full_winner_elo": float(
                full_ratings.iloc[0]
            ),
        },
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    existing = (
        load_existing_masks(
            output_dir
            / "evaluations.csv"
        )
    )

    print(
        f"\nAlready evaluated subsets: "
        f"{len(existing):,}"
    )

    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    if args.exact_max_removals > 0:

        run_exact_search(
            battles=battles,
            methods=methods,
            datasets=datasets,
            full_ranks=full_ranks,
            full_winner=full_winner,
            output_dir=output_dir,
            exact_max_removals=min(
                args.exact_max_removals,
                args.max_removals,
            ),
            tau_threshold=args.tau_threshold,
            already_evaluated=existing,
        )

    # --------------------------------------------------------
    # BEAM
    # --------------------------------------------------------

    if (
        not args.no_beam
        and args.max_removals
        > args.exact_max_removals
    ):

        run_beam_search(
            battles=battles,
            methods=methods,
            datasets=datasets,
            full_ranks=full_ranks,
            full_winner=full_winner,
            output_dir=output_dir,
            start_depth=(
                args.exact_max_removals
                + 1
            ),
            max_removals=args.max_removals,
            beam_width=args.beam_width,
            tau_threshold=args.tau_threshold,
            already_evaluated=existing,
            seed=args.seed,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    make_final_summary(
        output_dir,
        args.tau_threshold,
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SEARCH COMPLETE / CHECKPOINTED"
    )

    print(
        "=" * 78
    )

    for path in [
        output_dir / "evaluations.csv",
        output_dir / "depth_summary.csv",
        output_dir / "most_destabilizing_subsets.csv",
        output_dir / "summary.txt",
    ]:

        if path.exists():
            print(path)


if __name__ == "__main__":
    main()