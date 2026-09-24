from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, rankdata


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================
# This file is intended to live in Code/analysis/.
# Project root is therefore two levels above this script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "Code" / "analysis" / "results" / "score_method_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Code" / "analysis" / "results" / "dataset_removal_search" / "score"


# ============================================================
# DEFAULT SEARCH SETTINGS
# ============================================================
DEFAULT_EXACT_MAX_REMOVALS = 3
DEFAULT_BEAM_WIDTH = 50
DEFAULT_MAX_REMOVALS = 10
DEFAULT_TAU_THRESHOLD = 0.90
DEFAULT_SEED = 42

# Search can stop as soon as the target has been found, but all evaluation
# rows are retained so a run can be resumed and audited.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid exact + greedy/beam search for finding small dataset-removal "
            "subsets that destabilize a TabArena score-based ranking."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--exact-max-removals",
        type=int,
        default=DEFAULT_EXACT_MAX_REMOVALS,
        help="Exhaustively evaluate all subsets up to this many removed datasets.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help="Number of partial removal sets retained during beam search.",
    )
    parser.add_argument(
        "--max-removals",
        type=int,
        default=DEFAULT_MAX_REMOVALS,
        help="Maximum removal depth for the heuristic phase.",
    )
    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=DEFAULT_TAU_THRESHOLD,
        help="Ranking-instability threshold: candidate is unstable when Kendall tau < threshold.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for tie-breaking."
    )
    parser.add_argument(
        "--no-beam",
        action="store_true",
        help="Run exact search only; useful for validating small removal depths.",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Delete the search checkpoint/results in output-dir and start over.",
    )
    return parser.parse_args()


def rank_vector(scores: np.ndarray) -> np.ndarray:
    """1-based competition rank, larger score = better."""
    # rankdata ranks ascending, so negate scores for descending ranking.
    return rankdata(-scores, method="average")


def ranking_metrics(full_ranks: np.ndarray, candidate_ranks: np.ndarray) -> dict:
    tau, tau_p = kendalltau(full_ranks, candidate_ranks)
    rho, rho_p = spearmanr(full_ranks, candidate_ranks)

    rank_diff = np.abs(candidate_ranks - full_ranks)
    return {
        "kendall_tau": float(tau),
        "kendall_p_value": float(tau_p),
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "max_rank_change": float(np.max(rank_diff)),
        "mean_rank_change": float(np.mean(rank_diff)),
        "median_rank_change": float(np.median(rank_diff)),
        "num_rank_changes": int(np.sum(rank_diff > 1e-12)),
    }


def bitmask_from_indices(indices: Sequence[int]) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << int(i)
    return mask


def indices_from_mask(mask: int, n: int) -> tuple[int, ...]:
    return tuple(i for i in range(n) if (mask >> i) & 1)


def mask_hex(mask: int) -> str:
    return f"0x{mask:x}"


def load_score_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found:\n{path}")

    df = pd.read_csv(path)
    required = {"dataset", "method", "dataset_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["dataset", "method"]].duplicated().any():
        dupes = df.loc[df[["dataset", "method"]].duplicated(), ["dataset", "method"]]
        raise ValueError(
            "Input must contain exactly one row per dataset-method pair. "
            f"Found duplicates such as:\n{dupes.head(10)}"
        )

    pivot = df.pivot(index="method", columns="dataset", values="dataset_score")
    if pivot.isna().any().any():
        missing_pairs = int(pivot.isna().sum().sum())
        raise ValueError(
            f"Score matrix contains {missing_pairs} missing dataset-method values. "
            "Resolve missingness before running subset search so every subset is comparable."
        )

    # Stable alphabetical ordering makes dataset bitmasks portable/reproducible.
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    methods = pivot.index.tolist()
    datasets = pivot.columns.tolist()
    matrix = pivot.to_numpy(dtype=np.float64)

    return matrix, methods, datasets


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_masks(path: Path) -> set[int]:
    if not path.exists():
        return set()
    masks: set[int] = set()
    # Only read the mask column; this keeps resume startup cheap.
    for chunk in pd.read_csv(path, usecols=["removed_mask"], chunksize=100_000):
        for value in chunk["removed_mask"].dropna().tolist():
            masks.add(int(str(value), 16) if str(value).startswith("0x") else int(value))
    return masks


def load_evaluation_cache(path: Path) -> dict[int, dict]:
    """Load evaluated subset rows keyed by integer bitmask for beam-search resume."""
    if not path.exists():
        return {}
    cache: dict[int, dict] = {}
    for chunk in pd.read_csv(path, chunksize=100_000):
        for row in chunk.to_dict(orient="records"):
            raw = str(row["removed_mask"])
            mask = int(raw, 16) if raw.startswith("0x") else int(raw)
            cache[mask] = row
    return cache


def evaluate_subset(
    matrix: np.ndarray,
    methods: Sequence[str],
    datasets: Sequence[str],
    full_ranks: np.ndarray,
    full_winner: str,
    removed_indices: Sequence[int],
    tau_threshold: float,
) -> dict:
    n_datasets = matrix.shape[1]
    removed_indices = tuple(sorted(removed_indices))
    removed_mask = bitmask_from_indices(removed_indices)

    full_sum = matrix.sum(axis=1)
    if removed_indices:
        removed_sum = matrix[:, removed_indices].sum(axis=1)
    else:
        removed_sum = np.zeros(matrix.shape[0], dtype=np.float64)

    remaining = n_datasets - len(removed_indices)
    candidate_scores = (full_sum - removed_sum) / remaining
    candidate_ranks = rank_vector(candidate_scores)

    metrics = ranking_metrics(full_ranks, candidate_ranks)
    winner_idx = int(np.argmax(candidate_scores))
    candidate_winner = methods[winner_idx]
    winner_changed = candidate_winner != full_winner
    tau_unstable = metrics["kendall_tau"] < tau_threshold

    top5_full = set(np.argsort(-full_sum)[:5])
    top5_candidate = set(np.argsort(-candidate_scores)[:5])

    metrics.update(
        {
            "removed_mask": mask_hex(removed_mask),
            "removed_indices": ",".join(map(str, removed_indices)),
            "removed_datasets": "|".join(datasets[i] for i in removed_indices),
            "num_removed": len(removed_indices),
            "num_remaining": remaining,
            "winner": candidate_winner,
            "full_winner": full_winner,
            "winner_changed": winner_changed,
            "tau_unstable": tau_unstable,
            "top5_overlap": len(top5_full & top5_candidate),
            "instability_score": float(1.0 - metrics["kendall_tau"]),
        }
    )
    return metrics


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def best_rows(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Best = strongest instability, then winner change, then top-5 disruption."""
    return frame.sort_values(
        ["winner_changed", "kendall_tau", "top5_overlap", "max_rank_change"],
        ascending=[False, True, True, False],
    ).head(n)


def run_exact_search(
    matrix: np.ndarray,
    methods: list[str],
    datasets: list[str],
    full_ranks: np.ndarray,
    full_winner: str,
    output_dir: Path,
    exact_max_removals: int,
    tau_threshold: float,
    already_evaluated: set[int],
) -> tuple[dict[int, dict], set[int], bool]:
    """Exhaustively search all removal subsets for depths 1..exact_max_removals."""
    evaluations_file = output_dir / "evaluations.csv"
    summaries: dict[int, dict] = {}
    n = len(datasets)

    for depth in range(1, exact_max_removals + 1):
        total = math.comb(n, depth)
        print(f"\nEXACT SEARCH — remove {depth} dataset(s): {total:,} combinations")
        depth_start = time.perf_counter()
        rows: list[dict] = []
        best_tau = float("inf")
        best_row = None
        winner_change_row = None
        tau_row = None
        evaluated_this_depth = 0

        for combo in itertools.combinations(range(n), depth):
            mask = bitmask_from_indices(combo)
            if mask in already_evaluated:
                continue

            row = evaluate_subset(
                matrix, methods, datasets, full_ranks, full_winner, combo, tau_threshold
            )
            rows.append(row)
            already_evaluated.add(mask)
            evaluated_this_depth += 1

            if row["kendall_tau"] < best_tau:
                best_tau = row["kendall_tau"]
                best_row = row
            if row["winner_changed"] and winner_change_row is None:
                winner_change_row = row
            if row["tau_unstable"] and tau_row is None:
                tau_row = row

            # Flush periodically; this is important for overnight jobs.
            if len(rows) >= 5000:
                append_rows(evaluations_file, rows)
                rows.clear()

        append_rows(evaluations_file, rows)

        elapsed = time.perf_counter() - depth_start
        # If this depth was already fully evaluated before, derive the summary
        # from the checkpoint cache instead of printing misleading None values.
        if best_row is None:
            cached_depth_rows = []
            for mask, cached in load_evaluation_cache(evaluations_file).items():
                try:
                    if int(cached["num_removed"]) == depth:
                        cached_depth_rows.append(cached)
                except (KeyError, TypeError, ValueError):
                    pass
            if cached_depth_rows:
                best_cached = min(cached_depth_rows, key=lambda r: float(r["kendall_tau"]))
                best_row = best_cached
                winner_cached = next(
                    (r for r in cached_depth_rows if str(r["winner_changed"]).lower() == "true"),
                    None,
                )
                tau_cached = next(
                    (r for r in cached_depth_rows if float(r["kendall_tau"]) < tau_threshold),
                    None,
                )
                winner_change_row = winner_change_row or winner_cached
                tau_row = tau_row or tau_cached

        depth_summary = {
            "depth": depth,
            "total_combinations": total,
            "evaluated_new": evaluated_this_depth,
            "elapsed_seconds": elapsed,
            "best_kendall_tau": None if best_row is None else float(best_row["kendall_tau"]),
            "best_removed_datasets": None if best_row is None else best_row["removed_datasets"],
            "winner_change_found": winner_change_row is not None,
            "winner_change_datasets": None
            if winner_change_row is None
            else winner_change_row["removed_datasets"],
            "tau_threshold_found": tau_row is not None,
            "tau_threshold_datasets": None if tau_row is None else tau_row["removed_datasets"],
        }
        summaries[depth] = depth_summary

        save_json(output_dir / f"exact_depth_{depth}_summary.json", depth_summary)
        print(
            f"  evaluated: {evaluated_this_depth:,} new subsets in {elapsed/60:.2f} min\n"
            f"  best tau:  {depth_summary['best_kendall_tau']}\n"
            f"  best set:  {depth_summary['best_removed_datasets']}\n"
            f"  winner change: {depth_summary['winner_change_found']}\n"
            f"  tau < {tau_threshold}: {depth_summary['tau_threshold_found']}"
        )

        # The first depth at which a criterion is met is mathematically minimal
        # for that criterion, because all subsets at smaller depths were enumerated.
        if winner_change_row is not None:
            print("  => MINIMUM WINNER-CHANGE REMOVAL BUDGET FOUND")
        if tau_row is not None:
            print("  => MINIMUM TAU-THRESHOLD REMOVAL BUDGET FOUND")

    all_found = False
    return summaries, already_evaluated, all_found


def run_beam_search(
    matrix: np.ndarray,
    methods: list[str],
    datasets: list[str],
    full_ranks: np.ndarray,
    full_winner: str,
    output_dir: Path,
    start_depth: int,
    max_removals: int,
    beam_width: int,
    tau_threshold: float,
    already_evaluated: set[int],
    seed: int,
) -> None:
    """Beam search with greedy best-first scoring of partial removal sets."""
    rng = np.random.default_rng(seed)
    n = len(datasets)

    states_file = output_dir / "beam_states.csv"
    evaluations_file = output_dir / "evaluations.csv"
    evaluation_cache = load_evaluation_cache(evaluations_file)

    # Resume from a valid beam checkpoint only when the checkpoint frontier
    # really contains states of the recorded depth. Older versions of this
    # script initialized the beam from the empty set, which shifted the
    # reported depth by (start_depth - 1). Such checkpoints must be ignored.
    checkpoint_file = output_dir / "beam_checkpoint.json"
    beam: list[tuple[int, ...]] = []
    actual_start_depth = start_depth

    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        saved_depth = int(checkpoint.get("depth", -1))
        saved_states = checkpoint.get("beam_states", [])
        valid_checkpoint = bool(saved_states) and all(
            len(state) == saved_depth for state in saved_states
        )
        if valid_checkpoint and saved_depth >= start_depth - 1:
            beam = [tuple(int(i) for i in state) for state in saved_states]
            actual_start_depth = max(start_depth, saved_depth + 1)
            print(
                f"Resuming valid beam checkpoint from depth {saved_depth}; "
                f"next depth is {actual_start_depth}."
            )
        else:
            print(
                "Ignoring incompatible beam checkpoint; rebuilding the frontier "
                "from exact-search results.\n"
            )

    # If no valid beam checkpoint exists, seed the beam from the strongest
    # evaluated subsets at exactly start_depth - 1. This guarantees that beam
    # depth k really means exactly k removed datasets.
    if not beam:
        frontier_depth = start_depth - 1
        if frontier_depth <= 0:
            beam = [tuple()]
        else:
            frontier = []
            for mask, row in evaluation_cache.items():
                try:
                    num_removed = int(row["num_removed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if num_removed != frontier_depth:
                    continue

                combo = indices_from_mask(mask, n)
                winner_changed = str(row["winner_changed"]).lower() == "true"
                tau = float(row["kendall_tau"])
                max_rank_change = float(row["max_rank_change"])
                mean_rank_change = float(row["mean_rank_change"])
                tie_break = float(rng.random()) * 1e-12
                key = (
                    int(winner_changed),
                    -tau,
                    -max_rank_change,
                    -mean_rank_change,
                    tie_break,
                )
                frontier.append((key, combo))

            frontier.sort(key=lambda x: x[0], reverse=True)
            beam = [combo for _, combo in frontier[:beam_width]]
            if beam:
                print(
                    f"Initialized beam from exact depth {frontier_depth}: "
                    f"keeping {len(beam)} frontier states.\n"
                )
            else:
                raise RuntimeError(
                    f"Cannot initialize beam: no evaluated subsets at depth {frontier_depth}. "
                    "Run the exact search through that depth first."
                )

    for depth in range(actual_start_depth, max_removals + 1):
        candidate_states: set[tuple[int, ...]] = set()
        for state in beam:
            state_set = set(state)
            for i in range(n):
                if i not in state_set:
                    candidate_states.add(tuple(sorted((*state, i))))

        print(f"\nBEAM SEARCH — depth {depth}: {len(candidate_states):,} candidate states")
        rows = []
        state_candidates = []

        for combo in candidate_states:
            mask = bitmask_from_indices(combo)
            if mask in evaluation_cache:
                row = evaluation_cache[mask]
                # CSV round-tripping turns booleans/numbers into strings/numpy scalars.
                row = dict(row)
                row["winner_changed"] = str(row["winner_changed"]).lower() == "true"
                row["tau_unstable"] = str(row["tau_unstable"]).lower() == "true"
                row["kendall_tau"] = float(row["kendall_tau"])
                row["max_rank_change"] = float(row["max_rank_change"])
                row["mean_rank_change"] = float(row["mean_rank_change"])
            else:
                row = evaluate_subset(
                    matrix, methods, datasets, full_ranks, full_winner, combo, tau_threshold
                )
                rows.append(row)
                already_evaluated.add(mask)
                evaluation_cache[mask] = row

            # Random micro-jitter only resolves exact ties deterministically.
            tie_break = float(rng.random()) * 1e-12
            ranking_key = (
                int(row["winner_changed"]),
                -row["kendall_tau"],
                -row["max_rank_change"],
                -row["mean_rank_change"],
                tie_break,
            )
            state_candidates.append((ranking_key, combo, row))

            if len(rows) >= 5000:
                append_rows(evaluations_file, rows)
                rows.clear()

        append_rows(evaluations_file, rows)

        if not state_candidates:
            print("  No new states available; beam search cannot advance.")
            break

        state_candidates.sort(key=lambda x: x[0], reverse=True)
        selected = state_candidates[:beam_width]
        beam = [combo for _, combo, _ in selected]

        state_rows = []
        for rank, (_, combo, row) in enumerate(selected, start=1):
            state_rows.append(
                {
                    "depth": depth,
                    "beam_rank": rank,
                    "removed_mask": row["removed_mask"],
                    "removed_datasets": row["removed_datasets"],
                    "kendall_tau": row["kendall_tau"],
                    "spearman_rho": row["spearman_rho"],
                    "max_rank_change": row["max_rank_change"],
                    "mean_rank_change": row["mean_rank_change"],
                    "num_rank_changes": row["num_rank_changes"],
                    "top5_overlap": row["top5_overlap"],
                    "winner": row["winner"],
                    "winner_changed": row["winner_changed"],
                    "tau_unstable": row["tau_unstable"],
                }
            )
        append_rows(states_file, state_rows)

        best = selected[0][2]
        print(
            f"  best tau: {best['kendall_tau']:.6f}; "
            f"winner={best['winner']}; "
            f"removed={best['removed_datasets']}"
        )

        winner_rows = [row for _, _, row in selected if row["winner_changed"]]
        tau_rows = [row for _, _, row in selected if row["tau_unstable"]]
        if winner_rows:
            print("  => beam found a winner-changing subset at this depth")
        if tau_rows:
            print(f"  => beam found tau < {tau_threshold} at this depth")

        checkpoint = {
            "version": 2,
            "phase": "beam",
            "depth": depth,
            "beam_width": beam_width,
            "max_removals": max_removals,
            "beam_states": [list(x) for x in beam],
            "beam_states_datasets": [
                [datasets[i] for i in x] for x in beam
            ],
        }
        save_json(output_dir / "beam_checkpoint.json", checkpoint)

        if winner_rows and tau_rows:
            print("  Both target criteria reached; stopping beam search.")
            break


def make_final_summary(output_dir: Path, tau_threshold: float) -> None:
    evaluations_file = output_dir / "evaluations.csv"
    if not evaluations_file.exists():
        return

    df = pd.read_csv(evaluations_file)
    df["num_removed"] = pd.to_numeric(df["num_removed"], errors="coerce")
    df["kendall_tau"] = pd.to_numeric(df["kendall_tau"], errors="coerce")
    df["winner_changed"] = df["winner_changed"].astype(str).str.lower().eq("true")
    df["tau_unstable"] = df["tau_unstable"].astype(str).str.lower().eq("true")

    lines = [
        "TabArena dataset-removal search — score aggregation",
        "=" * 58,
        f"Evaluated subsets: {len(df):,}",
        f"Tau threshold: {tau_threshold}",
        "",
    ]

    for label, mask in [
        ("First winner-changing subset", df["winner_changed"]),
        (f"First subset with Kendall tau < {tau_threshold}", df["tau_unstable"]),
    ]:
        found = df.loc[mask]
        lines.append(label + ":")
        if found.empty:
            lines.append("  Not found in evaluated subsets.")
        else:
            min_depth = int(found["num_removed"].min())
            candidates = found[found["num_removed"] == min_depth].sort_values("kendall_tau")
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
        df.groupby("num_removed")
        .agg(
            evaluated=("removed_mask", "size"),
            best_tau=("kendall_tau", "min"),
            winner_change_rate=("winner_changed", "mean"),
            tau_unstable_rate=("tau_unstable", "mean"),
        )
        .reset_index()
    )
    by_depth.to_csv(output_dir / "depth_summary.csv", index=False)

    best_global = df.sort_values(["num_removed", "kendall_tau"]).head(20)
    best_global.to_csv(output_dir / "most_destabilizing_subsets.csv", index=False)

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.exact_max_removals < 0:
        raise ValueError("--exact-max-removals must be >= 0")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1")
    if args.max_removals < 1:
        raise ValueError("--max-removals must be >= 1")
    if args.max_removals > 50:
        raise ValueError("Refusing max_removals > 50 for a 51-dataset benchmark.")
    if not (0.0 < args.tau_threshold <= 1.0):
        raise ValueError("--tau-threshold must be in (0, 1].")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_new:
        for filename in [
            "evaluations.csv",
            "beam_states.csv",
            "beam_checkpoint.json",
            "config.json",
            "summary.txt",
            "depth_summary.csv",
            "most_destabilizing_subsets.csv",
        ]:
            path = output_dir / filename
            if path.exists():
                path.unlink()
        for path in output_dir.glob("exact_depth_*_summary.json"):
            path.unlink()

    print("=" * 78)
    print("TABARENA DATASET-REMOVAL SEARCH — SCORE AGGREGATION")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input:        {args.input}")
    print(f"Output:       {output_dir}")
    print()

    matrix, methods, datasets = load_score_matrix(args.input)
    print(f"Methods: {len(methods)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Matrix shape: {matrix.shape}")

    # Full benchmark reference ranking.
    full_scores = matrix.mean(axis=1)
    full_ranks = rank_vector(full_scores)
    full_winner = methods[int(np.argmax(full_scores))]
    print(f"Full benchmark winner: {full_winner}")

    # Save the immutable mapping/config before any expensive search.
    save_json(
        output_dir / "config.json",
        {
            "input_file": str(args.input.resolve()),
            "project_root": str(PROJECT_ROOT),
            "n_methods": len(methods),
            "n_datasets": len(datasets),
            "methods": methods,
            "datasets": datasets,
            "exact_max_removals": args.exact_max_removals,
            "beam_width": args.beam_width,
            "max_removals": args.max_removals,
            "tau_threshold": args.tau_threshold,
            "seed": args.seed,
            "full_winner": full_winner,
        },
    )

    existing = load_existing_masks(output_dir / "evaluations.csv")
    print(f"Already evaluated subsets from checkpoint: {len(existing):,}")

    if args.exact_max_removals > 0:
        run_exact_search(
            matrix,
            methods,
            datasets,
            full_ranks,
            full_winner,
            output_dir,
            min(args.exact_max_removals, args.max_removals),
            args.tau_threshold,
            existing,
        )

    if not args.no_beam and args.max_removals > args.exact_max_removals:
        run_beam_search(
            matrix,
            methods,
            datasets,
            full_ranks,
            full_winner,
            output_dir,
            start_depth=args.exact_max_removals + 1,
            max_removals=args.max_removals,
            beam_width=args.beam_width,
            tau_threshold=args.tau_threshold,
            already_evaluated=existing,
            seed=args.seed,
        )

    make_final_summary(output_dir, args.tau_threshold)

    print("\n" + "=" * 78)
    print("SEARCH COMPLETE / CHECKPOINTED")
    print("=" * 78)
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