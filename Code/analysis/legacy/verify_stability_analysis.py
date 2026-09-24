from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Code" / "analysis" / "results"
REMOVAL = RESULTS / "dataset_removal_search" / "score"
STABILITY = RESULTS / "stability_dimensions"

DEPTH_SUMMARY = REMOVAL / "depth_summary.csv"
LODO = RESULTS / "lodo_score_results.csv"
JOINT = STABILITY / "joint_removal_stability_by_depth.csv"
SUMMARY = STABILITY / "stability_dimensions_summary.csv"


def assert_close(a: float, b: float, tol: float = 1e-12) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"Values differ: {a} vs {b}")


def main() -> None:
    print("TABARENA STABILITY ANALYSIS REPRODUCIBILITY CHECK")
    print("=" * 62)

    for path in [DEPTH_SUMMARY, LODO]:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Input OK: {path.relative_to(ROOT)}")

    depth = pd.read_csv(DEPTH_SUMMARY)
    depth["num_removed"] = pd.to_numeric(depth["num_removed"], errors="raise").astype(int)
    depth["best_tau"] = pd.to_numeric(depth["best_tau"], errors="raise")
    depth["evaluated"] = pd.to_numeric(depth["evaluated"], errors="raise")
    depth["winner_change_rate"] = pd.to_numeric(depth["winner_change_rate"], errors="raise")
    depth["tau_unstable_rate"] = pd.to_numeric(depth["tau_unstable_rate"], errors="raise")

    # Reconstruct event counts exactly as the laptop analysis does.
    depth["winner_changes_recomputed"] = (
        depth["winner_change_rate"] * depth["evaluated"]
    ).round().astype(int)
    depth["tau_breaks_recomputed"] = (
        depth["tau_unstable_rate"] * depth["evaluated"]
    ).round().astype(int)

    # Basic combinatorial sanity check for exact depths.
    # The search has 51 datasets, so depth d has C(51,d) possible subsets.
    for _, row in depth.iterrows():
        d = int(row["num_removed"])
        expected = math.comb(51, d)
        evaluated = int(row["evaluated"])
        if evaluated > expected:
            raise AssertionError(
                f"Depth {d}: evaluated={evaluated} exceeds C(51,{d})={expected}"
            )

    winner_rows = depth.loc[depth["winner_changes_recomputed"] > 0, "num_removed"]
    tau_rows = depth.loc[depth["tau_breaks_recomputed"] > 0, "num_removed"]

    min_winner = int(winner_rows.min()) if not winner_rows.empty else None
    min_tau = int(tau_rows.min()) if not tau_rows.empty else None

    lodo = pd.read_csv(LODO)
    lodo["kendall_tau"] = pd.to_numeric(lodo["kendall_tau"], errors="raise")
    lodo["max_rank_change"] = pd.to_numeric(lodo["max_rank_change"], errors="raise")
    lodo["num_rank_changes"] = pd.to_numeric(lodo["num_rank_changes"], errors="raise")

    lodo_winner_changes = (
        lodo["winner_changed"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
        if "winner_changed" in lodo.columns
        else 0
    )
    lodo_min_tau = float(lodo["kendall_tau"].min())

    print("\nDepth summary:")
    print(
        depth[
            [
                "num_removed",
                "evaluated",
                "best_tau",
                "winner_changes_recomputed",
                "tau_breaks_recomputed",
            ]
        ].to_string(index=False)
    )

    print("\nIndependent conclusions:")
    print(f"  Minimum removals with winner change: {min_winner}")
    print(f"  Minimum removals with Kendall tau < 0.90: {min_tau}")
    print(f"  LODO winner changes: {int(lodo_winner_changes)}")
    print(f"  LODO minimum Kendall tau: {lodo_min_tau:.12f}")

    if min_winner != 5:
        raise AssertionError(f"Expected minimum winner-change depth 5, got {min_winner}")
    if min_tau != 6:
        raise AssertionError(f"Expected minimum tau-break depth 6, got {min_tau}")
    if int(lodo_winner_changes) != 0:
        raise AssertionError("Expected zero score-Lodo winner changes")
    assert_close(lodo_min_tau, 0.970402, tol=1e-5)

    # Compare the generated machine-readable stability summary when present.
    if SUMMARY.exists():
        s = pd.read_csv(SUMMARY).iloc[0]
        stored_winner = int(float(s["min_removals_winner_change"]))
        stored_tau = int(float(s["min_removals_tau_below_0_90"]))
        if stored_winner != min_winner or stored_tau != min_tau:
            raise AssertionError("Generated stability summary disagrees with independent reconstruction")

    print("\nPASS: stored removal-search summaries reproduce the headline stability claims.")


if __name__ == "__main__":
    main()
