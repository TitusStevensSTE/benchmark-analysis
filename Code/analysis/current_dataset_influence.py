from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# PROJECT PATHS
# ============================================================


RESULTS_ROOT = RESULTS_DIR

LODO_FILE = RESULTS_ROOT / "lodo_score_results.csv"

INFLUENCE_ROOT = (
    RESULTS_ROOT
    / "dataset_removal_search"
    / "score"
    / "influence"
)

WINNER_FREQ_FILE = (
    INFLUENCE_ROOT / "dataset_winner_change_frequency.csv"
)

TAU_BREAK_FREQ_FILE = (
    INFLUENCE_ROOT / "dataset_tau_break_frequency.csv"
)

OUTPUT_ROOT = RESULTS_ROOT / "dataset_influence_final"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TAU_BREAK_THRESHOLD = 0.90

# Minimum adversarial-removal depths found by the exact search.
MIN_WINNER_CHANGE_DEPTH = 5
MIN_TAU_BREAK_DEPTH = 6


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, description):
    """
    Find the first matching column from a list of candidate names.
    Matching is case-insensitive.
    """
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def load_frequency_file(path, frequency_name):
    """
    Load a dataset-frequency file and normalize it to:
        dataset
        <frequency_name>
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    dataset_col = find_column(
        df,
        ["dataset", "dataset_name"],
        f"dataset column in {path.name}",
    )

    frequency_col = find_column(
        df,
        [
            "frequency",
            "winner_change_frequency",
            "tau_break_frequency",
            "rate",
            "proportion",
        ],
        f"frequency column in {path.name}",
    )

    out = df[[dataset_col, frequency_col]].copy()
    out.columns = ["dataset", frequency_name]

    out["dataset"] = out["dataset"].astype(str)
    out[frequency_name] = pd.to_numeric(
        out[frequency_name],
        errors="coerce",
    )

    # Remove accidental duplicates conservatively.
    out = (
        out.groupby("dataset", as_index=False)[frequency_name]
        .max()
    )

    return out


def correlation_row(df, x_col, y_col):
    """
    Compute Pearson-free rank correlations between two columns.
    Returns both Spearman and Kendall.
    """

    tmp = df[[x_col, y_col]].dropna()

    if len(tmp) < 3:
        return {
            "x": x_col,
            "y": y_col,
            "n": len(tmp),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
        }

    spearman = spearmanr(
        tmp[x_col],
        tmp[y_col],
    )

    kendall = kendalltau(
        tmp[x_col],
        tmp[y_col],
    )

    return {
        "x": x_col,
        "y": y_col,
        "n": len(tmp),
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "kendall_tau": kendall.statistic,
        "kendall_p": kendall.pvalue,
    }


def top_k_overlap(df, col_a, col_b, k_values=(5, 10, 15)):
    """
    Compute overlap between the top-k datasets according to two signals.
    """

    rows = []

    for k in k_values:
        top_a = set(
            df.nlargest(k, col_a)["dataset"]
        )

        top_b = set(
            df.nlargest(k, col_b)["dataset"]
        )

        intersection = top_a & top_b
        union = top_a | top_b

        rows.append(
            {
                "signal_a": col_a,
                "signal_b": col_b,
                "k": k,
                "overlap_count": len(intersection),
                "jaccard": (
                    len(intersection) / len(union)
                    if union
                    else np.nan
                ),
                "datasets_in_both": "|".join(
                    sorted(intersection)
                ),
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET INFLUENCE ANALYSIS")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print("\nOutput directory:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # 1. LOAD FULL LODO RESULTS
    # --------------------------------------------------------

    if not LODO_FILE.exists():
        raise FileNotFoundError(
            f"Missing LODO results file:\n{LODO_FILE}"
        )

    lodo = pd.read_csv(LODO_FILE)

    dataset_col = find_column(
        lodo,
        ["dataset_removed", "dataset"],
        "dataset identifier in LODO results",
    )

    tau_col = find_column(
        lodo,
        ["kendall_tau", "kendall", "tau"],
        "Kendall tau in LODO results",
    )

    max_rank_col = find_column(
        lodo,
        ["max_rank_change"],
        "maximum rank change in LODO results",
    )

    num_rank_col = find_column(
        lodo,
        ["num_rank_changes"],
        "number of changed ranks in LODO results",
    )

    lodo = lodo[
        [
            dataset_col,
            tau_col,
            max_rank_col,
            num_rank_col,
        ]
    ].copy()

    lodo.columns = [
        "dataset",
        "kendall_tau",
        "max_rank_change",
        "num_rank_changes",
    ]

    lodo["dataset"] = lodo["dataset"].astype(str)

    for col in [
        "kendall_tau",
        "max_rank_change",
        "num_rank_changes",
    ]:
        lodo[col] = pd.to_numeric(
            lodo[col],
            errors="coerce",
        )

    # Instability increases when Kendall tau decreases.
    lodo["lodo_instability"] = (
        1.0 - lodo["kendall_tau"]
    )

    # --------------------------------------------------------
    # 2. LOAD ADVERSARIAL WINNER-CHANGE FREQUENCY
    # --------------------------------------------------------

    winner_freq = load_frequency_file(
        WINNER_FREQ_FILE,
        "winner_change_frequency",
    )

    # --------------------------------------------------------
    # 3. LOAD ADVERSARIAL TAU-BREAK FREQUENCY
    # --------------------------------------------------------

    tau_break_freq = load_frequency_file(
        TAU_BREAK_FREQ_FILE,
        "tau_break_frequency",
    )

    # --------------------------------------------------------
    # 4. MERGE EVERYTHING
    # --------------------------------------------------------

    influence = lodo.merge(
        winner_freq,
        on="dataset",
        how="left",
    )

    influence = influence.merge(
        tau_break_freq,
        on="dataset",
        how="left",
    )

    # Datasets that never occur in the corresponding minimum
    # adversarial set receive frequency 0.
    influence[
        [
            "winner_change_frequency",
            "tau_break_frequency",
        ]
    ] = influence[
        [
            "winner_change_frequency",
            "tau_break_frequency",
        ]
    ].fillna(0.0)

    # --------------------------------------------------------
    # 5. ADD RANKS
    # --------------------------------------------------------

    influence["rank_lodo_instability"] = (
        influence["lodo_instability"]
        .rank(method="min", ascending=False)
    )

    influence["rank_winner_change"] = (
        influence["winner_change_frequency"]
        .rank(method="min", ascending=False)
    )

    influence["rank_tau_break"] = (
        influence["tau_break_frequency"]
        .rank(method="min", ascending=False)
    )

    # Mean rank is only a descriptive aggregation of the
    # three separately meaningful rankings. It is NOT treated
    # as a causal or theoretically defined "influence score".
    influence["mean_rank"] = influence[
        [
            "rank_lodo_instability",
            "rank_winner_change",
            "rank_tau_break",
        ]
    ].mean(axis=1)

    influence = influence.sort_values(
        [
            "mean_rank",
            "rank_lodo_instability",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 6. SAVE COMPLETE DATASET TABLE
    # --------------------------------------------------------

    complete_file = (
        OUTPUT_ROOT / "dataset_influence_complete.csv"
    )

    influence.to_csv(
        complete_file,
        index=False,
    )

    # --------------------------------------------------------
    # 7. CORRELATIONS
    # --------------------------------------------------------

    correlation_pairs = [
        (
            "lodo_instability",
            "winner_change_frequency",
        ),
        (
            "lodo_instability",
            "tau_break_frequency",
        ),
        (
            "winner_change_frequency",
            "tau_break_frequency",
        ),
        (
            "lodo_instability",
            "max_rank_change",
        ),
        (
            "lodo_instability",
            "num_rank_changes",
        ),
    ]

    correlation_rows = []

    for x_col, y_col in correlation_pairs:
        correlation_rows.append(
            correlation_row(
                influence,
                x_col,
                y_col,
            )
        )

    correlations = pd.DataFrame(
        correlation_rows
    )

    correlations_file = (
        OUTPUT_ROOT / "influence_correlations.csv"
    )

    correlations.to_csv(
        correlations_file,
        index=False,
    )

    # --------------------------------------------------------
    # 8. TOP-K OVERLAP
    # --------------------------------------------------------

    signal_pairs = [
        (
            "lodo_instability",
            "winner_change_frequency",
        ),
        (
            "lodo_instability",
            "tau_break_frequency",
        ),
        (
            "winner_change_frequency",
            "tau_break_frequency",
        ),
    ]

    overlap_rows = []

    for col_a, col_b in signal_pairs:
        overlap_rows.extend(
            top_k_overlap(
                influence,
                col_a,
                col_b,
            )
        )

    overlaps = pd.DataFrame(
        overlap_rows
    )

    overlap_file = (
        OUTPUT_ROOT / "influence_topk_overlap.csv"
    )

    overlaps.to_csv(
        overlap_file,
        index=False,
    )

    # --------------------------------------------------------
    # 9. RANKED TABLE FOR THESIS USE
    # --------------------------------------------------------

    ranked_columns = [
        "dataset",
        "kendall_tau",
        "lodo_instability",
        "max_rank_change",
        "num_rank_changes",
        "winner_change_frequency",
        "tau_break_frequency",
        "rank_lodo_instability",
        "rank_winner_change",
        "rank_tau_break",
        "mean_rank",
    ]

    ranked = influence[
        ranked_columns
    ].copy()

    ranked_file = (
        OUTPUT_ROOT / "dataset_influence_ranked.csv"
    )

    ranked.to_csv(
        ranked_file,
        index=False,
    )

    # --------------------------------------------------------
    # 10. SUMMARY
    # --------------------------------------------------------

    summary_file = OUTPUT_ROOT / "summary.txt"

    lines = []

    lines.append(
        "Dataset Influence Analysis Summary"
    )
    lines.append(
        "=" * 50
    )
    lines.append("")

    lines.append(
        f"Number of datasets: {len(influence)}"
    )

    lines.append(
        f"Minimum winner-change depth: "
        f"{MIN_WINNER_CHANGE_DEPTH}"
    )

    lines.append(
        f"Minimum tau-break depth: "
        f"{MIN_TAU_BREAK_DEPTH}"
    )

    lines.append(
        f"Tau-break threshold: "
        f"{TAU_BREAK_THRESHOLD:.2f}"
    )

    lines.append("")

    lines.append(
        "Top 10 datasets by LODO instability:"
    )

    for _, row in influence.nlargest(
        10,
        "lodo_instability",
    ).iterrows():

        lines.append(
            f"  {row['dataset']}: "
            f"instability={row['lodo_instability']:.6f}, "
            f"tau={row['kendall_tau']:.6f}"
        )

    lines.append("")

    lines.append(
        "Top 10 datasets by minimum-depth winner-change frequency:"
    )

    for _, row in influence.nlargest(
        10,
        "winner_change_frequency",
    ).iterrows():

        lines.append(
            f"  {row['dataset']}: "
            f"{row['winner_change_frequency']:.6f}"
        )

    lines.append("")

    lines.append(
        "Top 10 datasets by minimum-depth tau-break frequency:"
    )

    for _, row in influence.nlargest(
        10,
        "tau_break_frequency",
    ).iterrows():

        lines.append(
            f"  {row['dataset']}: "
            f"{row['tau_break_frequency']:.6f}"
        )

    lines.append("")

    lines.append(
        "Rank correlations:"
    )

    for _, row in correlations.iterrows():

        lines.append(
            f"  {row['x']} vs {row['y']}: "
            f"Spearman rho={row['spearman_rho']:.4f} "
            f"(p={row['spearman_p']:.4g}), "
            f"Kendall tau={row['kendall_tau']:.4f} "
            f"(p={row['kendall_p']:.4g}), "
            f"n={row['n']}"
        )

    lines.append("")
    lines.append(
        "Top-k overlap:"
    )

    for _, row in overlaps.iterrows():

        lines.append(
            f"  {row['signal_a']} vs "
            f"{row['signal_b']}, "
            f"k={int(row['k'])}: "
            f"overlap={int(row['overlap_count'])}, "
            f"Jaccard={row['jaccard']:.4f}"
        )

    summary_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 11. CONSOLE OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP DATASETS")
    print("=" * 70)

    print(
        "\nBy LODO instability:"
    )

    print(
        influence[
            [
                "dataset",
                "kendall_tau",
                "lodo_instability",
                "max_rank_change",
                "num_rank_changes",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nBy winner-change frequency:"
    )

    print(
        influence.nlargest(
            10,
            "winner_change_frequency",
        )[
            [
                "dataset",
                "winner_change_frequency",
                "kendall_tau",
            ]
        ].to_string(index=False)
    )

    print(
        "\nBy tau-break frequency:"
    )

    print(
        influence.nlargest(
            10,
            "tau_break_frequency",
        )[
            [
                "dataset",
                "tau_break_frequency",
                "kendall_tau",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"\n{complete_file}")
    print(correlations_file)
    print(overlap_file)
    print(ranked_file)
    print(summary_file)

    print("\nDone.")


if __name__ == "__main__":
    main()