from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# <repo>/Code/analysis/diagnose_score_elo_comparison.py

RESULTS_DIR = (
    RESULTS_DIR
)

INPUT_FILE = (
    RESULTS_DIR
    / "score_vs_elo"
    / "same_subset_comparison.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "score_vs_elo"
    / "diagnostics"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 10


# ============================================================
# HELPERS
# ============================================================

def require_columns(
    df: pd.DataFrame,
    columns: set[str],
) -> None:

    missing = columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Input file is missing required "
            f"columns: {sorted(missing)}"
        )


def bool_series(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.lower()
        .eq("true")
    )


def print_section(
    title: str,
) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run compare_score_elo_same_subsets.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "removed_datasets",
        "num_removed",
        "score_kendall_tau",
        "elo_kendall_tau",
        "score_spearman_rho",
        "elo_spearman_rho",
        "score_max_rank_change",
        "elo_max_rank_change",
        "score_mean_rank_change",
        "elo_mean_rank_change",
        "score_winner",
        "elo_winner",
        "score_winner_changed",
        "elo_winner_changed",
        "winner_agrees",
        "full_winner_agrees",
    }

    require_columns(
        df,
        required,
    )

    # --------------------------------------------------------
    # Normalize numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "num_removed",
        "score_kendall_tau",
        "elo_kendall_tau",
        "score_spearman_rho",
        "elo_spearman_rho",
        "score_max_rank_change",
        "elo_max_rank_change",
        "score_mean_rank_change",
        "elo_mean_rank_change",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Normalize booleans
    # --------------------------------------------------------

    for column in [
        "score_winner_changed",
        "elo_winner_changed",
        "winner_agrees",
        "full_winner_agrees",
    ]:

        df[column] = bool_series(
            df[column]
        )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    # Positive:
    #   Elo has HIGHER Kendall tau and is therefore
    #   more stable for the same removal set.
    #
    # Negative:
    #   Score has higher Kendall tau and is therefore
    #   more stable.
    df["tau_difference_elo_minus_score"] = (
        df["elo_kendall_tau"]
        - df["score_kendall_tau"]
    )

    df["spearman_difference_elo_minus_score"] = (
        df["elo_spearman_rho"]
        - df["score_spearman_rho"]
    )

    # Positive:
    #   Score produces a larger maximum rank movement.
    df["max_rank_change_score_minus_elo"] = (
        df["score_max_rank_change"]
        - df["elo_max_rank_change"]
    )

    df["mean_rank_change_score_minus_elo"] = (
        df["score_mean_rank_change"]
        - df["elo_mean_rank_change"]
    )

    df["absolute_tau_difference"] = (
        df["tau_difference_elo_minus_score"]
        .abs()
    )

    # --------------------------------------------------------
    # Winner disagreement classification
    # --------------------------------------------------------

    def classify_winner_difference(
        row: pd.Series,
    ) -> str:

        score_changed = bool(
            row["score_winner_changed"]
        )

        elo_changed = bool(
            row["elo_winner_changed"]
        )

        winner_agrees = bool(
            row["winner_agrees"]
        )

        if winner_agrees:
            return "same_winner"

        if (
            score_changed
            and not elo_changed
        ):
            return "score_changes_only"

        if (
            elo_changed
            and not score_changed
        ):
            return "elo_changes_only"

        if (
            score_changed
            and elo_changed
        ):
            return "both_change_different_winners"

        return "winner_disagreement_other"

    df["winner_difference_type"] = (
        df.apply(
            classify_winner_difference,
            axis=1,
        )
    )

    return df


# ============================================================
# TOP ELO ADVANTAGE
# ============================================================

def make_top_elo_advantage(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "removed_datasets",
        "num_removed",
        "score_kendall_tau",
        "elo_kendall_tau",
        "tau_difference_elo_minus_score",
        "score_spearman_rho",
        "elo_spearman_rho",
        "score_max_rank_change",
        "elo_max_rank_change",
        "score_winner",
        "elo_winner",
        "score_winner_changed",
        "elo_winner_changed",
    ]

    # IMPORTANT:
    # Sort BEFORE selecting the output columns, so that
    # absolute_tau_difference remains available as a tie-break.
    result = (
        df.sort_values(
            [
                "tau_difference_elo_minus_score",
                "absolute_tau_difference",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(TOP_K)
    )

    return (
        result[
            columns
        ]
        .reset_index(drop=True)
    )


# ============================================================
# TOP SCORE ADVANTAGE
# ============================================================

def make_top_score_advantage(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "removed_datasets",
        "num_removed",
        "score_kendall_tau",
        "elo_kendall_tau",
        "tau_difference_elo_minus_score",
        "score_spearman_rho",
        "elo_spearman_rho",
        "score_max_rank_change",
        "elo_max_rank_change",
        "score_winner",
        "elo_winner",
        "score_winner_changed",
        "elo_winner_changed",
    ]

    result = (
        df.sort_values(
            [
                "tau_difference_elo_minus_score",
                "absolute_tau_difference",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(TOP_K)
    )

    return (
        result[
            columns
        ]
        .reset_index(drop=True)
    )


# ============================================================
# WINNER DISAGREEMENTS
# ============================================================

def make_winner_disagreements(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "removed_datasets",
        "num_removed",
        "winner_difference_type",
        "score_winner",
        "elo_winner",
        "score_winner_changed",
        "elo_winner_changed",
        "score_kendall_tau",
        "elo_kendall_tau",
        "tau_difference_elo_minus_score",
        "score_max_rank_change",
        "elo_max_rank_change",
        "score_mean_rank_change",
        "elo_mean_rank_change",
    ]

    return (
        df.loc[
            ~df["winner_agrees"]
        ][
            columns
        ]
        .sort_values(
            [
                "num_removed",
                "tau_difference_elo_minus_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# WINNER DISAGREEMENT SUMMARY
# ============================================================

def make_winner_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        df[
            "winner_difference_type"
        ]
        .value_counts()
        .rename_axis(
            "winner_difference_type"
        )
        .reset_index(
            name="count"
        )
    )

    counts["fraction"] = (
        counts["count"]
        / len(df)
    )

    return counts


# ============================================================
# DEPTH-WISE AGGREGATION EFFECT
# ============================================================

def make_depth_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        df.groupby(
            "num_removed"
        )
        .agg(
            n_subsets=(
                "removed_datasets",
                "size",
            ),

            mean_score_tau=(
                "score_kendall_tau",
                "mean",
            ),

            median_score_tau=(
                "score_kendall_tau",
                "median",
            ),

            mean_elo_tau=(
                "elo_kendall_tau",
                "mean",
            ),

            median_elo_tau=(
                "elo_kendall_tau",
                "median",
            ),

            mean_tau_difference=(
                "tau_difference_elo_minus_score",
                "mean",
            ),

            median_tau_difference=(
                "tau_difference_elo_minus_score",
                "median",
            ),

            std_tau_difference=(
                "tau_difference_elo_minus_score",
                "std",
            ),

            score_more_unstable=(
                "tau_difference_elo_minus_score",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),

            elo_more_unstable=(
                "tau_difference_elo_minus_score",
                lambda x: int(
                    (x < 0).sum()
                ),
            ),

            equal_tau=(
                "tau_difference_elo_minus_score",
                lambda x: int(
                    np.isclose(
                        x,
                        0.0,
                    ).sum()
                ),
            ),

            winner_disagreements=(
                "winner_agrees",
                lambda x: int(
                    (~x).sum()
                ),
            ),
        )
        .reset_index()
        .sort_values(
            "num_removed"
        )
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# RANKED WINNER DISAGREEMENTS
# ============================================================

def make_ranked_winner_disagreements(
    df: pd.DataFrame,
) -> pd.DataFrame:

    disagreement = df.loc[
        ~df["winner_agrees"]
    ].copy()

    if disagreement.empty:
        return disagreement

    return (
        disagreement
        .sort_values(
            [
                "num_removed",
                "absolute_tau_difference",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def make_overall_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    tau_difference = (
        df[
            "tau_difference_elo_minus_score"
        ]
    )

    summary = {
        "n_shared_subsets": len(df),

        "mean_score_tau": float(
            df[
                "score_kendall_tau"
            ].mean()
        ),

        "median_score_tau": float(
            df[
                "score_kendall_tau"
            ].median()
        ),

        "mean_elo_tau": float(
            df[
                "elo_kendall_tau"
            ].mean()
        ),

        "median_elo_tau": float(
            df[
                "elo_kendall_tau"
            ].median()
        ),

        "mean_elo_minus_score_tau": float(
            tau_difference.mean()
        ),

        "median_elo_minus_score_tau": float(
            tau_difference.median()
        ),

        "min_elo_minus_score_tau": float(
            tau_difference.min()
        ),

        "max_elo_minus_score_tau": float(
            tau_difference.max()
        ),

        "score_more_stable_count": int(
            (tau_difference < 0).sum()
        ),

        "elo_more_stable_count": int(
            (tau_difference > 0).sum()
        ),

        "equal_tau_count": int(
            np.isclose(
                tau_difference,
                0.0,
            ).sum()
        ),

        "winner_disagreement_count": int(
            (~df["winner_agrees"]).sum()
        ),

        "winner_change_disagreement_count": int(
            (
                df[
                    "score_winner_changed"
                ]
                !=
                df[
                    "elo_winner_changed"
                ]
            ).sum()
        ),

        "score_winner_change_count": int(
            df[
                "score_winner_changed"
            ].sum()
        ),

        "elo_winner_change_count": int(
            df[
                "elo_winner_changed"
            ].sum()
        ),
    }

    return pd.DataFrame(
        [summary]
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    overall: pd.DataFrame,
    depth_summary: pd.DataFrame,
    winner_summary: pd.DataFrame,
    top_elo: pd.DataFrame,
    top_score: pd.DataFrame,
    winner_disagreements: pd.DataFrame,
) -> None:

    lines: list[str] = []

    lines.append(
        "TabArena Score vs Elo "
        "same-subset diagnostic"
    )

    lines.append(
        "=" * 70
    )

    row = overall.iloc[0]

    lines.append(
        f"Shared removal subsets: "
        f"{int(row['n_shared_subsets'])}"
    )

    lines.append(
        f"Mean Score Kendall tau: "
        f"{row['mean_score_tau']:.6f}"
    )

    lines.append(
        f"Mean Elo Kendall tau: "
        f"{row['mean_elo_tau']:.6f}"
    )

    lines.append(
        f"Mean Elo - Score tau difference: "
        f"{row['mean_elo_minus_score_tau']:.6f}"
    )

    lines.append(
        f"Median Elo - Score tau difference: "
        f"{row['median_elo_minus_score_tau']:.6f}"
    )

    lines.append(
        f"Score more stable: "
        f"{int(row['score_more_stable_count'])}"
    )

    lines.append(
        f"Elo more stable: "
        f"{int(row['elo_more_stable_count'])}"
    )

    lines.append(
        f"Winner disagreements: "
        f"{int(row['winner_disagreement_count'])}"
    )

    lines.append(
        f"Winner-change disagreements: "
        f"{int(row['winner_change_disagreement_count'])}"
    )

    # --------------------------------------------------------
    # Depth-wise
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "DEPTH-WISE AGGREGATION EFFECT"
    )
    lines.append(
        "-" * 70
    )

    lines.append(
        depth_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Winner categories
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "WINNER DISAGREEMENT TYPES"
    )
    lines.append(
        "-" * 70
    )

    lines.append(
        winner_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Elo advantage
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "TOP SUBSETS WHERE ELO IS MORE STABLE"
    )
    lines.append(
        "-" * 70
    )

    lines.append(
        top_elo.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Score advantage
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "TOP SUBSETS WHERE SCORE IS MORE STABLE"
    )
    lines.append(
        "-" * 70
    )

    lines.append(
        top_score.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Winner disagreements
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "WINNER DISAGREEMENTS"
    )
    lines.append(
        "-" * 70
    )

    if winner_disagreements.empty:

        lines.append(
            "No winner disagreements."
        )

    else:

        lines.append(
            winner_disagreements.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    (
        OUTPUT_DIR
        / "diagnostic_report.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 88
    )

    print(
        "TABARENA SCORE vs ELO RESULTS DIAGNOSTIC"
    )

    print(
        "=" * 88
    )

    df = load_results()

    print(
        f"Loaded {len(df)} shared removal subsets."
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    top_elo = make_top_elo_advantage(
        df
    )

    top_score = make_top_score_advantage(
        df
    )

    winner_disagreements = (
        make_winner_disagreements(
            df
        )
    )

    winner_summary = (
        make_winner_summary(
            df
        )
    )

    depth_summary = (
        make_depth_summary(
            df
        )
    )

    overall = (
        make_overall_summary(
            df
        )
    )

    ranked_disagreements = (
        make_ranked_winner_disagreements(
            df
        )
    )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    top_elo.to_csv(
        OUTPUT_DIR
        / "top_elo_stability_advantage.csv",
        index=False,
    )

    top_score.to_csv(
        OUTPUT_DIR
        / "top_score_stability_advantage.csv",
        index=False,
    )

    winner_disagreements.to_csv(
        OUTPUT_DIR
        / "winner_disagreements.csv",
        index=False,
    )

    winner_summary.to_csv(
        OUTPUT_DIR
        / "winner_disagreement_summary.csv",
        index=False,
    )

    depth_summary.to_csv(
        OUTPUT_DIR
        / "depth_aggregation_effect.csv",
        index=False,
    )

    overall.to_csv(
        OUTPUT_DIR
        / "overall_diagnostic_summary.csv",
        index=False,
    )

    ranked_disagreements.to_csv(
        OUTPUT_DIR
        / "ranked_winner_disagreements.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    write_text_report(
        overall=overall,
        depth_summary=depth_summary,
        winner_summary=winner_summary,
        top_elo=top_elo,
        top_score=top_score,
        winner_disagreements=winner_disagreements,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print_section(
        "OVERALL SUMMARY"
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print_section(
        "TOP 10: ELO MORE STABLE THAN SCORE"
    )

    print(
        top_elo.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print_section(
        "TOP 10: SCORE MORE STABLE THAN ELO"
    )

    print(
        top_score.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print_section(
        "WINNER DISAGREEMENT SUMMARY"
    )

    print(
        winner_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print_section(
        "DEPTH-WISE AGGREGATION EFFECT"
    )

    print(
        depth_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print_section(
        "WINNER DISAGREEMENTS"
    )

    if winner_disagreements.empty:

        print(
            "No winner disagreements."
        )

    else:

        print(
            winner_disagreements.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print()
    print(
        "=" * 88
    )

    print(
        "Diagnostic outputs written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()