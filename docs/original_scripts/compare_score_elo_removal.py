from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from bencheval.elo_utils import EloHelper


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
)

TABARENA_RESULTS = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "data"
    / "df_results_tabarena_2026_08_13.csv"
)

SCORE_MATRIX = (
    RESULTS_DIR
    / "score_method_dataset.csv"
)

SCORE_SEARCH_DIR = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "score"
)

ELO_SEARCH_DIR = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "elo"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "score_vs_elo"
)


# ============================================================
# SETTINGS
# ============================================================

EXACT_MAX_REMOVALS = 3
TAU_THRESHOLD = 0.90


# ============================================================
# HELPERS
# ============================================================

def parse_dataset_set(value) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()

    return tuple(
        sorted(
            x
            for x in str(value).split("|")
            if x
        )
    )


def calculate_rank_metrics(
    ranking_a: pd.Series,
    ranking_b: pd.Series,
) -> dict:

    common = (
        ranking_a.index
        .intersection(ranking_b.index)
    )

    a = ranking_a.loc[common]
    b = ranking_b.loc[common]

    tau, tau_p = kendalltau(a, b)
    rho, rho_p = spearmanr(a, b)

    rank_diff = (a - b).abs()

    return {
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "max_rank_difference": float(
            rank_diff.max()
        ),
        "mean_rank_difference": float(
            rank_diff.mean()
        ),
    }


def safe_correlation(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """
    Correlate two influence vectors.

    If either vector is constant, the correlation is undefined.
    Return NaN rather than triggering a scipy ConstantInputWarning.
    """

    x = pd.to_numeric(
        x,
        errors="coerce",
    ).fillna(0.0)

    y = pd.to_numeric(
        y,
        errors="coerce",
    ).fillna(0.0)

    if len(x) < 2 or len(y) < 2:
        return {
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "correlation_defined": False,
            "reason": "fewer than two observations",
        }

    if (
        np.isclose(x.std(ddof=0), 0.0)
        or np.isclose(y.std(ddof=0), 0.0)
    ):
        return {
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "correlation_defined": False,
            "reason": "one or both vectors are constant",
        }

    rho, rho_p = spearmanr(
        x,
        y,
    )

    tau, tau_p = kendalltau(
        x,
        y,
    )

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "correlation_defined": True,
        "reason": None,
    }


# ============================================================
# FULL SCORE RANKING
# ============================================================

def load_score_ranking() -> pd.Series:

    df = pd.read_csv(
        SCORE_MATRIX
    )

    required = {
        "dataset",
        "method",
        "dataset_score",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Score file missing columns: "
            f"{sorted(missing)}"
        )

    matrix = df.pivot(
        index="method",
        columns="dataset",
        values="dataset_score",
    )

    if matrix.isna().any().any():
        raise ValueError(
            "Score matrix contains missing "
            "dataset-method values."
        )

    scores = matrix.mean(
        axis=1
    )

    return (
        scores
        .sort_values(
            ascending=False
        )
        .rank(
            ascending=False,
            method="average",
        )
    )


# ============================================================
# FULL ELO RANKING
# ============================================================

def load_elo_ranking() -> pd.Series:

    df = pd.read_csv(
        TABARENA_RESULTS
    )

    helper = EloHelper(
        method_col="method",
        task_col="dataset",
        error_col="metric_error",
        split_col="fold",
    )

    battles = (
        helper.convert_results_to_battles(
            df
        )
    )

    ratings = (
        helper.compute_mle_elo(
            battles=battles,
            SCALE=400,
            INIT_RATING=1000,
        )
    )

    return (
        ratings
        .sort_values(
            ascending=False
        )
        .rank(
            ascending=False,
            method="average",
        )
    )


# ============================================================
# LOAD SEARCH RESULTS
# ============================================================

def load_search(
    directory: Path,
    aggregation: str,
) -> pd.DataFrame:

    path = (
        directory
        / "evaluations.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Search results not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "removed_datasets",
        "num_removed",
        "kendall_tau",
        "winner_changed",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{aggregation} search is missing "
            f"columns: {sorted(missing)}"
        )

    df["aggregation"] = aggregation

    df["removed_dataset_set"] = (
        df["removed_datasets"]
        .apply(parse_dataset_set)
    )

    # IMPORTANT:
    # This is the search phase, not a claim about proof.
    df["search_phase"] = np.where(
        df["num_removed"]
        <= EXACT_MAX_REMOVALS,
        "exact",
        "beam",
    )

    df["winner_changed"] = (
        df["winner_changed"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    if "tau_unstable" in df.columns:

        df["tau_unstable"] = (
            df["tau_unstable"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    else:

        df["tau_unstable"] = (
            df["kendall_tau"]
            < TAU_THRESHOLD
        )

    return df


# ============================================================
# DEPTH SUMMARY
# ============================================================

def build_depth_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for depth, group in (
        df.groupby("num_removed")
    ):

        depth = int(depth)

        rows.append(
            {
                "aggregation": group[
                    "aggregation"
                ].iloc[0],

                "depth": depth,

                "search_phase": (
                    "exact"
                    if depth <= EXACT_MAX_REMOVALS
                    else "beam"
                ),

                "evaluated_subsets": len(
                    group
                ),

                "best_kendall_tau": float(
                    group[
                        "kendall_tau"
                    ].min()
                ),

                "best_spearman_rho": (
                    float(
                        group[
                            "spearman_rho"
                        ].min()
                    )
                    if "spearman_rho"
                    in group.columns
                    else np.nan
                ),

                "max_rank_change_observed": (
                    float(
                        group[
                            "max_rank_change"
                        ].max()
                    )
                    if "max_rank_change"
                    in group.columns
                    else np.nan
                ),

                "winner_changes": int(
                    group[
                        "winner_changed"
                    ].sum()
                ),

                "winner_change_observed": bool(
                    group[
                        "winner_changed"
                    ].any()
                ),

                "tau_break_observed": bool(
                    group[
                        "tau_unstable"
                    ].any()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "aggregation",
                "depth",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# FIRST OBSERVED EVENT
# ============================================================

def find_first_observed_event(
    df: pd.DataFrame,
    event_column: str,
) -> dict:

    found = df.loc[
        df[event_column]
    ]

    if found.empty:

        return {
            "found": False,
            "first_observed_depth": np.nan,
            "evidence": None,
            "removed_datasets": None,
            "kendall_tau": np.nan,
        }

    depth = int(
        found["num_removed"].min()
    )

    candidates = (
        found[
            found["num_removed"]
            == depth
        ]
        .sort_values(
            [
                "kendall_tau",
                "removed_datasets",
            ]
        )
    )

    best = candidates.iloc[0]

    evidence = (
        "exact"
        if depth <= EXACT_MAX_REMOVALS
        else "beam"
    )

    return {
        "found": True,
        "first_observed_depth": depth,
        "evidence": evidence,
        "removed_datasets": (
            best["removed_datasets"]
        ),
        "kendall_tau": float(
            best["kendall_tau"]
        ),
    }


def build_event_summary(
    score_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for aggregation, df in [
        ("Score", score_df),
        ("Elo", elo_df),
    ]:

        winner = find_first_observed_event(
            df,
            "winner_changed",
        )

        tau = find_first_observed_event(
            df,
            "tau_unstable",
        )

        rows.append(
            {
                "aggregation": aggregation,

                "winner_change_observed": (
                    winner["found"]
                ),

                "first_observed_winner_change_depth": (
                    winner[
                        "first_observed_depth"
                    ]
                ),

                "winner_change_evidence": (
                    winner["evidence"]
                ),

                "winner_change_set": (
                    winner[
                        "removed_datasets"
                    ]
                ),

                "tau_break_observed": (
                    tau["found"]
                ),

                "first_observed_tau_break_depth": (
                    tau[
                        "first_observed_depth"
                    ]
                ),

                "tau_break_evidence": (
                    tau["evidence"]
                ),

                "tau_break_set": (
                    tau[
                        "removed_datasets"
                    ]
                ),

                "tau_at_first_observed_break": (
                    tau["kendall_tau"]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DATASET INFLUENCE
# ============================================================

def calculate_influence_frequency(
    df: pd.DataFrame,
    event_column: str,
) -> pd.DataFrame:

    event_df = df.loc[
        df[event_column]
    ].copy()

    aggregation = df[
        "aggregation"
    ].iloc[0]

    if event_df.empty:

        return pd.DataFrame(
            columns=[
                "dataset",
                "frequency",
                "count",
                "n_events",
                "aggregation",
            ]
        )

    counts: dict[str, int] = {}

    for dataset_set in (
        event_df[
            "removed_dataset_set"
        ]
    ):

        for dataset in dataset_set:
            counts[dataset] = (
                counts.get(
                    dataset,
                    0,
                )
                + 1
            )

    n_events = len(
        event_df
    )

    rows = []

    for dataset, count in (
        counts.items()
    ):

        rows.append(
            {
                "dataset": dataset,
                "frequency": (
                    count / n_events
                ),
                "count": count,
                "n_events": n_events,
                "aggregation": aggregation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "frequency",
                "dataset",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# INFLUENCE CORRELATION
# ============================================================

def compare_influence_vectors(
    score_freq: pd.DataFrame,
    elo_freq: pd.DataFrame,
) -> dict:

    score = (
        score_freq
        .set_index("dataset")[
            "frequency"
        ]
        .rename("score")
    )

    elo = (
        elo_freq
        .set_index("dataset")[
            "frequency"
        ]
        .rename("elo")
    )

    combined = pd.concat(
        [
            score,
            elo,
        ],
        axis=1,
    ).fillna(
        0.0
    )

    corr = safe_correlation(
        combined["score"],
        combined["elo"],
    )

    return {
        "n_datasets": len(
            combined
        ),
        **corr,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 78
    )

    print(
        "TABARENA SCORE vs ELO "
        "DATASET-REMOVAL COMPARISON"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    score_df = load_search(
        SCORE_SEARCH_DIR,
        "Score",
    )

    elo_df = load_search(
        ELO_SEARCH_DIR,
        "Elo",
    )

    print(
        f"Score evaluated subsets: "
        f"{len(score_df):,}"
    )

    print(
        f"Elo evaluated subsets:   "
        f"{len(elo_df):,}"
    )

    # --------------------------------------------------------
    # Depth comparison
    # --------------------------------------------------------

    depth_summary = pd.concat(
        [
            build_depth_summary(
                score_df
            ),
            build_depth_summary(
                elo_df
            ),
        ],
        ignore_index=True,
    )

    depth_summary.to_csv(
        OUTPUT_DIR
        / "depth_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Event summary
    # --------------------------------------------------------

    event_summary = build_event_summary(
        score_df,
        elo_df,
    )

    event_summary.to_csv(
        OUTPUT_DIR
        / "event_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Full ranking comparison
    # --------------------------------------------------------

    print(
        "\nComputing full Score/Elo ranking comparison..."
    )

    score_ranking = load_score_ranking()
    elo_ranking = load_elo_ranking()

    full_metrics = calculate_rank_metrics(
        score_ranking,
        elo_ranking,
    )

    full_summary = {
        "score_winner": score_ranking.index[0],
        "elo_winner": elo_ranking.index[0],
        "winner_agrees": (
            score_ranking.index[0]
            == elo_ranking.index[0]
        ),
        **full_metrics,
    }

    pd.DataFrame(
        [full_summary]
    ).to_csv(
        OUTPUT_DIR
        / "full_ranking_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Influence analysis
    # --------------------------------------------------------

    for event_name, column_name in [
        (
            "tau_break",
            "tau_unstable",
        ),
        (
            "winner_change",
            "winner_changed",
        ),
    ]:

        score_freq = (
            calculate_influence_frequency(
                score_df,
                column_name,
            )
        )

        elo_freq = (
            calculate_influence_frequency(
                elo_df,
                column_name,
            )
        )

        score_freq.to_csv(
            OUTPUT_DIR
            / f"score_{event_name}_influence.csv",
            index=False,
        )

        elo_freq.to_csv(
            OUTPUT_DIR
            / f"elo_{event_name}_influence.csv",
            index=False,
        )

        influence = (
            compare_influence_vectors(
                score_freq,
                elo_freq,
            )
        )

        pd.DataFrame(
            [influence]
        ).to_csv(
            OUTPUT_DIR
            / f"{event_name}_influence_correlation.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\nEVENT SUMMARY"
    )

    print(
        event_summary.to_string(
            index=False
        )
    )

    print(
        "\nFULL SCORE vs ELO RANKING"
    )

    print(
        pd.DataFrame(
            [full_summary]
        ).to_string(
            index=False
        )
    )

    print(
        "\nDEPTH COMPARISON"
    )

    print(
        depth_summary.to_string(
            index=False
        )
    )

    print(
        "\nResults written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()