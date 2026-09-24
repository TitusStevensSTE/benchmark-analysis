from __future__ import annotations

from analysis_config import (PROJECT_ROOT, RESULTS_DIR, TABARENA_DATA_FILE)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================


RESULTS_DIR = (
    RESULTS_DIR
)

COMPARISON_DIR = (
    RESULTS_DIR
    / "score_vs_elo"
)

PLOT_DIR = (
    COMPARISON_DIR
    / "plots"
)


# ============================================================
# DATA
# ============================================================

def load_depth_data():

    path = (
        COMPARISON_DIR
        / "depth_comparison.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Depth comparison not found:\n{path}\n"
            "Run compare_score_elo_removal.py first."
        )

    return pd.read_csv(
        path
    ).sort_values(
        [
            "aggregation",
            "depth",
        ]
    )


# ============================================================
# FIGURE 1
# BEST KENDALL TAU VS REMOVAL DEPTH
# ============================================================

def plot_tau_by_depth(
    df: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for aggregation in [
        "Score",
        "Elo",
    ]:

        sub = df.loc[
            df["aggregation"]
            == aggregation
        ]

        ax.plot(
            sub["depth"],
            sub["best_kendall_tau"],
            marker="o",
            label=aggregation,
        )

    ax.axhline(
        0.90,
        linestyle="--",
        linewidth=1,
        label="Kendall tau = 0.90",
    )

    ax.set_xlabel(
        "Number of removed datasets"
    )

    ax.set_ylabel(
        "Best Kendall's tau vs. full ranking"
    )

    ax.set_title(
        "Ranking stability under dataset removal"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "score_vs_elo_kendall_tau.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# FIGURE 2
# WINNER CHANGES BY DEPTH
# ============================================================

def plot_winner_events(
    df: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for aggregation in [
        "Score",
        "Elo",
    ]:

        sub = df.loc[
            df["aggregation"]
            == aggregation
        ]

        winner_indicator = (
            sub[
                "winner_change_observed"
            ]
            .astype(int)
        )

        ax.step(
            sub["depth"],
            winner_indicator,
            where="mid",
            label=aggregation,
        )

    ax.set_xlabel(
        "Number of removed datasets"
    )

    ax.set_ylabel(
        "Winner change observed"
    )

    ax.set_yticks(
        [0, 1],
        [
            "No",
            "Yes",
        ],
    )

    ax.set_title(
        "Winner stability under dataset removal"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "score_vs_elo_winner_changes.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# FIGURE 3
# SAME SUBSETS: SCORE VS ELO
# ============================================================

def plot_same_subsets() -> None:

    path = (
        COMPARISON_DIR
        / "same_subset_comparison.csv"
    )

    if not path.exists():
        print(
            "Skipping same-subset plot: "
            f"{path} not found."
        )
        return

    df = pd.read_csv(
        path
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        df["score_kendall_tau"],
        df["elo_kendall_tau"],
    )

    limits = [
        min(
            df["score_kendall_tau"].min(),
            df["elo_kendall_tau"].min(),
        ),
        max(
            df["score_kendall_tau"].max(),
            df["elo_kendall_tau"].max(),
        ),
    ]

    ax.plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Kendall's tau under Score aggregation"
    )

    ax.set_ylabel(
        "Kendall's tau under Elo aggregation"
    )

    ax.set_title(
        "Aggregation sensitivity for identical dataset removals"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "score_vs_elo_same_subsets.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# FIGURE 4
# DATASET INFLUENCE
# ============================================================

def plot_influence() -> None:

    score_path = (
        COMPARISON_DIR
        / "score_tau_break_influence.csv"
    )

    elo_path = (
        COMPARISON_DIR
        / "elo_tau_break_influence.csv"
    )

    if not (
        score_path.exists()
        and elo_path.exists()
    ):

        print(
            "Skipping influence plot: "
            "influence files not found."
        )

        return

    score = (
        pd.read_csv(
            score_path
        )
        .rename(
            columns={
                "frequency":
                    "score_frequency"
            }
        )
    )

    elo = (
        pd.read_csv(
            elo_path
        )
        .rename(
            columns={
                "frequency":
                    "elo_frequency"
            }
        )
    )

    merged = pd.merge(
        score[
            [
                "dataset",
                "score_frequency",
            ]
        ],
        elo[
            [
                "dataset",
                "elo_frequency",
            ]
        ],
        on="dataset",
        how="outer",
    ).fillna(
        0.0
    )

    # Use the top datasets by combined influence rather than
    # only sorting by Score.
    merged["combined_frequency"] = (
        merged["score_frequency"]
        + merged["elo_frequency"]
    )

    merged = (
        merged
        .sort_values(
            "combined_frequency",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "combined_frequency"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    positions = list(
        range(len(merged))
    )

    # barh(y, width): use width as the SECOND positional
    # argument. Do not also pass width=...
    bar_height = 0.35

    score_positions = [
        x - bar_height / 2
        for x in positions
    ]

    elo_positions = [
        x + bar_height / 2
        for x in positions
    ]

    ax.barh(
        score_positions,
        merged[
            "score_frequency"
        ],
        bar_height,
        label="Score",
    )

    ax.barh(
        elo_positions,
        merged[
            "elo_frequency"
        ],
        bar_height,
        label="Elo",
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        merged["dataset"]
    )

    ax.set_xlabel(
        "Frequency among tau-breaking subsets"
    )

    ax.set_title(
        "Dataset influence under Score and Elo"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "score_vs_elo_dataset_influence.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# FIGURE 5
# SAME-SUBSET TAU DIFFERENCE
# ============================================================

def plot_tau_difference() -> None:

    path = (
        COMPARISON_DIR
        / "same_subset_comparison.csv"
    )

    if not path.exists():
        print(
            "Skipping tau-difference plot: "
            f"{path} not found."
        )
        return

    df = pd.read_csv(
        path
    )

    df = df.sort_values(
        "tau_difference_elo_minus_score"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    positions = list(
        range(len(df))
    )

    ax.barh(
        positions,
        df[
            "tau_difference_elo_minus_score"
        ],
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        [
            x[:55]
            for x in df[
                "removed_datasets"
            ]
        ],
        fontsize=8,
    )

    ax.set_xlabel(
        "Elo Kendall's tau minus Score Kendall's tau"
    )

    ax.set_title(
        "Aggregation effect for identical dataset removals"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "score_vs_elo_tau_difference.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_depth_data()

    plot_tau_by_depth(
        df
    )

    plot_winner_events(
        df
    )

    plot_same_subsets()

    plot_influence()

    plot_tau_difference()

    print(
        "Plots written to:"
    )

    print(
        PLOT_DIR
    )


if __name__ == "__main__":
    main()