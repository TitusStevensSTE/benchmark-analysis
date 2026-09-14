from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ============================================================
# REPOSITORY-RELATIVE PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "Code"
    / "analysis"
    / "results"
)

LODO_DEFAULT = RESULTS_DIR / "lodo_score_results.csv"

REMOVAL_DIR = (
    RESULTS_DIR
    / "dataset_removal_search"
    / "score"
)

# Preferred compact file. The 819 MB evaluations.csv is only a fallback.
REMOVAL_SUMMARY_DEFAULT = REMOVAL_DIR / "most_destabilizing_subsets.csv"
REMOVAL_EVALUATIONS_DEFAULT = REMOVAL_DIR / "evaluations.csv"

OUTPUT_DIR = RESULTS_DIR / "dataset_influence"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# DEFAULT SETTINGS
# ============================================================
DEFAULT_TOP_K_PER_DEPTH = 100
DEFAULT_RANDOM_TOP_K = 100


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dataset influence across LODO, adversarial "
            "dataset-removal search, and optional random-subset analysis."
        )
    )

    parser.add_argument(
        "--lodo-input",
        type=Path,
        default=LODO_DEFAULT,
        help="Score-based LODO results CSV.",
    )

    parser.add_argument(
        "--removal-input",
        type=Path,
        default=None,
        help=(
            "Optional adversarial-search input. If omitted, the script first "
            "uses most_destabilizing_subsets.csv and then falls back to evaluations.csv."
        ),
    )

    parser.add_argument(
        "--random-input",
        type=Path,
        default=None,
        help="Optional random-subset results CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--top-k-per-depth",
        type=int,
        default=DEFAULT_TOP_K_PER_DEPTH,
        help="Number of most destabilizing subsets retained per removal depth.",
    )

    parser.add_argument(
        "--random-top-k",
        type=int,
        default=DEFAULT_RANDOM_TOP_K,
        help=(
            "If random-subset results contain an instability metric, retain this "
            "many most unstable subsets for participation analysis."
        ),
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================
def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def parse_dataset_list(value) -> list[str]:
    """
    Convert common dataset-list representations to a Python list.

    Supported:
      - Python lists
      - JSON lists
      - pipe-separated strings
      - semicolon-separated strings
      - comma-separated strings
      - literal Python list strings
    """
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text:
        return []

    # Try JSON / Python literal representations first.
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(x).strip()
                        for x in parsed
                        if str(x).strip()
                    ]
            except Exception:
                pass

    # Pipe-separated is the representation used by the removal search.
    if "|" in text:
        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    if ";" in text:
        return [
            item.strip()
            for item in text.split(";")
            if item.strip()
        ]

    # Avoid splitting ordinary dataset names on commas unless clearly useful.
    if "," in text:
        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    return [text]


def dataset_name_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "removed_datasets",
        "removed_dataset",
        "datasets",
        "dataset_set",
        "subset_datasets",
        "adversarial_datasets",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# LODO
# ============================================================
def load_lodo(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"LODO input file not found:\n{path}"
        )

    df = pd.read_csv(path)

    dataset_col = find_column(
        df,
        [
            "dataset_removed",
            "removed_dataset",
            "dataset",
            "removed_task",
        ],
    )

    if dataset_col is None:
        raise ValueError(
            "Could not identify dataset column in LODO results.\n"
            f"Columns: {df.columns.tolist()}"
        )

    tau_col = find_column(
        df,
        [
            "kendall_tau",
            "tau",
        ],
    )

    if tau_col is None:
        raise ValueError(
            "Could not identify Kendall tau column in LODO results.\n"
            f"Columns: {df.columns.tolist()}"
        )

    result = pd.DataFrame()
    result["dataset"] = df[dataset_col].astype(str)

    result["kendall_tau"] = pd.to_numeric(
        df[tau_col],
        errors="coerce",
    )

    result["lodo_instability"] = 1.0 - result["kendall_tau"]

    for source, target in [
        (
            ["max_rank_change", "maximum_rank_change"],
            "max_rank_change",
        ),
        (
            ["num_rank_changes", "number_rank_changes"],
            "num_rank_changes",
        ),
        (
            ["spearman_rho"],
            "spearman_rho",
        ),
    ]:
        col = find_column(df, source)

        if col is not None:
            result[target] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    winner_changed_col = find_column(
        df,
        [
            "winner_changed",
        ],
    )

    if winner_changed_col is not None:
        result["lodo_winner_changed"] = df[winner_changed_col].map(
            normalize_bool
        )

    return (
        result
        .drop_duplicates("dataset")
        .reset_index(drop=True)
    )


# ============================================================
# ADVERSARIAL REMOVAL SEARCH
# ============================================================
def resolve_removal_input(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"Specified removal-search file does not exist:\n{explicit_path}"
            )
        return explicit_path

    if REMOVAL_SUMMARY_DEFAULT.exists():
        return REMOVAL_SUMMARY_DEFAULT

    if REMOVAL_EVALUATIONS_DEFAULT.exists():
        return REMOVAL_EVALUATIONS_DEFAULT

    raise FileNotFoundError(
        "Could not find adversarial removal-search results.\n\n"
        "Expected one of:\n"
        f"  {REMOVAL_SUMMARY_DEFAULT}\n"
        f"  {REMOVAL_EVALUATIONS_DEFAULT}\n\n"
        "Copy the compact most_destabilizing_subsets.csv from the machine "
        "that ran the removal search."
    )


def load_removal(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Removal-search input not found:\n{path}"
        )

    df = pd.read_csv(path)

    dataset_col = dataset_name_column(df)

    if dataset_col is None:
        raise ValueError(
            "Could not identify the removed-dataset column in removal results.\n"
            f"Columns: {df.columns.tolist()}"
        )

    # Normalize common fields.
    df["_removed_list"] = df[dataset_col].apply(
        parse_dataset_list
    )

    depth_col = find_column(
        df,
        [
            "num_removed",
            "removed_count",
            "depth",
        ],
    )

    if depth_col is not None:
        df["num_removed"] = pd.to_numeric(
            df[depth_col],
            errors="coerce",
        )
    else:
        df["num_removed"] = df["_removed_list"].map(len)

    tau_col = find_column(
        df,
        [
            "kendall_tau",
            "tau",
        ],
    )

    if tau_col is not None:
        df["kendall_tau"] = pd.to_numeric(
            df[tau_col],
            errors="coerce",
        )
    else:
        df["kendall_tau"] = np.nan

    winner_changed_col = find_column(
        df,
        [
            "winner_changed",
        ],
    )

    if winner_changed_col is not None:
        df["winner_changed"] = df[winner_changed_col].map(
            normalize_bool
        )
    else:
        df["winner_changed"] = False

    tau_unstable_col = find_column(
        df,
        [
            "tau_unstable",
            "kendall_unstable",
        ],
    )

    if tau_unstable_col is not None:
        df["tau_unstable"] = df[tau_unstable_col].map(
            normalize_bool
        )
    else:
        df["tau_unstable"] = (
            df["kendall_tau"].notna()
            & (df["kendall_tau"] < 0.90)
        )

    max_rank_col = find_column(
        df,
        [
            "max_rank_change",
        ],
    )

    if max_rank_col is not None:
        df["max_rank_change"] = pd.to_numeric(
            df[max_rank_col],
            errors="coerce",
        )
    else:
        df["max_rank_change"] = np.nan

    mean_rank_col = find_column(
        df,
        [
            "mean_rank_change",
        ],
    )

    if mean_rank_col is not None:
        df["mean_rank_change"] = pd.to_numeric(
            df[mean_rank_col],
            errors="coerce",
        )
    else:
        df["mean_rank_change"] = np.nan

    return df


def select_top_adversarial_subsets(
    df: pd.DataFrame,
    top_k_per_depth: int,
) -> pd.DataFrame:
    """
    Select the strongest adversarial subsets independently per removal depth.

    Primary criterion:
        lowest Kendall tau

    Secondary criteria:
        maximum rank change
        mean rank change
    """
    work = df.copy()

    work["_tau_sort"] = work["kendall_tau"].fillna(np.inf)

    work["_max_sort"] = work["max_rank_change"].fillna(-np.inf)

    work["_mean_sort"] = work["mean_rank_change"].fillna(-np.inf)

    work = work.sort_values(
        [
            "num_removed",
            "_tau_sort",
            "_max_sort",
            "_mean_sort",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )

    selected = (
        work
        .groupby("num_removed", sort=True)
        .head(top_k_per_depth)
        .copy()
    )

    return selected


# ============================================================
# ADVERSARIAL DATASET FREQUENCIES
# ============================================================
def calculate_adversarial_frequencies(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    winner_rows = []
    tau_rows = []

    for _, row in selected.iterrows():
        datasets = row["_removed_list"]

        for dataset in datasets:
            all_rows.append(
                {
                    "dataset": dataset,
                    "num_removed": row["num_removed"],
                    "winner_changed": bool(row["winner_changed"]),
                    "tau_unstable": bool(row["tau_unstable"]),
                }
            )

            if bool(row["winner_changed"]):
                winner_rows.append(
                    {
                        "dataset": dataset,
                        "num_removed": row["num_removed"],
                    }
                )

            if bool(row["tau_unstable"]):
                tau_rows.append(
                    {
                        "dataset": dataset,
                        "num_removed": row["num_removed"],
                    }
                )

    # All adversarial subset participation.
    if all_rows:
        all_df = pd.DataFrame(all_rows)
        adversarial = (
            all_df
            .groupby("dataset")
            .agg(
                adversarial_occurrences=("dataset", "size"),
                adversarial_depths=(
                    "num_removed",
                    lambda x: ",".join(
                        map(
                            str,
                            sorted(set(x)),
                        )
                    ),
                ),
            )
            .reset_index()
        )
    else:
        adversarial = pd.DataFrame(
            columns=[
                "dataset",
                "adversarial_occurrences",
                "adversarial_depths",
            ]
        )

    # Winner-changing subsets.
    if winner_rows:
        winner_df = pd.DataFrame(winner_rows)

        winner_freq = (
            winner_df
            .groupby("dataset")
            .size()
            .reset_index(
                name="winner_change_occurrences"
            )
        )
    else:
        winner_freq = pd.DataFrame(
            columns=[
                "dataset",
                "winner_change_occurrences",
            ]
        )

    # Tau-breaking subsets.
    if tau_rows:
        tau_df = pd.DataFrame(tau_rows)

        tau_freq = (
            tau_df
            .groupby("dataset")
            .size()
            .reset_index(
                name="tau_break_occurrences"
            )
        )
    else:
        tau_freq = pd.DataFrame(
            columns=[
                "dataset",
                "tau_break_occurrences",
            ]
        )

    # We need the denominator: number of top subsets considered.
    total_subsets = len(selected)

    adversarial["adversarial_participation"] = (
        adversarial["adversarial_occurrences"]
        / total_subsets
        if total_subsets > 0
        else 0.0
    )

    winner_freq["winner_change_participation"] = (
        winner_freq["winner_change_occurrences"]
        / total_subsets
        if total_subsets > 0
        else 0.0
    )

    tau_freq["tau_break_participation"] = (
        tau_freq["tau_break_occurrences"]
        / total_subsets
        if total_subsets > 0
        else 0.0
    )

    return (
        adversarial,
        winner_freq,
        tau_freq,
    )


# ============================================================
# RANDOM SUBSET ANALYSIS
# ============================================================
def load_random_subset(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(
            f"Random-subset input does not exist:\n{path}"
        )

    df = pd.read_csv(path)

    dataset_col = dataset_name_column(df)

    if dataset_col is None:
        raise ValueError(
            "Could not identify dataset-list column in random-subset results.\n"
            f"Columns: {df.columns.tolist()}"
        )

    df["_dataset_list"] = df[dataset_col].apply(
        parse_dataset_list
    )

    tau_col = find_column(
        df,
        [
            "kendall_tau",
            "tau",
        ],
    )

    if tau_col is not None:
        df["kendall_tau"] = pd.to_numeric(
            df[tau_col],
            errors="coerce",
        )
    else:
        df["kendall_tau"] = np.nan

    winner_col = find_column(
        df,
        [
            "winner",
            "lodo_winner",
            "subset_winner",
        ],
    )

    if winner_col is not None:
        df["winner"] = df[winner_col].astype(str)
    else:
        df["winner"] = ""

    return df


def random_subset_participation(
    random_df: pd.DataFrame,
    random_top_k: int,
) -> pd.DataFrame:
    work = random_df.copy()

    # If a tau exists, focus on the most unstable random subsets.
    if work["kendall_tau"].notna().any():
        selected = work.sort_values(
            "kendall_tau",
            ascending=True,
        ).head(random_top_k)
    else:
        selected = work.copy()

    rows = []

    for _, row in selected.iterrows():
        for dataset in row["_dataset_list"]:
            rows.append(
                {
                    "dataset": dataset,
                    "kendall_tau": row["kendall_tau"],
                    "winner": row["winner"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "random_subset_occurrences",
                "random_subset_participation",
            ]
        )

    expanded = pd.DataFrame(rows)

    result = (
        expanded
        .groupby("dataset")
        .agg(
            random_subset_occurrences=("dataset", "size"),
        )
        .reset_index()
    )

    result["random_subset_participation"] = (
        result["random_subset_occurrences"]
        / len(selected)
    )

    # Winner-changing participation can be defined once the
    # full-benchmark winner is known later; for now we preserve
    # the occurrence count only.
    return result


# ============================================================
# MERGING + CORRELATIONS
# ============================================================
def merge_influence_tables(
    lodo: pd.DataFrame,
    adversarial: pd.DataFrame,
    winner_freq: pd.DataFrame,
    tau_freq: pd.DataFrame,
    random_freq: pd.DataFrame | None,
) -> pd.DataFrame:
    datasets = set(lodo["dataset"])

    for table in [
        adversarial,
        winner_freq,
        tau_freq,
    ]:
        if "dataset" in table.columns:
            datasets.update(
                table["dataset"].astype(str)
            )

    if random_freq is not None:
        datasets.update(
            random_freq["dataset"].astype(str)
        )

    result = pd.DataFrame(
        {
            "dataset": sorted(datasets),
        }
    )

    for table in [
        lodo,
        adversarial,
        winner_freq,
        tau_freq,
        random_freq,
    ]:
        if table is None or table.empty:
            continue

        result = result.merge(
            table,
            on="dataset",
            how="left",
        )

    numeric_defaults = {
        "lodo_instability": 0.0,
        "max_rank_change": 0.0,
        "num_rank_changes": 0.0,
        "adversarial_occurrences": 0.0,
        "adversarial_participation": 0.0,
        "winner_change_occurrences": 0.0,
        "winner_change_participation": 0.0,
        "tau_break_occurrences": 0.0,
        "tau_break_participation": 0.0,
        "random_subset_occurrences": 0.0,
        "random_subset_participation": 0.0,
    }

    for col, default in numeric_defaults.items():
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            ).fillna(default)

    return result


def correlation_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> dict | None:
    if x not in df.columns or y not in df.columns:
        return None

    work = df[[x, y]].dropna()

    if len(work) < 3:
        return None

    if work[x].nunique() < 2 or work[y].nunique() < 2:
        return None

    rho, rho_p = spearmanr(
        work[x],
        work[y],
    )

    tau, tau_p = kendalltau(
        work[x],
        work[y],
    )

    return {
        "x": x,
        "y": y,
        "n": len(work),
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p_value": float(tau_p),
    }


def calculate_correlations(
    influence: pd.DataFrame,
) -> pd.DataFrame:
    pairs = [
        (
            "lodo_instability",
            "adversarial_participation",
        ),
        (
            "lodo_instability",
            "winner_change_participation",
        ),
        (
            "lodo_instability",
            "tau_break_participation",
        ),
        (
            "lodo_instability",
            "random_subset_participation",
        ),
        (
            "max_rank_change",
            "adversarial_participation",
        ),
        (
            "num_rank_changes",
            "adversarial_participation",
        ),
        (
            "adversarial_participation",
            "random_subset_participation",
        ),
    ]

    rows = []

    for x, y in pairs:
        result = correlation_pair(
            influence,
            x,
            y,
        )

        if result is not None:
            rows.append(result)

    return pd.DataFrame(rows)


def calculate_top_overlap(
    influence: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "lodo_instability",
        "max_rank_change",
        "num_rank_changes",
        "adversarial_participation",
        "winner_change_participation",
        "tau_break_participation",
        "random_subset_participation",
    ]

    available = [
        m
        for m in metrics
        if m in influence.columns
    ]

    rows = []

    for k in [5, 10, 15]:
        top_sets = {}

        for metric in available:
            ranked = influence.sort_values(
                metric,
                ascending=False,
            )

            top_sets[metric] = set(
                ranked.head(k)["dataset"]
            )

        pairs = [
            (
                "lodo_instability",
                "adversarial_participation",
            ),
            (
                "lodo_instability",
                "random_subset_participation",
            ),
            (
                "adversarial_participation",
                "random_subset_participation",
            ),
            (
                "lodo_instability",
                "winner_change_participation",
            ),
        ]

        for a, b in pairs:
            if a not in top_sets or b not in top_sets:
                continue

            rows.append(
                {
                    "top_k": k,
                    "metric_a": a,
                    "metric_b": b,
                    "overlap_count": len(
                        top_sets[a] & top_sets[b]
                    ),
                    "overlap_fraction_of_a": len(
                        top_sets[a] & top_sets[b]
                    ) / k,
                    "intersection": " | ".join(
                        sorted(
                            top_sets[a] & top_sets[b]
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================
def save_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if x not in df.columns or y not in df.columns:
        return

    work = df[[x, y, "dataset"]].dropna()

    if len(work) < 3:
        return

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        work[x],
        work[y],
        alpha=0.75,
    )

    # Label the most extreme 5 points in y.
    top = work.nlargest(
        min(5, len(work)),
        y,
    )

    for _, row in top.iterrows():
        ax.annotate(
            str(row["dataset"]),
            (row[x], row[y]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(
        axis="both",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_top_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        return

    top = (
        df
        .sort_values(metric, ascending=False)
        .head(top_n)
        .sort_values(metric, ascending=True)
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        top["dataset"],
        top[metric],
    )

    ax.set_xlabel(ylabel)
    ax.set_ylabel("Dataset")
    ax.set_title(title)
    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================
def write_summary(
    output_dir: Path,
    influence: pd.DataFrame,
    correlations: pd.DataFrame,
    overlaps: pd.DataFrame,
    removal_path: Path,
    random_path: Path | None,
) -> None:
    lines = [
        "TabArena dataset-influence summary",
        "=" * 45,
        "",
        f"Datasets in merged table: {len(influence)}",
        f"LODO input: {LODO_DEFAULT}",
        f"Removal input: {removal_path}",
        f"Random input: {random_path if random_path else 'Not supplied'}",
        "",
        "TOP DATASETS BY LODO INSTABILITY",
        "--------------------------------",
    ]

    top_lodo = (
        influence
        .sort_values(
            "lodo_instability",
            ascending=False,
        )
        .head(10)
    )

    for _, row in top_lodo.iterrows():
        lines.append(
            f"{row['dataset']}: "
            f"LODO instability={row['lodo_instability']:.6f}, "
            f"max_rank_change={row.get('max_rank_change', 0):.1f}, "
            f"num_rank_changes={row.get('num_rank_changes', 0):.0f}"
        )

    lines.extend(
        [
            "",
            "TOP DATASETS BY ADVERSARIAL PARTICIPATION",
            "------------------------------------------",
        ]
    )

    top_adv = (
        influence
        .sort_values(
            "adversarial_participation",
            ascending=False,
        )
        .head(10)
    )

    for _, row in top_adv.iterrows():
        lines.append(
            f"{row['dataset']}: "
            f"adversarial_participation="
            f"{row['adversarial_participation']:.6f}"
        )

    lines.extend(
        [
            "",
            "TOP DATASETS IN WINNER-CHANGING SUBSETS",
            "---------------------------------------",
        ]
    )

    top_winner = (
        influence
        .sort_values(
            "winner_change_participation",
            ascending=False,
        )
        .head(10)
    )

    for _, row in top_winner.iterrows():
        lines.append(
            f"{row['dataset']}: "
            f"winner_change_participation="
            f"{row['winner_change_participation']:.6f}"
        )

    lines.extend(
        [
            "",
            "CORRELATIONS",
            "------------",
        ]
    )

    if correlations.empty:
        lines.append("No valid correlation pairs available.")
    else:
        for _, row in correlations.iterrows():
            lines.append(
                f"{row['x']} vs {row['y']}: "
                f"Spearman rho={row['spearman_rho']:.4f} "
                f"(p={row['spearman_p_value']:.4g}), "
                f"Kendall tau={row['kendall_tau']:.4f} "
                f"(p={row['kendall_p_value']:.4g}), "
                f"n={int(row['n'])}"
            )

    lines.extend(
        [
            "",
            "TOP-K OVERLAP",
            "-------------",
        ]
    )

    if overlaps.empty:
        lines.append("No overlap results available.")
    else:
        for _, row in overlaps.iterrows():
            lines.append(
                f"top-{int(row['top_k'])}: "
                f"{row['metric_a']} vs {row['metric_b']} -> "
                f"{int(row['overlap_count'])} shared datasets"
            )

    (output_dir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()

    ensure_output_dirs(output_dir)

    removal_path = resolve_removal_input(
        args.removal_input
    )

    print("=" * 78)
    print("TABARENA DATASET-INFLUENCE SUMMARY")
    print("=" * 78)

    print(f"Project root:  {PROJECT_ROOT}")
    print(f"LODO input:    {args.lodo_input.resolve()}")
    print(f"Removal input: {removal_path.resolve()}")

    if args.random_input is not None:
        print(
            f"Random input:  {args.random_input.resolve()}"
        )
    else:
        print(
            "Random input:  not supplied"
        )

    print()

    # --------------------------------------------------------
    # LODO
    # --------------------------------------------------------
    lodo = load_lodo(
        args.lodo_input
    )

    print(
        f"Loaded LODO results: {len(lodo)} datasets"
    )

    # --------------------------------------------------------
    # ADVERSARIAL SEARCH
    # --------------------------------------------------------
    removal = load_removal(
        removal_path
    )

    print(
        f"Loaded removal-search rows: {len(removal):,}"
    )

    selected = select_top_adversarial_subsets(
        removal,
        args.top_k_per_depth,
    )

    print(
        f"Selected adversarial rows: {len(selected):,}"
    )

    (
        adversarial,
        winner_freq,
        tau_freq,
    ) = calculate_adversarial_frequencies(
        selected
    )

    # --------------------------------------------------------
    # RANDOM SUBSETS
    # --------------------------------------------------------
    random_freq = None

    if args.random_input is not None:
        random_df = load_random_subset(
            args.random_input
        )

        random_freq = random_subset_participation(
            random_df,
            args.random_top_k,
        )

        print(
            f"Loaded random-subset rows: {len(random_df):,}"
        )
    else:
        print(
            "No random-subset file supplied; "
            "random participation will remain empty."
        )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------
    influence = merge_influence_tables(
        lodo=lodo,
        adversarial=adversarial,
        winner_freq=winner_freq,
        tau_freq=tau_freq,
        random_freq=random_freq,
    )

    # --------------------------------------------------------
    # SAVE INTERMEDIATE TABLES
    # --------------------------------------------------------
    influence_path = (
        output_dir
        / "dataset_influence_summary.csv"
    )

    influence.to_csv(
        influence_path,
        index=False,
    )

    adversarial.to_csv(
        output_dir
        / "adversarial_topk_frequency.csv",
        index=False,
    )

    winner_freq.to_csv(
        output_dir
        / "adversarial_winner_change_frequency.csv",
        index=False,
    )

    tau_freq.to_csv(
        output_dir
        / "adversarial_tau_break_frequency.csv",
        index=False,
    )

    if random_freq is not None:
        random_freq.to_csv(
            output_dir
            / "random_subset_participation.csv",
            index=False,
        )

    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------
    correlations = calculate_correlations(
        influence
    )

    correlations.to_csv(
        output_dir
        / "influence_correlations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # TOP-K OVERLAP
    # --------------------------------------------------------
    overlaps = calculate_top_overlap(
        influence
    )

    overlaps.to_csv(
        output_dir
        / "influence_rank_overlaps.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    save_scatter(
        influence,
        "lodo_instability",
        "adversarial_participation",
        "LODO instability (1 - Kendall tau)",
        "Adversarial participation",
        "LODO influence vs. adversarial-removal participation",
        PLOTS_DIR / "lodo_vs_adversarial_frequency.png",
    )

    save_scatter(
        influence,
        "lodo_instability",
        "random_subset_participation",
        "LODO instability (1 - Kendall tau)",
        "Random-subset participation",
        "LODO influence vs. random-subset participation",
        PLOTS_DIR / "lodo_vs_random_participation.png",
    )

    save_top_bar(
        influence,
        "adversarial_participation",
        "Adversarial participation",
        "Datasets most frequently selected by adversarial search",
        PLOTS_DIR / "top_adversarial_datasets.png",
    )

    save_top_bar(
        influence,
        "lodo_instability",
        "LODO instability (1 - Kendall tau)",
        "Datasets most influential under LODO",
        PLOTS_DIR / "top_lodo_datasets.png",
    )

    if "random_subset_participation" in influence.columns:
        save_top_bar(
            influence,
            "random_subset_participation",
            "Random-subset participation",
            "Datasets most frequent in unstable random subsets",
            PLOTS_DIR / "top_random_subset_datasets.png",
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    write_summary(
        output_dir=output_dir,
        influence=influence,
        correlations=correlations,
        overlaps=overlaps,
        removal_path=removal_path,
        random_path=args.random_input,
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("TOP DATASETS BY LODO INSTABILITY")
    print("=" * 78)

    cols = [
        "dataset",
        "lodo_instability",
        "max_rank_change",
        "num_rank_changes",
        "adversarial_participation",
        "winner_change_participation",
        "tau_break_participation",
        "random_subset_participation",
    ]

    available_cols = [
        c
        for c in cols
        if c in influence.columns
    ]

    print(
        influence
        .sort_values(
            "lodo_instability",
            ascending=False,
        )
        .head(15)[available_cols]
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("CORRELATIONS")
    print("=" * 78)

    if correlations.empty:
        print("No correlations available.")
    else:
        print(
            correlations.to_string(
                index=False
            )
        )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Main table:\n  {influence_path}"
    )

    print(
        f"Summary:\n  "
        f"{output_dir / 'summary.txt'}"
    )

    print(
        f"Plots:\n  {PLOTS_DIR}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()

