from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

# Repository root:
#   benchmark-analysis/
#
# This script lives in:
#   benchmark-analysis/Code/analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "benchmark_comparison.csv"
OUTPUT_MD = RESULTS_DIR / "benchmark_comparison.md"

# Existing result files
TABARENA_FILE = RESULTS_DIR / "tabarena_lodo_results.csv"
GLUE_FILE = RESULTS_DIR / "glue_lodo_results.csv"
OPENLLM_FILE = RESULTS_DIR / "openllm_lodo_results.csv"
IMAGENET_FILE = RESULTS_DIR / "imagenet_lodo_results.csv"
IMAGENET_ARCH_FILE = (
    RESULTS_DIR / "imagenet_unique_architecture_lodo_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_lodo_file(path: Path, benchmark_name: str) -> pd.DataFrame:
    """
    Load a LODO result CSV and validate the required columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{benchmark_name}: result file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "kendall_tau",
        "spearman_rho",
        "max_rank_change",
        "num_rank_changes",
        "winner_changed",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{benchmark_name}: missing required columns "
            f"{sorted(missing)} in:\n{path}"
        )

    return df


def extract_summary(
    benchmark: str,
    df: pd.DataFrame,
    tasks: int,
    competitors: int | None = None,
) -> dict:
    """
    Extract directly comparable summary statistics from a LODO table.
    """

    # Row with lowest Kendall tau
    worst_tau_row = df.loc[df["kendall_tau"].idxmin()]

    # Row with lowest Spearman rho
    worst_spearman_row = df.loc[df["spearman_rho"].idxmin()]

    # Row with largest maximum rank change
    worst_mrc_row = df.loc[df["max_rank_change"].idxmax()]

    # Largest number of changed rank positions
    worst_num_changed_row = df.loc[df["num_rank_changes"].idxmax()]

    # Number of removals that change the winner
    winner_change_count = int(df["winner_changed"].sum())

    return {
        "benchmark": benchmark,
        "tasks": tasks,
        "competitors": competitors,

        # Main ranking-stability measures
        "worst_kendall_tau": float(worst_tau_row["kendall_tau"]),
        "worst_kendall_removed_task": str(
            worst_tau_row.get("removed_task", "")
        ),

        "worst_spearman_rho": float(worst_spearman_row["spearman_rho"]),
        "worst_spearman_removed_task": str(
            worst_spearman_row.get("removed_task", "")
        ),

        "max_rank_change": float(worst_mrc_row["max_rank_change"]),
        "max_rank_change_removed_task": str(
            worst_mrc_row.get("removed_task", "")
        ),

        "max_num_rank_changes": int(
            worst_num_changed_row["num_rank_changes"]
        ),
        "max_num_rank_changes_removed_task": str(
            worst_num_changed_row.get("removed_task", "")
        ),

        "winner_changed_removals": winner_change_count,
        "winner_changed": winner_change_count > 0,

        # Number of LODO experiments
        "lodo_cases": len(df),
    }


def format_value(value):
    """
    Pretty formatting for the Markdown table.
    """
    if pd.isna(value):
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 80)
print("CROSS-BENCHMARK LODO COMPARISON")
print("=" * 80)

print(f"\nProject root:")
print(PROJECT_ROOT)

rows = []


# ------------------------------------------------------------
# TabArena
# ------------------------------------------------------------
if TABARENA_FILE.exists():
    tabarena_df = load_lodo_file(TABARENA_FILE, "TabArena")

    # Current project setup:
    # 51 datasets, 2574 frameworks
    rows.append(
        extract_summary(
            benchmark="TabArena",
            df=tabarena_df,
            tasks=51,
            competitors=2574,
        )
    )
else:
    print("\nWARNING: TabArena file not found.")
    print(TABARENA_FILE)


# ------------------------------------------------------------
# GLUE
# ------------------------------------------------------------
if GLUE_FILE.exists():
    glue_df = load_lodo_file(GLUE_FILE, "GLUE")

    # Nine GLUE tasks:
    # CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI
    #
    # Number of competitors is taken from the CSV if possible.
    glue_competitors = None

    if "full_winner" in glue_df.columns:
        # This is not necessarily sufficient to infer the number
        # of competitors, so leave it as N/A unless another column
        # explicitly provides it.
        glue_competitors = None

    rows.append(
        extract_summary(
            benchmark="GLUE",
            df=glue_df,
            tasks=9,
            competitors=glue_competitors,
        )
    )
else:
    print("\nWARNING: GLUE file not found.")
    print(GLUE_FILE)


# ------------------------------------------------------------
# OpenLLM
# ------------------------------------------------------------
if OPENLLM_FILE.exists():
    openllm_df = load_lodo_file(OPENLLM_FILE, "OpenLLM")

    # Current OpenLLM benchmark uses:
    # ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K
    rows.append(
        extract_summary(
            benchmark="OpenLLM",
            df=openllm_df,
            tasks=6,
            competitors=None,
        )
    )
else:
    print("\nWARNING: OpenLLM file not found.")
    print(OPENLLM_FILE)


# ------------------------------------------------------------
# ImageNet -- weights
# ------------------------------------------------------------
if IMAGENET_FILE.exists():
    imagenet_df = load_lodo_file(IMAGENET_FILE, "ImageNet weights")

    rows.append(
        extract_summary(
            benchmark="ImageNet (weights)",
            df=imagenet_df,
            tasks=1000,
            competitors=112,
        )
    )
else:
    print("\nWARNING: ImageNet weight file not found.")
    print(IMAGENET_FILE)


# ------------------------------------------------------------
# ImageNet -- unique architectures
# ------------------------------------------------------------
if IMAGENET_ARCH_FILE.exists():
    imagenet_arch_df = load_lodo_file(
        IMAGENET_ARCH_FILE,
        "ImageNet architectures",
    )

    rows.append(
        extract_summary(
            benchmark="ImageNet (architectures)",
            df=imagenet_arch_df,
            tasks=1000,
            competitors=79,
        )
    )
else:
    print("\nWARNING: ImageNet architecture file not found.")
    print(IMAGENET_ARCH_FILE)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

if not rows:
    raise RuntimeError(
        "No benchmark result files were found. "
        "Run the individual benchmark analyses first."
    )

comparison_df = pd.DataFrame(rows)


# ============================================================
# SAVE CSV
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(
    OUTPUT_CSV,
    index=False,
    float_format="%.6f",
)


# ============================================================
# SAVE MARKDOWN TABLE
# ============================================================

display_columns = [
    "benchmark",
    "tasks",
    "competitors",
    "worst_kendall_tau",
    "worst_spearman_rho",
    "max_rank_change",
    "max_num_rank_changes",
    "winner_changed_removals",
]

markdown_df = comparison_df[display_columns].copy()

markdown_df.columns = [
    "Benchmark",
    "Tasks",
    "Competitors",
    "Worst Kendall tau",
    "Worst Spearman rho",
    "Max rank change",
    "Max # changed ranks",
    "Winner-changing removals",
]


# Make values presentation-friendly
markdown_df["Worst Kendall tau"] = markdown_df[
    "Worst Kendall tau"
].map(lambda x: f"{x:.4f}")

markdown_df["Worst Spearman rho"] = markdown_df[
    "Worst Spearman rho"
].map(lambda x: f"{x:.4f}")

markdown_df["Max rank change"] = markdown_df[
    "Max rank change"
].map(lambda x: f"{x:.0f}")

markdown_df["Max # changed ranks"] = markdown_df[
    "Max # changed ranks"
].map(lambda x: f"{x:.0f}")

markdown_text = (
    "# Cross-Benchmark LODO Comparison\n\n"
    "Generated automatically from the saved LODO result files.\n\n"
    + markdown_df.to_markdown(index=False)
    + "\n"
)

OUTPUT_MD.write_text(
    markdown_text,
    encoding="utf-8",
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(markdown_text)

print("\nSaved:")
print(f"  CSV:      {OUTPUT_CSV}")
print(f"  Markdown: {OUTPUT_MD}")


# ============================================================
# ADDITIONAL DETAILS
# ============================================================

print("\n" + "=" * 80)
print("MOST INFLUENTIAL REMOVALS")
print("=" * 80)

for _, row in comparison_df.iterrows():
    print(f"\n{row['benchmark']}")
    print(
        f"  Worst Kendall tau: "
        f"{row['worst_kendall_tau']:.6f}"
        f"  (removed: {row['worst_kendall_removed_task']})"
    )
    print(
        f"  Worst Spearman rho: "
        f"{row['worst_spearman_rho']:.6f}"
        f"  (removed: {row['worst_spearman_removed_task']})"
    )
    print(
        f"  Maximum rank change: "
        f"{row['max_rank_change']:.0f}"
        f"  (removed: {row['max_rank_change_removed_task']})"
    )
    print(
        f"  Maximum number of changed ranks: "
        f"{row['max_num_rank_changes']}"
        f"  (removed: {row['max_num_rank_changes_removed_task']})"
    )
    print(
        f"  Winner-changing removals: "
        f"{row['winner_changed_removals']}"
    )
