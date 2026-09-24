import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\elo_winner_analysis.csv")

# ============================================================
# PREPARE DATA
# ============================================================

# Count how often each method won for each subset size
winner_counts = (
    df.groupby(["num_datasets", "winner"])
      .size()
      .unstack(fill_value=0)
)

# Convert counts to percentages
winner_percentages = winner_counts.div(
    winner_counts.sum(axis=1),
    axis=0
) * 100

# Sort by number of datasets
winner_percentages = winner_percentages.sort_index()


# ============================================================
# GROUP SMALL / RARE METHODS
# ============================================================

# Methods you want to show individually
main_methods = [
    "REALMLP (tuned + ensemble)",
    "TABM (tuned + ensemble)",
    "GBM (tuned + ensemble)",
    "CAT (tuned + ensemble)",
    "CAT (tuned)",
    "XGB (tuned + ensemble)",
    "MNCA (tuned + ensemble)",
]

# Everything else becomes "Other methods"
other_columns = [
    col for col in winner_percentages.columns
    if col not in main_methods
]

if other_columns:
    winner_percentages["Other methods"] = winner_percentages[
        other_columns
    ].sum(axis=1)

    winner_percentages = winner_percentages[
        main_methods + ["Other methods"]
    ]

# Make sure all desired columns exist
for method in main_methods + ["Other methods"]:
    if method not in winner_percentages.columns:
        winner_percentages[method] = 0

winner_percentages = winner_percentages[
    main_methods + ["Other methods"]
]


# ============================================================
# CREATE STACKPLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.stackplot(
    winner_percentages.index,
    *[
        winner_percentages[method]
        for method in winner_percentages.columns
    ],
    labels=winner_percentages.columns
)

ax.set_xlabel("Number of datasets")
ax.set_ylabel("Probability of being ranked #1 (%)")

ax.set_title(
    "Winner Stability as a Function of Benchmark Size"
)

ax.set_xticks(winner_percentages.index)
ax.set_ylim(0, 100)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

output_path = os.path.join(
    r"C:\Users\Titus Stevens\Bachelorarbeit\Code\analysis\data",
    "elo_winner_stackplot.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nPlot saved to:\n{output_path}")

plt.show()