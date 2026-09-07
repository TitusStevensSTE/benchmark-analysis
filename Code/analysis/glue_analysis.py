import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\Glue_Leaderboard.csv")

#print(df.head())
#print(df.columns)
#print(df.shape)

#Working on the csv

#helper function, averaging two values as glue score
def average_pair(value):
    if pd.isna(value):
        return None

    parts = str(value).split("/")

    if len(parts) == 2:
        return (float(parts[0]) + float(parts[1])) / 2

    return float(value)

#one score
df["MRPC_score"] = df["MRPC"].apply(average_pair)
df["STS-B_score"] = df["STS-B"].apply(average_pair)
df["QQP_score"] = df["QQP"].apply(average_pair)

#combine mnli-m and mnli-mm
df["MNLI_score"] = (
    df["MNLI-m"] + df["MNLI-mm"]
) / 2

#creating 9-task glue matrix
task_columns = [
    "CoLA",
    "SST-2",
    "MRPC_score",
    "STS-B_score",
    "QQP_score",
    "MNLI_score",
    "QNLI",
    "RTE",
    "WNLI"
]
glue_scores = df[["Model"] + task_columns].copy()

# verifying that the 4 new scores are present with one score
print(df.head())
print(df.columns)
print(df.shape)

#creating the 9task dataframe
task_columns = [
    "CoLA",
    "SST-2",
    "MRPC_score",
    "STS-B_score",
    "QQP_score",
    "MNLI_score",
    "QNLI",
    "RTE",
    "WNLI"
]

glue_scores = df[["Model"] + task_columns].copy()

# Convert all task scores to numeric (some are strings)
for col in task_columns:
    glue_scores[col] = pd.to_numeric(glue_scores[col], errors="coerce")


print("\nGLUE analysis dataframe:")
print(glue_scores.head())

print("\nMissing values:")
print(glue_scores[task_columns].isna().sum())

#calculating full glue score (gives the ranking that LODO analysis will use)
glue_scores["overall_score"] = (
    glue_scores[task_columns].mean(axis=1)
)

calculated_ranking = (
    glue_scores[
        ["Model", "overall_score"]
    ]
    .sort_values(
        "overall_score",
        ascending=False
    )
)

print("\nCalculated GLUE ranking:")
print(calculated_ranking.head(20))

#comparing to original glue ranking
comparison = df[
    ["Rank", "Model", "Score"]
].copy()

comparison = comparison.merge(
    calculated_ranking,
    on="Model"
)

comparison = comparison.sort_values("Rank")

print(comparison.head(20))


#2 missing values
print(
    glue_scores[
        glue_scores["QNLI"].isna()
    ][["Model", "QNLI"]]
)

#####################
#### Final Prepartion of GLUE dataframe 
####################

# Keep only models with complete results for all 9 GLUE tasks
glue_scores = glue_scores.dropna(subset=task_columns).copy()

print("Number of complete models:", len(glue_scores))
# expecting all zeros
print(glue_scores[task_columns].isna().sum())

# recalculate glue ranking
glue_scores["overall_score"] = (
    glue_scores[task_columns].mean(axis=1)
)

full_ranking = (
    glue_scores
    .set_index("Model")["overall_score"]
    .sort_values(ascending=False)
)

print("\nFull GLUE ranking:")
print(full_ranking.head(20))

# glue winner for full ranking
full_winner = full_ranking.index[0]

print("\nFull GLUE winner:")
print(full_winner)

print("\nWinner score:")
print(full_ranking.iloc[0])

full_rank = full_ranking.rank(
    ascending=False,
    method="min"
)

###########################################
######### LODO ANALYSIS ###################
###########################################

results = []

tasks = [
    "CoLA",
    "SST-2",
    "MRPC_score",
    "STS-B_score",
    "QQP_score",
    "MNLI_score",
    "QNLI",
    "RTE",
    "WNLI"
]



####
for removed_task in tasks:
    remaining_tasks = [
        task for task in tasks
        if task != removed_task
    ]

    
    lodo_scores = (
        glue_scores
        .set_index("Model")[remaining_tasks]
        .mean(axis=1)
    )

    lodo_ranking = lodo_scores.sort_values(
        ascending=False
    )

    print(f"\nRanking without {removed_task}:")
    print(lodo_ranking.head(20))

    # comparing to full ranking
    ####

    lodo_rank = lodo_ranking.rank(
        ascending=False,
        method="min"
    )

    # Same 107 models in both rankings
    common_models = full_rank.index.intersection(
        lodo_rank.index
    )

    # Kendall tau
    tau, _ = kendalltau(
        full_rank.loc[common_models],
        lodo_rank.loc[common_models]
    )

    # Absolute rank changes
    rank_difference = (
        full_rank.loc[common_models]
        - lodo_rank.loc[common_models]
    ).abs()

    # Maximum rank change
    max_rank_change = rank_difference.max()

    # Number of models whose rank changed
    num_rank_changes = (
        rank_difference > 0
    ).sum()

    # Winners
    full_winner = full_ranking.index[0]
    lodo_winner = lodo_ranking.index[0]

    winner_changed = (
        full_winner != lodo_winner
    )

    # Where did the original winner go?
    original_winner_new_rank = lodo_rank.loc[full_winner]

    results.append({
        "removed_task": removed_task,
        "kendall_tau": tau,
        "max_rank_change": max_rank_change,
        "num_rank_changes": num_rank_changes,
        "full_winner": full_winner,
        "lodo_winner": lodo_winner,
        "winner_changed": winner_changed,
        "original_winner_new_rank": original_winner_new_rank
    })

lodo_results = pd.DataFrame(results)

print("\nGlue LODO Results")
print(lodo_results.to_string(index=False))


###########################################
######### WINNER ANALYSIS ################
###########################################

# ============================================================
# SETTINGS
# ============================================================

N_REPETITIONS = 1000

TASK_SIZES = [
    2, 3, 4, 5, 6, 7, 8, 9
]

RANDOM_SEED = 42

OUTPUT_PATH = (
    "C:\\Users\\Titus Stevens\\Bachelorarbeit\\"
    "Code\\analysis\\data\\glue_winner_analysis.csv"
)


# ============================================================
# 1. COMPUTE FULL-BENCHMARK RANKING
# ============================================================

print("\n" + "=" * 60)
print("GLUE WINNER ANALYSIS")
print("=" * 60)

# full_ranking already exists from your previous analysis

baseline_ranking = (
    full_ranking
    .reset_index()
)

baseline_ranking.columns = [
    "Model",
    "overall_score"
]

baseline_ranking["rank"] = np.arange(
    1,
    len(baseline_ranking) + 1
)


print("\nFull GLUE ranking:")

print(
    baseline_ranking.head(15).to_string(
        index=False
    )
)


# ============================================================
# 2. IDENTIFY BASELINE TOP 10
# ============================================================

baseline_top10 = (
    baseline_ranking
    .head(10)
)

baseline_top10_models = (
    baseline_top10["Model"]
    .tolist()
)


print("\n" + "=" * 60)
print("BASELINE TOP 10")
print("=" * 60)


for _, row in baseline_top10.iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['Model']} "
        f"(Score = {row['overall_score']:.2f})"
    )


# ============================================================
# 3. RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 4. RANDOM TASK SUBSAMPLING
# ============================================================

print("\n" + "=" * 60)
print("STARTING WINNER ANALYSIS")
print("=" * 60)


winner_results = []


for n_tasks in TASK_SIZES:

    print(
        f"\nSubset size: {n_tasks} tasks"
    )

    for repetition in range(
        N_REPETITIONS
    ):

        # ----------------------------------------------------
        # Randomly sample tasks
        # ----------------------------------------------------

        sampled_tasks = rng.choice(
            tasks,
            size=n_tasks,
            replace=False
        )


        # ----------------------------------------------------
        # Calculate score using only sampled tasks
        # ----------------------------------------------------

        subset_scores = (
            glue_scores
            .set_index("Model")[list(sampled_tasks)]
            .mean(axis=1)
        )


        # ----------------------------------------------------
        # Determine winner
        # ----------------------------------------------------

        winner = subset_scores.idxmax()


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        winner_results.append({

            "num_tasks": n_tasks,

            "repetition": repetition,

            "winner": winner,

        })


    print(
        f"  Finished "
        f"{N_REPETITIONS:,} repetitions."
    )


# ============================================================
# 5. CREATE RESULTS DATAFRAME
# ============================================================

winner_results = pd.DataFrame(
    winner_results
)


print("\n" + "=" * 60)
print("WINNER ANALYSIS FINISHED")
print("=" * 60)


print(
    winner_results.head()
)


# ============================================================
# 6. SAVE RAW RESULTS
# ============================================================

winner_results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nRaw results saved to:\n"
    f"{OUTPUT_PATH}"
)


# ============================================================
# 7. CALCULATE WINNER PERCENTAGES
# ============================================================

winner_counts = (
    winner_results
    .groupby(
        [
            "num_tasks",
            "winner"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


winner_counts["percentage"] = (
    winner_counts["count"]
    / N_REPETITIONS
    * 100
)


# ============================================================
# 8. KEEP BASELINE TOP 10 + OTHER
# ============================================================

winner_counts["winner_group"] = (
    winner_counts["winner"]
    .astype(str)
)


winner_counts.loc[
    ~winner_counts["winner"].isin(
        baseline_top10_models
    ),
    "winner_group"
] = "Other models"


# Aggregate all non-top-10 models

winner_counts = (
    winner_counts
    .groupby(
        [
            "num_tasks",
            "winner_group"
        ],
        as_index=False
    )["percentage"]
    .sum()
)


# ============================================================
# 9. CREATE PIVOT TABLE
# ============================================================

winner_pivot = (
    winner_counts
    .pivot(
        index="num_tasks",
        columns="winner_group",
        values="percentage"
    )
    .fillna(0)
)


# Make sure all baseline top-10 models exist

for model in baseline_top10_models:

    if model not in winner_pivot.columns:

        winner_pivot[model] = 0.0


# Add Other models if necessary

if "Other models" not in winner_pivot.columns:

    winner_pivot["Other models"] = 0.0


# ============================================================
# 10. ORDER STACKS BY ORIGINAL BASELINE RANK
# ============================================================

plot_columns = (
    baseline_top10_models
    + ["Other models"]
)


winner_pivot = winner_pivot[
    plot_columns
]


# ============================================================
# 11. PRINT WINNER PERCENTAGES
# ============================================================

print("\n" + "=" * 60)
print("WINNER PERCENTAGES")
print("=" * 60)


print(
    winner_pivot.round(2).to_string()
)


# ============================================================
# 12. CHECK THAT EACH ROW SUMS TO 100%
# ============================================================

row_sums = (
    winner_pivot.sum(axis=1)
)


print("\nPercentage sums:")

print(
    row_sums.round(2)
)


if not np.allclose(
    row_sums.values,
    100,
    atol=0.01
):

    print(
        "\nWARNING: Some rows do not sum to 100%."
    )

else:

    print(
        "\nAll rows sum to approximately 100%."
    )


# ============================================================
# 13. CREATE STACKPLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 9)
)


x = winner_pivot.index.values


y = [
    winner_pivot[model].values
    for model in plot_columns
]


labels = [
    f"Rank {rank}: {model}"
    for rank, model
    in enumerate(
        baseline_top10_models,
        start=1
    )
]


labels.append(
    "Other models"
)


ax.stackplot(
    x,
    y,
    labels=labels,
    alpha=0.85
)


# ============================================================
# 14. FORMAT PLOT
# ============================================================

ax.set_title(
    "GLUE Winner Stability Under Task Subsampling",
    fontsize=16,
    pad=15
)


ax.set_xlabel(
    "Number of Sampled Tasks",
    fontsize=12
)


ax.set_ylabel(
    "Percentage of Repetitions",
    fontsize=12
)


ax.set_ylim(
    0,
    100
)


ax.set_xticks(
    TASK_SIZES
)


ax.set_yticks(
    np.arange(
        0,
        101,
        10
    )
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    title="Winner"
)


plt.tight_layout()


# ============================================================
# 15. SAVE FIGURE
# ============================================================

plt.savefig(
    "C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\glue_winner_stackplot.png",
    dpi=300,
    bbox_inches="tight"
)


print(
    "\nPlot saved to:"
    "\n..C:\\Users\\Titus Stevens\\Bachelorarbeit\\Code\\analysis\\data\\glue_winner_stackplot.png"
)


# ============================================================
# 16. SHOW PLOT
# ============================================================

plt.show()