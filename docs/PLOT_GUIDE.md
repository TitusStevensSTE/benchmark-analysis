# Plot Guide

The consolidated thesis figures are stored in:

```text
results/final_figures/
```

Each figure is generated in both PNG and PDF format.

## Figure 1 — Analysis pipeline

**File:** `figure_01_pipeline`

Purpose: summarize the workflow from benchmark data and baseline ranking through perturbations, stability diagnostics, and cross-benchmark synthesis.

Use: methodology overview.

Avoid: presenting the schematic itself as an empirical result.

## Figure 2 — TabArena LODO: Score versus Elo

**File:** `figure_02_lodo_score_vs_elo`

Purpose: compare Score and Elo ranking stability over the complete 51-dataset TabArena LODO perturbation set.

Key measure: Kendall's tau.

Interpretation: both ranking formulations show high stability under individual dataset removal in the current saved results.

Avoid: quoting the obsolete 93-shared-subset comparison as the main Score-vs-Elo result.

## Figure 3 — LODO rank movement

**File:** `figure_03_lodo_rank_movement`

Purpose: show the largest individual rank movements and the datasets affecting the largest numbers of method ranks.

Interpretation: a high global Kendall tau can coexist with material movement of individual competitors.

## Figure 4 — Random subset stability

**File:** `figure_04_random_subset_stability`

Purpose: show how winner frequencies vary with benchmark composition under the saved random-subset experiment.

Interpretation: composition sensitivity can be visible even when single-dataset LODO remains highly stable.

## Figure 5 — Coordinated removal boundary

**File:** `figure_05_joint_removal_boundary`

Purpose: compare global ranking deterioration with decision-level winner changes as increasingly many datasets are removed.

Current saved Score results identify the first observed winner-changing subset at depth 5 and the first observed tau-break at depth 6.

Interpretation: global ranking failure and winner failure need not occur at the same perturbation strength.

Important: later beam-search depths are targeted observations, not exhaustive coverage of the full perturbation space.

## Figure 6 — Dataset influence

**File:** `figure_06_dataset_influence`

Purpose: compare the datasets ranked as sensitive under LODO instability, winner changes, and tau-breaks.

Interpretation: sensitivity depends on the outcome dimension being examined.

Important: the displayed influence ranking is descriptive, not causal.

## Figure 7 — Performance gap

**File:** `figure_07_performance_gap`

Purpose: examine the association between winner–runner-up performance gap and LODO instability.

Interpretation: the figure addresses a plausible explanation for ranking fragility, but association must not be described as causation.

## Figure 8 — Cross-benchmark LODO

**File:** `figure_08_crossbenchmark_lodo`

Purpose: compare mean and worst-case LODO Kendall tau across benchmark families and summarize winner-changing removals.

Current synthesis includes TabArena, GLUE, OpenLLM, and ImageNet. The main OpenLLM synthesis representation uses the unique-model-mean result.

Interpretation: robustness profiles differ substantially across benchmarks.

## Figure 9 — Diversity versus sensitivity

**File:** `figure_09_diversity_vs_sensitivity`

Purpose: connect the stability analysis with the saved BenchBench diversity/sensitivity metrics.

Interpretation: this is a descriptive cross-benchmark synthesis and should not be used to claim a universal causal trade-off from a small number of benchmark observations.

## Supporting plots

Additional diagnostics remain under their analysis-specific result directories, including:

```text
results/fold_stability/figures/
results/performance_gap/
results/phase1_crossbenchmark/figures/
results/phase1_crossbenchmark/glue/performance_gap/
results/score_vs_elo/plots/
results/dataset_influence/plots/
```

These are not part of the nine-figure consolidated thesis set unless explicitly selected as supplementary material.
