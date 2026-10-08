# Methodology

## 1. Ranking robustness framework

The benchmark is represented as competitors evaluated across multiple datasets or tasks. A full-benchmark ranking is computed first. The benchmark composition is then perturbed and the resulting ranking is compared with the baseline ranking.

The analysis deliberately separates three levels of sensitivity:

1. **global ranking stability**;
2. **local rank movement**;
3. **decision-level winner stability**.

This avoids treating one ranking statistic as a complete description of robustness.

## 2. LODO

LODO means leave-one-dataset-out for TabArena and leave-one-task-out for the task-oriented cross-benchmark analyses.

For each removable dataset/task:

1. remove it;
2. recompute the ranking;
3. compare the perturbed ranking with the full ranking;
4. record global and local stability measures and whether the winner changed.

## 3. Kendall's tau and instability

Kendall's tau is used as a global ranking-agreement measure.

The analysis also uses:

```text
LODO instability = 1 - Kendall's tau
```

Higher tau indicates stronger agreement with the original ranking; higher `1 - tau` indicates more global reordering.

The coordinated-removal analysis uses `tau < 0.90` as an operational stability-break threshold.

## 4. Local rank movement

The LODO result tables also contain local movement measures such as:

- maximum rank change;
- number of methods whose rank changed;
- mean rank change where available.

These measures capture material competitor-level movement that can be understated by a single global correlation value.

## 5. Winner change

A winner change is recorded when the top-ranked competitor after a perturbation differs from the full-benchmark winner.

Winner change is intentionally treated as a separate decision-level outcome. A ranking may retain the same winner while individual ranks move substantially, or the winner may change without an extreme global ranking reversal.

## 6. Random benchmark subsets

The saved TabArena random-subset experiment samples distinct datasets without replacement for each subset. The current saved configuration uses seed 42, 1000 repetitions per subset size, subset sizes from 1 through 51, and the 51 available datasets / 44 methods.

The main output is winner frequency across sampled benchmark compositions.

This evaluates composition sensitivity across a distribution of sampled subsets. It should not be interpreted as a probability of future benchmark outcomes outside that sampling procedure.

## 7. Coordinated dataset removal

Joint removal evaluates stronger perturbations in which multiple datasets are removed together.

The current Score search uses exhaustive exact enumeration for smaller removal depths and targeted beam search at later depths. Exact enumeration and targeted search have different coverage properties and must not be treated as equivalent.

The saved depth summary contains the best observed Kendall tau and event rates such as winner-change rate and tau-unstable rate.

Accordingly, a plot of this search describes **observed failure boundaries under the evaluated search**, not a probability over all possible removal subsets.

## 8. Dataset influence

Dataset influence is computed descriptively from observed instability dimensions in the saved LODO / removal outputs.

Relevant dimensions include:

- LODO instability;
- winner-change frequency;
- tau-break frequency.

The ranked output places datasets on these separate dimensions and provides a combined descriptive rank.

The term “influence” must therefore not be interpreted causally. The analysis identifies datasets that are associated with stronger observed sensitivity under the tested perturbations; it does not establish that the dataset causes the instability.

## 9. Performance gap

The performance-gap analyses examine whether benchmark cases with a small winner–runner-up gap also tend to exhibit larger ranking instability.

The analysis is observational/associational. A relationship between the quantities does not demonstrate that the performance gap causes the instability.

For GLUE, the pairwise reversal-rate analysis evaluates specific model pairs. Reversal rate asks how frequently a pair's relative ordering changes across LODO perturbations. It is therefore a local pair-order statistic and should not be described as an alternative definition of global Kendall tau.

## 10. Score versus Elo

TabArena rankings are examined under both the benchmark Score aggregation and an MLE Elo formulation.

The primary reproducible comparison is the full 51-dataset LODO evaluation for each ranking formulation.

The repository also contains a diagnostic same-subset comparison. That shared subset set is produced by independently selecting representative removals for Score and Elo and then taking their intersection. It is a diagnostic subset and not a general sample of the full coordinated-removal space.

The older historical 93-shared-subset result is intentionally excluded from the primary claims because the required historical evaluation tables are not present in the committed current result package.

## 11. Cross-benchmark analysis

The robustness framework is applied to TabArena, GLUE, OpenLLM, and ImageNet.

TabArena uses its TabArena-specific pipeline. GLUE, OpenLLM, and ImageNet use the benchmark-native BenchBench loading path in the active cross-benchmark implementation.

Benchmark dimensions differ substantially, including the number of competitors and tasks. Cross-benchmark comparisons are therefore descriptive comparisons of robustness profiles, not controlled experiments in which all benchmark characteristics are equalized.

## 12. BenchBench diversity and sensitivity

The saved BenchBench result table records benchmark-level diversity and sensitivity measures, including:

- diversity W;
- diversity MRC;
- sensitivity Kendall tau;
- sensitivity MRC.

The final synthesis plot uses these saved measures descriptively. With only a small number of benchmark-level observations, the plot should not be presented as proof of a universal causal diversity-to-sensitivity law.
