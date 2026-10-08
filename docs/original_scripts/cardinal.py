# Sensitivity
# In the worst case, by adding label noises to different tasks, how largely the final ranking would change?
from benchbench.data import load_cardinal_benchmark, cardinal_benchmark_list
from benchbench.measures.cardinal import get_sensitivity

sensitivity = {}  # the variance for all rankings based on only one task
for dataset_name in cardinal_benchmark_list:
    data, cols = load_cardinal_benchmark(dataset_name)
    sensitivity[dataset_name] = get_sensitivity(data, cols)
    print("%s %.2lf %.2lf" % (dataset_name, sensitivity[dataset_name][0], sensitivity[dataset_name][1]))

#Diversity
#How largely the rankings on each task differ from each other?
from benchbench.data import load_cardinal_benchmark, cardinal_benchmark_list
from benchbench.measures.cardinal import get_diversity

diversity = {}  # the variance for all rankings based on only one task
for dataset_name in cardinal_benchmark_list:
    data, cols = load_cardinal_benchmark(dataset_name)
    diversity[dataset_name] = get_diversity(data, cols)
    print("%s %.2lf %.2lf" % (dataset_name, diversity[dataset_name][0], diversity[dataset_name][1]))

#Dummy Baselines
import numpy as np
from benchbench.data import load_cardinal_benchmark
from benchbench.measures.cardinal import get_diversity, get_sensitivity

num_seed = 5
for dataset_name in ["ImageNet", "Constant", "Random"]:
    for seed in range(num_seed):
        data, cols = load_cardinal_benchmark(dataset_name, seed=seed)
        if dataset_name not in sensitivity.keys():
            sensitivity[dataset_name] = np.array(get_sensitivity(data, cols))
        else:
            sensitivity[dataset_name] += np.array(get_sensitivity(data, cols))
    sensitivity[dataset_name] = sensitivity[dataset_name] / float(num_seed)
    print("%s, %.2lf, %.2lf" % (dataset_name, sensitivity[dataset_name][0], sensitivity[dataset_name][1]))

for dataset_name in ["ImageNet", "Constant", "Random"]:
    for seed in range(num_seed):
        data, cols = load_cardinal_benchmark(dataset_name, seed=seed)
        if dataset_name not in diversity.keys():
            diversity[dataset_name] = np.array(get_diversity(data, cols))
        else:
            diversity[dataset_name] += np.array(get_diversity(data, cols))
    diversity[dataset_name] = diversity[dataset_name] / float(num_seed)
    print("%s, %.2lf, %.2lf" % (dataset_name, diversity[dataset_name][0], diversity[dataset_name][1]))

#Draw the figures
import numpy as np
from zarth_utils.drawer import Drawer
from benchbench.data import cardinal_benchmark_list

dataset_names = cardinal_benchmark_list + ["ImageNet", "Constant", "Random"]
x = [diversity[i][0] for i in dataset_names]
y = [sensitivity[i][0] for i in dataset_names]
print("Corr=%.2lf" % float(np.corrcoef(np.array(x), np.array(y))[0, 1]))

drawer = Drawer(unit_row_length=6.4, unit_col_length=4.8)
ax = drawer.draw_one_axes(x, [y], index=1, xlabel=r"Diversity ($W$)", ylabel=r"Sensitivity ($\tau$)",
                          mode="scatter", use_marker=True, linewidth=80)
for i, label in enumerate(dataset_names):
    if label in ["GLUE", "Constant", "VTAB"]:
        ax.text(x[i], y[i], label, fontsize=14, ha='left', va='top')
    else:
        ax.text(x[i], y[i], label, fontsize=14, ha='right', va='bottom')
ax.grid()

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
clf = Pipeline([
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("lin_reg", Ridge(fit_intercept=False, alpha=0.0))
])
clf.fit(x, y)

x2 = np.linspace(0.00, 1.00, 100).reshape(-1, 1)
y2 = clf.predict(x2.reshape(-1, 1)).reshape(-1, 1)
ax.plot(x2, y2, color="green", linewidth=4, linestyle="--", label="Regression")
ax.legend()
#
drawer.save("cardinal_trade_off_tau.pdf")

import numpy as np
from zarth_utils.drawer import Drawer
from benchbench.data import cardinal_benchmark_list

dataset_names = cardinal_benchmark_list + ["ImageNet", "Constant", "Random"]
x = [diversity[i][0] for i in dataset_names]
y = [sensitivity[i][1] for i in dataset_names]
print("Corr=%.2lf" % float(np.corrcoef(np.array(x), np.array(y))[0, 1]))

drawer = Drawer(unit_row_length=6.4, unit_col_length=4.8)
ax = drawer.draw_one_axes(x, [y], index=1, xlabel=r"Diversity ($W$)", ylabel=r"Sensitivity ($MRC$)",
                          mode="scatter", use_marker=True, linewidth=80)
for i, label in enumerate(dataset_names):
    if label in ["SuperGLUE", "MTEB", "Constant", "VTAB"]:
        ax.text(x[i], y[i], label, fontsize=14, ha='left', va='top')
    else:
        ax.text(x[i], y[i], label, fontsize=14, ha='right', va='bottom')
ax.grid()

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
clf = Pipeline([
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("lin_reg", Ridge(fit_intercept=False, alpha=0.0))
])
clf.fit(x, y)

x2 = np.linspace(0.00, 1.00, 100).reshape(-1, 1)
y2 = clf.predict(x2.reshape(-1, 1)).reshape(-1, 1)
idxs = [i for i in range(len(y2)) if y2[i] <= 1.0]
x2 = x2[idxs]
y2 = y2[idxs]
ax.plot(x2, y2, color="green", linewidth=4, linestyle="--", label="Regression")
ax.legend()

drawer.save("cardinal_trade_off_MRC.pdf")

#Sensitivity versus minimal noise portion  ϵ
from collections import defaultdict
from benchbench.data import cardinal_benchmark_list, load_cardinal_benchmark
from benchbench.measures.cardinal import get_sensitivity

sensitivity_vs_minvalue = defaultdict(dict)

minimal_value_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
dataset_name_list = cardinal_benchmark_list + ["ImageNet", "Constant", "Random"]

for minimal_value in minimal_value_list:
    for dataset_name in dataset_name_list:
        data, cols = load_cardinal_benchmark(dataset_name)
        sensitivity_vs_minvalue[minimal_value][dataset_name] = get_sensitivity(data, cols, min_value=minimal_value, normalize_epsilon=False)
        print("%.2lf %s %.2lf %.2lf" % (
            minimal_value,
            dataset_name,
            sensitivity_vs_minvalue[minimal_value][dataset_name][0],
            sensitivity_vs_minvalue[minimal_value][dataset_name][1]
        ))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = list(np.arange(len(minimal_value_list) + 1))
y = [[
         sensitivity_vs_minvalue[minimal_value][dataset_name][0]
         for minimal_value in minimal_value_list] + [0.0]
     for dataset_name in dataset_name_list
     ]
labels = dataset_name_list
data = pd.DataFrame({
    "Minimal Preserving Portion (r'$\epsilon$')": x * len(dataset_name_list),
    "Sensitivity ($\tau$)": np.array(y).flatten(),
    "Benchmark": np.array(labels).repeat(len(minimal_value_list) + 1)
})

plt.figure(figsize=(6.4, 4.8))
plt.tight_layout()
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")  #
sns.set_palette('tab10')
sns.lineplot(data, x="Minimal Preserving Portion ($\epsilon$)", y="Sensitivity ($\tau$)", hue="Benchmark",
             style="Benchmark",
             markers=True, dashes=False,
             linewidth=4
             )
plt.xticks(x, minimal_value_list + [1.0])
plt.savefig("figures/cardinal_tau_vs_minvalue.pdf", dpi=300, bbox_inches='tight')

y = [
    [
        sensitivity_vs_minvalue[minimal_value][dataset_name][1]
        for minimal_value in minimal_value_list
    ] + [0.0]
    for dataset_name in dataset_name_list
]
data = pd.DataFrame({
    "Minimal Preserving Portion ($\epsilon$)": x * len(dataset_name_list),
    "Sensitivity ($MRC$)": np.array(y).flatten(),
    "Benchmark": np.array(labels).repeat(len(minimal_value_list) + 1)
})
plt.figure(figsize=(6.4, 4.8))
plt.tight_layout()
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")  #
sns.set_palette('tab10')
sns.lineplot(data, x="Minimal Preserving Portion ($\epsilon$)", y="Sensitivity ($MRC$)", hue="Benchmark",
             style="Benchmark",
             markers=True, dashes=False,
             linewidth=4
             )
plt.xticks(x, minimal_value_list + [1.0])
plt.savefig('figures/cardinal_mrc_vs_minvalue.pdf', dpi=300, bbox_inches='tight')

from zarth_utils.drawer import Drawer

drawer = Drawer(num_col=1, unit_row_length=6.4, unit_col_length=4.8)
x = list(np.arange(len(minimal_value_list) + 1))
y = [[
         sensitivity_vs_minvalue[minimal_value][dataset_name][0]
         for minimal_value in minimal_value_list] + [0.0]
     for dataset_name in dataset_name_list
     ]
labels = dataset_name_list
ax = drawer.draw_one_axes(x, y, index=1,
                          xlabel=r"Minimal Preserving Portion ($\epsilon$)", ylabel=r"Sensitivity ($\tau$)",
                          mode="plot", use_marker=True, labels=labels, fontsize=14, linewidth=4)
ax.set_xticks(x)
ax.set_xticklabels(minimal_value_list + [1.0])
ax.grid()
# drawer.save("cardinal_tau_vs_minvalue.pdf")

drawer = Drawer(num_col=1, unit_row_length=6.4, unit_col_length=4.8)
y = [[
         sensitivity_vs_minvalue[minimal_value][dataset_name][1]
         for minimal_value in minimal_value_list
     ] + [0.0]
     for dataset_name in dataset_name_list
     ]
labels = dataset_name_list
ax = drawer.draw_one_axes(x, y, index=1,
                          xlabel=r"Minimal Preserving Portion ($\epsilon$)", ylabel=r"Sensitivity ($MRC$)",
                          mode="plot", use_marker=True, labels=labels, fontsize=14, linewidth=4)
ax.set_xticks(x)
ax.set_xticklabels(minimal_value_list + [1.0])
ax.grid()
# drawer.save("cardinal_mrc_vs_minvalue.pdf")

#Subset Rank Change Experiment
from benchbench.utils.base import rankdata
from benchbench.utils.metric import get_rank_diff
from benchbench.data import cardinal_benchmark_list, load_cardinal_benchmark

np.random.seed(0)

x = list(np.arange(1, 7))
y1_all, y2_all = [], []
dataset_name_list = ["GLUE", "SuperGLUE", "MTEB", "BigBenchHard", "MMLU", "OpenLLM", "VTAB"] + ["ImageNet", "Constant",
                                                                                                "Random"]
for dataset_name in dataset_name_list:
    data, cols = load_cardinal_benchmark(dataset_name)
    old_rank = rankdata(-data[cols].values.mean(1), method="average")
    y1, y2 = [], []
    for subset_size in x:
        min_y = (1, 1)
        for _ in range(1000):
            subset = list(np.random.choice(cols, subset_size, replace=False))
            new_rank = rankdata(-data[subset].values.mean(1), method="average")
            res = get_rank_diff(new_rank, old_rank)
            if res < min_y:
                min_y = res
        y1.append(min_y[0])
        y2.append(min_y[1])
    y1_all.append(y1)
    y2_all.append(y2)

from zarth_utils.drawer import Drawer
import os

drawer = Drawer(num_col=1, num_row=1, unit_row_length=6.4, unit_col_length=4.8)
ax = drawer.draw_one_axes(x, y1_all, index=1, xlabel="Subset size", ylabel=r"Smallest Rank Distance ($\tau$)",
                          labels=dataset_name_list, mode="plot",
                          linewidth=4, fontsize=14, use_marker=True, markersize=10)
ax.grid()
drawer.figure.savefig(os.path.join("figures", "cardinal_subset_min_tau.pdf"), bbox_inches='tight')

drawer = Drawer(num_col=1, num_row=1, unit_row_length=6.4, unit_col_length=4.8)
ax = drawer.draw_one_axes(x, y2_all, index=1, xlabel="Subset size", ylabel=r"Smallest Rank Distance ($MRC$)",
                          labels=dataset_name_list, mode="plot",
                          linewidth=4, fontsize=14, use_marker=True, markersize=10)
ax.grid()
drawer.figure.savefig(os.path.join("figures", "cardinal_subset_min_mrc.pdf"), bbox_inches='tight')

