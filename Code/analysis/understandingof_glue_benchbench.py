import sys
from pathlib import Path

BENCHBENCH_ROOT = Path(
    r"D:\Users\titus\Informatik-DO\benchmark-analysis\benchbench"
)
sys.path.insert(0, str(BENCHBENCH_ROOT))

from benchbench.data import load_cardinal_benchmark
from benchbench.measures.cardinal import get_diversity, get_sensitivity


data, cols = load_cardinal_benchmark("GLUE")

print("=" * 70)
print("OFFICIAL BENCHBENCH GLUE ANALYSIS")
print("=" * 70)

print(f"Number of models: {len(data)}")
print(f"Number of tasks: {len(cols)}")
print(f"Tasks: {cols}")

diversity = get_diversity(data, cols)

print("\nDIVERSITY")
print(f"Diversity (W):   {diversity[0]:.6f}")
print(f"Diversity (MRC): {diversity[1]:.6f}")

sensitivity = get_sensitivity(data, cols)

print("\nSENSITIVITY")
print(f"Sensitivity (tau): {sensitivity[0]:.6f}")
print(f"Sensitivity (MRC): {sensitivity[1]:.6f}")

print("=" * 70)