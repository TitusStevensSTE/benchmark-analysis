from benchbench.data import load_cardinal_benchmark
from benchbench.measures.cardinal import get_diversity as get_diversity_card, get_sensitivity as get_sensitivity_card
from benchbench.data import load_ordinal_benchmark
from benchbench.measures.ordinal import get_diversity as get_diversity_ord, get_sensitivity as get_sensitivity_ord
import matplotlib.pyplot as plt

# Load GLUE (cardinal)
dataCard, colsCard = load_cardinal_benchmark('GLUE')
diversityCard = get_diversity_card(dataCard, colsCard)
sensitivityCard = get_sensitivity_card(dataCard, colsCard)

print("Diversity Cardinal:", diversityCard)
print("Sensitivity Cardinal:", sensitivityCard)

# Load HELM-accuracy (ordinal)
dataOrd, colsOrd = load_ordinal_benchmark('HELM-accuracy')
diversityOrd = get_diversity_ord(dataOrd, colsOrd)
sensitivityOrd = get_sensitivity_ord(dataOrd, colsOrd)

print("Ordinal Diversity:", diversityOrd)
print("Ordinal Sensitivity:", sensitivityOrd)

# Scatter both series on the same axes: ordinal in blue, cardinal in red
plt.figure(figsize=(8, 6))
plt.scatter(diversityOrd, sensitivityOrd, color='blue', label='Ordinal')
plt.scatter(diversityCard, sensitivityCard, color='red', label='Cardinal')
plt.xlabel("Diversity")
plt.ylabel("Sensitivity")
plt.title("Diversity vs Sensitivity (Ordinal: blue, Cardinal: red)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()