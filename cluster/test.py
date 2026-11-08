import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

X = np.array([[1], [2], [6], [10]])
Z = linkage(X, method="single")
print(Z)
# Cột: [idx1, idx2, height, số điểm trong cluster mới]

for h in [0.5, 2, 3.9, 4]:
    print(f"h={h}:", fcluster(Z, t=h, criterion="distance"))

dendrogram(Z, labels=["A", "B", "C", "D"])
plt.axhline(y=2, color="r", linestyle="--", label="cut h=2 (3 clusters)")
plt.ylabel("Height")
plt.legend()
plt.show()