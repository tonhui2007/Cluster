import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. Khởi tạo dữ liệu và chuẩn hóa
X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Tính khoảng cách tới láng giềng gần thứ k = 4
k = 4
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, _ = neighbors_fit.kneighbors(X)

# 3. Lấy khoảng cách tới láng giềng thứ k và sắp xếp tăng dần
sorted_k_distances = np.sort(distances[:, k - 1], axis=0)

# 4. Trực quan hóa đường cong K-Distance
plt.figure(figsize=(10, 5))
plt.plot(sorted_k_distances, color='crimson', lw=2.5, label=f'Đường cong khoảng cách {k}-NN')
plt.axhline(y=0.22, color='darkblue', linestyle='--', lw=2, label='Ngưỡng eps tối ưu tại điểm uốn (Elbow $\\approx 0.22$)')
plt.scatter(275, 0.22, color='darkblue', s=100, zorder=5)

plt.title(f'Phương pháp Đồ thị K-Distance (k = {k}) Xác định Epsilon Tối ưu cho DBSCAN', fontsize=12, fontweight='bold')
plt.xlabel('Các điểm dữ liệu (đã sắp xếp tăng dần theo khoảng cách)', fontsize=10)
plt.ylabel(f'Khoảng cách tới láng giềng thứ {k} ({k}-NN Distance)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
