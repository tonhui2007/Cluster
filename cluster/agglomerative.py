import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs

# 1. Khởi tạo tập dữ liệu mô phỏng gồm 3 cụm với 30 điểm
X, _ = make_blobs(n_samples=30, centers=3, cluster_std=1.0, random_state=42)

# 2. Huấn luyện mô hình AgglomerativeClustering với K = 3
clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = clustering.fit_predict(X)

# 3. Huấn luyện mô hình toàn vẹn để trích xuất đầy đủ cây phân cấp phục vụ vẽ Dendrogram
# Để lưu trữ thông tin khoảng cách, ta đặt distance_threshold=0 và compute_distances=True
agg_full = AgglomerativeClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
agg_full.fit(X)

# 4. Hàm chuyển đổi cấu trúc children_ của scikit-learn thành ma trận liên kết cho SciPy dendrogram
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # nút lá
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)

# 5. Vẽ đồ thị phân cụm và biểu đồ Dendrogram cạnh nhau
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Đồ thị phân cụm
scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=70, edgecolors='k')
ax1.set_title("Kết quả Phân cụm Agglomerative (K = 3)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Đặc trưng 1 (Feature 1)")
ax1.set_ylabel("Đặc trưng 2 (Feature 2)")
ax1.grid(True, linestyle='--', alpha=0.5)

# Đồ thị Dendrogram
plt.sca(ax2)
plot_dendrogram(agg_full, truncate_mode='level', p=5)
plt.title("Biểu đồ Cây Phân cấp (Dendrogram)", fontsize=13, fontweight='bold')
plt.xlabel("Chỉ số mẫu dữ liệu (Sample index)")
plt.ylabel("Khoảng cách sáp nhập (Distance / Ward)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()