import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# 1. Tạo tập dữ liệu tổng hợp gồm 600 mẫu với 4 cụm tự nhiên
X_raw, y_true = make_blobs(
    n_samples=600, 
    centers=4, 
    cluster_std=0.75, 
    random_state=42
)

# 2. BẮT BUỘC: Chuẩn hóa dữ liệu về mean=0, std=1
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Kích thước tập dữ liệu: {X.shape}")

distortions = []
inertias = []
k_range = range(1, 11)

for k in k_range:
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    model.fit(X)
    
    # 1. Inertia: Tổng bình phương khoảng cách
    inertias.append(model.inertia_)
    
    # 2. Distortion: Trung bình bình phương khoảng cách Euclidean tới tâm cụm gần nhất
    distortion = np.mean(np.min(cdist(X, model.cluster_centers_, 'euclidean')**2, axis=1))
    distortions.append(distortion)

# Vẽ đồ thị kép biểu diễn Elbow Method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Đồ thị Distortion
ax1.plot(k_range, distortions, 'bx-', linewidth=2, markersize=8)
ax1.set_xlabel('Số lượng cụm (k)', fontsize=12)
ax1.set_ylabel('Độ biến dạng (Distortion)', fontsize=12)
ax1.set_title('Elbow Method sử dụng Distortion', fontsize=14, fontweight='bold')
ax1.axvline(x=4, color='red', linestyle='--', label='Điểm khuỷu tay (Optimal k=4)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)

# Đồ thị Inertia
ax2.plot(k_range, inertias, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Số lượng cụm (k)', fontsize=12)
ax2.set_ylabel('Quán tính (Inertia / WCSS)', fontsize=12)
ax2.set_title('Elbow Method sử dụng Inertia', fontsize=14, fontweight='bold')
ax2.axvline(x=4, color='blue', linestyle='--', label='Điểm khuỷu tay (Optimal k=4)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()

#############################################################################################################
############################################################################################################
#########################################################################################################
# Huấn luyện mô hình tối ưu với k=4
optimal_kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = optimal_kmeans.fit_predict(X)
centroids = optimal_kmeans.cluster_centers_

# Tạo lưới mịn để vẽ vùng Voronoi
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = optimal_kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ biểu đồ
plt.figure(figsize=(9, 6))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

# Vẽ các điểm dữ liệu
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1', s=35, edgecolors='k', linewidth=0.5)

# Vẽ các tâm cụm (Centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=250, linewidths=2,
            color='black', zorder=10, label='Tâm cụm (Centroids)')

plt.title('Kết quả Phân cụm K-Means++ và Ranh giới Ô Voronoi (k=4)', fontsize=14, fontweight='bold')
plt.xlabel('Đặc trưng X1 (Đã chuẩn hóa)', fontsize=12)
plt.ylabel('Đặc trưng X2 (Đã chuẩn hóa)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.4)
plt.show()