import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Generate 3D Data ---
# Creating 3 centers in a 3D space to classify
X, y = make_blobs(n_samples=50, centers=2, n_features=3, random_state=42)

# Create and train SVM model with a linear kernel
# A linear kernel in 3D creates a 2D plane (Hyperplane)
model = svm.SVC(kernel='linear')
model.fit(X, y)

# --- 2. Visualization ---
fig = plt.figure(figsize=(15, 8))

# --- Subplot 1: 2D Projection (Hyperspace conceptualization) ---
ax1 = fig.add_subplot(121)
ax1.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k')
ax1.set_title("2D Projection (Feature 1 vs Feature 2)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Subplot 2: 3D Hyperplane ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=50, cmap='coolwarm', edgecolors='k')

# Create a meshgrid for the plane
z = lambda x, y: (-model.intercept_[0] - model.coef_[0][0] * x - model.coef_[0][1] * y) / model.coef_[0][2]

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                     np.linspace(ylim[0], ylim[1], 10))

# Plot the 2D Hyperplane in 3D space
ax2.plot_surface(xx, yy, z(xx, yy), alpha=0.3, color='green', label='Decision Hyperplane')

ax2.set_title("3D SVM Hyperplane")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.set_zlabel("Feature 3")

# Set labels for the legend
import matplotlib.lines as mlines
blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Class 0')
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Class 1')
plane = mlines.Line2D([], [], color='green', alpha=0.3, linewidth=10, label='Hyperplane')
ax2.legend(handles=[blue_dot, red_dot, plane])

plt.tight_layout()
print("2D and 3D Hyperplane visualization generated.")
plt.show()
