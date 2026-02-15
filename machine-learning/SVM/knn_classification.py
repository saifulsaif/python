import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

# 1. Create a "Moons" dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# 2. Train KNN with K=5
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

# 3. Create mesh for visualization
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Visualize
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
plt.title(f'K-Nearest Neighbors (K={k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
