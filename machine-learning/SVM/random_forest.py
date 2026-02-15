import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 1. Create a complex dataset
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# 2. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Decision Boundary Visualization
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Visualize
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap='PiYG', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='PiYG', alpha=0.8)
plt.title('Random Forest Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
