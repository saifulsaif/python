import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 1. Create a non-linear dataset (Circles)
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define different kernels to compare
kernels = ['rbf', 'poly', 'sigmoid']
plt.figure(figsize=(18, 5))

for i, kernel in enumerate(kernels):
    # Create and train model
    model = svm.SVC(kernel=kernel, gamma='auto', degree=3)
    model.fit(X_train, y_train)
    
    # Plotting
    plt.subplot(1, 3, i + 1)
    
    # Create grid for decision boundary
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f'SVM with {kernel.upper()} Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
print("Non-linear classification plot generated.")
plt.show()
