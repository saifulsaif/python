import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
X = np.array([
    [2, 6],   # fail
    [1, 5],   # fail
    [4, 8],   # pass
    [5, 9],   # pass
    [3, 7],   # pass
    [1, 4]    # fail
])

y = np.array([0, 0, 1, 1, 1, 0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create SVM model
model = svm.SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Visualization ---

def plot_svc_decision_function(model, ax=None, plot_scatter=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=300, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.figure(figsize=(10, 6))

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='k', label='Data Points')

# Set labels
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Linear Classifier Visualization')

# Plot decision boundary
plot_svc_decision_function(model)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
