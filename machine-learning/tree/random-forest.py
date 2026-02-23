# Random Forest Classifier
# Import libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define the Random Forest model
clf = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=3,        # same depth as your Decision Tree slide
    random_state=42
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot Feature Importances
fig, ax = plt.subplots(figsize=(8, 5))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = iris.feature_names

ax.bar(range(len(importances)), importances[indices], color='steelblue')
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=15)
ax.set_title("Random Forest - Feature Importances (Iris Dataset)")
ax.set_ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Optional: Visualize one tree from the forest
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf.estimators_[0], filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names, ax=ax)
plt.title("One Tree from the Random Forest")
plt.show()