from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Load the famous Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train a Decision Tree
# We limit depth to 3 so it's easy to visualize
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# 3. Visualize the Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=iris.feature_names,  
          class_names=iris.target_names,
          filled=True, 
          rounded=True, 
          fontsize=12)

plt.title("Decision Tree Visualization (Iris Dataset)")
plt.show()
