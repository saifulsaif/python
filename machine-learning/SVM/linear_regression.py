import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise

# 2. Train the model
model = LinearRegression()
model.fit(X, y)

# 3. Make predictions
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# 4. Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_new, y_predict, color='red', linewidth=3, label='Linear Regression Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print(f"Model Equation: y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}x")
plt.show()
