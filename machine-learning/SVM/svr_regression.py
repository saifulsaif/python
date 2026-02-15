import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 1. Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# 2. Fit regression models
# We compare different kernels for SVR
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, epsilon=0.1)
svr_poly = SVR(kernel="poly", C=100, degree=3, epsilon=0.1, coef0=1)

# Fit models
svr_rbf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)

# 3. Visualization
lw = 2
plt.figure(figsize=(12, 7))

plt.scatter(X, y, color="darkorange", label="Data Points", alpha=0.6)

# Generate points for prediction line
X_plot = np.linspace(0, 5, 100)[:, None]

plt.plot(X_plot, svr_rbf.predict(X_plot), color="navy", lw=lw, label="RBF model")
plt.plot(X_plot, svr_lin.predict(X_plot), color="c", lw=lw, label="Linear model")
plt.plot(X_plot, svr_poly.predict(X_plot), color="cornflowerblue", lw=lw, label="Polynomial model")

plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.title("Support Vector Regression (SVR)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print("SVR (Support Vector Regression) plot generated.")
plt.show()
