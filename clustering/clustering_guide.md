# Beginner's Guide to Clustering Algorithms

This guide explains four popular clustering algorithms used in machine learning: K-Means, Gaussian Mixture Models (GMM), Agglomerative Hierarchical Clustering, and DBSCAN.

---

## 1. K-Means Clustering

### Concept
K-Means is one of the simplest and most popular clustering algorithms. It tries to partition the data into **K** distinct clusters.

### How it works
1. **Initialization**: Choose **K** random points as initial centroids (centers).
2. **Assignment**: Assign each data point to the nearest centroid.
3. **Update**: Calculate the new mean (center) of the assigned points for each cluster.
4. **Repeat**: Repeat steps 2 and 3 until the centroids stop moving (convergence).

### Pros
- Fast and efficient for large datasets.
- Easy to understand and implement.

### Cons
- You must specify the number of clusters (**K**) beforehand.
- Sensitive to outliers.
- Assumes clusters are spherical (round) and of similar size.

---

## 2. Gaussian Mixture Models (GMM)

### Concept
GMM is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

### How it works
It uses the **Expectation-Maximization (EM)** algorithm:
1. **E-Step (Expectation)**: Calculate the probability that each data point belongs to each cluster.
2. **M-Step (Maximization)**: Update the parameters (mean, covariance, and mixing coefficient) of the Gaussian distributions to maximize the likelihood of the data.

### Pros
- Can model clusters of different sizes and elliptical shapes (not just spheres).
- Provides "soft clustering" (gives probabilities of belonging to each cluster).

### Cons
- More computationally expensive than K-Means.
- Can get stuck in local optima.

---

## 3. Agglomerative Hierarchical Clustering

### Concept
This is a "bottom-up" approach where each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

### How it works
1. Start with **N** clusters (each point is a cluster).
2. Find the two closest clusters and merge them into a single cluster.
3. Repeat step 2 until only **K** clusters remain (or one big cluster).

### Pros
- Does not require specifying the number of clusters upfront (you can "cut" the tree/dendrogram where you want).
- can capture hierarchical structures.

### Cons
- Computationally expensive for large datasets ($O(N^3)$ or $O(N^2)$).

---

## 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Concept
DBSCAN groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions.

### How it works
It uses two parameters:
- **eps** ($\epsilon$): The maximum distance between two samples for them to be considered as in the same neighborhood.
- **min_samples**: The number of samples in a neighborhood for a point to be considered as a core point.

1. Find the points in the $\epsilon$-neighborhood of every point.
2. Identify **core points** (have more than `min_samples` neighbors).
3. Find connected components of core points and assign non-core neighbors to the same cluster.
4. Points not assigned to any cluster are labeled as **Noise**.

### Pros
- Does **not** require specifying the number of clusters beforehand.
- Can find arbitrarily shaped clusters (not just round ones).
- Handles noise/outliers well.

### Cons
- Sensitive to the choice of `eps` and `min_samples`.
- Struggles with clusters of varying densities.

---

## Comparison Summary

| Algorithm | Type | Geometry | Needs K? | Handles Noise? |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means** | Centroid-based | Spherical | Yes | No |
| **GMM** | Probabilistic | Elliptical | Yes | "Softly" |
| **Agglomerative** | Hierarchical | Tree-like | No* | No |
| **DBSCAN** | Density-based | Arbitrary | No | Yes |

*Agglomerative clustering creates a hierarchy, but you typically cut it at a specific level to get K clusters.
