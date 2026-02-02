import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

def run_agglomerative_example():
    """
    Runs the Agglomerative Hierarchical Clustering example.
    """
    print("---------------------------------------------------------")
    print("Running Agglomerative Clustering Example...")
    print("---------------------------------------------------------")

    # 1. Generate the dataset
    # We use the same dataset generation to compare with other methods.
    print("Generating dataset...")
    X, _ = make_blobs(n_samples=100, random_state=12)
    d = pd.DataFrame(X)

    # 2. Visualize the raw data
    print("Plotting raw data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(d[0], d[1], c='gray', marker='o', edgecolor='k', s=50)
    plt.title("Raw Data (Before Agglomerative Clustering)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

    # 3. Apply Agglomerative Clustering
    # This is a "bottom-up" approach: each observation starts in its own cluster,
    # and pairs of clusters are merged as one moves up the hierarchy.
    # n_clusters=3: We want to find 3 clusters.
    print("Applying Agglomerative Clustering with 3 clusters...")
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    
    # Fit and predict in one step
    # Unlike KMeans, AgglomerativeClustering doesn't have a separate predict method 
    # for new data in the standard sense (it builds a hierarchy on existing data).
    # So we use fit_predict.
    labels = agg_clustering.fit_predict(d)

    # Add labels to DataFrame
    d['labels'] = labels

    # 4. Visualize the Clustered Data
    d0 = d[d['labels'] == 0]
    d1 = d[d['labels'] == 1]
    d2 = d[d['labels'] == 2]

    print("Plotting clustered data...")
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster
    plt.scatter(d0[0], d0[1], c='red', label='Cluster 0', s=50, edgecolor='k')
    plt.scatter(d1[0], d1[1], c='yellow', label='Cluster 1', s=50, edgecolor='k')
    plt.scatter(d2[0], d2[1], c='green', label='Cluster 2', s=50, edgecolor='k')
    
    plt.title("Agglomerative Clustering Results")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()
    
    print("Agglomerative Clustering Example Completed.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_agglomerative_example()
