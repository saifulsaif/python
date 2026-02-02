import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

def run_dbscan_example():
    """
    Runs the DBSCAN clustering example.
    """
    print("---------------------------------------------------------")
    print("Running DBSCAN Clustering Example...")
    print("---------------------------------------------------------")

    # 1. Generate the dataset
    print("Generating dataset...")
    X, _ = make_blobs(n_samples=100, random_state=12)
    d = pd.DataFrame(X)

    # 2. Visualize the raw data
    print("Plotting raw data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(d[0], d[1], c='gray', marker='o', edgecolor='k', s=50)
    plt.title("Raw Data (Before DBSCAN)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

    # 3. Apply DBSCAN
    # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    # eps=1: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples=10: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    print("Applying DBSCAN with eps=1 and min_samples=10...")
    dbscan = DBSCAN(eps=1, min_samples=10)
    
    # Fit and predict labels
    # Labels of -1 imply noise (outliers).
    labels = dbscan.fit_predict(d)

    # Add labels to DataFrame
    d['labels'] = labels

    # Check unique labels to see how many clusters were found (and if -1 exists)
    unique_labels = set(labels)
    print(f"Labels found: {unique_labels}")
    if -1 in unique_labels:
        print("Note: Label -1 represents noise points.")

    # 4. Visualize the Clustered Data
    print("Plotting clustered data...")
    plt.figure(figsize=(8, 6))
    
    # Generate colors for clusters
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            label_name = "Noise"
        else:
            label_name = f"Cluster {k}"

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10, label=label_name)

    plt.title("DBSCAN Clustering Results")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()
    
    print("DBSCAN Example Completed.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_dbscan_example()
