import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def run_kmeans_example():
   
    print("Generating dataset...")
    X, _ = make_blobs(n_samples=100, random_state=12)

    # Convert the data into a Pandas DataFrame for easier handling
    d = pd.DataFrame(X)
    
    # 2. Visualize the raw data (Before Clustering)
    print("Plotting raw data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(d[0], d[1], c='gray', marker='o', edgecolor='k', s=50)
    plt.title("Raw Data (Before Clustering)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show() # This will open a window with the plot

    # 3. Apply K-Means Clustering
    print("Applying K-Means clustering with 3 clusters...")
    kmeans = KMeans(n_clusters=3, random_state=12)
    
    # Fit the model to our data
    kmeans.fit(d)

    # 4. Predict the cluster labels
    labels = kmeans.predict(d)
    
    # Add the labels to our DataFrame so we can see which cluster each point belongs to
    d['labels'] = labels
    
    # 5. Get Cluster Centers
    # The centroids are the center points of each cluster.
    centroids = kmeans.cluster_centers_
    print("Cluster Centers:\n", centroids)

    # 6. Visualize the Clustered Data
    # We separate the data points based on their assigned label.
    d0 = d[d['labels'] == 0]
    d1 = d[d['labels'] == 1]
    d2 = d[d['labels'] == 2]

    print("Plotting clustered data...")
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster with a different color
    plt.scatter(d0[0], d0[1], c='red', label='Cluster 0', s=50, edgecolor='k')
    plt.scatter(d1[0], d1[1], c='yellow', label='Cluster 1', s=50, edgecolor='k')
    plt.scatter(d2[0], d2[1], c='green', label='Cluster 2', s=50, edgecolor='k')
    
    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=200, marker='X', label='Centroids')

    plt.title("K-Means Clustering Results")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show() # Open the second plot

if __name__ == "__main__":
    run_kmeans_example()
