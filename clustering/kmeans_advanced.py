import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def run_advanced_kmeans():

    # 1. Load data from CSV file
    # Features: Annual Income (k$) and Spending Score (1-100)
    print("Loading data from customer_data.csv...")
    try:
        df = pd.read_csv('customer_data.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: customer_data.csv not found. Please ensure the file exists.")
        return

    # Extracting features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

    # 2. Visualize the Raw Data
    print("Visualizing raw data...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', s=60)
    plt.title('Customer Data Before Clustering')
    plt.show()


    #. Apply K-Means with the optimal number of clusters (k=5 based on blobs)
    optimal_k = 5
    print(f"Applying K-Means clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(df)
    
    df['Cluster'] = y_kmeans
    centroids = kmeans.cluster_centers_

    #. Visualize the Clustered Clusters
    print("Visualizing the resulting clusters...")
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    
    for i in range(optimal_k):
        sns.scatterplot(
            data=df[df['Cluster'] == i], 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)', 
            s=60, 
            label=f'Cluster {i}',
            color=colors[i]
        )
        
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', label='Centroids', marker='X', edgecolor='black')
    
    plt.title('Clusters of Customers (k=5)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_advanced_kmeans()
