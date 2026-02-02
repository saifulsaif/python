import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

def run_gmm_example():
    """
    Runs the Gaussian Mixture Model (GMM) clustering example.
    """
    print("---------------------------------------------------------")
    print("Running GMM Clustering Example...")
    print("---------------------------------------------------------")

    # 1. Generate the dataset
    # We create a synthetic dataset using make_blobs, similar to the K-Means example.
    # We use the same random_state=12 to allow for comparison between the two methods.
    print("Generating dataset...")
    X, _ = make_blobs(n_samples=100, random_state=12)

    # Convert to DataFrame
    d = pd.DataFrame(X)

    # 2. Visualize the raw data
    print("Plotting raw data...")
    plt.figure(figsize=(8, 6))
    plt.scatter(d[0], d[1], c='gray', marker='o', edgecolor='k', s=50)
    plt.title("Raw Data (Before GMM Clustering)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

    # 3. Apply GMM Clustering
    # We initialize the GaussianMixture model.
    # n_components=3: This is analogous to K in K-Means, asking for 3 gaussian distributions.
    print("Applying GMM clustering with 3 components (clusters)...")
    gmm = GaussianMixture(n_components=3, random_state=12)

    # Fit the data
    # GMM tries to find the best mix of Gaussian distributions that generated this data.
    gmm.fit(d)

    # 4. Predict the cluster labels
    # Assign each data point to the most likely Gaussian component.
    labels = gmm.predict(d)

    # Check for convergence
    print('Converged:', gmm.converged_) 

    # Get the final means and covariances
    means = gmm.means_
    covariances = gmm.covariances_
    print("Means:\n", means)
    print("Covariances:\n", covariances)

    # Add labels to DataFrame
    d['labels'] = labels

    # 5. Visualize the Clustered Data
    d0 = d[d['labels'] == 0]
    d1 = d[d['labels'] == 1]
    d2 = d[d['labels'] == 2]

    print("Plotting clustered data...")
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster
    plt.scatter(d0[0], d0[1], c='red', label='Cluster 0', s=50, edgecolor='k')
    plt.scatter(d1[0], d1[1], c='yellow', label='Cluster 1', s=50, edgecolor='k')
    plt.scatter(d2[0], d2[1], c='green', label='Cluster 2', s=50, edgecolor='k')
    
    # Plot the means (centers of the Gaussians)
    plt.scatter(means[:, 0], means[:, 1], c='blue', s=200, marker='X', label='Means')

    plt.title("GMM Clustering Results")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()

    print("GMM Example Completed.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_gmm_example()
