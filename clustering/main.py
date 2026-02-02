import sys
import kmeans_clustering
import gmm_clustering
import agglomerative_clustering
import dbscan_clustering

def main():
    """
    Main entry point to run the clustering examples.
    """
    print("Welcome to the Scikit-Learn Clustering Demo!")
    print("Select an algorithm to run:")
    print("1. K-Means Clustering")
    print("2. Gaussian Mixture Model (GMM) Clustering")
    print("3. Agglomerative Hierarchical Clustering")
    print("4. DBSCAN Clustering")
    print("5. Run All")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            kmeans_clustering.run_kmeans_example()
        elif choice == '2':
            gmm_clustering.run_gmm_example()
        elif choice == '3':
            agglomerative_clustering.run_agglomerative_example()
        elif choice == '4':
            dbscan_clustering.run_dbscan_example()
        elif choice == '5':
            kmeans_clustering.run_kmeans_example()
            gmm_clustering.run_gmm_example()
            agglomerative_clustering.run_agglomerative_example()
            dbscan_clustering.run_dbscan_example()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
