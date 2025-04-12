import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def plot_clusters(df):
    """
    Plot the clusters using Seaborn scatter plot.
    """
    # Check if 'Cluster' exists or use 'Predicted_Cluster' for user input
    if 'Cluster' in df.columns:
        hue_column = 'Cluster'
    elif 'Predicted_Cluster' in df.columns:
        hue_column = 'Predicted_Cluster'
    else:
        raise ValueError("Neither 'Cluster' nor 'Predicted_Cluster' column found in the dataframe.")
    
    # Create a scatterplot to visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue=hue_column, palette='colorblind', s=100)
    
    plt.title('Mall Customer Segmentation')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend(title='Cluster')
    plt.show()

def plot_elbow_method(X, max_clusters=10):
    """
    Plot the Elbow Method to find the optimal number of clusters.
    """
    wcss = []  # List to store the Within-Cluster Sum of Squares (WCSS)
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # inertia is WCSS
    
    # Plotting the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

def plot_silhouette_score(X, max_clusters=10):
    """
    Plot silhouette score to evaluate clustering quality for different values of k.
    """
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Plotting the Silhouette Score for different cluster counts
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score For Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

def plot_silhouette_values(X, n_clusters=5):
    """
    Plot silhouette values for individual data points to assess cluster consistency.
    """
    from sklearn.metrics import silhouette_samples
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(X)
    silhouette_vals = silhouette_samples(X, kmeans.labels_)
    
    # Plotting silhouette values for each point
    plt.figure(figsize=(8, 6))
    plt.hist(silhouette_vals, bins=50, alpha=0.7)
    plt.title(f'Silhouette Values for {n_clusters} Clusters')
    plt.xlabel('Silhouette Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
