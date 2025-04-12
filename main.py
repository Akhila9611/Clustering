import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.data.make_dataset import load_data, inspect_data
from src.features.build_features import preprocess_data, create_clusters
from src.models.train_model import train_kmeans
from src.models.predict_model import predict_clusters
from src.visualization.visualize import plot_clusters, plot_elbow_method, plot_silhouette_score, plot_silhouette_values

def main():
    # Load and inspect data
    df = load_data('data/raw/mall_customers.csv')
    
    # Check columns in the dataframe to ensure 'CustomerID' exists
    print("Columns in the dataset:", df.columns)

    inspect_data(df)
    
    # Preprocess the data and create clusters
    df = create_clusters(df)  # Creates 'Cluster' column with cluster labels
    
    # Extract features and preprocess them (scaling)
    X_scaled, y, scaler = preprocess_data(df)
    
    # Train the KMeans model
    n_clusters = 5  # You can change this based on your model
    kmeans, scaler = train_kmeans(X_scaled, n_clusters=n_clusters)

    # Make predictions on the dataset
    predictions = predict_clusters(kmeans, X_scaled)
    df['Predicted_Cluster'] = predictions
    
    # Display the cluster assignment
    if 'CustomerID' in df.columns:
        print("Cluster Assignments for the First 10 Entries:")
        print(df[['CustomerID', 'Predicted_Cluster']].head(10))
    else:
        print("CustomerID column not found. Here are the cluster assignments:")
        print(df[['Predicted_Cluster']].head(10))

    # Visualize the clusters
    plot_clusters(df)  # Visualizes the clustering of customers based on spending score and income
    plot_elbow_method(X_scaled, max_clusters=10)  # Elbow plot for optimal clusters
    plot_silhouette_score(X_scaled, max_clusters=10)  # Silhouette score plot for optimal clusters
    plot_silhouette_values(X_scaled, n_clusters=n_clusters)  # Silhouette values for individual points

if __name__ == "__main__":
    main()
