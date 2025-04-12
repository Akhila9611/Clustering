import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(X, n_clusters=5, model_save_path='src/models/kmeans_model.pkl', scaler_save_path='src/models/scaler.pkl'):
    """
    Train KMeans model, evaluate using Silhouette score, and save the model and scaler to pickle files.
    """
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=123)
    kmeans.fit(X)
    
    # Save the KMeans model to a pickle file
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(kmeans, model_file)
    
    # Save the scaler used to transform the data to a pickle file
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    # Calculate Silhouette score for evaluation
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
    
    return kmeans, scaler
