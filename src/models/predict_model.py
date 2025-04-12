def predict_clusters(kmeans, X_input_scaled):
    """
    Make predictions using the trained KMeans model on new input data.
    Args:
    - kmeans: The trained KMeans model
    - X_input_scaled: The scaled input data for which we want to predict the cluster
    Returns:
    - predictions: Predicted cluster labels for the input data
    """
    predictions = kmeans.predict(X_input_scaled)
    return predictions
