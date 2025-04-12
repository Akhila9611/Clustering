import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from src.models.predict_model import predict_clusters
from src.visualization.visualize import plot_clusters, plot_elbow_method, plot_silhouette_score, plot_silhouette_values
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data  # Import the preprocess_data function

# Load the trained KMeans model and scaler
model_path = 'src/models/kmeans_model.pkl'
scaler_path = 'src/models/scaler.pkl'

with open(model_path, 'rb') as file:
    kmeans = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI for user input
st.title("Mall Customer Segmentation")

# Collect user input
age = st.slider("Age", 18, 100, 30)
annual_income = st.slider("Annual Income (in 1000 dollars)", 10, 150, 50)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Female" else 0  # Convert gender to 0 (Male) or 1 (Female)

# Prepare the input data for prediction (as a DataFrame)
input_data = pd.DataFrame([[age, annual_income, spending_score, gender]],
                          columns=["Age", "Annual_Income", "Spending_Score", "Gender"])

# Scale the input data using the trained scaler
input_scaled = scaler.transform(input_data)

# Make the prediction using the trained KMeans model
prediction = predict_clusters(kmeans, input_scaled)

# Add the predicted cluster to the input data
input_data['Predicted_Cluster'] = prediction

# Show the user input and the predicted cluster
st.write("### Your Input Data:")
st.write(input_data)

st.write(f"### Predicted Cluster: {prediction[0]}")

# Load the original dataset and preprocess it for visualization
df = load_data('data/raw/mall_customers.csv')
df_scaled, _, _ = preprocess_data(df)  # Preprocess the data (scaling)

# Predict clusters for the original dataset
df['Predicted_Cluster'] = predict_clusters(kmeans, df_scaled)

# Add the user input data to the original dataframe for visualization
df_with_input = pd.concat([df, input_data], ignore_index=True)

# Visualize the clusters, including the user input data
plot_clusters(df_with_input)  # Plot the clusters using scatterplot

# Plot the Elbow Method to determine the optimal number of clusters
plot_elbow_method(df_scaled, max_clusters=10)

# Plot the Silhouette Score for different numbers of clusters
plot_silhouette_score(df_scaled, max_clusters=10)

# Plot the Silhouette Values for the clusters
plot_silhouette_values(df_scaled, n_clusters=5)
