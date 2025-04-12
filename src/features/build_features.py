import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Preprocess the dataset: One-hot encode categorical variables and scale numerical features.
    """
    # One-hot encode categorical columns (Gender)
    df['Gender'] = pd.get_dummies(df['Gender'], drop_first=True)
    
    # Feature selection: Select relevant columns for clustering
    X = df[['Age', 'Annual_Income', 'Spending_Score', 'Gender']]
    
    # Scale numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return the scaled data, the original target labels (if any), and the scaler
    # If you have no target variable (y), return just X_scaled and the scaler
    return X_scaled, df['Cluster'] if 'Cluster' in df.columns else None, scaler


def create_clusters(df, n_clusters=5):
    """
    Perform clustering using KMeans and return the dataframe with cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    df['Cluster'] = kmeans.fit_predict(df[['Annual_Income', 'Spending_Score']])
    
    return df

def save_preprocessed_data(df, output_path='data/processed/preprocessed.csv'):
    """
    Save the preprocessed dataset to a new CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Example usage of preprocessing and clustering functions
if __name__ == "__main__":
    # Load the raw dataset
    df = pd.read_csv('data/raw/mall_customers.csv')

    # Preprocess the dataset (scale the features)
    X_scaled, scaler = preprocess_data(df)
    
    # Perform clustering and add cluster labels
    df_with_clusters = create_clusters(df)
    
    # Save the preprocessed data to CSV
    save_preprocessed_data(df_with_clusters)
