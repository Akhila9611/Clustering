import pandas as pd

def load_data(file_path='data/raw/mall_customers.csv'):
    """
    Load dataset from a CSV file and return it as a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def inspect_data(df):
    """
    Display basic information and summary statistics about the dataset.
    """
    print(df.shape)
    print(df.info())
    print(df.describe())
    return df.head()
