import numpy as np
import pandas as pd
from skmultiflow.data import AGRAWALGenerator, SEAGenerator
import requests
import os

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate and save synthetic datasets with abrupt drifts
def generate_synthetic_data():
    # 1. AGRAWALGenerator with two abrupt drifts
    agrawal_data = []
    agrawal_labels = []

    # First segment: instances 0 to 34,999 (classification_function=0)
    agrawal_stream = AGRAWALGenerator(random_state=42, classification_function=0)
    for _ in range(35000):
        X, y = agrawal_stream.next_sample()
        agrawal_data.append(X[0])
        agrawal_labels.append(y[0])

    # Second segment: instances 35,000 to 59,999 (classification_function=1)
    agrawal_stream = AGRAWALGenerator(random_state=42, classification_function=1)
    for _ in range(25000):
        X, y = agrawal_stream.next_sample()
        agrawal_data.append(X[0])
        agrawal_labels.append(y[0])

    # Third segment: instances 60,000 to 99,999 (classification_function=2)
    agrawal_stream = AGRAWALGenerator(random_state=42, classification_function=2)
    for _ in range(40000):
        X, y = agrawal_stream.next_sample()
        agrawal_data.append(X[0])
        agrawal_labels.append(y[0])

    # Convert to DataFrame
    agrawal_df = pd.DataFrame(agrawal_data, columns=[f'feature_{i}' for i in range(len(agrawal_data[0]))])
    agrawal_df['label'] = agrawal_labels

    # Save to file
    agrawal_df.to_csv('AGRAWALGenerator.csv', index=False)
    print("AGRAWALGenerator dataset with drifts at 35k and 60k saved to AGRAWALGenerator.csv")

    # 2. SEAGenerator with two abrupt drifts
    sea_data = []
    sea_labels = []

    # First segment: instances 0 to 34,999 (classification_function=0)
    sea_stream = SEAGenerator(random_state=42, classification_function=0, noise_percentage=0.1)
    for _ in range(35000):
        X, y = sea_stream.next_sample()
        sea_data.append(X[0])
        sea_labels.append(y[0])

    # Second segment: instances 35,000 to 59,999 (classification_function=1)
    sea_stream = SEAGenerator(random_state=42, classification_function=1, noise_percentage=0.1)
    for _ in range(25000):
        X, y = sea_stream.next_sample()
        sea_data.append(X[0])
        sea_labels.append(y[0])

    # Third segment: instances 60,000 to 99,999 (classification_function=2)
    sea_stream = SEAGenerator(random_state=42, classification_function=2, noise_percentage=0.1)
    for _ in range(40000):
        X, y = sea_stream.next_sample()
        sea_data.append(X[0])
        sea_labels.append(y[0])

    # Convert to DataFrame
    sea_df = pd.DataFrame(sea_data, columns=[f'feature_{i}' for i in range(len(sea_data[0]))])
    sea_df['label'] = sea_labels

    # Save to file
    sea_df.to_csv('SEADataset.csv', index=False)
    print("SEADataset with drifts at 35k and 60k saved to SEADataset.csv")

# Function to download and load real datasets
def load_real_datasets():
    # URLs for the datasets from the GitHub repository
    spam_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv"
    electricity_url = "https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/elec.csv"

    # Download and save Spam dataset
    try:
        response = requests.get(spam_url)
        response.raise_for_status()
        with open('Spam.csv', 'w') as f:
            f.write(response.text)
        print("Spam dataset downloaded and saved to Spam.csv")
    except Exception as e:
        print(f"Error downloading Spam dataset: {e}")

    # Download and save Electricity dataset
    try:
        response = requests.get(electricity_url)
        response.raise_for_status()
        with open('Electricity.csv', 'w') as f:
            f.write(response.text)
        print("Electricity dataset downloaded and saved to Electricity.csv")
    except Exception as e:
        print(f"Error downloading Electricity dataset: {e}")

# Execute the functions
if __name__ == "__main__":
    # Generate and save synthetic datasets
    generate_synthetic_data()

    # Download and save real datasets
    load_real_datasets()