"""
This script is responsible for fitting an anomaly detection model on preprocessed data
Training only on features (X) 
Isolation Forest will be used as it is the best model choice for anomaly detection. 
It detects anomalies by isolating points that require fewer splits in a random tree structure

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
# Load the data 
from load_data import load_creditcard_data
from preprocess import preprocess_data

def train_model(df):
    # Preprocess the data, uses the function created in preprocess.py. X = preprocessed features the model will train on. 
    # y = labels (Class) that indicate fraud or normal transaction, this is not being used for training.
    X, y = preprocess_data(df)

    # Create the Isolation Forest. random_state ensures results are reproducible. 
    # Contamination estimates the proportion of anomalies in the dataset (1% for fraud)
    model = IsolationForest(random_state = 42, contamination = 0.01)

    # Fit only on features. The model learns patterns of normal transactions
    # Later, it will be able to flag transactions that deviate significantly from these patterns
    model.fit(X)

    # Return the trained model. The model can now be used to predict anomalies on new transactional data
    return model

# allows this script to be run directly for testing
if __name__ == "__main__":

    # Load the dataset to test
    df = load_creditcard_data()

    # Train the Isolation Forest model on the data
    model = train_model(df)

    # Confirm it ran
    print("Model trained successfully!")