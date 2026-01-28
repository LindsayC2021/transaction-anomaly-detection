"""
This script evaluates the trained anomaly detection model. It will compare the model predictions to known fraud labels
"""

import pandas as pd
import numpy as np

from load_data import load_creditcard_data
from preprocess import preprocess_data
from train_model import train_model
from sklearn.metrics import precision_score, recall_score


if __name__ == "__main__":
    # Load the dataset
    df = load_creditcard_data()

    # Preprocess
    X, y = preprocess_data(df)

    # Train the model
    model = train_model(df)

    # Generate anomaly predictions
    predictions = model.predict(X)

    # Convert Isolation Forest predictions to fraud labels
    # -1 anomaly --> 1 fraud
    # 1 normal --> 0 normal 
    y_pred = np.where(predictions == -1, 1, 0)

    # Calculate precision and recall
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)


    # How many rows the model sees
    print("Total transactions:", len(X))

    # How many transactions the model thinks are anomalies
    print("Anomalies detected by model:", np.sum(predictions == -1))

    # The fraud count according to the model
    print("Actual fraud cases in data:", np.sum(y == 1))
    
    print("Precision:", precision)
    print("Recall:", recall)