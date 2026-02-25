import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from load_data import load_creditcard_data
from preprocess import preprocess_data

def train_model(df):
    """
    Train an Isolation Forest model on preprocessed data.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        model (IsolationForest): Trained Isolation Forest model
    """
    X, y = preprocess_data(df)

    model = IsolationForest(random_state=42, contamination=0.01)
    model.fit(X)

    return model

if __name__ == "__main__":
    df = load_creditcard_data()
    model = train_model(df)
    print("Model trained successfully!")