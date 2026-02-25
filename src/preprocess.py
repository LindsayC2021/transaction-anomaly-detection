import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess raw transaction data for model training.

    - Separates labels (Class)
    - Scales features to mean 0 and variance 1

    Args:
        df (pd.DataFrame): Raw dataset with 'Class' column.

    Returns:
        X_scaled (np.ndarray): Normalized features.
        y (pd.Series): Labels (0 = normal, 1 = fraud).
    """
    y = df['Class']
    X = df.drop('Class', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y