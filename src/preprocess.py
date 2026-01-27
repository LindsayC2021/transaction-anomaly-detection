# This script prepares raw transactional data for ML by separating labels and scaling numerical features.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# This function takes in a raw pandas DataFrame and returns model-ready features and labels 
def preprocess_data(df):
    # 'Class' is the truth label, 0 = normal transaction, 1 = fraudulent (anomalous) transaction.
    # We separate it so it isn't used as a feature
    y= df['Class']

    # Drop the label column from the feature set. axis = 1 means we are dropping a column and not a row
    X = df.drop('Class', axis = 1)

    # Initialize a StandardScaler to normalize the features to ensure all features have mean 0 and variance 1
    # This helps the model perform better
    scaler = StandardScaler()

    # Fit the scaler on the data and transform the features. The output is a NumPy array
    X_scaled = scaler.fit_transform(X)

    # Return the preprocessed features and the labels
    return X_scaled, y
