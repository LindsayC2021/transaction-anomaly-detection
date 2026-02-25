import os
import pandas as pd

def load_creditcard_data(filename="creditcard.csv"):
    """
    Load the credit card transactions dataset.

    Args:
        filename (str): Name of the CSV file in the `data/` folder.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # Path to the CSV
    DATA_PATH = os.path.join("data", filename)

    # Read the CSV into a DataFrame
    df = pd.read_csv(DATA_PATH)

    return df

if __name__=="__main__":
    df = load_creditcard_data()
    print("Data loaded successfully!")
    print(df.head())