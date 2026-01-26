import os
import pandas as pd

def load_creditcard_data(filename="creditcard.csv"):
    #Path to the CSV
    DATA_PATH = os.path.join("data", filename)

    #Read the CSV into the Dataframe
    df = pd.read_csv(DATA_PATH)

    #Return the Dataframe so other scripts can use it
    return df

if __name__=="__main__":
    df = load_creditcard_data()
    print("Data loaded successfully!")
    print(df.head())