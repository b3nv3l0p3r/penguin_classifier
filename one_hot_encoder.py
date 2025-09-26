#one_hot_encoder.py
import pandas as pd

def preprocess_penguin_data(filepath):
    """
    Cleans and encodes the penguin dataset.

    Args:
        df (pd.DataFrame): Original data

    Returns:
        X (pd.DataFrame): Cleaned and one-hot encoded feature dataframe
        y (pd.Series): Target variable (penguin species)
    """
    # reading data and loading the dataframe
    df = pd.read_csv(filepath)

    # Dropping incomplete data
    df_clean = df.dropna()

    # Transforming categorical data into numeric values
    df_encoded = pd.get_dummies(df_clean, columns=['island', 'sex'])

    # Selecting target variable and features
    X = df_encoded.drop(columns=['species'])
    y = df_encoded['species']

    return X, y

