import pandas as pd

def preprocess_penguin_data(filepath):
    """
    Lädt, bereinigt und kodiert (One-Hot-Encoding) den Pinguin-Datensatz.

    Args:
        filepath (str): Pfad zur CSV-Datei.

    Returns:
        X (pd.DataFrame): Bereinigter und kodierter DataFrame mit Features.
        y (pd.Series): Zielvariable (Pinguin-Arten).
    """
    # Daten einlesen und DataFrame laden
    df = pd.read_csv(filepath)

    # Zeilen mit fehlenden Werten (NaN) entfernen
    df_clean = df.dropna()

    # One-Hot-Encoding für kategoriale Spalten 'island' und 'sex'
    df_encoded = pd.get_dummies(df_clean, columns=['island', 'sex'])

    # Features (X) sind alle Spalten außer der Zielvariable 'species'
    X = df_encoded.drop(columns=['species'])
    # Zielvariable (y) ist die Spalte 'species'
    y = df_encoded['species']

    return X, y