import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Exemplo básico
    df = df.dropna()

    # Converter categóricos
    df = pd.get_dummies(df, drop_first=True)

    return df