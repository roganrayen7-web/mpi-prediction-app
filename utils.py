import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target, features):
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    return X, y