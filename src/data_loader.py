# data_loader.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    # Additional data preprocessing steps
    # ...
    X = df[['Glucose', 'BMI', 'Age']]  # Example feature selection
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
