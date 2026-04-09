# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = pd.read_csv(path)
    return data


def preprocess_data(data):
    X = data.drop(columns='Class', axis=1)
    Y = data['Class']

    scaler = StandardScaler()

    # Normalize important columns
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])

    return X, Y
