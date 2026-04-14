# src/utils.py

import numpy as np

def create_dataset(X, y, look_back=5):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i + look_back, 0])
        ys.append(y[i + look_back, 0])
    return np.array(Xs), np.array(ys)