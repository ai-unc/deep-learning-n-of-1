# src/train_xgboost.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils import create_dataset


def main():
    df = pd.read_csv("data/wellness.csv")

    df["effective_time_frame"] = pd.to_datetime(df["effective_time_frame"])
    df = df.sort_values("effective_time_frame").dropna().reset_index(drop=True)

    # AHP weights
    W_SLEEP_DURATION = 0.30
    W_SLEEP_QUALITY = 0.25
    W_FATIGUE = 0.20
    W_STRESS = 0.15
    W_MOOD = 0.10

    df["wellness_score"] = (
        W_SLEEP_DURATION * df["sleep_duration_h"] +
        W_SLEEP_QUALITY * df["sleep_quality"] +
        W_FATIGUE * (5 - df["fatigue"]) +
        W_STRESS * (5 - df["stress"]) +
        W_MOOD * df["mood"]
    )

    df["readiness_smooth"] = df["readiness"].rolling(5).mean()
    df = df.dropna()

    X_raw = df[["wellness_score"]].values.astype("float32")
    y_raw = df[["readiness_smooth"]].values.astype("float32")

    train_size = int(len(X_raw) * 0.7)

    X_train_raw = X_raw[:train_size]
    X_test_raw = X_raw[train_size:]
    y_train_raw = y_raw[:train_size]
    y_test_raw = y_raw[train_size:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    look_back = 5
    trainX, trainY = create_dataset(X_train, y_train, look_back)
    testX, testY = create_dataset(X_test, y_test, look_back)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(trainX, trainY)

    pred = model.predict(testX)
    pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
    testY = scaler_y.inverse_transform(testY.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(testY, pred))
    print(f"Test RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
