import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import make_weekly_dataset

FEATURES = ["ret_1w","ret_2w","ret_4w","vol_4w","vol_12w","dist_ma20","dist_ma50"]

def train_model(train_df):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    X = train_df[FEATURES].values
    y = train_df["y"].values
    model.fit(X, y)
    return model

def make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=0.55):
    dataset = make_weekly_dataset(df, hold_weeks, cost_bps)

    # Train on ALL history up to each point (simple version)
    def signal_fn(df_prices, i):
        date = pd.to_datetime(df_prices.loc[i, "date"])
        hist = dataset[dataset["date"] < date]
        if len(hist) < 200:
            return False

        model = train_model(hist)
        row = dataset[dataset["date"] == date]
        if row.empty:
            return False

        p = model.predict_proba(row[FEATURES].values)[0,1]
        return p >= prob_threshold

    return signal_fn
