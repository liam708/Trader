import pandas as pd
import numpy as np

def make_weekly_dataset(df: pd.DataFrame, hold_weeks: int = 4, cost_bps: int = 2) -> pd.DataFrame:
    d = df.copy()

    # Ensure sorted
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    # Weekly returns
    d["ret_1w"] = d["Close"].pct_change(1)
    d["ret_2w"] = d["Close"].pct_change(2)
    d["ret_4w"] = d["Close"].pct_change(4)

    # Volatility (rolling std of 1w returns)
    d["vol_4w"] = d["ret_1w"].rolling(4).std()
    d["vol_12w"] = d["ret_1w"].rolling(12).std()

    # Trend distance from MAs
    d["ma_20w"] = d["Close"].rolling(20).mean()
    d["ma_50w"] = d["Close"].rolling(50).mean()
    d["dist_ma20"] = (d["Close"] / d["ma_20w"]) - 1.0
    d["dist_ma50"] = (d["Close"] / d["ma_50w"]) - 1.0

    # Forward return label over hold_weeks
    d["fwd_ret"] = d["Close"].shift(-hold_weeks) / d["Close"] - 1.0

    # Subtract simple round-trip cost proxy
    cost = cost_bps / 10_000
    d["fwd_net"] = d["fwd_ret"] - cost

    # Classification label: trade worth it?
    d["y"] = (d["fwd_net"] > 0).astype(int)

    # Drop rows that can't be computed
    feature_cols = ["ret_1w","ret_2w","ret_4w","vol_4w","vol_12w","dist_ma20","dist_ma50"]
    out = d[["date"] + feature_cols + ["fwd_net","y"]].dropna().reset_index(drop=True)
    return out
