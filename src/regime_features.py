import pandas as pd
import numpy as np

FEATURES = ["ret_1w", "vol_12w", "dist_ma20", "dist_ma50", "ma20_slope8"]

def add_regime_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    d = df_prices.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    d["ret_1w"] = d["Close"].pct_change(1)
    d["vol_12w"] = d["ret_1w"].rolling(12).std()

    d["ma_20w"] = d["Close"].rolling(20).mean()
    d["ma_50w"] = d["Close"].rolling(50).mean()
    d["dist_ma20"] = (d["Close"] / d["ma_20w"]) - 1.0
    d["dist_ma50"] = (d["Close"] / d["ma_50w"]) - 1.0

    # Trend slope (8-week change in MA20)
    d["ma20_slope8"] = d["ma_20w"] / d["ma_20w"].shift(8) - 1.0

    return d
