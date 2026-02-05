import pandas as pd
import numpy as np

def make_regime_dataset(df: pd.DataFrame):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    # Weekly return
    d["ret_1w"] = d["Close"].pct_change(1)

    # Rolling vol proxy
    d["vol_12w"] = d["ret_1w"].rolling(12).std()

    # Trend features
    d["ma_20w"] = d["Close"].rolling(20).mean()
    d["ma_50w"] = d["Close"].rolling(50).mean()
    d["dist_ma20"] = (d["Close"] / d["ma_20w"]) - 1.0
    d["dist_ma50"] = (d["Close"] / d["ma_50w"]) - 1.0

    # Trend slope: compare current MA20 to MA20 from 8 weeks ago
    d["ma20_slope8"] = d["ma_20w"] / d["ma_20w"].shift(8) - 1.0

    # Forward downside risk label: max drawdown over next 4 weeks
    # Compute future min close over next 4 weeks
    fwd_min = (
        d["Close"]
        .shift(-1)
        .rolling(4, min_periods=4)
        .min()
        .shift(-3)
    )
    d["fwd_dd_4w"] = (fwd_min / d["Close"]) - 1.0  # negative = drawdown

    # Regime definition (tune thresholds later, but start here)
    # STRESS: expect >= 5% drawdown within next 4 weeks
    stress = d["fwd_dd_4w"] <= -0.05

    # TREND: price above MA20 and MA20 slope positive
    trend = (d["Close"] > d["ma_20w"]) & (d["ma20_slope8"] > 0)

    # Encode regimes: 2=STRESS, 1=TREND, 0=CHOP
    d["regime"] = 0
    d.loc[trend, "regime"] = 1
    d.loc[stress, "regime"] = 2  # stress overrides everything

    feature_cols = ["ret_1w", "vol_12w", "dist_ma20", "dist_ma50", "ma20_slope8"]
    out = d[["date"] + feature_cols + ["regime"]].dropna().reset_index(drop=True)
    return out
