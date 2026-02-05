import pandas as pd
import numpy as np

def add_regime_labels(d: pd.DataFrame, stress_dd_4w: float = -0.05) -> pd.DataFrame:
    """
    Regime encoding:
      2 = STRESS  (future 4w drawdown <= threshold)
      1 = TREND   (price above MA20 and MA20 slope positive)
      0 = CHOP    (everything else)
    """
    out = d.copy()

    # Future min Close over next 4 weeks (for drawdown)
    fwd_min = (
        out["Close"]
        .shift(-1)
        .rolling(4, min_periods=4)
        .min()
        .shift(-3)
    )
    out["fwd_dd_4w"] = (fwd_min / out["Close"]) - 1.0  # negative = drawdown

    stress = out["fwd_dd_4w"] <= stress_dd_4w
    trend = (out["Close"] > out["ma_20w"]) & (out["ma20_slope8"] > 0)

    out["regime"] = 0
    out.loc[trend, "regime"] = 1
    out.loc[stress, "regime"] = 2  # stress overrides

    return out
