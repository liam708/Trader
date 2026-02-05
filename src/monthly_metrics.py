import numpy as np
import pandas as pd


def rolling_monthly_metrics(curve: pd.DataFrame, window=4) -> pd.DataFrame:
    """
    curve must have columns:
    ['date','equity','net_ret','weight','invested_dollars']
    window = number of weeks (4 ≈ 1 month)
    """
    rows = []

    for i in range(len(curve) - window):
        start_eq = curve.iloc[i]["equity"]
        end_eq = curve.iloc[i + window]["equity"]

        window_slice = curve.iloc[i:i+window+1]
        eq_path = window_slice["equity"].values

        # 1-month return
        ret = (end_eq / start_eq) - 1.0

        # max drawdown within window
        peak = np.maximum.accumulate(eq_path)
        dd = (eq_path / peak - 1.0).min()

        avg_invested = window_slice["invested_dollars"].mean()

        rows.append({
            "start_date": curve.iloc[i]["date"],
            "end_date": curve.iloc[i + window]["date"],
            "month_ret": ret,
            "month_dd": dd,
            "avg_invested": avg_invested,
            "loss": ret < 0,
        })

    return pd.DataFrame(rows)
