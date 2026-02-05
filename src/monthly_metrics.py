import numpy as np
import pandas as pd


def rolling_monthly_metrics(curve: pd.DataFrame, window=4, start_capital: float = 10000.0) -> pd.DataFrame:
    """
    curve must have: ['date','equity','weight']
    Optional: ['invested_dollars'] or ['prev_equity']
    window = number of weeks (4 ≈ 1 month)
    """
    c = curve.copy()

    # If invested_dollars isn't present, compute it safely
    if "invested_dollars" not in c.columns:
        if "prev_equity" not in c.columns:
            c["prev_equity"] = c["equity"].shift(1)
            c.loc[c.index[0], "prev_equity"] = float(start_capital)
        c["invested_dollars"] = c["prev_equity"] * c["weight"]

    rows = []
    for i in range(len(c) - window):
        start_eq = c.iloc[i]["equity"]
        end_eq = c.iloc[i + window]["equity"]

        window_slice = c.iloc[i:i+window+1]
        eq_path = window_slice["equity"].values

        ret = (end_eq / start_eq) - 1.0

        peak = np.maximum.accumulate(eq_path)
        dd = (eq_path / peak - 1.0).min()

        avg_invested = window_slice["invested_dollars"].mean()

        rows.append({
            "start_date": c.iloc[i]["date"],
            "end_date": c.iloc[i + window]["date"],
            "month_ret": ret,
            "month_dd": dd,
            "avg_invested": avg_invested,
            "loss": ret < 0,
        })

    return pd.DataFrame(rows)
