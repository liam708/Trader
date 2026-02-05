import pandas as pd
import numpy as np

def compute_metrics(equity_curve: pd.DataFrame) -> dict:
    """
    equity_curve columns: date, equity, net_ret, weight
    """
    d = equity_curve.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    start = float(d["equity"].iloc[0])
    end = float(d["equity"].iloc[-1])

    # Weekly returns (already net_ret, but compute from equity too)
    weekly_ret = d["equity"].pct_change().dropna()

    # CAGR using year fraction
    years = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cagr = (end / start) ** (1 / years) - 1 if years > 0 else np.nan

    # Max drawdown
    peak = d["equity"].cummax()
    dd = (peak - d["equity"]) / peak
    max_dd = float(dd.max())

    # Vol + Sharpe (rough, weekly->annualized)
    vol_ann = float(weekly_ret.std() * np.sqrt(52)) if len(weekly_ret) > 2 else np.nan
    ret_ann = float(weekly_ret.mean() * 52) if len(weekly_ret) > 2 else np.nan
    sharpe = (ret_ann / vol_ann) if (vol_ann and vol_ann > 0) else np.nan

    return {
        "start_equity": start,
        "final_equity": end,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "ann_return": ret_ann,
        "ann_vol": vol_ann,
        "sharpe": sharpe,
        "weeks": int(len(d)),
        "avg_weight": float(d["weight"].mean()),
    }
