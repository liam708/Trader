import pandas as pd

def backtest_weekly_sized(df: pd.DataFrame, weight_fn, hold_weeks=4, cost_bps=2, start_capital=10_000.0):
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    trades = []

    # We'll rebalance weekly, but still measure return over next week and compound.
    # This is simpler and more realistic for sizing than "hold 4 weeks then exit".
    # weight_fn(df, i) returns weight in [0,1]
    cost = cost_bps / 10_000

    for i in range(len(df) - 1):
        w = float(weight_fn(df, i))
        w = max(0.0, min(1.0, w))

        px0 = float(df.loc[i, "Close"])
        px1 = float(df.loc[i+1, "Close"])
        ret = (px1 / px0) - 1.0

        # Apply cost only when weight changes materially (rough proxy)
        # Simple: pay cost if w > 0 (enter/hold), not perfect but ok for now
        net_ret = w * ret - (cost if w > 0 else 0.0)

        equity *= (1.0 + net_ret)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

        trades.append({
            "date": df.loc[i, "date"],
            "weight": w,
            "week_ret": ret,
            "net_ret": net_ret,
            "equity": equity
        })

    return {
        "final_equity": equity,
        "max_drawdown_pct": max_dd,
        "history": pd.DataFrame(trades)
    }
