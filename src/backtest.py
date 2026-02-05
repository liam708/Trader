import pandas as pd

def backtest_weekly(df: pd.DataFrame, signal_fn, hold_weeks=4, cost_bps=2, start_capital=10_000.0):
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    trades = []

    in_pos = False
    entry_i = None
    entry_px = None

    for i in range(len(df) - hold_weeks - 1):
        px = float(df.loc[i, "Close"])

        # exit
        if in_pos and i == entry_i + hold_weeks:
            exit_px = float(df.loc[i, "Close"])
            gross_ret = (exit_px / entry_px) - 1.0
            cost = (cost_bps / 10_000)  # round-trip
            net_ret = gross_ret - cost

            equity *= (1.0 + net_ret)
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

            trades.append({
                "entry_date": df.loc[entry_i, "date"],
                "exit_date": df.loc[i, "date"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "net_ret": net_ret,
                "equity": equity,
            })

            in_pos = False
            entry_i = None
            entry_px = None

        # enter
        if (not in_pos) and signal_fn(df, i):
            in_pos = True
            entry_i = i
            entry_px = px

    return {
        "final_equity": equity,
        "max_drawdown_pct": max_dd,
        "n_trades": len(trades),
        "trades": pd.DataFrame(trades),
    }
