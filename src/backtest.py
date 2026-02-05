import pandas as pd

def backtest_weekly(df: pd.DataFrame, signal_fn, hold_weeks=4, cost_bps=2):
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = []

    in_pos = False
    entry_i = None
    entry_px = None

    for i in range(len(df) - hold_weeks - 1):
        px = float(df.loc[i, "Close"])

        # Exit after fixed holding period
        if in_pos and i == entry_i + hold_weeks:
            exit_px = float(df.loc[i, "Close"])
            gross_ret = (exit_px / entry_px) - 1.0
            cost = (cost_bps / 10_000)  # round-trip cost
            net_ret = gross_ret - cost

            equity += net_ret
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

            trades.append({
                "entry_date": df.loc[entry_i, "date"],
                "exit_date": df.loc[i, "date"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "net_ret": net_ret,
            })

            in_pos = False
            entry_i = None
            entry_px = None

        # Enter only if flat
        if (not in_pos) and signal_fn(df, i):
            in_pos = True
            entry_i = i
            entry_px = px

    return {
        "equity": equity,
        "max_drawdown": max_dd,
        "n_trades": len(trades),
        "trades": pd.DataFrame(trades),
    }
