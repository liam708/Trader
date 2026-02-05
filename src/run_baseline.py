import pandas as pd
from backtest import backtest_weekly
from baselines import ma20_trend_signal

df = pd.read_csv("data/spy_weekly.csv")
res = backtest_weekly(df, ma20_trend_signal, hold_weeks=4, cost_bps=2, start_capital=10_000)

print("Trades:", res["n_trades"])
print("Final equity:", round(res["final_equity"], 2))
print("Max drawdown (%):", round(100 * res["max_drawdown_pct"], 2))
print(res["trades"].tail())
