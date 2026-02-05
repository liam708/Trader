import pandas as pd
from backtest import backtest_weekly
from baselines import ma20_trend_signal

df = pd.read_csv("data/spy_weekly.csv")
res = backtest_weekly(df, ma20_trend_signal, hold_weeks=4, cost_bps=2)

print("Trades:", res["n_trades"])
print("Equity (sum net returns):", round(res["equity"], 4))
print("Max DD (same units):", round(res["max_drawdown"], 4))
print(res["trades"].tail())
