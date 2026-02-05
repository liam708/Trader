import pandas as pd
from backtest import backtest_weekly
from model_signal import make_signal_fn

df = pd.read_csv("data/spy_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

signal_fn = make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=0.55)

res = backtest_weekly(df, signal_fn, hold_weeks=4, cost_bps=2, start_capital=10_000)

print("MODEL RESULTS")
print("Trades:", res["n_trades"])
print("Final equity:", round(res["final_equity"], 2))
print("Max drawdown (%):", round(100 * res["max_drawdown_pct"], 2))
print(res["trades"].tail())
