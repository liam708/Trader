import pandas as pd
import numpy as np

from baselines import ma20_trend_signal
from model_signal import make_signal_fn
from backtest_sized import backtest_weekly_sized

df = pd.read_csv("data/spy_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

# We'll use model probability via make_signal_fn, but we need p not just True/False.
# Quick hack: rebuild here using model_signal internals would be ideal later.
# For now: approximate sizing using multiple thresholds.
# Better version comes next step.

# Start with a simple rule: if model says trade at t=0.50 -> full size, else 0.
model_trade = make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=0.50)

def weight_fn(df_prices, i):
    # Only allow exposure when MA20 trend is positive
    if not ma20_trend_signal(df_prices, i):
        return 0.0
    return 1.0 if model_trade(df_prices, i) else 0.25  # fallback small exposure

res = backtest_weekly_sized(df, weight_fn, cost_bps=2, start_capital=10_000)

print("SIZED RESULTS (MA20 + model-weight)")
print("Final equity:", round(res["final_equity"], 2))
print("Max drawdown (%):", round(100 * res["max_drawdown_pct"], 2))
print(res["history"].tail())
