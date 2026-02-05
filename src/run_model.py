import pandas as pd

from backtest import backtest_weekly
from baselines import ma20_trend_signal
from model_signal import make_signal_fn

df = pd.read_csv("data/spy_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

# This is your model-based signal (probability threshold inside)
model_signal = make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=0.55)

def combined_signal(df_prices, i):
    # Only trade when the MA20 trend filter says "allowed"
    if not ma20_trend_signal(df_prices, i):
        return False
    # Then require the model to also agree
    return model_signal(df_prices, i)

res = backtest_weekly(df, combined_signal, hold_weeks=4, cost_bps=2, start_capital=10_000)

print("COMBINED (MA20 + MODEL FILTER) RESULTS")
print("Trades:", res["n_trades"])
print("Final equity:", round(res["final_equity"], 2))
print("Max drawdown (%):", round(100 * res["max_drawdown_pct"], 2))
print(res["trades"].tail())
