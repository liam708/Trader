import pandas as pd

from backtest import backtest_weekly
from baselines import ma20_trend_signal
from model_signal import make_signal_fn

df = pd.read_csv("data/spy_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]

print("THRESHOLD SWEEP (MODEL-ONLY)")
for t in thresholds:
    model_signal = make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=t)
    res = backtest_weekly(df, model_signal, hold_weeks=4, cost_bps=2, start_capital=10_000)
    print(f"t={t:.2f}  trades={res['n_trades']:3d}  final={res['final_equity']:10.2f}  maxDD={100*res['max_drawdown_pct']:.2f}%")

print("\nTHRESHOLD SWEEP (MA20 + MODEL FILTER)")
for t in thresholds:
    model_signal = make_signal_fn(df, hold_weeks=4, cost_bps=2, prob_threshold=t)

    def combined(df_prices, i):
        return ma20_trend_signal(df_prices, i) and model_signal(df_prices, i)

    res = backtest_weekly(df, combined, hold_weeks=4, cost_bps=2, start_capital=10_000)
    print(f"t={t:.2f}  trades={res['n_trades']:3d}  final={res['final_equity']:10.2f}  maxDD={100*res['max_drawdown_pct']:.2f}%")
