# config.py

CONFIG = {
    # Data / costs
    "start_capital": 10_000.0,
    "cost_bps": 2,  # round-trip proxy per week exposure

    # Walk-forward training
    "train_years": 10,    # rolling window
    "min_train_rows": 300,

    # Regime labeling (for training targets)
    "stress_dd_4w": -0.05,  # STRESS if future 4w drawdown <= -5%

    # Policy sizing
    "w_stress": 0.0,
    "w_trend": 1.0,
    "w_chop_base": 0.0,

    # CHOP mean-reversion rule (simple, interpretable)
    "mr_dist_ma20_entry": -0.02,  # if price is >2% below MA20 in CHOP => go long
    "w_chop_mr": 0.5,

    # Optional: if you want trend to require price above MA20 too (extra caution)
    "trend_requires_above_ma20": True,
}
