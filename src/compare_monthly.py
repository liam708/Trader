import pandas as pd
import numpy as np

from monthly_metrics import rolling_monthly_metrics
from compare_strategies import simulate_from_weights
from master_backtest import run_master_backtest
from regime_features import add_regime_features
from config import CONFIG


def summarize_monthly(df, label):
    return {
        "Strategy": label,
        "Mean 1M Ret (%)": 100 * df["month_ret"].mean(),
        "Median 1M Ret (%)": 100 * df["month_ret"].median(),
        "Worst 1M Ret (%)": 100 * df["month_ret"].min(),
        "5% Tail Ret (%)": 100 * df["month_ret"].quantile(0.05),
        "Loss Months (%)": 100 * df["loss"].mean(),
        "Avg $ Invested": df["avg_invested"].mean(),
        "Obs": len(df),
    }


def main():
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ===== MASTER =====
    master = run_master_backtest(df)
    m_month = rolling_monthly_metrics(master)

    # ===== BUY & HOLD =====
    feat = add_regime_features(df).dropna(subset=["ma_20w"]).reset_index(drop=True)
    w_bh = pd.Series([1.0] * (len(feat) - 1))
    bh = simulate_from_weights(feat, w_bh, "BuyHold")
    bh_month = rolling_monthly_metrics(bh)

    # ===== MA20 =====
    w_ma20 = (feat["Close"] > feat["ma_20w"]).astype(float).iloc[:-1]
    ma20 = simulate_from_weights(feat, w_ma20, "MA20")
    ma20_month = rolling_monthly_metrics(ma20)

    table = pd.DataFrame([
        summarize_monthly(bh_month, "Buy & Hold"),
        summarize_monthly(ma20_month, "MA20 Trend"),
        summarize_monthly(m_month, "Master Strategy"),
    ])

    pd.set_option("display.width", 200)
    print("\nROLLING 1-MONTH PERFORMANCE (4-week windows)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
