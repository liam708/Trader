import pandas as pd
import numpy as np

from config import CONFIG
from regime_features import add_regime_features
from master_backtest import run_master_backtest


def simulate_from_weights(d: pd.DataFrame, weights: pd.Series, name: str) -> pd.DataFrame:
    """
    d: dataframe with at least ['date','Close'] sorted weekly, no NaNs
    weights: same length as d-1 (each week i weight applies from i->i+1)
    returns a curve dataframe with columns aligned to master.
    """
    start_cap = float(CONFIG["start_capital"])
    cost = float(CONFIG["cost_bps"]) / 10_000.0

    equity = start_cap
    prev_w = 0.0
    logs = []

    for i in range(len(d) - 1):
        date = pd.to_datetime(d.iloc[i]["date"])
        next_date = pd.to_datetime(d.iloc[i + 1]["date"])

        w = float(weights.iloc[i])

        px0 = float(d.iloc[i]["Close"])
        px1 = float(d.iloc[i + 1]["Close"])
        ret = (px1 / px0) - 1.0

        turnover = abs(w - prev_w)
        t_cost = turnover * cost
        cost_dollars = equity * t_cost  # dollar cost at time of trade

        net_ret = w * ret - t_cost
        prev_equity = equity
        equity *= (1.0 + net_ret)

        logs.append({
            "strategy": name,
            "date": date,
            "next_date": next_date,
            "pred_regime": 0.0,  # benchmark strategies are always "active"
            "weight": w,
            "turnover": turnover,
            "t_cost": t_cost,
            "cost_dollars": cost_dollars,
            "week_ret": ret,
            "net_ret": net_ret,
            "prev_equity": prev_equity,
            "equity": equity,
        })

        prev_w = w

    curve = pd.DataFrame(logs)
    curve["invested_dollars"] = curve["prev_equity"] * curve["weight"]
    return curve


def dollar_summary(curve: pd.DataFrame, label: str) -> dict:
    start_cap = float(CONFIG["start_capital"])
    final_cap = float(curve["equity"].iloc[-1])
    profit = final_cap - start_cap

    return {
        "Strategy": label,
        "Final Portfolio ($)": round(final_cap, 2),
        "Pure Profit ($)": round(profit, 2),
        "Avg Weekly Invested ($)": round(curve["invested_dollars"].mean(), 2),
        "Total Capital Deployed ($)": round(curve["invested_dollars"].sum(), 2),
        "Avg Weight": round(curve["weight"].mean(), 3),
        "Total Turnover": round(curve["turnover"].sum(), 2),
        "Total Cost Paid ($)": round(curve["cost_dollars"].sum(), 2),
        "Weeks": int(len(curve)),
    }


def main():
    # Load and prep prices
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add MA20 etc (safe features)
    feat = add_regime_features(df).dropna(subset=["ma_20w"]).reset_index(drop=True)

    # We will apply weights from i->i+1, so weights length = len(feat)-1
    n = len(feat)
    if n < 60:
        raise ValueError("Not enough data after feature construction.")

    # 1) Buy & Hold: always 1.0 exposure once we start
    w_bh = pd.Series([1.0] * (n - 1))

    # 2) MA20 Trend: long when Close > MA20, else flat
    w_ma20 = (feat["Close"] > feat["ma_20w"]).astype(float).iloc[:-1].reset_index(drop=True)

    # 3) Master: use your existing walk-forward model/controller
    master = run_master_backtest(df).copy()
    # Ensure master has invested_dollars for summary consistency
    master["prev_equity"] = master["equity"].shift(1)
    master.loc[master.index[0], "prev_equity"] = float(CONFIG["start_capital"])
    if "cost_dollars" not in master.columns:
        master["cost_dollars"] = master["prev_equity"] * master.get("t_cost", 0.0)
    master["invested_dollars"] = master["prev_equity"] * master["weight"]

    # Build benchmark curves using SAME cost model
    bh = simulate_from_weights(feat, w_bh, "Buy&Hold")
    ma20 = simulate_from_weights(feat, w_ma20, "MA20")

    # Align date ranges: master curve starts later (because of training window).
    # For fair comparison, trim bh and ma20 to master start date.
    master_start = pd.to_datetime(master["date"].iloc[0])
    bh = bh[bh["date"] >= master_start].reset_index(drop=True)
    ma20 = ma20[ma20["date"] >= master_start].reset_index(drop=True)

    # Summaries
    table = pd.DataFrame([
        dollar_summary(bh, "Buy & Hold SPY"),
        dollar_summary(ma20, "MA20 Trend"),
        dollar_summary(master, "Master (Regime + Policy)"),
    ])

    # Pretty print
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nSTRATEGY COMPARISON (aligned to Master start date)")
    print(table.to_string(index=False))

    # Save curves for plotting
    bh.to_csv("cache/curve_buyhold.csv", index=False)
    ma20.to_csv("cache/curve_ma20.csv", index=False)
    master.to_csv("cache/curve_master.csv", index=False)


if __name__ == "__main__":
    main()
