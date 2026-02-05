import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import CONFIG
from regime_features import add_regime_features, FEATURES
from regime_labels import add_regime_labels
from metrics import compute_metrics


def policy_weight(row: pd.Series, pred_regime: int) -> float:
    """
    row is a single row (pd.Series) from the feature dataframe
    pred_regime: 2=STRESS, 1=TREND, 0=CHOP
    """
    close = float(row["Close"])
    ma20 = float(row["ma_20w"])
    dist = float(row["dist_ma20"])

    if pred_regime == 2:
        return float(CONFIG["w_stress"])

    if pred_regime == 1:
        if CONFIG.get("trend_requires_above_ma20", True) and not (close > ma20):
            return 0.0
        return float(CONFIG["w_trend"])

    # CHOP: mean-reversion entry if far below MA20
    if dist <= float(CONFIG["mr_dist_ma20_entry"]):
        return float(CONFIG["w_chop_mr"])
    return float(CONFIG["w_chop_base"])


def run_master_backtest(df_prices: pd.DataFrame) -> pd.DataFrame:
    # Build features (safe, no future)
    d = add_regime_features(df_prices)

    # Add labels (uses future, but only to train targets)
    d = add_regime_labels(d, stress_dd_4w=CONFIG["stress_dd_4w"])

    # Keep only what we need and drop NaNs
    needed = ["date", "Close", "ma_20w"] + FEATURES + ["regime"]    
    d = d[needed].dropna().reset_index(drop=True)

    # Rolling walk-forward model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
    ])

    equity = float(CONFIG["start_capital"])
    cost = float(CONFIG["cost_bps"]) / 10_000.0

    logs = []
    prev_w = 0.0  # previous week's weight

    for i in range(len(d) - 1):
        row = d.iloc[i]
        date = pd.to_datetime(row["date"])
        next_date = pd.to_datetime(d.iloc[i + 1]["date"])

        # Rolling training window: last N years ending at "date" (exclusive)
        train_end = date
        train_start = train_end - pd.Timedelta(days=int(float(CONFIG["train_years"]) * 365.25))

        # Filter training data strictly before current date
        dates = pd.to_datetime(d["date"])
        train = d[(dates < train_end) & (dates >= train_start)]

        if len(train) < int(CONFIG["min_train_rows"]):
            pred = None
            w = 0.0
        else:
            Xtr = train[FEATURES].values
            ytr = train["regime"].values
            model.fit(Xtr, ytr)

            x = row[FEATURES].values.reshape(1, -1)
            pred = int(model.predict(x)[0])
            w = float(policy_weight(row, pred))

        # Next week return
        px0 = float(row["Close"])
        px1 = float(d.iloc[i + 1]["Close"])
        ret = (px1 / px0) - 1.0

        # Transaction cost paid only when weight changes (turnover-based)
        turnover = abs(w - prev_w)
        t_cost = turnover * cost

        net_ret = w * ret - t_cost
        cost_dollars = equity * t_cost
        equity *= (1.0 + net_ret)
        
        logs.append({
            "date": date,
            "next_date": next_date,
            "pred_regime": pred,
            "weight": w,
            "turnover": turnover,
            "t_cost": t_cost,
            "cost_dollars": cost_dollars,
            "week_ret": ret,
            "net_ret": net_ret,
            "equity": equity,
        })
        prev_w = w
    return pd.DataFrame(logs)


if __name__ == "__main__":
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])

    curve = run_master_backtest(df)
    curve["prev_equity"] = curve["equity"].shift(1)
    curve.loc[curve.index[0], "prev_equity"] = CONFIG["start_capital"]

    curve["invested_dollars"] = curve["prev_equity"] * curve["weight"]
    active = curve[curve["pred_regime"].notna()].copy()
    m = compute_metrics(curve)
    start_cap = float(CONFIG["start_capital"])
    final_cap = curve["equity"].iloc[-1]
    profit = final_cap - start_cap
    def dollar_summary(df, name):
    start_cap = float(CONFIG["start_capital"])
    final_cap = float(df["equity"].iloc[-1])
    profit = final_cap - start_cap
    return {
        "Period": name,
        "Final Portfolio ($)": round(final_cap, 2),
        "Pure Profit ($)": round(profit, 2),
        "Avg Weekly Invested ($)": round(df["invested_dollars"].mean(), 2),
        "Total Capital Deployed ($)": round(df["invested_dollars"].sum(), 2),
        "Avg Weight": round(df["weight"].mean(), 3),
        "Total Turnover": round(df["turnover"].sum(), 2),
        "Total Cost Paid ($)": round(df["cost_dollars"].sum(), 2),
        "Weeks": int(len(df)),
    }

    summary = pd.DataFrame([
        dollar_summary(curve, "FULL"),
        dollar_summary(active, "ACTIVE (pred_regime!=None)")
    ])

    print(summary.to_string(index=False))
    summary = pd.DataFrame([{
        "Start Capital ($)": round(start_cap, 2),
        "Final Portfolio ($)": round(final_cap, 2),
        "Pure Profit ($)": round(profit, 2),
        "Avg Weekly Invested ($)": round(curve["invested_dollars"].mean(), 2),
        "Total Capital Deployed ($)": round(curve["invested_dollars"].sum(), 2),
        "Avg Weight": round(curve["weight"].mean(), 3),
        "Total Turnover": round(curve["turnover"].sum(), 2),
        "Total Cost Paid ($)": round(curve["cost_dollars"].sum(), 2),
    }])

    print("\nDOLLAR SUMMARY")
    print(summary.to_string(index=False))
    print("MASTER REGIME + POLICY BACKTEST")
    print(f"Weeks: {m['weeks']}")
    print(f"Start: {m['start_equity']:.2f}  Final: {m['final_equity']:.2f}")
    print(f"CAGR: {100*m['cagr']:.2f}%")
    print(f"Max DD: {100*m['max_drawdown']:.2f}%")
    print(f"AnnRet: {100*m['ann_return']:.2f}%  AnnVol: {100*m['ann_vol']:.2f}%  Sharpe: {m['sharpe']:.2f}")
    print(f"Avg weight: {m['avg_weight']:.2f}")

    # 🔹 THIS GOES HERE 🔹
    print(f"Total turnover: {curve['turnover'].sum():.2f}")
    print(f"Avg weekly turnover: {curve['turnover'].mean():.4f}")
    print(f"Total cost paid (sum t_cost): {curve['t_cost'].sum():.4f}")
    print(f"Total cost paid ($): {curve['cost_dollars'].sum():.2f}")
    print(f"Avg weekly cost ($): {curve['cost_dollars'].mean():.2f}")

    print(curve.tail())

    # Save for plotting / analysis
    curve.to_csv("cache/master_equity_curve.csv", index=False)
    summary.to_csv("cache/master_summary.csv", index=False)
