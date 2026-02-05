import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import CONFIG
from regime_features import add_regime_features, FEATURES
from regime_labels import add_regime_labels
from metrics import compute_metrics

def year_fraction(d0, d1) -> float:
    return (d1 - d0).days / 365.25

def policy_weight(row, pred_regime: int) -> float:
    """
    pred_regime: 2=STRESS, 1=TREND, 0=CHOP
    """
    if pred_regime == 2:
        return CONFIG["w_stress"]

    if pred_regime == 1:
        if CONFIG["trend_requires_above_ma20"] and not (row["Close"] > row["ma_20w"]):
            return 0.0
        return CONFIG["w_trend"]

    # CHOP: mean-reversion entry if far below MA20
    if row["dist_ma20"] <= CONFIG["mr_dist_ma20_entry"]:
        return CONFIG["w_chop_mr"]
    return CONFIG["w_chop_base"]

def run_master_backtest(df_prices: pd.DataFrame) -> pd.DataFrame:
    # Build features (safe)
    d = add_regime_features(df_prices)

    # Add labels (only used for training targets)
    d = add_regime_labels(d, stress_dd_4w=CONFIG["stress_dd_4w"])

    # Drop rows without features/labels
    needed = ["date", "Close", "ma_20w", "dist_ma20"] + FEATURES + ["regime"]
    d = d[needed].dropna().reset_index(drop=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
    ])

    equity = CONFIG["start_capital"]
    cost = CONFIG["cost_bps"] / 10_000

    logs = []

    for i in range(len(d) - 1):
        date = pd.to_datetime(d.loc[i, "date"])
        next_date = pd.to_datetime(d.loc[i + 1, "date"])

        # Rolling training window by years (no lookahead)
        train_end = date
        train_start = train_end - pd.Timedelta(days=int(CONFIG["train_years"] * 365.25))

        train = d[(pd.to_datetime(d["date"]) < train_end) & (pd.to_datetime(d["date"]) >= train_start)]
        if len(train) < CONFIG["min_train_rows"]:
            # not enough history yet; stay flat
            w = 0.0
            pred = None
        else:
            Xtr = train[FEATURES].values
            ytr = train["regime"].values
            model.fit(Xtr, ytr)

            x = d.loc[i, FEATURES].values.reshape(1, -1)
            pred = int(model.predict(x)[0])
            w = float(policy_weight(d.loc[i], pred))

        # Apply return to next week
        px0 = float(d.loc[i, "Close"])
        px1 = float(d.loc[i + 1, "Close"])
        ret = (px1 / px0) - 1.0

        # Cost if exposed this week
        net_ret = w * ret - (cost if w > 0 else 0.0)
        equity *= (1.0 + net_ret)

        logs.append({
            "date": date,
            "next_date": next_date,
            "pred_regime": pred,
            "weight": w,
            "week_ret": ret,
            "net_ret": net_ret,
            "equity": equity,
        })

    return pd.DataFrame(logs)

if __name__ == "__main__":
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])

    curve = run_master_backtest(df)
    m = compute_metrics(curve)

    print("MASTER REGIME + POLICY BACKTEST")
    print(f"Weeks: {m['weeks']}")
    print(f"Start: {m['start_equity']:.2f}  Final: {m['final_equity']:.2f}")
    print(f"CAGR: {100*m['cagr']:.2f}%")
    print(f"Max DD: {100*m['max_drawdown']:.2f}%")
    print(f"AnnRet: {100*m['ann_return']:.2f}%  AnnVol: {100*m['ann_vol']:.2f}%  Sharpe: {m['sharpe']:.2f}")
    print(f"Avg weight: {m['avg_weight']:.2f}")
    print(curve.tail())
    curve.to_csv("cache/master_equity_curve.csv", index=False)
