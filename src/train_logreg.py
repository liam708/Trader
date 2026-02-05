import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from features import make_weekly_dataset

FEATURES = ["ret_1w","ret_2w","ret_4w","vol_4w","vol_12w","dist_ma20","dist_ma50"]

def walk_forward_trade_accuracy(data: pd.DataFrame, train_years=10, test_years=2):
    data = data.copy()
    data["year"] = data["date"].dt.year

    years = sorted(data["year"].unique())
    results = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    for start in range(0, len(years) - (train_years + test_years) + 1):
        train_yrs = years[start:start+train_years]
        test_yrs = years[start+train_years:start+train_years+test_years]

        train = data[data["year"].isin(train_yrs)]
        test = data[data["year"].isin(test_yrs)]

        Xtr, ytr = train[FEATURES].values, train["y"].values
        Xte, yte = test[FEATURES].values, test["y"].values

        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:,1]

        # simple threshold rule
        pred = (p >= 0.55).astype(int)

        results.append({
            "train_years": f"{train_yrs[0]}-{train_yrs[-1]}",
            "test_years": f"{test_yrs[0]}-{test_yrs[-1]}",
            "acc": float(accuracy_score(yte, pred)),
            "trades": int(pred.sum()),
            "n": int(len(test))
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])

    dataset = make_weekly_dataset(df, hold_weeks=4, cost_bps=2)
    res = walk_forward_trade_accuracy(dataset, train_years=10, test_years=2)

    print(res.tail(10))
    print("\nAvg acc:", round(res["acc"].mean(), 3))
    print("Avg trades per test window:", round(res["trades"].mean(), 1))
