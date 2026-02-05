import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from regime import make_regime_dataset

FEATURES = ["ret_1w", "vol_12w", "dist_ma20", "dist_ma50", "ma20_slope8"]

def walk_forward_regime(data: pd.DataFrame, train_years=10, test_years=2):
    data = data.copy()
    data["year"] = data["date"].dt.year
    years = sorted(data["year"].unique())

    # Multiclass logistic regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
    ])

    rows = []
    for start in range(0, len(years) - (train_years + test_years) + 1):
        train_yrs = years[start:start+train_years]
        test_yrs = years[start+train_years:start+train_years+test_years]

        train = data[data["year"].isin(train_yrs)]
        test  = data[data["year"].isin(test_yrs)]

        Xtr, ytr = train[FEATURES].values, train["regime"].values
        Xte, yte = test[FEATURES].values, test["regime"].values

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        # simple summary
        acc = (pred == yte).mean()
        rows.append({
            "train": f"{train_yrs[0]}-{train_yrs[-1]}",
            "test":  f"{test_yrs[0]}-{test_yrs[-1]}",
            "acc": float(acc),
            "n": int(len(test)),
            "stress_rate": float((yte == 2).mean()),
            "pred_stress_rate": float((pred == 2).mean())
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_csv("data/spy_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])

    data = make_regime_dataset(df)
    res = walk_forward_regime(data)

    print(res.tail(10))
    print("\nAvg acc:", round(res["acc"].mean(), 3))
    print("Avg true stress rate:", round(res["stress_rate"].mean(), 3))
    print("Avg predicted stress rate:", round(res["pred_stress_rate"].mean(), 3))
