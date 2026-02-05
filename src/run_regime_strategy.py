import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from regime import make_regime_dataset
from backtest_sized import backtest_weekly_sized

FEATURES = ["ret_1w", "vol_12w", "dist_ma20", "dist_ma50", "ma20_slope8"]

df = pd.read_csv("data/spy_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

data = make_regime_dataset(df)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
])

# Train on most history except last 5 years (basic out-of-sample holdout)
cutoff_year = data["date"].dt.year.max() - 5
train = data[data["date"].dt.year <= cutoff_year]
test  = data[data["date"].dt.year > cutoff_year]

model.fit(train[FEATURES].values, train["regime"].values)

# Build a weight function that uses model prediction at that date
lookup = test.set_index("date")[FEATURES]

def weight_fn(df_prices, i):
    date = pd.to_datetime(df_prices.loc[i, "date"])
    if date not in lookup.index:
        return 0.0  # outside test window, stay flat
    x = lookup.loc[date].values.reshape(1, -1)
    pred = int(model.predict(x)[0])

    # Policy:
    # STRESS (2) -> cash
    # TREND (1)  -> long
    # CHOP (0)   -> cash (for now)
    return 1.0 if pred == 1 else 0.0

res = backtest_weekly_sized(df, weight_fn, cost_bps=2, start_capital=10_000)

print("REGIME STRATEGY (last 5y out-of-sample)")
print("Final equity:", round(res["final_equity"], 2))
print("Max drawdown (%):", round(100 * res["max_drawdown_pct"], 2))
print(res["history"].tail())
