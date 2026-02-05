import yfinance as yf
import pandas as pd
from pathlib import Path

OUT = Path("data/spy_weekly.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = yf.download(
    "SPY",
    period="max",
    interval="1wk",
    auto_adjust=True,
    group_by="column"
)

# Flatten columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# Force numeric types (THIS is the important part)
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df.to_csv(OUT, index=False)
print("Saved:", OUT, "rows:", len(df))
