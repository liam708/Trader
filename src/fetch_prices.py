import yfinance as yf
from pathlib import Path

OUT = Path("data/spy_weekly.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = yf.download("SPY", period="max", interval="1wk", auto_adjust=True)
df = df.dropna().reset_index()
df.rename(columns={"Date": "date"}, inplace=True)

df.to_csv(OUT, index=False)
print("Saved:", OUT, "rows:", len(df))
