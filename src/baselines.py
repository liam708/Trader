import pandas as pd

def ma20_trend_signal(df: pd.DataFrame, i: int) -> bool:
    if i < 20:
        return False
    close = df.loc[i, "Close"]
    ma20 = df.loc[i-20:i-1, "Close"].mean()
    return close > ma20
