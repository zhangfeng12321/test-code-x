from __future__ import annotations

import pandas as pd


def make_triple_barrier_labels(daily: pd.DataFrame, profit_take: float = 0.03, stop_loss: float = 0.02, horizon: int = 3) -> pd.DataFrame:
    """生成短线三分类标签：1 看涨，-1 看跌，0 震荡。"""
    df = daily.copy().sort_values(["code", "date"] if "code" in daily.columns else ["date"])
    rows = []
    for code, x in df.groupby("code") if "code" in df.columns else [(None, df)]:
        x = x.reset_index(drop=True)
        labels = []
        max_fwd_ret = []
        min_fwd_ret = []
        for i in range(len(x)):
            entry = x.loc[i, "close"]
            future = x.iloc[i + 1:i + 1 + horizon]
            if future.empty or not entry:
                labels.append(pd.NA); max_fwd_ret.append(pd.NA); min_fwd_ret.append(pd.NA); continue
            mx = future["high"].max() / entry - 1
            mn = future["low"].min() / entry - 1
            max_fwd_ret.append(mx); min_fwd_ret.append(mn)
            if mx >= profit_take and mn > -stop_loss:
                labels.append(1)
            elif mn <= -stop_loss or mx < profit_take / 3:
                labels.append(-1)
            else:
                labels.append(0)
        y = x[["code", "date"]].copy() if "code" in x.columns else x[["date"]].copy()
        y["label"] = labels
        y["max_fwd_ret"] = max_fwd_ret
        y["min_fwd_ret"] = min_fwd_ret
        rows.append(y)
    return pd.concat(rows, ignore_index=True)
