from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import kdj, macd, rsi


def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """V1 日线特征。输入至少包含 date/open/high/low/close/volume/amount。"""
    df = daily.copy().sort_values(["code", "date"] if "code" in daily.columns else ["date"])
    g = df.groupby("code", group_keys=False) if "code" in df.columns else [(None, df)]

    parts = []
    iterator = g if hasattr(g, "__iter__") and not isinstance(g, list) else g
    for _, x in iterator:
        x = x.copy().sort_values("date")
        for n in [1, 3, 5, 10, 20]:
            x[f"ret_{n}d"] = x["close"].pct_change(n)
            x[f"ma{n}"] = x["close"].rolling(n).mean()
            x[f"close_ma{n}_ratio"] = x["close"] / x[f"ma{n}"] - 1
        x["ma5_slope"] = x["ma5"].pct_change(3)
        x["ma10_slope"] = x["ma10"].pct_change(3)
        x["volume_ratio_5"] = x["volume"] / x["volume"].rolling(5).mean()
        x["volume_ratio_20"] = x["volume"] / x["volume"].rolling(20).mean()
        x["amount_ratio_20"] = x["amount"] / x["amount"].rolling(20).mean()
        x["amplitude"] = (x["high"] - x["low"]) / x["close"].shift(1)
        x["upper_shadow"] = (x["high"] - x[["open", "close"]].max(axis=1)) / x["close"].replace(0, np.nan)
        x["lower_shadow"] = (x[["open", "close"]].min(axis=1) - x["low"]) / x["close"].replace(0, np.nan)
        x["rsi_6"] = rsi(x["close"], 6)
        x["rsi_14"] = rsi(x["close"], 14)
        x = pd.concat([x, macd(x["close"]), kdj(x)], axis=1)
        x["atr_14"] = np.maximum.reduce([
            (x["high"] - x["low"]).to_numpy(),
            (x["high"] - x["close"].shift(1)).abs().to_numpy(),
            (x["low"] - x["close"].shift(1)).abs().to_numpy(),
        ])
        x["atr_14"] = pd.Series(x["atr_14"], index=x.index).rolling(14).mean() / x["close"]
        if "turnover_rate" not in x.columns:
            x["turnover_rate"] = np.nan
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


V1_FEATURE_COLUMNS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
    "ma5_slope", "ma10_slope", "volume_ratio_5", "volume_ratio_20", "amount_ratio_20",
    "amplitude", "upper_shadow", "lower_shadow", "rsi_6", "rsi_14",
    "macd_dif", "macd_dea", "macd_hist", "kdj_k", "kdj_d", "kdj_j", "atr_14", "turnover_rate",
]
