from __future__ import annotations

import numpy as np
import pandas as pd


def build_level2_features(snapshots: pd.DataFrame, depth: int = 10) -> pd.DataFrame:
    """V4 盘口特征。支持 CSV 宽表格式。"""
    df = snapshots.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    bid_p1, ask_p1 = "bid_price_1", "ask_price_1"
    bid_v1, ask_v1 = "bid_volume_1", "ask_volume_1"
    df["spread"] = df[ask_p1] - df[bid_p1]
    df["mid_price"] = (df[ask_p1] + df[bid_p1]) / 2
    df["relative_spread"] = df["spread"] / df["mid_price"].replace(0, np.nan)
    df["microprice"] = (df[ask_p1] * df[bid_v1] + df[bid_p1] * df[ask_v1]) / (df[bid_v1] + df[ask_v1]).replace(0, np.nan)
    bid_vol_cols = [f"bid_volume_{i}" for i in range(1, depth + 1) if f"bid_volume_{i}" in df.columns]
    ask_vol_cols = [f"ask_volume_{i}" for i in range(1, depth + 1) if f"ask_volume_{i}" in df.columns]
    df["bid_depth"] = df[bid_vol_cols].sum(axis=1)
    df["ask_depth"] = df[ask_vol_cols].sum(axis=1)
    df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / (df["bid_depth"] + df["ask_depth"]).replace(0, np.nan)
    df["level1_imbalance"] = (df[bid_v1] - df[ask_v1]) / (df[bid_v1] + df[ask_v1]).replace(0, np.nan)
    if "last_price" in df.columns:
        df["last_mid_deviation"] = df["last_price"] / df["mid_price"] - 1
    else:
        df["last_mid_deviation"] = np.nan
    return df.replace([np.inf, -np.inf], np.nan)


V4_FEATURE_COLUMNS = [
    "spread", "relative_spread", "microprice", "bid_depth", "ask_depth",
    "depth_imbalance", "level1_imbalance", "last_mid_deviation",
]
