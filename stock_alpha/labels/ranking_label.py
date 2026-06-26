"""排序标签生成：将未来收益率转化为横截面排名标签。

核心思路：同一天内所有股票按未来N日收益率排序，输出五档排名标签。
LightGBM Ranker 使用这些标签学习"谁比谁更值得买"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ranking_labels(daily: pd.DataFrame, horizon: int = 5, n_bins: int = 5) -> pd.DataFrame:
    """生成排序标签。

    Args:
        daily: 日线数据，包含 code, date, close
        horizon: 预测未来 N 天收益率
        n_bins: 标签分档数（默认5档：0=最差20%, 4=最好20%）

    Returns:
        DataFrame with columns: code, date, rank_label(0~n_bins-1), fwd_return
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["code", "date"])

    # 计算未来 N 日收益率
    parts = []
    for code, x in df.groupby("code"):
        x = x.copy().sort_values("date").reset_index(drop=True)
        # 未来 horizon 天后的收盘价
        x["fwd_close"] = x["close"].shift(-horizon)
        x["fwd_return"] = x["fwd_close"] / x["close"] - 1
        parts.append(x[["code", "date", "fwd_return"]])

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["fwd_return"])

    # 按日分组做横截面排名 → 五档标签
    results = []
    for d, group in out.groupby("date"):
        g = group.copy()
        n = len(g)
        if n < n_bins:
            g["rank_label"] = (g["fwd_return"].rank(pct=True) * (n_bins - 1)).round().astype(int)
        else:
            g["rank_label"] = pd.qcut(
                g["fwd_return"], q=n_bins, labels=False, duplicates="drop"
            )
            if g["rank_label"].isna().any():
                g["rank_label"] = (g["fwd_return"].rank(pct=True) * (n_bins - 1)).round().astype(int)
        results.append(g)
    
    out = pd.concat(results, ignore_index=True)
    out["rank_label"] = out["rank_label"].astype(int).clip(0, n_bins - 1)
    
    return out[["code", "date", "rank_label", "fwd_return"]].reset_index(drop=True)
