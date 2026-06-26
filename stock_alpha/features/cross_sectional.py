"""横截面排名特征：当日每只股票在全市场中的相对位置。

核心思路：同一天所有股票放在一起做排名，输出分位数 (0~1)。
让模型知道"这只股票今天在全市场中是强还是弱"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CROSS_SECTIONAL_FEATURES = [
    "ret_1d_rank",         # 当日收益率在全市场的分位数 (0~1)
    "ret_5d_rank",         # 5日收益率排名分位
    "amount_rank",         # 当日成交额排名分位
    "turnover_rank",       # 当日换手率排名分位
    "volume_ratio_5_rank", # 5日量比排名分位
    "amplitude_rank",      # 当日振幅排名分位
    "close_ma20_rank",     # 20日均线偏离度排名分位
    "rsi_14_rank",         # RSI排名分位
    "strength_20d",        # 20日强度（涨幅分位 0~1）
    "weakness_5d",         # 5日弱势程度（收益越低排名越高）
]


def build_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    """构建横截面排名特征。

    输入：已包含 V1 基础特征的 DataFrame（需包含 date, code, 以及 V1 特征列）。
    输出：与输入等长的 DataFrame，新增横截面排名列。
    """
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 需要排名的源列 → 目标列
    rank_mapping = {
        "ret_1d": "ret_1d_rank",
        "ret_5d": "ret_5d_rank",
        "amount": "amount_rank",
        "turnover_rate": "turnover_rank",
        "volume_ratio_5": "volume_ratio_5_rank",
        "amplitude": "amplitude_rank",
        "close_ma20_ratio": "close_ma20_rank",
        "rsi_14": "rsi_14_rank",
    }

    # 按日期分组做横截面排名
    for src_col, dst_col in rank_mapping.items():
        if src_col in df.columns:
            df[dst_col] = df.groupby("date")[src_col].rank(pct=True, na_option="keep")
        else:
            df[dst_col] = np.nan

    # strength_20d: 20日涨幅在横截面的分位数（越高越强）
    if "ret_20d" in df.columns:
        df["strength_20d"] = df.groupby("date")["ret_20d"].rank(pct=True, na_option="keep")
    else:
        df["strength_20d"] = np.nan

    # weakness_5d: 5日跌幅程度（收益越低，weakness越高）
    if "ret_5d" in df.columns:
        # 取反排名：收益越低排名越高（越弱）
        df["weakness_5d"] = df.groupby("date")["ret_5d"].rank(pct=True, ascending=False, na_option="keep")
    else:
        df["weakness_5d"] = np.nan

    return df
