"""北向资金特征：陆股通资金流向信号。

核心思路：
1. 市场级别：每日北向净流入额、5日累计净流入、净流入动量（反映外资整体情绪）
2. 个股级别：北向持股比例变化、持股集中度排名（反映外资对个股的偏好）

北向资金是 A 股重要的边际定价力量，其行为有较强的趋势性和前瞻性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- 市场级北向资金特征（全市场共享，类似 market_env）---
NORTHBOUND_MARKET_FEATURES = [
    "north_net_amount",       # 当日北向净流入额（亿元）
    "north_net_5d",           # 5日北向净流入累计（亿元）
    "north_net_10d",          # 10日北向净流入累计（亿元）
    "north_momentum",         # 北向净流入动量：5日均值 / 20日均值
    "north_net_ratio",        # 北向净流入 / 全市场成交额
]

# --- 个股级北向资金特征 ---
NORTHBOUND_STOCK_FEATURES = [
    "north_hold_ratio",       # 北向持股占比（%）
    "north_hold_ratio_chg5",  # 5日持股比例变化
    "north_hold_ratio_chg10", # 10日持股比例变化
    "north_hold_rank",        # 北向持股比例全市场排名分位 (0~1)
]

NORTHBOUND_FEATURES = NORTHBOUND_MARKET_FEATURES + NORTHBOUND_STOCK_FEATURES


def build_northbound_market_features(
    northbound_flow: pd.DataFrame,
    daily_amount: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建市场级北向资金特征。

    参数:
        northbound_flow: 北向资金每日数据，至少包含 date, north_net_amount（万元）
        daily_amount: 全市场每日总成交额，包含 date, total_amount（用于计算比率）

    返回:
        按 date 聚合的市场级北向资金特征表，每天一行。
    """
    if northbound_flow.empty:
        return pd.DataFrame(columns=["date"] + NORTHBOUND_MARKET_FEATURES)

    df = northbound_flow.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["north_net_amount"] = pd.to_numeric(df["north_net_amount"], errors="coerce")
    df = df.dropna(subset=["date", "north_net_amount"]).sort_values("date").drop_duplicates("date")

    # 转换为亿元（原始数据为万元）
    df["north_net_amount"] = df["north_net_amount"] / 10000.0

    # 5日/10日累计
    df["north_net_5d"] = df["north_net_amount"].rolling(5, min_periods=1).sum()
    df["north_net_10d"] = df["north_net_amount"].rolling(10, min_periods=1).sum()

    # 动量指标：5日均值 / 20日均值
    ma5 = df["north_net_amount"].rolling(5, min_periods=1).mean()
    ma20 = df["north_net_amount"].rolling(20, min_periods=5).mean()
    df["north_momentum"] = ma5 / ma20.replace(0, np.nan)

    # 北向净流入 / 全市场成交额
    if daily_amount is not None and not daily_amount.empty:
        amt = daily_amount.copy()
        amt["date"] = pd.to_datetime(amt["date"], errors="coerce")
        df = df.merge(amt[["date", "total_amount"]], on="date", how="left")
        df["north_net_ratio"] = df["north_net_amount"] / (df["total_amount"] / 1e8).replace(0, np.nan)
        df = df.drop(columns=["total_amount"], errors="ignore")
    else:
        df["north_net_ratio"] = np.nan

    df = df[["date"] + NORTHBOUND_MARKET_FEATURES].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def build_northbound_stock_features(
    daily: pd.DataFrame,
    northbound_stock: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建个股级北向资金特征。

    参数:
        daily: 全量日线 DataFrame，包含 date, code
        northbound_stock: 个股北向持股数据，包含 date, code, north_hold_ratio

    返回:
        与 daily 等长的 DataFrame，仅包含北向个股特征列。
    """
    n = len(daily)
    result = pd.DataFrame(index=daily.index)

    if northbound_stock is None or northbound_stock.empty:
        for col in NORTHBOUND_STOCK_FEATURES:
            result[col] = np.nan
        return result

    ns = northbound_stock.copy()
    ns["date"] = pd.to_datetime(ns["date"], errors="coerce")
    ns["code"] = ns["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    ns["north_hold_ratio"] = pd.to_numeric(ns["north_hold_ratio"], errors="coerce")
    ns = ns.dropna(subset=["date", "code"]).sort_values(["code", "date"])

    # 持股比例变化
    ns["north_hold_ratio_chg5"] = ns.groupby("code")["north_hold_ratio"].transform(
        lambda x: x - x.shift(5)
    )
    ns["north_hold_ratio_chg10"] = ns.groupby("code")["north_hold_ratio"].transform(
        lambda x: x - x.shift(10)
    )

    # 持股比例全市场排名
    ns["north_hold_rank"] = ns.groupby("date")["north_hold_ratio"].rank(pct=True)

    # merge 到 daily
    df = daily[["date", "code"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)

    merge_cols = ["date", "code"] + NORTHBOUND_STOCK_FEATURES
    available = [c for c in merge_cols if c in ns.columns]
    merged = df.merge(ns[available], on=["date", "code"], how="left")

    for col in NORTHBOUND_STOCK_FEATURES:
        result[col] = merged[col].values if col in merged.columns else np.nan

    return result


def merge_northbound_features(
    features: pd.DataFrame,
    northbound_flow: pd.DataFrame,
    northbound_stock: pd.DataFrame | None = None,
    daily_amount: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将北向资金特征 merge 到个股特征表。

    参数:
        features: 已有特征 DataFrame（包含 date, code）
        northbound_flow: 北向资金市场级流入数据
        northbound_stock: 个股北向持股数据（可选）
        daily_amount: 全市场每日总成交额（可选）
    """
    # 市场级特征
    market_feat = build_northbound_market_features(northbound_flow, daily_amount)
    if not market_feat.empty:
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
        market_feat["date"] = pd.to_datetime(market_feat["date"], errors="coerce")
        features = features.merge(market_feat, on="date", how="left", suffixes=("", "_nb"))
    else:
        for col in NORTHBOUND_MARKET_FEATURES:
            features[col] = np.nan

    # 个股级特征
    stock_feat = build_northbound_stock_features(features, northbound_stock)
    for col in NORTHBOUND_STOCK_FEATURES:
        features[col] = stock_feat[col].values if col in stock_feat.columns else np.nan

    return features
