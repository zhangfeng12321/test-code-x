"""市场环境特征：全市场级别的宏观信号。

核心思路：每天计算全市场聚合指标（涨跌家数、成交额、涨停数等），
同一天所有股票共享这些特征值。让模型知道"今天大盘环境如何"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_ENV_FEATURES = [
    "market_ret_1d",       # 全市场等权平均收益
    "market_ret_5d",       # 5日市场收益
    "market_breadth",      # 上涨家数 / 总家数
    "market_vol_ratio",    # 全市场成交额 / 20日均值
    "up_limit_count",      # 当日涨停家数
    "down_limit_count",    # 当日跌停家数
    "market_amplitude",    # 全市场平均振幅
    "day_of_week",         # 周几 (0=周一, 4=周五)
    "large_amount_ratio",  # 成交额Top10%股票的平均涨幅（大资金方向代理）
    "hot_sector_count",    # 板块涨幅>2%的数量（市场热点集中度）
]


def build_market_env_features(daily: pd.DataFrame) -> pd.DataFrame:
    """构建市场环境特征。

    输入：全量 daily DataFrame（至少包含 date, code, close, volume, amount, open, high, low）。
    输出：按 date 聚合的市场环境表，每天一行。需要 merge 回原 DataFrame。
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 计算个股日收益率（如果没有 pct_chg）
    if "pct_chg" not in df.columns or df["pct_chg"].isna().all():
        df = df.sort_values(["code", "date"])
        df["pct_chg"] = df.groupby("code")["close"].pct_change()
    else:
        # pct_chg 可能是百分比形式，统一为小数
        if df["pct_chg"].abs().median() > 1:
            df["pct_chg"] = df["pct_chg"] / 100.0

    # 个股振幅
    df["_amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1).where(lambda s: s > 0)

    # 按日聚合
    market_rows = []
    for d, x in df.groupby("date"):
        n = len(x)
        if n == 0:
            continue
        ret = x["pct_chg"].dropna()
        row = {
            "date": d,
            "market_ret_1d": ret.mean() if not ret.empty else 0.0,
            "market_breadth": (ret > 0).sum() / max(n, 1),
            "up_limit_count": (ret >= 0.095).sum(),
            "down_limit_count": (ret <= -0.095).sum(),
            "market_amplitude": x["_amplitude"].mean() if "_amplitude" in x.columns else 0.0,
            "_total_amount": x["amount"].sum() if "amount" in x.columns else 0.0,
            "day_of_week": pd.Timestamp(d).dayofweek,
            # 大资金方向代理：成交额Top10%股票的平均涨幅
            "large_amount_ratio": ret[x["amount"].nlargest(max(n // 10, 1)).index].mean() if "amount" in x.columns and not ret.empty else 0.0,
            # 热门板块数：代码前缀分组后涨幅>2%的板块数
            "hot_sector_count": 0,  # 先占位，下面计算
        }
        # 计算热门板块数
        if "code" in x.columns and not ret.empty:
            sector_ret = x.assign(_ret=ret).groupby(x["code"].astype(str).str[:3])["_ret"].mean()
            row["hot_sector_count"] = int((sector_ret > 0.02).sum())
        market_rows.append(row)

    if not market_rows:
        return pd.DataFrame(columns=["date"] + MARKET_ENV_FEATURES)

    market = pd.DataFrame(market_rows).sort_values("date").reset_index(drop=True)

    # market_ret_5d: 5日滚动市场收益
    market["market_ret_5d"] = market["market_ret_1d"].rolling(5).sum()

    # market_vol_ratio: 当日成交额 / 20日均值
    market["market_vol_ratio"] = market["_total_amount"] / market["_total_amount"].rolling(20).mean()

    # 只保留需要的列
    market = market[["date"] + MARKET_ENV_FEATURES].copy()
    market = market.replace([np.inf, -np.inf], np.nan)

    return market


def merge_market_env(features: pd.DataFrame, market_env: pd.DataFrame) -> pd.DataFrame:
    """将市场环境特征 merge 到个股特征表。"""
    if market_env.empty:
        for col in MARKET_ENV_FEATURES:
            features[col] = np.nan
        return features
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    market_env["date"] = pd.to_datetime(market_env["date"], errors="coerce")
    return features.merge(market_env, on="date", how="left", suffixes=("", "_mkt"))
