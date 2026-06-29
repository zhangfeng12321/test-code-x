"""龙虎榜特征：异常交易活动信号。

核心思路：
1. 个股级别：当日买入额、净买入额、机构参与数、近期上榜频率
2. 市场级别：全市场当日龙虎榜上榜家数、机构净买入总额（反映市场活跃度/资金博弈强度）

龙虎榜记录了异常交易（涨停/跌停/振幅异常等）的席位明细，
机构席位出现通常意味着中长线资金介入，对后续走势有一定预测价值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- 个股级龙虎榜特征 ---
DRAGON_TIGER_STOCK_FEATURES = [
    "lhb_buy_amount",    # 当日龙虎榜买入额（万元）
    "lhb_net_amount",    # 当日龙虎榜净买入额（万元）
    "lhb_org_count",     # 当日参与机构数
    "lhb_count_5d",      # 近5日上榜次数
    "lhb_count_10d",     # 近10日上榜次数
    "lhb_net_5d",        # 近5日龙虎榜净买入累计（万元）
]

# --- 市场级龙虎榜特征（全市场共享）---
DRAGON_TIGER_MARKET_FEATURES = [
    "lhb_market_count",      # 当日全市场龙虎榜上榜家数
    "lhb_market_net_amount", # 当日全市场龙虎榜净买入总额（万元）
    "lhb_org_net_ratio",     # 机构净买入占比
]

DRAGON_TIGER_FEATURES = DRAGON_TIGER_STOCK_FEATURES + DRAGON_TIGER_MARKET_FEATURES


def build_dragon_tiger_stock_features(
    daily: pd.DataFrame,
    lhb_data: pd.DataFrame,
) -> pd.DataFrame:
    """构建个股级龙虎榜特征。

    参数:
        daily: 全量日线 DataFrame，包含 date, code
        lhb_data: 龙虎榜数据，包含 date, code, buy_amount, sell_amount, net_amount, org_count

    返回:
        与 daily 等长的 DataFrame，仅包含龙虎榜个股特征列。
    """
    result = pd.DataFrame(index=daily.index)

    if lhb_data is None or lhb_data.empty:
        for col in DRAGON_TIGER_STOCK_FEATURES:
            result[col] = np.nan
        return result

    lhb = lhb_data.copy()
    lhb["date"] = pd.to_datetime(lhb["date"], errors="coerce")
    lhb["code"] = lhb["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    for c in ["buy_amount", "sell_amount", "net_amount", "org_count"]:
        if c in lhb.columns:
            lhb[c] = pd.to_numeric(lhb[c], errors="coerce")

    # 按 (date, code) 聚合（同一天同一只股票可能有多条记录）
    agg_dict = {}
    if "buy_amount" in lhb.columns:
        agg_dict["buy_amount"] = "sum"
    if "sell_amount" in lhb.columns:
        agg_dict["sell_amount"] = "sum"
    if "net_amount" in lhb.columns:
        agg_dict["net_amount"] = "sum"
    if "org_count" in lhb.columns:
        agg_dict["org_count"] = "sum"

    if not agg_dict:
        for col in DRAGON_TIGER_STOCK_FEATURES:
            result[col] = np.nan
        return result

    lhb_agg = lhb.groupby(["date", "code"]).agg(agg_dict).reset_index()
    lhb_agg = lhb_agg.rename(columns={
        "buy_amount": "lhb_buy_amount",
        "net_amount": "lhb_net_amount",
        "org_count": "lhb_org_count",
    })

    # 计算上榜标记（用于统计频率）
    lhb_agg["_on_list"] = 1

    # merge 到 daily
    df = daily[["date", "code"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)

    merge_cols = ["date", "code"]
    feat_cols = [c for c in ["lhb_buy_amount", "lhb_net_amount", "lhb_org_count", "_on_list"] if c in lhb_agg.columns]
    merged = df.merge(lhb_agg[merge_cols + feat_cols], on=merge_cols, how="left")

    # 填充未上榜的为 0
    for c in ["lhb_buy_amount", "lhb_net_amount", "lhb_org_count", "_on_list"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    # 近5日/10日上榜次数和净买入累计（按个股滚动）
    merged = merged.sort_values(["code", "date"])
    merged["lhb_count_5d"] = merged.groupby("code")["_on_list"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    merged["lhb_count_10d"] = merged.groupby("code")["_on_list"].transform(
        lambda x: x.rolling(10, min_periods=1).sum()
    )
    merged["lhb_net_5d"] = merged.groupby("code")["lhb_net_amount"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    # 按原始 index 顺序恢复
    merged = merged.sort_index()

    for col in DRAGON_TIGER_STOCK_FEATURES:
        result[col] = merged[col].values if col in merged.columns else np.nan

    return result


def build_dragon_tiger_market_features(lhb_data: pd.DataFrame) -> pd.DataFrame:
    """构建市场级龙虎榜特征。

    参数:
        lhb_data: 龙虎榜数据，包含 date, code, buy_amount, sell_amount, net_amount, org_count

    返回:
        按 date 聚合的市场级龙虎榜特征表，每天一行。
    """
    if lhb_data is None or lhb_data.empty:
        return pd.DataFrame(columns=["date"] + DRAGON_TIGER_MARKET_FEATURES)

    lhb = lhb_data.copy()
    lhb["date"] = pd.to_datetime(lhb["date"], errors="coerce")
    for c in ["buy_amount", "sell_amount", "net_amount", "org_count"]:
        if c in lhb.columns:
            lhb[c] = pd.to_numeric(lhb[c], errors="coerce")

    # 按日期聚合
    market_rows = []
    for d, x in lhb.groupby("date"):
        unique_codes = x["code"].nunique() if "code" in x.columns else 0
        total_buy = x["buy_amount"].sum() if "buy_amount" in x.columns else 0
        total_sell = x["sell_amount"].sum() if "sell_amount" in x.columns else 0
        total_net = x["net_amount"].sum() if "net_amount" in x.columns else 0
        total_org_net = x.loc[x.get("org_count", pd.Series(dtype=float)) > 0, "net_amount"].sum() \
            if "org_count" in x.columns and "net_amount" in x.columns else 0
        row = {
            "date": d,
            "lhb_market_count": unique_codes,
            "lhb_market_net_amount": total_net,
            "lhb_org_net_ratio": total_org_net / total_buy if total_buy > 0 else 0.0,
        }
        market_rows.append(row)

    if not market_rows:
        return pd.DataFrame(columns=["date"] + DRAGON_TIGER_MARKET_FEATURES)

    market = pd.DataFrame(market_rows).sort_values("date").reset_index(drop=True)
    market = market[["date"] + DRAGON_TIGER_MARKET_FEATURES].copy()
    market = market.replace([np.inf, -np.inf], np.nan)
    return market


def merge_dragon_tiger_features(
    features: pd.DataFrame,
    lhb_data: pd.DataFrame,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将龙虎榜特征 merge 到个股特征表。

    参数:
        features: 已有特征 DataFrame（包含 date, code）
        lhb_data: 龙虎榜原始数据
        daily: 原始日线数据（用于个股级特征构建的 index 对齐）
    """
    ref_daily = daily if daily is not None else features

    # 个股级特征
    stock_feat = build_dragon_tiger_stock_features(ref_daily, lhb_data)
    for col in DRAGON_TIGER_STOCK_FEATURES:
        features[col] = stock_feat[col].values if col in stock_feat.columns else np.nan

    # 市场级特征
    market_feat = build_dragon_tiger_market_features(lhb_data)
    if not market_feat.empty:
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
        market_feat["date"] = pd.to_datetime(market_feat["date"], errors="coerce")
        features = features.merge(market_feat, on="date", how="left", suffixes=("", "_lhb"))
    else:
        for col in DRAGON_TIGER_MARKET_FEATURES:
            features[col] = np.nan

    return features
