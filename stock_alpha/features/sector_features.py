"""板块/行业强度特征：个股所属板块的横截面强弱。

通过代码前缀粗分板块（000/002/300/301/600/601/603/688），
计算板块级聚合指标，让模型知道"这只票所在板块今天强不强"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SECTOR_FEATURES = [
    "sector_ret_1d",       # 所属板块当日平均收益
    "sector_ret_5d",       # 所属板块5日累计收益
    "sector_rank",         # 板块强度排名（板块间横截面, 0~1）
    "stock_vs_sector",     # 个股收益 - 板块收益（相对强度）
    "sector_money_flow",   # 板块资金净流入排名（用成交额增幅代理）
    "sector_breadth",      # 板块内上涨比例
]


def _code_to_sector(code: str) -> str:
    """按代码前缀粗分板块。"""
    c = str(code).zfill(6)
    if c.startswith("000") or c.startswith("001"):
        return "SZ_MAIN"      # 深市主板
    elif c.startswith("002") or c.startswith("003"):
        return "SZ_SME"       # 中小板
    elif c.startswith("300") or c.startswith("301"):
        return "CHINEXT"      # 创业板
    elif c.startswith("600") or c.startswith("601"):
        return "SH_MAIN"      # 沪市主板（大盘）
    elif c.startswith("603") or c.startswith("605"):
        return "SH_MAIN2"     # 沪市主板（次新/小盘）
    elif c.startswith("688") or c.startswith("689"):
        return "STAR"         # 科创板
    else:
        return "OTHER"


def build_sector_features(daily: pd.DataFrame) -> pd.DataFrame:
    """构建板块强度特征。

    输入：全量 daily DataFrame（需包含 code, date, close, amount, pct_chg 或可推算）。
    输出：与输入等长的 DataFrame，新增板块特征列。
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    for c in ["close", "amount", "pct_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 分配板块
    df["_sector"] = df["code"].apply(_code_to_sector)

    # 计算个股日收益率
    if "pct_chg" not in df.columns or df["pct_chg"].isna().all():
        df = df.sort_values(["code", "date"])
        df["pct_chg"] = df.groupby("code")["close"].pct_change()
    else:
        if df["pct_chg"].abs().median() > 1:
            df["pct_chg"] = df["pct_chg"] / 100.0

    # 按 (date, sector) 聚合
    sector_daily = df.groupby(["date", "_sector"]).agg(
        sector_ret_1d=("pct_chg", "mean"),
        sector_amount=("amount", "sum"),
        sector_up_count=("pct_chg", lambda x: (x > 0).sum()),
        sector_total=("pct_chg", "count"),
    ).reset_index()

    # 板块内上涨比例
    sector_daily["sector_breadth"] = sector_daily["sector_up_count"] / sector_daily["sector_total"].clip(lower=1)

    # 板块5日收益（滚动）
    sector_daily = sector_daily.sort_values(["_sector", "date"])
    sector_daily["sector_ret_5d"] = sector_daily.groupby("_sector")["sector_ret_1d"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    # 板块间排名（当日哪个板块最强）
    sector_daily["sector_rank"] = sector_daily.groupby("date")["sector_ret_1d"].rank(pct=True)

    # 板块资金流排名（用成交额变化代理）
    sector_daily["_amount_ma5"] = sector_daily.groupby("_sector")["sector_amount"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    sector_daily["_amount_ratio"] = sector_daily["sector_amount"] / sector_daily["_amount_ma5"].clip(lower=1)
    sector_daily["sector_money_flow"] = sector_daily.groupby("date")["_amount_ratio"].rank(pct=True)

    # merge 回个股
    merge_cols = ["date", "_sector", "sector_ret_1d", "sector_ret_5d", "sector_rank", "sector_breadth", "sector_money_flow"]
    out = df.merge(sector_daily[merge_cols], on=["date", "_sector"], how="left", suffixes=("", "_sec"))

    # 个股相对板块强度
    out["stock_vs_sector"] = out["pct_chg"] - out["sector_ret_1d"]

    # 清理临时列
    out = out.drop(columns=["_sector"], errors="ignore")

    return out[SECTOR_FEATURES] if set(SECTOR_FEATURES) <= set(out.columns) else out[[c for c in SECTOR_FEATURES if c in out.columns]]


def merge_sector_features(features: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """将板块特征计算后 merge 到特征表。"""
    sector_df = build_sector_features(daily)
    if sector_df.empty:
        for col in SECTOR_FEATURES:
            features[col] = np.nan
        return features

    # sector_df 与 features 等长（同源 daily），直接按位置对齐
    # 但为安全起见，用 index 对齐
    for col in SECTOR_FEATURES:
        if col in sector_df.columns:
            features[col] = sector_df[col].values if len(sector_df) == len(features) else np.nan
        else:
            features[col] = np.nan

    return features
