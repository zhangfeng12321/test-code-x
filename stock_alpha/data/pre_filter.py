"""首轮预过滤：下载数据前基于 stock_basic 排除明确不需要的股票。

排除规则：
1. ST / *ST 股票
2. 北交所（代码 8/4 开头）
3. 上市不足 min_list_days 天的次新股
4. 已退市/暂停上市

这一步的目的是节省下载时间和存储空间，不影响后续更精细的 universe filter。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class PreFilterConfig:
    """首轮预过滤配置。"""
    exclude_st: bool = True
    exclude_bj_exchange: bool = True
    min_list_days: int = 365  # 上市至少1年
    exclude_delisted: bool = True
    as_of_date: str | None = None  # 默认用当前日期


def pre_filter_stock_basic(stock_basic: pd.DataFrame, cfg: PreFilterConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 stock_basic 执行首轮预过滤。

    Args:
        stock_basic: 股票基础信息表，至少包含 code 列
        cfg: 预过滤配置

    Returns:
        (passed, excluded): 通过的股票, 被排除的股票（含排除原因）
    """
    cfg = cfg or PreFilterConfig()
    df = stock_basic.copy()

    # 标准化 code
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)

    # 用于记录排除原因
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def add_reason(mask: pd.Series, reason: str) -> None:
        for i, flag in enumerate(mask.fillna(False).to_numpy()):
            if flag:
                reasons[i].append(reason)

    # 1. 排除 ST
    if cfg.exclude_st:
        is_st = pd.Series(False, index=df.index)
        if "is_st" in df.columns:
            is_st |= df["is_st"].fillna(False).astype(bool)
        if "name" in df.columns:
            is_st |= df["name"].astype(str).str.contains(r"ST|退市", case=False, na=False)
        add_reason(is_st, "ST/退市风险")

    # 2. 排除北交所
    if cfg.exclude_bj_exchange:
        is_bj = df["code"].str.match(r"^(8|4)\d{5}$")
        add_reason(is_bj, "北交所")

    # 3. 排除次新股
    if cfg.min_list_days > 0 and "list_date" in df.columns:
        as_of = pd.to_datetime(cfg.as_of_date) if cfg.as_of_date else pd.Timestamp(datetime.now())
        list_date = pd.to_datetime(df["list_date"], errors="coerce")
        too_new = (as_of - list_date).dt.days < cfg.min_list_days
        add_reason(too_new, f"上市不足{cfg.min_list_days}天")

    # 4. 排除已退市
    if cfg.exclude_delisted:
        if "status" in df.columns:
            is_delisted = df["status"].astype(str).str.contains(r"退市|暂停|终止", case=False, na=False)
            add_reason(is_delisted, "已退市/暂停上市")
        if "delist_date" in df.columns:
            has_delist = pd.to_datetime(df["delist_date"], errors="coerce").notna()
            add_reason(has_delist, "有退市日期")

    # 分类
    df["exclude_reasons"] = ["|".join(r) for r in reasons]
    df["pre_filter_pass"] = df["exclude_reasons"].eq("")

    passed = df[df["pre_filter_pass"]].drop(columns=["exclude_reasons", "pre_filter_pass"])
    excluded = df[~df["pre_filter_pass"]].copy()

    return passed, excluded


def pre_filter_summary(passed: pd.DataFrame, excluded: pd.DataFrame) -> str:
    """输出预过滤摘要。"""
    lines = [
        f"首轮预过滤结果：",
        f"  总股票数：{len(passed) + len(excluded)}",
        f"  通过：{len(passed)}",
        f"  排除：{len(excluded)}",
    ]
    if not excluded.empty and "exclude_reasons" in excluded.columns:
        reason_counts = {}
        for reasons_str in excluded["exclude_reasons"]:
            for r in reasons_str.split("|"):
                if r:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
        lines.append("  排除原因分布：")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason}: {count}")
    return "\n".join(lines)
