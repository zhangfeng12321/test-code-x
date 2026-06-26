from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stock_alpha.data.trade_calendar import TradeCalendar


@dataclass
class QualityIssue:
    code: str
    issue: str
    severity: str
    count: int
    detail: str = ""


def check_daily_quality(daily: pd.DataFrame, start: str | None = None, end: str | None = None, trade_calendar: "TradeCalendar | None" = None) -> pd.DataFrame:
    """日线数据质量检查：重复、缺日期、价格异常、成交量异常。

    如果提供了 trade_calendar，使用精确交易日数量判断缺失；否则回退到工作日估算。
    """
    if daily.empty:
        return pd.DataFrame([QualityIssue("", "empty_dataset", "high", 0, "daily data is empty").__dict__])
    df = daily.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    issues: list[QualityIssue] = []
    for code, x in df.groupby("code"):
        x = x.sort_values("date")
        dup = x.duplicated(["code", "date"]).sum()
        if dup:
            issues.append(QualityIssue(code, "duplicate_code_date", "high", int(dup)))
        null_dates = x["date"].isna().sum()
        if null_dates:
            issues.append(QualityIssue(code, "null_date", "high", int(null_dates)))
        for col in ["open", "high", "low", "close"]:
            if col in x.columns:
                bad = (pd.to_numeric(x[col], errors="coerce") <= 0).sum()
                if bad:
                    issues.append(QualityIssue(code, f"non_positive_{col}", "high", int(bad)))
        if {"high", "low", "open", "close"} <= set(x.columns):
            bad_ohlc = ((x["high"] < x[["open", "close", "low"]].max(axis=1)) | (x["low"] > x[["open", "close", "high"]].min(axis=1))).sum()
            if bad_ohlc:
                issues.append(QualityIssue(code, "invalid_ohlc_relation", "high", int(bad_ohlc)))
        if "volume" in x.columns:
            zero_vol = (pd.to_numeric(x["volume"], errors="coerce").fillna(0) <= 0).sum()
            if zero_vol:
                issues.append(QualityIssue(code, "zero_or_null_volume", "medium", int(zero_vol)))
        s = pd.to_datetime(start) if start else x["date"].min()
        e = pd.to_datetime(end) if end else x["date"].max()
        # 使用交易日历精确计算期望交易日，否则回退到工作日估算
        if trade_calendar is not None and trade_calendar.total_days > 0:
            expected_days = trade_calendar.count_trade_days(s, e)
        else:
            expected_days = len(pd.date_range(s, e, freq="B")) if pd.notna(s) and pd.notna(e) else 0
        actual = x["date"].dt.normalize().nunique()
        # 交易日低于期望的80%才报问题（有交易日历时更精确）
        threshold = 0.80 if trade_calendar is not None else 0.60
        if expected_days and actual < expected_days * threshold:
            issues.append(QualityIssue(code, "too_few_trading_days", "medium", int(expected_days - actual), f"actual={actual}, expected={expected_days}"))
    return pd.DataFrame([i.__dict__ for i in issues]) if issues else pd.DataFrame(columns=["code", "issue", "severity", "count", "detail"])


def summarize_quality(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame([{"severity": "ok", "issue_count": 0, "affected_codes": 0}])
    return issues.groupby("severity").agg(issue_count=("issue", "count"), affected_codes=("code", "nunique")).reset_index()
