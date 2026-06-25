from __future__ import annotations

import pandas as pd


def missing_date_ranges(existing: pd.DataFrame, start: str, end: str, date_col: str = "date") -> list[tuple[str, str]]:
    """返回需要补拉的头尾日期区间。

    不再按自然工作日检查中间缺口，避免把 A 股节假日/停牌误判为缺失并触发无意义补拉。
    数据质量缺口由 quality-check 报告，不在下载器里强行补中间日期。
    """
    req_start = pd.to_datetime(start)
    req_end = pd.to_datetime(end)
    if existing.empty or date_col not in existing.columns:
        return [(req_start.strftime("%Y%m%d"), req_end.strftime("%Y%m%d"))]
    dates = pd.to_datetime(existing[date_col], errors="coerce").dropna()
    if dates.empty:
        return [(req_start.strftime("%Y%m%d"), req_end.strftime("%Y%m%d"))]
    cur_min, cur_max = dates.min(), dates.max()
    ranges = []
    if cur_min > req_start:
        ranges.append((req_start.strftime("%Y%m%d"), (cur_min - pd.Timedelta(days=1)).strftime("%Y%m%d")))
    if cur_max < req_end:
        ranges.append(((cur_max + pd.Timedelta(days=1)).strftime("%Y%m%d"), req_end.strftime("%Y%m%d")))
    return [(s, e) for s, e in ranges if pd.to_datetime(s) <= pd.to_datetime(e)]


def missing_datetime_ranges(existing: pd.DataFrame, start: str, end: str, datetime_col: str = "datetime") -> list[tuple[str, str]]:
    req_start = pd.to_datetime(start)
    req_end = pd.to_datetime(end)
    if existing.empty or datetime_col not in existing.columns:
        return [(str(start), str(end))]
    dts = pd.to_datetime(existing[datetime_col], errors="coerce").dropna()
    if dts.empty:
        return [(str(start), str(end))]
    ranges = []
    if dts.min() > req_start:
        ranges.append((str(start), (dts.min() - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")))
    if dts.max() < req_end:
        ranges.append(((dts.max() + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"), str(end)))
    return ranges
