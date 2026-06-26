"""A股交易日历模块。

从已下载的日线数据反推交易日历（取所有股票日期的并集），
提供精确的交易日判断能力，替代 pd.date_range(freq="B") 工作日估算。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class TradeCalendar:
    """A股交易日历，基于已有数据反推。"""

    def __init__(self, dates: list[pd.Timestamp] | None = None):
        self._dates: list[pd.Timestamp] = sorted(dates) if dates else []
        self._date_set: set[pd.Timestamp] = set(self._dates)

    @classmethod
    def from_daily_data(cls, daily: pd.DataFrame) -> "TradeCalendar":
        """从全量日线数据中提取交易日历（所有股票日期的并集）。"""
        if daily.empty or "date" not in daily.columns:
            return cls()
        dates = pd.to_datetime(daily["date"], errors="coerce").dropna().dt.normalize().unique()
        return cls(sorted(dates.tolist()))

    @classmethod
    def from_csv(cls, path: Path | str) -> "TradeCalendar":
        """从缓存的交易日历 CSV 加载。"""
        path = Path(path)
        if not path.exists():
            return cls()
        df = pd.read_csv(path)
        if "date" not in df.columns or df.empty:
            return cls()
        dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
        return cls(sorted(dates.tolist()))

    def to_csv(self, path: Path | str) -> Path:
        """保存交易日历到 CSV。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in self._dates]})
        df.to_csv(path, index=False)
        return path

    def get_trade_dates(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[pd.Timestamp]:
        """获取区间内的交易日列表。"""
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        return [d for d in self._dates if s <= d <= e]

    def count_trade_days(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> int:
        """计算区间内交易日数量。"""
        return len(self.get_trade_dates(start, end))

    def is_trade_date(self, date: str | pd.Timestamp) -> bool:
        """判断某日是否为交易日。"""
        return pd.Timestamp(date).normalize() in self._date_set

    @property
    def total_days(self) -> int:
        return len(self._dates)

    @property
    def date_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not self._dates:
            return None, None
        return self._dates[0], self._dates[-1]


def build_trade_calendar(daily: pd.DataFrame, cache_path: Path | str | None = None) -> TradeCalendar:
    """从日线数据构建交易日历，可选缓存。"""
    cal = TradeCalendar.from_daily_data(daily)
    if cache_path and cal.total_days > 0:
        cal.to_csv(cache_path)
    return cal
