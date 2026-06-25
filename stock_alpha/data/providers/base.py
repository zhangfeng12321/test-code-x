from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd


class MarketDataProvider(ABC):
    """统一行情数据源抽象。业务层只能依赖这个接口。"""

    @abstractmethod
    def get_daily_bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        """返回日线：date, open, high, low, close, volume, amount, turnover_rate(optional)."""

    @abstractmethod
    def get_minute_bars(self, code: str, start: str, end: str, period: str = "5") -> pd.DataFrame:
        """返回分钟线：datetime, open, high, low, close, volume, amount."""

    def get_stock_basic(self, as_of: Optional[str] = None) -> pd.DataFrame:
        """返回股票基础信息。Provider 不支持时可返回空表。"""
        return pd.DataFrame()

    def get_index_daily(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """返回指数日线。"""
        return pd.DataFrame()

    def get_industry_rank(self, trade_date: str) -> pd.DataFrame:
        """返回行业/板块涨幅排名。"""
        return pd.DataFrame()

    def get_level2_snapshots(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        """V4 Level-2 快照流。真实源通常由付费接口实现。"""
        raise NotImplementedError("Level-2 snapshots are not supported by this provider")


class Level2DataProvider(ABC):
    """V4 专用盘口 Provider 抽象。"""

    @abstractmethod
    def stream_order_book(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        """逐条返回盘口快照 dict。"""

    def stream_ticks(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        """逐笔成交流，可选。"""
        return iter(())
