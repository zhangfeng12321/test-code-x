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

    def get_northbound_flow(self, start: str, end: str) -> pd.DataFrame:
        """返回北向资金每日净流入数据：date, north_net_amount（万元）。"""
        return pd.DataFrame()

    def get_northbound_stock(self, code: str, start: str, end: str) -> pd.DataFrame:
        """返回个股北向资金持股数据：date, code, north_hold_vol, north_hold_ratio。"""
        return pd.DataFrame()

    def get_dragon_tiger_list(self, start: str, end: str) -> pd.DataFrame:
        """返回龙虎榜数据：date, code, reason, buy_amount, sell_amount, net_amount, org_count。"""
        return pd.DataFrame()

    def get_financial_indicators(self, code: str) -> pd.DataFrame:
        """返回个股财务指标：report_date, code, roe, net_profit_growth, revenue_growth, eps, bps。"""
        return pd.DataFrame()

    def get_margin_data(self, start: str, end: str) -> pd.DataFrame:
        """返回融资融券汇总：date, margin_balance, margin_buy, short_volume, short_balance。"""
        return pd.DataFrame()


class Level2DataProvider(ABC):
    """V4 专用盘口 Provider 抽象。"""

    @abstractmethod
    def stream_order_book(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        """逐条返回盘口快照 dict。"""

    def stream_ticks(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        """逐笔成交流，可选。"""
        return iter(())
