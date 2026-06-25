from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


@dataclass(frozen=True)
class StockBasic:
    code: str
    name: str
    exchange: Optional[str] = None
    industry: Optional[str] = None
    is_st: bool = False
    list_date: Optional[date] = None


@dataclass(frozen=True)
class DailyBar:
    code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover_rate: Optional[float] = None
    adj_factor: Optional[float] = None


@dataclass(frozen=True)
class MinuteBar:
    code: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    volume: float


@dataclass(frozen=True)
class Level2Snapshot:
    code: str
    ts: datetime
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    last_price: Optional[float] = None
    total_volume: Optional[float] = None
    total_amount: Optional[float] = None


@dataclass(frozen=True)
class TradeTick:
    code: str
    ts: datetime
    price: float
    volume: float
    amount: float
    side: Optional[str] = None  # buy/sell/unknown
