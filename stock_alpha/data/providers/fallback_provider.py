from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from .base import MarketDataProvider


class FallbackMarketDataProvider(MarketDataProvider):
    """按顺序尝试多个 Provider，某个源失败时自动切换下一个。"""

    def __init__(self, providers: list[MarketDataProvider]):
        if not providers:
            raise ValueError("providers cannot be empty")
        self.providers = providers

    def _try(self, method: str, *args, **kwargs):
        errors = []
        for p in self.providers:
            try:
                result = getattr(p, method)(*args, **kwargs)
                if isinstance(result, pd.DataFrame) and result.empty:
                    errors.append(f"{p.__class__.__name__}: empty")
                    continue
                return result
            except Exception as e:
                errors.append(f"{p.__class__.__name__}: {e!r}")
        raise RuntimeError("all providers failed: " + " | ".join(errors))

    def get_daily_bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        return self._try("get_daily_bars", code, start, end, adjust=adjust)

    def get_minute_bars(self, code: str, start: str, end: str, period: str = "5") -> pd.DataFrame:
        return self._try("get_minute_bars", code, start, end, period=period)

    def get_stock_basic(self, as_of: str | None = None) -> pd.DataFrame:
        return self._try("get_stock_basic", as_of=as_of)

    def get_index_daily(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        return self._try("get_index_daily", index_code, start, end)

    def get_industry_rank(self, trade_date: str) -> pd.DataFrame:
        return self._try("get_industry_rank", trade_date)

    def get_level2_snapshots(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        return self._try("get_level2_snapshots", code, start, end)

    def close(self) -> None:
        for p in self.providers:
            if hasattr(p, "close"):
                try:
                    p.close()
                except Exception:
                    pass
