from __future__ import annotations

import pandas as pd

from .base import MarketDataProvider


class AkShareProvider(MarketDataProvider):
    """AkShare 免费数据源 Provider。需要安装 akshare。"""

    def __init__(self):
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise RuntimeError("AkShareProvider requires: pip install akshare") from exc
        self.ak = ak

    def get_daily_bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        df = self.ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace('-', ''), end_date=end.replace('-', ''), adjust=adjust)
        mapping = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
            "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate", "涨跌幅": "pct_chg",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        return df

    def get_minute_bars(self, code: str, start: str, end: str, period: str = "5") -> pd.DataFrame:
        df = self.ak.stock_zh_a_hist_min_em(symbol=code, start_date=start, end_date=end, period=period, adjust="")
        mapping = {"时间": "datetime", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        df["code"] = code
        return df

    def get_stock_basic(self, as_of: str | None = None) -> pd.DataFrame:
        df = self.ak.stock_info_a_code_name()
        return df.rename(columns={"code": "code", "name": "name"})
