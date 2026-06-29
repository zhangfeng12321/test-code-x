from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from stock_alpha.data.providers.base import MarketDataProvider
from stock_alpha.storage.cache import DataLake
from stock_alpha.storage.status import TaskStatusStore
from stock_alpha.data.incremental import missing_date_ranges, missing_datetime_ranges


def _ts() -> str:
    """返回当前时间戳，用于日志打印。"""
    return datetime.now().strftime('%H:%M:%S')


def normalize_codes(codes: Iterable[str]) -> list[str]:
    return [str(c).zfill(6)[-6:] for c in codes]


@dataclass
class MarketDataDownloader:
    provider: MarketDataProvider
    lake: DataLake
    sleep_seconds: float = 0.2
    max_retries: int = 3

    def get_stock_universe(self, limit: Optional[int] = None, force_industry: bool = False) -> pd.DataFrame:
        """获取股票列表。若本地已有含行业分类的 stock_basic 则复用缓存，避免重复拉取。"""
        # 检查本地缓存是否已有行业数据
        if not force_industry:
            cached = self.lake.read_parquet("meta", "stock_basic")
            if not cached.empty and "industry" in cached.columns and cached["industry"].notna().sum() > 100:
                print(f"[{_ts()}] stock_basic 已有行业分类缓存（{cached['industry'].notna().sum()} 只），跳过重新拉取")
                df = cached
                if limit:
                    df = df.head(limit)
                return df

        df = self.provider.get_stock_basic()
        if df.empty:
            raise RuntimeError("provider did not return stock universe")
        if "code" not in df.columns:
            raise RuntimeError(f"stock basic missing code column: {df.columns.tolist()}")
        df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        df = df.dropna(subset=["code"]).drop_duplicates("code")
        if "name" in df.columns:
            df["is_st"] = df["name"].astype(str).str.contains("ST", case=False, na=False)
        if limit:
            df = df.head(limit)
        self.lake.write_parquet("meta", "stock_basic", df)
        return df

    @property
    def status_store(self) -> TaskStatusStore:
        return TaskStatusStore(self.lake)

    def _cache_covers_daily(self, code: str, start: str, end: str) -> bool:
        old = self.lake.read_parquet("daily", code)
        if old.empty or "date" not in old.columns:
            return False
        dates = pd.to_datetime(old["date"], errors="coerce")
        return dates.min() <= pd.to_datetime(start) and dates.max() >= pd.to_datetime(end)

    def _cache_covers_minute(self, code: str, start: str, end: str, period: str) -> bool:
        old = self.lake.read_parquet("minute", f"{code}_{period}m")
        if old.empty or "datetime" not in old.columns:
            return False
        dts = pd.to_datetime(old["datetime"], errors="coerce")
        return dts.min() <= pd.to_datetime(start) and dts.max() >= pd.to_datetime(end)

    def download_daily(self, codes: Iterable[str], start: str, end: str, adjust: str = "qfq", limit: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        codes = normalize_codes(codes)
        if limit:
            codes = codes[:limit]
        all_parts = []
        errors = []
        for i, code in enumerate(codes, 1):
            try:
                existing = self.lake.read_parquet("daily", code)
                ranges = [(start, end)] if force else missing_date_ranges(existing, start, end, "date")
                if not ranges:
                    df = existing
                    print(f"[{_ts()}] [{i}/{len(codes)}] daily {code}: cached {len(df)}")
                    all_parts.append(df)
                    self.status_store.upsert("daily", code, start, end, "cached", len(df))
                    continue
                fetched_parts = []
                for rs, re in ranges:
                    last_error = None
                    for attempt in range(self.max_retries + 1):
                        try:
                            part = self.provider.get_daily_bars(code, rs, re, adjust=adjust)
                            break
                        except Exception as e:
                            last_error = e
                            if attempt >= self.max_retries:
                                raise
                            time.sleep(self.sleep_seconds * (attempt + 1) * 2)
                    if not part.empty:
                        fetched_parts.append(part)
                df = pd.concat(fetched_parts, ignore_index=True) if fetched_parts else pd.DataFrame()
                if not df.empty:
                    df["code"] = code
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    self.lake.append_unique("daily", code, df, keys=["code", "date"])
                    all_parts.append(self.lake.read_parquet("daily", code))
                self.status_store.upsert("daily", code, start, end, "success", len(df))
                print(f"[{_ts()}] [{i}/{len(codes)}] daily {code}: fetched {len(df)} ranges={len(ranges)}")
            except Exception as e:
                errors.append({"code": code, "error": repr(e)})
                self.status_store.upsert("daily", code, start, end, "failed", 0, repr(e))
                print(f"[{_ts()}] [{i}/{len(codes)}] daily {code} ERROR: {e}")
            time.sleep(self.sleep_seconds)
        if errors:
            self.lake.write_parquet("logs", "daily_errors", pd.DataFrame(errors))
        return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

    def download_minute(self, codes: Iterable[str], start: str, end: str, period: str = "5", limit: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        codes = normalize_codes(codes)
        if limit:
            codes = codes[:limit]
        all_parts = []
        errors = []
        for i, code in enumerate(codes, 1):
            try:
                existing = self.lake.read_parquet("minute", f"{code}_{period}m")
                ranges = [(start, end)] if force else missing_datetime_ranges(existing, start, end, "datetime")
                if not ranges:
                    df = existing
                    print(f"[{_ts()}] [{i}/{len(codes)}] minute {period}m {code}: cached {len(df)}")
                    all_parts.append(df)
                    self.status_store.upsert(f"minute_{period}m", code, start, end, "cached", len(df))
                    continue
                fetched_parts = []
                for rs, re in ranges:
                    for attempt in range(self.max_retries + 1):
                        try:
                            part = self.provider.get_minute_bars(code, rs, re, period=period)
                            break
                        except Exception:
                            if attempt >= self.max_retries:
                                raise
                            time.sleep(self.sleep_seconds * (attempt + 1) * 2)
                    if not part.empty:
                        fetched_parts.append(part)
                df = pd.concat(fetched_parts, ignore_index=True) if fetched_parts else pd.DataFrame()
                if not df.empty:
                    df["code"] = code
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    self.lake.append_unique("minute", f"{code}_{period}m", df, keys=["code", "datetime"])
                    all_parts.append(self.lake.read_parquet("minute", f"{code}_{period}m"))
                self.status_store.upsert(f"minute_{period}m", code, start, end, "success", len(df))
                print(f"[{_ts()}] [{i}/{len(codes)}] minute {period}m {code}: fetched {len(df)} ranges={len(ranges)}")
            except Exception as e:
                errors.append({"code": code, "error": repr(e)})
                self.status_store.upsert(f"minute_{period}m", code, start, end, "failed", 0, repr(e))
                print(f"[{_ts()}] [{i}/{len(codes)}] minute {code} ERROR: {e}")
            time.sleep(self.sleep_seconds)
        if errors:
            self.lake.write_parquet("logs", "minute_errors", pd.DataFrame(errors))
        return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

    def download_northbound_flow(self, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """下载北向资金每日净流入数据并缓存。"""
        cached = self.lake.read_parquet("northbound", "daily_flow")
        if not force and not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            if cached["date"].max() >= pd.to_datetime(end):
                print(f"[{_ts()}] northbound flow: cached {len(cached)} rows")
                return cached
        try:
            df = self.provider.get_northbound_flow(start, end)
            if not df.empty:
                self.lake.append_unique("northbound", "daily_flow", df, keys=["date"])
                result = self.lake.read_parquet("northbound", "daily_flow")
                print(f"[{_ts()}] northbound flow: fetched {len(df)} rows")
                return result
        except Exception as e:
            print(f"[{_ts()}] northbound flow ERROR: {e}")
        return cached if not cached.empty else pd.DataFrame()

    def download_dragon_tiger(self, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """下载龙虎榜数据并缓存。"""
        cached = self.lake.read_parquet("dragon_tiger", "lhb_detail")
        if not force and not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            if cached["date"].max() >= pd.to_datetime(end):
                print(f"[{_ts()}] dragon tiger: cached {len(cached)} rows")
                return cached
        try:
            df = self.provider.get_dragon_tiger_list(start, end)
            if not df.empty:
                self.lake.append_unique("dragon_tiger", "lhb_detail", df, keys=["date", "code"])
                result = self.lake.read_parquet("dragon_tiger", "lhb_detail")
                print(f"[{_ts()}] dragon tiger: fetched {len(df)} rows")
                return result
        except Exception as e:
            print(f"[{_ts()}] dragon tiger ERROR: {e}")
        return cached if not cached.empty else pd.DataFrame()

    def download_fundamentals(self, codes: Iterable[str], force: bool = False) -> pd.DataFrame:
        """下载个股财务指标并缓存。"""
        codes = normalize_codes(codes)
        cached = self.lake.read_parquet("fundamental", "financial_indicators")
        if not force and not cached.empty and cached["code"].nunique() >= len(codes) * 0.8:
            print(f"[{_ts()}] fundamentals: cached {len(cached)} rows, {cached['code'].nunique()} codes")
            return cached
        all_parts = []
        for i, code in enumerate(codes, 1):
            try:
                df = self.provider.get_financial_indicators(code)
                if not df.empty:
                    all_parts.append(df)
                if i % 50 == 0:
                    print(f"[{_ts()}]   fundamentals progress: {i}/{len(codes)}")
            except Exception as e:
                print(f"[{_ts()}]   fundamental {code} ERROR: {e}")
            time.sleep(self.sleep_seconds)
        if all_parts:
            result = pd.concat(all_parts, ignore_index=True)
            self.lake.write_parquet("fundamental", "financial_indicators", result)
            print(f"[{_ts()}] fundamentals: fetched {len(result)} rows, {result['code'].nunique()} codes")
            return result
        return cached if not cached.empty else pd.DataFrame()

    def download_margin_data(self, start: str, end: str, force: bool = False) -> pd.DataFrame:
        """下载全市场融资融券数据并缓存。"""
        cached = self.lake.read_parquet("margin", "daily_summary")
        if not force and not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            if cached["date"].max() >= pd.to_datetime(end):
                print(f"[{_ts()}] margin data: cached {len(cached)} rows")
                return cached
        try:
            df = self.provider.get_margin_data(start, end)
            if not df.empty:
                self.lake.append_unique("margin", "daily_summary", df, keys=["date"])
                result = self.lake.read_parquet("margin", "daily_summary")
                print(f"[{_ts()}] margin data: fetched {len(df)} rows")
                return result
        except Exception as e:
            print(f"[{_ts()}] margin data ERROR: {e}")
        return cached if not cached.empty else pd.DataFrame()

    def download_northbound_stock(self, codes: Iterable[str], start: str, end: str, force: bool = False) -> pd.DataFrame:
        """下载个股北向持股数据并缓存。"""
        codes = normalize_codes(codes)
        cached = self.lake.read_parquet("northbound", "stock_holdings")
        if not force and not cached.empty and cached["code"].nunique() >= len(codes) * 0.8:
            print(f"[{_ts()}] northbound stock: cached {len(cached)} rows, {cached['code'].nunique()} codes")
            return cached
        all_parts = []
        for i, code in enumerate(codes, 1):
            try:
                df = self.provider.get_northbound_stock(code, start, end)
                if not df.empty:
                    all_parts.append(df)
                if i % 50 == 0:
                    print(f"[{_ts()}]   northbound stock progress: {i}/{len(codes)}")
            except Exception as e:
                print(f"[{_ts()}]   northbound stock {code} ERROR: {e}")
            time.sleep(self.sleep_seconds)
        if all_parts:
            result = pd.concat(all_parts, ignore_index=True)
            self.lake.write_parquet("northbound", "stock_holdings", result)
            print(f"[{_ts()}] northbound stock: fetched {len(result)} rows, {result['code'].nunique()} codes")
            return result
        return cached if not cached.empty else pd.DataFrame()
