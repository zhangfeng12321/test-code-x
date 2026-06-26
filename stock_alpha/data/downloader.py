from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from stock_alpha.data.providers.base import MarketDataProvider
from stock_alpha.storage.cache import DataLake
from stock_alpha.storage.status import TaskStatusStore
from stock_alpha.data.incremental import missing_date_ranges, missing_datetime_ranges


def normalize_codes(codes: Iterable[str]) -> list[str]:
    return [str(c).zfill(6)[-6:] for c in codes]


@dataclass
class MarketDataDownloader:
    provider: MarketDataProvider
    lake: DataLake
    sleep_seconds: float = 0.2
    max_retries: int = 3

    def get_stock_universe(self, limit: Optional[int] = None) -> pd.DataFrame:
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
                    print(f"[{i}/{len(codes)}] daily {code}: cached {len(df)}")
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
                print(f"[{i}/{len(codes)}] daily {code}: fetched {len(df)} ranges={len(ranges)}")
            except Exception as e:
                errors.append({"code": code, "error": repr(e)})
                self.status_store.upsert("daily", code, start, end, "failed", 0, repr(e))
                print(f"[{i}/{len(codes)}] daily {code} ERROR: {e}")
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
                    print(f"[{i}/{len(codes)}] minute {period}m {code}: cached {len(df)}")
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
                print(f"[{i}/{len(codes)}] minute {period}m {code}: fetched {len(df)} ranges={len(ranges)}")
            except Exception as e:
                errors.append({"code": code, "error": repr(e)})
                self.status_store.upsert(f"minute_{period}m", code, start, end, "failed", 0, repr(e))
                print(f"[{i}/{len(codes)}] minute {code} ERROR: {e}")
            time.sleep(self.sleep_seconds)
        if errors:
            self.lake.write_parquet("logs", "minute_errors", pd.DataFrame(errors))
        return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
