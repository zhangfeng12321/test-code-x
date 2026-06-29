from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from stock_alpha.data.downloader import MarketDataDownloader
from stock_alpha.storage.status import TaskStatusStore


def _ts() -> str:
    """返回当前时间戳，用于日志打印。"""
    return datetime.now().strftime('%H:%M:%S')


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


@dataclass
class BatchDownloadRunner:
    downloader: MarketDataDownloader
    batch_size: int = 50
    workers: int = 1  # 固定单线程串行下载

    def download_daily_batches(self, codes: list[str], start: str, end: str, adjust: str = "qfq", force: bool = False) -> pd.DataFrame:
        parts = []
        batches = list(chunks(codes, self.batch_size))
        for idx, batch in enumerate(batches, 1):
            print(f"[{_ts()}] batch {idx}/{len(batches)} size={len(batch)}")
            df = self.downloader.download_daily(batch, start, end, adjust=adjust, force=force)
            if not df.empty:
                parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def retry_failed_daily(self, start: str | None = None, end: str | None = None, force: bool = True) -> pd.DataFrame:
        status = TaskStatusStore(self.downloader.lake).failed()
        if status.empty:
            print(f"[{_ts()}] no failed tasks")
            return pd.DataFrame()
        daily_failed = status[status["task"].astype(str) == "daily"]
        if daily_failed.empty:
            print(f"[{_ts()}] no failed daily tasks")
            return pd.DataFrame()
        codes = daily_failed["code"].astype(str).str.zfill(6).drop_duplicates().tolist()
        s = start or str(daily_failed["start"].iloc[0])
        e = end or str(daily_failed["end"].iloc[0])
        return self.download_daily_batches(codes, s, e, force=force)
