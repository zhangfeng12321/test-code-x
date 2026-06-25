from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import Level2DataProvider


class CsvLevel2Provider(Level2DataProvider):
    """V4 CSV Level-2 Provider。

    CSV 列约定：
    code,datetime,last_price,bid_price_1,bid_volume_1,ask_price_1,ask_volume_1,...bid_price_10,bid_volume_10,ask_price_10,ask_volume_10
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def stream_order_book(self, code: str, start: datetime, end: datetime) -> Iterable[dict]:
        df = pd.read_csv(self.csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["code"].astype(str) == str(code)) & (df["datetime"] >= start) & (df["datetime"] <= end)]
        for row in df.sort_values("datetime").to_dict("records"):
            yield row
