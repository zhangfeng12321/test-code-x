from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from stock_alpha.storage.cache import DataLake


@dataclass
class TaskStatusStore:
    lake: DataLake
    dataset: str = "meta"
    name: str = "download_status"

    def load(self) -> pd.DataFrame:
        df = self.lake.read_parquet(self.dataset, self.name)
        if not df.empty:
            for c in ["task", "code", "start", "end", "status", "error", "updated_at"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)
        if df.empty:
            return pd.DataFrame(columns=["task", "code", "start", "end", "status", "rows", "error", "updated_at"])
        return df

    def upsert(self, task: str, code: str, start: str, end: str, status: str, rows: int = 0, error: str = "") -> None:
        df = self.load()
        code = str(code).zfill(6)[-6:]
        row = {
            "task": task,
            "code": code,
            "start": str(start),
            "end": str(end),
            "status": status,
            "rows": int(rows),
            "error": error,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        key = (df["task"].astype(str) == task) & (df["code"].astype(str).str.zfill(6) == code)
        if key.any():
            df = df.loc[~key].copy()
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.lake.write_parquet(self.dataset, self.name, df)

    def failed(self) -> pd.DataFrame:
        df = self.load()
        return df[df["status"] == "failed"] if not df.empty else df
