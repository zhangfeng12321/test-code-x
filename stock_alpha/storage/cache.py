from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


@dataclass
class DataLake:
    """本地数据湖。

    默认使用 CSV 缓存，避免强依赖大型二进制包；如果环境安装 duckdb/pyarrow，后续可无缝扩展为 Parquet。
    """

    root: Path | str = Path("data")
    db_path: Path | str = Path("data/stock_alpha.duckdb")

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.db_path = Path(self.db_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset: str) -> Path:
        p = self.root / dataset
        p.mkdir(parents=True, exist_ok=True)
        return p

    def data_path(self, dataset: str, name: str) -> Path:
        return self._dataset_dir(dataset) / f"{name}.csv"

    # 兼容旧调用名
    def parquet_path(self, dataset: str, name: str) -> Path:
        return self.data_path(dataset, name)

    def write_parquet(self, dataset: str, name: str, df: pd.DataFrame) -> Path:
        return self.write_table(dataset, name, df)

    def read_parquet(self, dataset: str, name: str) -> pd.DataFrame:
        return self.read_table(dataset, name)

    def write_table(self, dataset: str, name: str, df: pd.DataFrame) -> Path:
        path = self.data_path(dataset, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        if "code" in out.columns:
            out["code"] = out["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        out.to_csv(path, index=False)
        return path

    def read_table(self, dataset: str, name: str) -> pd.DataFrame:
        path = self.data_path(dataset, name)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, dtype={"code": str})
        except EmptyDataError:
            return pd.DataFrame()
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        return df

    def append_unique(self, dataset: str, name: str, df: pd.DataFrame, keys: list[str]) -> Path:
        old = self.read_table(dataset, name)
        if old.empty:
            out = df.copy()
        else:
            out = pd.concat([old, df], ignore_index=True)
            out = out.drop_duplicates(subset=keys, keep="last")
        return self.write_table(dataset, name, out)

    def query_csv(self, dataset: str, name: str) -> pd.DataFrame:
        return self.read_table(dataset, name)

    def connect(self):
        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("duckdb not installed; use read_table/write_table or install duckdb") from exc
        return duckdb.connect(str(self.db_path))

    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).df()

    def register_views(self) -> None:
        """DuckDB 可选视图注册。当前 CSV 缓存下不是必须。"""
        with self.connect() as con:
            daily_glob = str((self.root / "daily" / "*.csv").resolve())
            minute_glob = str((self.root / "minute" / "*.csv").resolve())
            pred_glob = str((self.root / "predictions" / "*.csv").resolve())
            con.execute(f"CREATE OR REPLACE VIEW daily_bars AS SELECT * FROM read_csv_auto('{daily_glob}', union_by_name=true)")
            con.execute(f"CREATE OR REPLACE VIEW minute_bars AS SELECT * FROM read_csv_auto('{minute_glob}', union_by_name=true)")
            con.execute(f"CREATE OR REPLACE VIEW predictions AS SELECT * FROM read_csv_auto('{pred_glob}', union_by_name=true)")
