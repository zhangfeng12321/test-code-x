from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd

from stock_alpha.backtest.ashare_backtest import AShareBacktestConfig, AShareBacktester
from stock_alpha.storage.cache import DataLake


@dataclass
class GridSearchRunner:
    lake: DataLake
    top_n_values: list[int] = field(default_factory=lambda: [3, 5, 10])
    hold_days_values: list[int] = field(default_factory=lambda: [2, 3, 5])
    min_score_values: list[float] = field(default_factory=lambda: [0.35, 0.45, 0.55])
    take_profit_values: list[float | None] = field(default_factory=lambda: [None, 0.03, 0.05])
    stop_loss_values: list[float | None] = field(default_factory=lambda: [None, 0.02, 0.03])

    def run(self, daily: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for top_n in self.top_n_values:
            for hold_days in self.hold_days_values:
                for min_score in self.min_score_values:
                    for take_profit in self.take_profit_values:
                        for stop_loss in self.stop_loss_values:
                            cfg = AShareBacktestConfig(top_n=top_n, hold_days=hold_days, min_score=min_score, take_profit=take_profit, stop_loss=stop_loss)
                            bt = AShareBacktester(cfg).run(daily, predictions)
                            m = bt["metrics"].iloc[0].to_dict() if not bt["metrics"].empty else {}
                            rows.append({"top_n": top_n, "hold_days": hold_days, "min_score": min_score, "take_profit": take_profit, "stop_loss": stop_loss, **m})
        out = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)
        self.lake.write_parquet("optimization", "grid_search", out)
        if not out.empty:
            self.lake.write_parquet("optimization", "best_params", pd.DataFrame([out.iloc[0].to_dict()]))
        return out

    def write_recommended_config(self, base_config: dict, path: str | Path) -> Path:
        best = self.lake.read_parquet("optimization", "best_params")
        if best.empty:
            raise RuntimeError("best_params not found; run grid search first")
        row = best.iloc[0]
        cfg = dict(base_config)
        def none_or_float(v):
            return None if pd.isna(v) else float(v)
        cfg.update({
            "top_n": int(row["top_n"]),
            "hold_days": int(row["hold_days"]),
            "min_score": float(row["min_score"]),
            "take_profit": none_or_float(row.get("take_profit")),
            "stop_loss": none_or_float(row.get("stop_loss")),
        })
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p
