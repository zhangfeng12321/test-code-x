from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_alpha.backtest.ashare_backtest import AShareBacktester, AShareBacktestConfig
from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.labels.triple_barrier import make_triple_barrier_labels
from stock_alpha.models.v1_daily_model import V1DailyAlphaModel
from stock_alpha.storage.cache import DataLake


@dataclass
class WalkForwardRunner:
    lake: DataLake
    train_days: int = 120
    test_days: int = 30
    step_days: int = 30
    backtest_config: AShareBacktestConfig | None = None
    label_profit_take: float = 0.03
    label_stop_loss: float = 0.02
    label_horizon: int = 3

    def run(self, daily: pd.DataFrame, score_col: str = "final_score") -> dict[str, pd.DataFrame]:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["code"] = daily["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        dates = sorted(daily["date"].dropna().unique())
        if len(dates) < self.train_days + self.test_days:
            raise RuntimeError("not enough dates for walk-forward")
        features = build_daily_features(daily)
        labels = make_triple_barrier_labels(daily, profit_take=self.label_profit_take, stop_loss=self.label_stop_loss, horizon=self.label_horizon)
        all_preds = []
        windows = []
        start_idx = 0
        while start_idx + self.train_days + self.test_days <= len(dates):
            train_start = pd.Timestamp(dates[start_idx])
            train_end = pd.Timestamp(dates[start_idx + self.train_days - 1])
            test_start = pd.Timestamp(dates[start_idx + self.train_days])
            test_end = pd.Timestamp(dates[start_idx + self.train_days + self.test_days - 1])
            train_f = features[(features["date"] >= train_start) & (features["date"] <= train_end)]
            train_l = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
            test_f = features[(features["date"] >= test_start) & (features["date"] <= test_end)]
            if not train_f.empty and not test_f.empty:
                model = V1DailyAlphaModel().fit(train_f, train_l)
                pred = model.predict(test_f)
                pred["wf_train_start"] = train_start
                pred["wf_train_end"] = train_end
                pred["wf_test_start"] = test_start
                pred["wf_test_end"] = test_end
                all_preds.append(pred)
                windows.append({"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end, "rows": len(pred), "backend": model.backend})
            start_idx += self.step_days
        predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
        windows_df = pd.DataFrame(windows)
        if not predictions.empty:
            self.lake.write_parquet("predictions", "walk_forward", predictions)
            cfg = self.backtest_config or AShareBacktestConfig(score_col=score_col)
            bt = AShareBacktester(cfg).run(daily, predictions)
            for k, v in bt.items():
                self.lake.write_parquet("walk_forward", f"{k}", v)
        self.lake.write_parquet("walk_forward", "windows", windows_df)
        return {"predictions": predictions, "windows": windows_df}
