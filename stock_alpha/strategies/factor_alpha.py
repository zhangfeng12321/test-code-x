"""因子选股策略：封装现有 EnsembleModel 为统一策略接口。

这是现有系统的核心策略，基于89维特征 + LightGBM Ranker + Classifier 融合。
"""
from __future__ import annotations

import pandas as pd

from stock_alpha.strategies.base import BaseStrategy
from stock_alpha.storage.cache import DataLake
from stock_alpha.training.train_v1 import V1Trainer


class FactorAlphaStrategy(BaseStrategy):
    """因子选股策略：多因子模型打分 → 选股。"""

    name = "factor_alpha"

    def __init__(self, lake: DataLake, model_type: str = "ensemble", ensemble_alpha: float = 0.6, label_horizon: int = 3):
        self.lake = lake
        self.model_type = model_type
        self.ensemble_alpha = ensemble_alpha
        self.label_horizon = label_horizon
        self._cached_signals: pd.DataFrame | None = None
        self._cached_daily_hash: int | None = None

    def generate_signals(self, daily: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
        """使用因子模型生成信号。"""
        if daily.empty:
            return self._empty_signals()

        # 缓存检查：用 daily 的行数和最新日期作为指纹，避免重复训练
        daily_hash = hash((len(daily), str(daily["date"].max()) if "date" in daily.columns else None))
        if self._cached_signals is not None and self._cached_daily_hash == daily_hash:
            pred = self._cached_signals
        else:
            trainer = V1Trainer(self.lake)
            # 训练并获取预测
            try:
                codes = daily["code"].unique().tolist() if "code" in daily.columns else None
                result = trainer.train(
                    codes=codes,
                    model_type=self.model_type,
                    ensemble_alpha=self.ensemble_alpha,
                    label_horizon=self.label_horizon,
                )
                pred = self.lake.read_parquet("predictions", "v1_latest")
            except Exception:
                return self._empty_signals()
            self._cached_signals = pred
            self._cached_daily_hash = daily_hash

        if pred.empty:
            return self._empty_signals()

        # 转换为统一信号格式
        pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
        if date is not None:
            pred = pred[pred["date"] == pd.to_datetime(date)]

        signals = []
        for _, row in pred.iterrows():
            score = float(row.get("final_score", 0))
            action = row.get("suggest_action", "HOLD")
            if action == "BUY" and score >= 0.6:
                action = "BUY"
            elif action == "AVOID" or score < 0.3:
                action = "AVOID"
            else:
                action = "HOLD"
            signals.append(self._make_signal(
                code=row["code"], date=row["date"],
                score=score, action=action, strategy=self.name,
            ))

        return pd.DataFrame(signals) if signals else self._empty_signals()
