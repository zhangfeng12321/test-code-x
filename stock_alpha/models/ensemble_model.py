"""Ensemble 模型：Ranker + Classifier 加权融合。

核心思路：
- Ranker 擅长横截面排序（谁比谁强）
- Classifier 擅长方向判断（涨 / 跌 / 震荡）
- 加权融合兼顾两者优势，提升选股稳定性

融合公式：
    final_score = alpha * ranker_score + (1 - alpha) * classifier_score

其中 ranker_score 和 classifier_score 均归一化到 [0, 1]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_alpha.models.v1_daily_model import V1DailyAlphaModel
from stock_alpha.models.v2_ranker_model import V2RankerModel
from stock_alpha.features.v1_daily import V1_FEATURE_COLUMNS


class EnsembleModel:
    """Ranker + Classifier 加权融合模型。"""

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Ranker 权重（0~1），Classifier 权重为 1-alpha。
                   默认 0.6 表示 Ranker 占主导（横截面排序更匹配 TopN 选股）。
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.ranker = V2RankerModel()
        self.classifier = V1DailyAlphaModel()
        self.feature_columns = V1_FEATURE_COLUMNS
        self.backend = "ensemble"

    def fit(
        self,
        features: pd.DataFrame,
        ranking_labels: pd.DataFrame,
        classification_labels: pd.DataFrame,
    ) -> "EnsembleModel":
        """同时训练 Ranker 和 Classifier。

        Args:
            features: 特征表（需包含 code, date, V1_FEATURE_COLUMNS）
            ranking_labels: 排序标签（code, date, rank_label, fwd_return）
            classification_labels: 三分类标签（code, date, label）
        """
        self.ranker.fit(features, ranking_labels)
        self.classifier.fit(features, classification_labels)
        self.backend = f"ensemble(ranker={self.ranker.backend},classifier={self.classifier.backend})"
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """融合两个模型的预测结果。

        融合策略：
        1. Ranker 输出 rank_score → 按日归一化到 [0, 1] 作为 ranker_norm
        2. Classifier 输出 up_probability 作为 classifier_norm（已在 [0, 1]）
        3. final_score = alpha * ranker_norm + (1 - alpha) * classifier_norm
        """
        # 获取 Ranker 预测
        ranker_pred = self.ranker.predict(features)
        # 获取 Classifier 预测
        classifier_pred = self.classifier.predict(features)

        # 以 Ranker 输出为基础表
        out = ranker_pred[["code", "date"]].copy()
        out["ranker_score"] = ranker_pred["final_score"].values  # 已按日归一化到 0~1
        out["classifier_score"] = self._align_classifier_score(
            out, classifier_pred
        )

        # 加权融合
        alpha = self.alpha
        out["final_score"] = (
            alpha * out["ranker_score"] + (1 - alpha) * out["classifier_score"]
        )

        # 输出兼容字段
        out["up_probability"] = out["final_score"]
        # down_probability：用 Classifier 的 down_prob 按对齐方式取值
        out["down_probability"] = self._align_column(
            out, classifier_pred, "down_probability", default=1 - out["final_score"]
        )
        out["neutral_probability"] = self._align_column(
            out, classifier_pred, "neutral_probability", default=0.0
        )
        out["risk_score"] = self._align_column(
            out, ranker_pred, "risk_score", default=0.0
        )

        # suggest_action：基于融合后 final_score
        out["suggest_action"] = np.where(
            out["final_score"] >= 0.8, "BUY",
            np.where(out["final_score"] <= 0.2, "AVOID", "WATCH")
        )

        # 保留子模型分数，便于后续分析
        out["ranker_contribution"] = alpha * out["ranker_score"]
        out["classifier_contribution"] = (1 - alpha) * out["classifier_score"]

        return out.sort_values("final_score", ascending=False)

    def _align_classifier_score(
        self, base: pd.DataFrame, classifier_pred: pd.DataFrame
    ) -> pd.Series:
        """将 Classifier 的 up_probability 对齐到 Ranker 输出的行顺序。"""
        cls_score = classifier_pred[["code", "date", "up_probability"]].copy()
        cls_score["date"] = pd.to_datetime(cls_score["date"], errors="coerce")
        base_align = base[["code", "date"]].copy()
        base_align["date"] = pd.to_datetime(base_align["date"], errors="coerce")
        merged = base_align.merge(
            cls_score, on=["code", "date"], how="left"
        )
        # 缺失值用 0.5（中性分）填充
        return merged["up_probability"].fillna(0.5).values

    def _align_column(
        self,
        base: pd.DataFrame,
        source_pred: pd.DataFrame,
        col: str,
        default=0.0,
    ) -> pd.Series:
        """通用列对齐辅助函数，支持标量和 Series 类型的 default 值。"""
        if col not in source_pred.columns:
            if isinstance(default, (int, float)):
                return pd.Series(default, index=base.index)
            return default
        src = source_pred[["code", "date", col]].copy()
        src["date"] = pd.to_datetime(src["date"], errors="coerce")
        base_align = base[["code", "date"]].copy()
        base_align["date"] = pd.to_datetime(base_align["date"], errors="coerce")
        merged = base_align.merge(src, on=["code", "date"], how="left")
        # 支持 Series 类型的 default：按 base 行顺序对齐后填充
        if isinstance(default, pd.Series):
            fill_values = default.reindex(base.index, fill_value=0.0)
            result = merged[col].fillna(fill_values)
        else:
            result = merged[col].fillna(default)
        return result.values
