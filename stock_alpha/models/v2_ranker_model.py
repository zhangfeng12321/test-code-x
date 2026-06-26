"""V2 Ranker 模型：使用 LightGBM LambdaRank 学习横截面选股排序。

核心优势：直接学习"同一天内哪只股票更值得买"，比三分类更匹配 TopN 选股。
输出兼容现有 pipeline（final_score / suggest_action / up_probability 等）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_alpha.features.v1_daily import V1_FEATURE_COLUMNS


class V2RankerModel:
    """LightGBM Ranker 模型：学习横截面排序。"""

    def __init__(self):
        self.model = None
        self.backend = "heuristic"
        self.feature_columns = V1_FEATURE_COLUMNS

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        """训练 Ranker 模型。

        Args:
            features: 特征表（需包含 code, date, V1_FEATURE_COLUMNS）
            labels: 排序标签表（需包含 code, date, rank_label, fwd_return）
        """
        data = features.merge(labels[["code", "date", "rank_label", "fwd_return"]], on=["code", "date"], how="inner")
        data = data.dropna(subset=["rank_label"])
        data = data.sort_values("date")  # Ranker 需要按 group 顺序排列

        X = data[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = data["rank_label"].astype(int)

        # 计算每个 date group 的大小（LGBMRanker 需要 group 参数）
        group_sizes = data.groupby("date").size().values

        try:
            from lightgbm import LGBMRanker  # type: ignore
            self.model = LGBMRanker(
                objective="lambdarank",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=63,
                min_child_samples=50,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X, y, group=group_sizes)
            self.backend = "lightgbm_ranker"
        except Exception:
            # Fallback: 用回归模型预测未来收益率（近似排序效果）
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
                self.model = HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05, max_depth=6, random_state=42
                )
                # 回归目标用 fwd_return（连续值）
                y_reg = data["fwd_return"].fillna(0).values
                self.model.fit(X, y_reg)
                self.backend = "sklearn_regressor"
            except Exception:
                # 最终 fallback: 简单线性组合
                self.model = _HeuristicRanker()
                self.model.fit(X, y)
                self.backend = "heuristic_ranker"

        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """预测并输出兼容 pipeline 的格式。"""
        X = features[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        # 获取原始排序分数
        if self.backend == "lightgbm_ranker":
            rank_score = self.model.predict(X)
        elif self.backend == "sklearn_regressor":
            rank_score = self.model.predict(X)
        else:
            rank_score = self.model.predict(X)

        out = features[["code", "date"]].copy()
        out["rank_score"] = rank_score

        # 按日归一化到 0~1（横截面内相对位置）
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["final_score"] = out.groupby("date")["rank_score"].rank(pct=True)

        # 兼容字段
        out["up_probability"] = out["final_score"]
        out["down_probability"] = 1 - out["final_score"]
        out["neutral_probability"] = 0.0

        # 风险分沿用 ATR
        out["risk_score"] = features.get("atr_14", pd.Series(0, index=features.index)).fillna(0).clip(0, 0.2) * 5

        # suggest_action: 基于当日分位
        out["suggest_action"] = np.where(
            out["final_score"] >= 0.8, "BUY",
            np.where(out["final_score"] <= 0.2, "AVOID", "WATCH")
        )

        return out.sort_values("final_score", ascending=False)


class _HeuristicRanker:
    """无ML依赖时的简单排序备用模型。"""

    def fit(self, X: pd.DataFrame, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 简单线性组合作为排序分
        score = (
            X.get("ret_3d", pd.Series(0)).fillna(0) * 0.3
            + X.get("ret_5d", pd.Series(0)).fillna(0) * 0.2
            + X.get("close_ma5_ratio", pd.Series(0)).fillna(0) * 0.2
            + X.get("volume_ratio_5", pd.Series(0)).fillna(0).clip(0, 5) * 0.1
            + X.get("market_ret_1d", pd.Series(0)).fillna(0) * 0.1
            - X.get("atr_14", pd.Series(0)).fillna(0) * 0.1
        )
        return score.values
