from __future__ import annotations

import numpy as np
import pandas as pd

from stock_alpha.features.v1_daily import V1_FEATURE_COLUMNS


class HeuristicShortTermModel:
    """无第三方 ML 依赖时的可运行备用模型。用于 smoke/基线，不用于严肃实盘。"""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = (
            X.get("ret_3d", 0).fillna(0) * 1.2
            + X.get("close_ma5_ratio", 0).fillna(0) * 1.5
            + np.log1p(X.get("volume_ratio_5", 1).fillna(1).clip(lower=0)) * 0.15
            + X.get("macd_hist", 0).fillna(0) * 0.2
            - X.get("upper_shadow", 0).fillna(0) * 1.0
            - X.get("atr_14", 0).fillna(0) * 0.5
        )
        up = 1 / (1 + np.exp(-8 * z))
        down = 1 - up
        neutral = 1 - np.abs(up - 0.5) * 1.4
        neutral = np.clip(neutral, 0.05, 0.8)
        up2 = up * (1 - neutral)
        down2 = down * (1 - neutral)
        return np.vstack([down2, neutral, up2]).T


class V1DailyAlphaModel:
    """V1 日线模型包装器：优先 LightGBM，其次 sklearn，最后启发式。"""

    def __init__(self):
        self.model = None
        self.backend = "heuristic"
        self.feature_columns = V1_FEATURE_COLUMNS

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        data = features.merge(labels[["code", "date", "label"]], on=["code", "date"], how="inner")
        data = data.dropna(subset=["label"])
        X = data[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = data["label"].astype(int).map({-1: 0, 0: 1, 1: 2})
        try:
            from lightgbm import LGBMClassifier  # type: ignore
            self.model = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=-1, objective="multiclass", random_state=42, verbose=-1)
            self.backend = "lightgbm"
        except Exception:
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
                self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, random_state=42)
                self.backend = "sklearn_hgb"
            except Exception:
                self.model = HeuristicShortTermModel()
                self.backend = "heuristic"
        self.model.fit(X, y)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        proba = self.model.predict_proba(X)
        out = features[["code", "date"]].copy()
        # 类顺序按备用模型约定：down, neutral, up。sklearn/lightgbm 若缺类则需兼容。
        if proba.shape[1] == 3:
            out["down_probability"] = proba[:, 0]
            out["neutral_probability"] = proba[:, 1]
            out["up_probability"] = proba[:, 2]
        else:
            out["up_probability"] = proba[:, -1]
            out["down_probability"] = 1 - out["up_probability"]
            out["neutral_probability"] = 0.0
        out["risk_score"] = features.get("atr_14", pd.Series(0, index=features.index)).fillna(0).clip(0, 0.2) * 5
        out["final_score"] = 0.7 * out["up_probability"] - 0.2 * out["down_probability"] - 0.1 * out["risk_score"]
        out["suggest_action"] = np.where(out["final_score"] >= 0.45, "BUY", np.where(out["final_score"] <= 0.1, "AVOID", "WATCH"))
        return out.sort_values("final_score", ascending=False)
