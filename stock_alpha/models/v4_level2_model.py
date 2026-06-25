from __future__ import annotations

import pandas as pd

from stock_alpha.features.v4_level2 import V4_FEATURE_COLUMNS


class Level2ModelBase:
    """V4 盘口模型基类。真实 DeepLOB/Transformer 后续继承实现。"""

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Level2HeuristicScorer(Level2ModelBase):
    """V4 框架内置盘口启发式评分，便于无付费接口时验证链路。"""

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame | None = None):
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        score = (
            df["depth_imbalance"].fillna(0) * 0.45
            + df["level1_imbalance"].fillna(0) * 0.35
            - df["relative_spread"].fillna(0).clip(0, 0.01) * 20
            + df["last_mid_deviation"].fillna(0).clip(-0.005, 0.005) * 10
        )
        out_cols = [c for c in ["code", "datetime"] if c in df.columns]
        out = df[out_cols].copy()
        out["level2_score"] = score.clip(-1, 1)
        out["level2_signal"] = pd.cut(out["level2_score"], bins=[-2, -0.2, 0.2, 2], labels=["SELL_PRESSURE", "NEUTRAL", "BUY_PRESSURE"])
        return out


class DeepLOBFramework(Level2ModelBase):
    """DeepLOB 框架占位。

    预期输入：shape = [samples, time_window, levels * fields]
    推荐字段：bid/ask price + volume for 10 levels。
    真实实现需要 Level-2 历史样本和 torch/tensorflow，本项目先保留工程接口。
    """

    def __init__(self, levels: int = 10, window: int = 100):
        self.levels = levels
        self.window = window

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        raise NotImplementedError("DeepLOB training requires paid Level-2 dataset and deep learning backend")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("DeepLOB inference requires a trained model checkpoint")
