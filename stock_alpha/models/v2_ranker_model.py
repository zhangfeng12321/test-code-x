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

        # 保存标签中的实际收益，用于概率校准
        self._calibration_data = data[["rank_label", "fwd_return"]].copy()

        try:
            from lightgbm import LGBMRanker  # type: ignore
            self.model = LGBMRanker(
                objective="lambdarank",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=50,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
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

        # 构建概率校准映射：rank_percentile → 实际上涨概率
        self._build_calibration()

        return self

    def _build_calibration(self):
        """基于训练数据构建校准映射：将 rank 分位数映射到真实上涨概率。

        原理：统计训练集中不同 rank_percentile 区间的实际正收益率，
        用分箱平滑得到 rank_pct → P(up) 的映射表。
        """
        self._calibration_map = None
        if not hasattr(self, '_calibration_data') or self._calibration_data.empty:
            return
        cal = self._calibration_data.copy()
        cal["is_positive"] = (cal["fwd_return"] > 0).astype(float)
        # 按 rank_label 分 20 档统计实际胜率
        cal["rank_pct"] = cal.groupby(cal.index // max(1, len(cal) // 20))["rank_label"].rank(pct=True)
        # 更简单的方式：按 rank_label 直接排序分箱
        cal = cal.sort_values("rank_label")
        n_bins = 20
        cal["bin"] = pd.cut(np.arange(len(cal)), bins=n_bins, labels=False)
        bin_stats = cal.groupby("bin").agg(
            mean_pct=("rank_label", "mean"),
            actual_win_rate=("is_positive", "mean"),
            count=("is_positive", "count"),
        ).reset_index()
        # 存储：每个 rank_label 均值对应的实际胜率
        if not bin_stats.empty and bin_stats["count"].sum() > 100:
            self._calibration_map = bin_stats[["mean_pct", "actual_win_rate"]].values

    def _calibrate_probability(self, rank_percentile: pd.Series) -> pd.Series:
        """将 rank 分位数映射到校准后的真实概率。

        优先使用 OOS 校准表，fallback 到训练集校准。
        """
        cal_map = self._calibration_map
        # 优先使用 OOS 校准表
        if hasattr(self, '_oos_calibration_map') and self._oos_calibration_map is not None:
            cal_map = self._oos_calibration_map

        if cal_map is None or len(cal_map) < 2:
            # 无校准数据时，用保守的线性映射：0.3 ~ 0.7
            return 0.3 + rank_percentile * 0.4
        # 线性插值：分位数 → 实际胜率
        xp = cal_map[:, 0]
        yp = cal_map[:, 1]
        x_min, x_max = xp.min(), xp.max()
        mapped_x = x_min + rank_percentile * (x_max - x_min)
        calibrated = np.interp(mapped_x, xp, yp)
        return pd.Series(calibrated, index=rank_percentile.index).clip(0.05, 0.95)

    def calibrate_from_oos(self, wf_predictions: pd.DataFrame, daily: pd.DataFrame, horizon: int = 5) -> np.ndarray | None:
        """用 Walk-Forward OOS 实际结果重新校准概率映射。

        步骤：
        1. 从 WF predictions 提取 final_score
        2. 与 daily 数据 merge 计算实际 N 日后收益
        3. 按 final_score 分 20 档统计实际正收益比例
        4. 用 OOS 统计替换 in-sample 校准表

        Returns:
            OOS 校准映射表 (n_bins x 2 ndarray) 或 None
        """
        if wf_predictions.empty:
            return None

        pred = wf_predictions.copy()
        pred["code"] = pred["code"].astype(str).str.zfill(6)
        pred["date"] = pd.to_datetime(pred["date"], errors="coerce")

        d = daily.copy()
        d["code"] = d["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.sort_values(["code", "date"])

        # 计算每只股票未来 N 日收益
        parts = []
        for code, x in d.groupby("code"):
            x = x.sort_values("date").reset_index(drop=True)
            x["fwd_ret"] = x["close"].pct_change(horizon).shift(-horizon)
            parts.append(x[["code", "date", "fwd_ret"]])
        fwd = pd.concat(parts, ignore_index=True).dropna(subset=["fwd_ret"])

        # Merge
        merged = pred.merge(fwd, on=["code", "date"], how="inner")
        if len(merged) < 200:
            return None

        # 按 final_score 分 20 档统计实际胜率
        merged["is_positive"] = (merged["fwd_ret"] > 0).astype(float)
        merged = merged.sort_values("final_score")
        n_bins = 20
        merged["bin"] = pd.cut(np.arange(len(merged)), bins=n_bins, labels=False)
        bin_stats = merged.groupby("bin").agg(
            mean_score=("final_score", "mean"),
            actual_win_rate=("is_positive", "mean"),
            count=("is_positive", "count"),
        ).reset_index()

        if bin_stats.empty or bin_stats["count"].sum() < 100:
            return None

        oos_map = bin_stats[["mean_score", "actual_win_rate"]].values
        self._oos_calibration_map = oos_map
        return oos_map

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """预测并输出兼容 pipeline 的格式。概率经过校准。"""
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

        # 概率校准：将 rank percentile 映射到真实胜率
        out["up_probability"] = self._calibrate_probability(out["final_score"])
        out["down_probability"] = (1 - out["up_probability"]).clip(0.05, 0.95)
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
