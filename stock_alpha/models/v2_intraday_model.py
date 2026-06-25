from __future__ import annotations

import numpy as np
import pandas as pd


class V2IntradayScorer:
    """V2 分时强弱评分器。可与 V1 模型结果合并。"""

    def score(self, intraday_features: pd.DataFrame) -> pd.DataFrame:
        df = intraday_features.copy()
        vol_score = np.log1p(df["open_30m_volume_ratio"].fillna(0) * 5) * 0.25
        vwap_score = df["above_vwap_ratio"].fillna(0.5) * 0.35 + df["vwap_deviation"].fillna(0).clip(-0.03, 0.03) * 5
        late_penalty = (df["late_30m_volume_ratio"].fillna(0) > 0.35).astype(float) * 0.10
        spike_score = np.log1p(df["minute_volume_spike_count"].fillna(0)) * 0.05
        df["intraday_score"] = (vol_score + vwap_score + spike_score - late_penalty).clip(0, 1)
        return df[["code", "date", "intraday_score"]]

    def merge_with_v1(self, v1_predictions: pd.DataFrame, intraday_scores: pd.DataFrame) -> pd.DataFrame:
        out = v1_predictions.merge(intraday_scores, on=["code", "date"], how="left")
        out["intraday_score"] = out["intraday_score"].fillna(0.5)
        out["final_score_v2"] = 0.75 * out["final_score"] + 0.25 * out["intraday_score"]
        out["suggest_action_v2"] = np.where(out["final_score_v2"] >= 0.50, "BUY", np.where(out["final_score_v2"] <= 0.15, "AVOID", "WATCH"))
        return out.sort_values("final_score_v2", ascending=False)
