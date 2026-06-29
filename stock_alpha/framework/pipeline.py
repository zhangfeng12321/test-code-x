from __future__ import annotations

import pandas as pd

from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.features.v2_intraday import build_intraday_features
from stock_alpha.features.v4_level2 import build_level2_features
from stock_alpha.labels.triple_barrier import make_triple_barrier_labels
from stock_alpha.models.v1_daily_model import V1DailyAlphaModel
from stock_alpha.models.v2_intraday_model import V2IntradayScorer
from stock_alpha.models.v4_level2_model import Level2HeuristicScorer


class V1Pipeline:
    def fit_predict(self, daily: pd.DataFrame, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
        features = build_daily_features(daily, stock_basic=stock_basic)
        labels = make_triple_barrier_labels(daily)
        model = V1DailyAlphaModel().fit(features, labels)
        pred = model.predict(features)
        pred.attrs["backend"] = model.backend
        return pred


class V2Pipeline:
    def score(self, v1_predictions: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
        intraday = build_intraday_features(minute)
        scores = V2IntradayScorer().score(intraday)
        return V2IntradayScorer().merge_with_v1(v1_predictions, scores)


class V4Pipeline:
    def score_level2(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        features = build_level2_features(snapshots)
        return Level2HeuristicScorer().fit(features).predict(features)
