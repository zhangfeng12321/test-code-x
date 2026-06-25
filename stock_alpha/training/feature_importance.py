from __future__ import annotations

import pandas as pd


def extract_feature_importance(model_bundle: dict) -> pd.DataFrame:
    model = model_bundle.get("model") if isinstance(model_bundle, dict) else model_bundle
    feature_columns = getattr(model, "feature_columns", None) or model_bundle.get("feature_columns", [])
    estimator = getattr(model, "model", model)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": feature_columns, "importance": 0})
