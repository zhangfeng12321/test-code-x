from __future__ import annotations

import pandas as pd


def candidate_risk_tags(predictions: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """给候选股打风险标签。"""
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("code", as_index=False).tail(1).copy()
    tags = []
    for r in latest.itertuples(index=False):
        t = []
        if getattr(r, "risk_score", 0) >= 0.18:
            t.append("高波动")
        if getattr(r, "down_probability", 0) >= 0.55:
            t.append("下跌概率高")
        if getattr(r, "up_probability", 0) >= 0.65:
            t.append("强上涨概率")
        if getattr(r, "final_score", 0) < 0.15:
            t.append("综合分偏低")
        if hasattr(r, "intraday_score") and getattr(r, "intraday_score", 0) < 0.35:
            t.append("分时弱")
        if getattr(r, "is_limit_up", False):
            t.append("涨停风险/可能买不进")
        if getattr(r, "is_limit_down", False):
            t.append("跌停风险/流动性差")
        if getattr(r, "risk_blocked", False):
            t.append("风控阻断")
        tags.append("、".join(t) if t else "正常")
    latest["risk_tags"] = tags
    return latest[["code", "date", "risk_tags"]]


def explain_candidates(features: pd.DataFrame, predictions: pd.DataFrame, feature_importance: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """基于特征重要性和当前特征值生成简易解释。

    不是 SHAP，只是工程可用的“重要特征快照”。
    """
    if features.empty or predictions.empty or feature_importance.empty:
        return pd.DataFrame()
    f = features.copy()
    p = predictions.copy()
    f["code"] = f["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    p["code"] = p["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    f["date"] = pd.to_datetime(f["date"])
    p["date"] = pd.to_datetime(p["date"])
    latest_pred = p.sort_values("date").groupby("code", as_index=False).tail(1)[["code", "date"]]
    data = latest_pred.merge(f, on=["code", "date"], how="left")
    top_features = feature_importance.sort_values("importance", ascending=False)["feature"].head(20).tolist()
    rows = []
    for r in data.itertuples(index=False):
        items = []
        for feat in top_features:
            if hasattr(r, feat):
                val = getattr(r, feat)
                try:
                    if pd.notna(val):
                        items.append((feat, float(val)))
                except Exception:
                    pass
            if len(items) >= top_k:
                break
        explain = "；".join([f"{k}={v:.4f}" for k, v in items]) if items else "暂无解释"
        rows.append({"code": r.code, "date": r.date, "why_selected": explain})
    return pd.DataFrame(rows)
