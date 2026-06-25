from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class OrderPlanConfig:
    capital: float = 1_000_000.0
    top_n: int = 5
    min_score: float = 0.45
    max_position_pct: float = 0.2
    lot_size: int = 100
    take_profit: float | None = None
    stop_loss: float | None = None
    selection_mode: str = "threshold"
    score_quantile: float = 0.95
    max_down_probability: float | None = None
    max_risk_score: float | None = None
    min_avg_amount_20: float | None = None


def generate_next_day_orders(predictions: pd.DataFrame, daily: pd.DataFrame, cfg: OrderPlanConfig, universe_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if predictions.empty or daily.empty:
        return pd.DataFrame()
    pred = predictions.copy()
    pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"])
    score_col = "final_score_v2" if "final_score_v2" in pred.columns else "final_score"
    latest_date = pred["date"].max()
    x = pred[pred["date"] == latest_date].copy()
    if "risk_blocked" in x.columns:
        x = x[~x["risk_blocked"].fillna(False)]
    if cfg.max_down_probability is not None and "down_probability" in x.columns:
        x = x[x["down_probability"] <= cfg.max_down_probability]
    if cfg.max_risk_score is not None and "risk_score" in x.columns:
        x = x[x["risk_score"] <= cfg.max_risk_score]
    if universe_metrics is not None and not universe_metrics.empty and cfg.min_avg_amount_20 is not None:
        um = universe_metrics.copy()
        um["code"] = um["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        x = x.merge(um[["code", "avg_amount_20"]], on="code", how="left")
        x = x[x["avg_amount_20"].fillna(0) >= cfg.min_avg_amount_20]
    x = x.sort_values(score_col, ascending=False)
    if cfg.selection_mode == "topn":
        x = x.head(cfg.top_n)
    elif cfg.selection_mode == "quantile":
        q = x[score_col].quantile(cfg.score_quantile)
        x = x[x[score_col] >= q].head(cfg.top_n)
    else:
        x = x[x[score_col] >= cfg.min_score].head(cfg.top_n)
    d = daily.copy(); d["code"] = d["code"].astype(str).str.zfill(6); d["date"] = pd.to_datetime(d["date"])
    last_close = d.sort_values("date").groupby("code", as_index=False).tail(1)[["code", "close"]].rename(columns={"close":"ref_price"})
    x = x.merge(last_close, on="code", how="left")
    rows = []
    budget = cfg.capital * cfg.max_position_pct
    for r in x.itertuples(index=False):
        ref = getattr(r, "ref_price", None)
        if pd.isna(ref) or ref <= 0:
            continue
        shares = int((budget / ref) // cfg.lot_size * cfg.lot_size)
        if shares <= 0:
            continue
        rows.append({
            "signal_date": latest_date,
            "code": r.code,
            "action": "BUY",
            "ref_price": ref,
            "shares": shares,
            "planned_amount": shares * ref,
            "score": getattr(r, score_col),
            "up_probability": getattr(r, "up_probability", None),
            "down_probability": getattr(r, "down_probability", None),
            "take_profit_price": ref * (1 + cfg.take_profit) if cfg.take_profit else None,
            "stop_loss_price": ref * (1 - cfg.stop_loss) if cfg.stop_loss else None,
            "note": "次日按开盘/流动性/涨跌停复核后执行",
        })
    return pd.DataFrame(rows)
