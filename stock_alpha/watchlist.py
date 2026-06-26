from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_alpha.risk_rules import RiskRuleConfig, attach_latest_context, apply_hard_risk_filters


@dataclass
class WatchlistConfig:
    """Observation-pool gates.

    WATCH is not a buy list. It is a shortlist for next-session confirmation.
    The gates are deliberately softer than BUY, but still block weak-rebound traps.
    """

    top_n: int = 20
    min_score: float = 0.18
    max_down_probability: float = 0.55
    max_risk_score: float = 0.45
    min_avg_amount_20: float | None = 200_000_000.0


def generate_watchlist(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: WatchlistConfig | None = None,
    universe_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = cfg or WatchlistConfig()
    if predictions.empty or daily.empty:
        return pd.DataFrame()
    pred = attach_latest_context(predictions, daily)
    pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], format="mixed")
    latest_date = pred["date"].max()
    x = pred[pred["date"] == latest_date].copy()
    score_col = "final_score_v2" if "final_score_v2" in x.columns else "final_score"

    # Softer than BUY, but do not require up_probability > down_probability because this model's
    # 10%/5d label is conservative and tends to overstate downside for volatile pools.
    risk_cfg = RiskRuleConfig(
        min_score=cfg.min_score,
        max_down_probability=cfg.max_down_probability,
        max_risk_score=cfg.max_risk_score,
        require_up_gt_down=False,
    )
    x = apply_hard_risk_filters(x, risk_cfg, score_col=score_col)
    x = x[~x["risk_blocked"].fillna(False)].copy()

    if universe_metrics is not None and not universe_metrics.empty and cfg.min_avg_amount_20 is not None:
        um = universe_metrics.copy()
        um["code"] = um["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        um = um[["code", "avg_amount_20"]].rename(columns={"avg_amount_20": "avg_amount_20_universe"})
        x = x.merge(um, on="code", how="left")
        amount_col = x["avg_amount_20_universe"].combine_first(x["avg_amount_20"]) if "avg_amount_20" in x.columns else x["avg_amount_20_universe"]
        x = x[amount_col.fillna(0) >= cfg.min_avg_amount_20]

    x = x.sort_values(score_col, ascending=False).head(cfg.top_n)
    if x.empty:
        return pd.DataFrame()

    rows = []
    for i, r in enumerate(x.itertuples(index=False), 1):
        rows.append({
            "signal_date": latest_date,
            "rank": i,
            "code": r.code,
            "action": "WATCH",
            "score": getattr(r, score_col),
            "up_probability": getattr(r, "up_probability", None),
            "down_probability": getattr(r, "down_probability", None),
            "risk_score": getattr(r, "risk_score", None),
            "ret_20d": getattr(r, "ret_20d", None),
            "pct_chg": getattr(r, "pct_chg", None),
            "amplitude": getattr(r, "amplitude", None),
            "atr_14": getattr(r, "atr_14", None),
            "turnover_rate": getattr(r, "turnover_rate", None),
            "confirmation_required": "次日开盘后确认：不低开>2%、不放量下杀、站上昨收、板块不弱；否则不买",
        })
    return pd.DataFrame(rows)
