from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from stock_alpha.risk_rules import RiskRuleConfig, attach_latest_context, apply_hard_risk_filters


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
    require_up_gt_down: bool = True
    use_unadjusted_ref_price: bool = True


def _fetch_unadjusted_last_close(codes: list[str], fallback: pd.DataFrame) -> pd.DataFrame:
    """Best-effort latest unadjusted close for executable order prices.

    Model features may use adjusted historical prices, but order sizing/stop lines must use
    real trade prices. If AkShare/network is unavailable, fall back to cached prices and mark it.
    """
    base = fallback.copy()
    if base.empty or not codes:
        return base
    base["price_source"] = "cached"
    try:
        from stock_alpha.data.providers.akshare_provider import AkShareProvider
        provider = AkShareProvider()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")
        rows = []
        for code in codes:
            df = provider.get_daily_bars(code, start, end, adjust="")
            if df.empty:
                continue
            df = df.sort_values("date").tail(1)
            close = pd.to_numeric(df["close"].iloc[0], errors="coerce")
            if pd.notna(close) and close > 0:
                rows.append({"code": code, "ref_price": float(close), "price_source": "unadjusted_akshare"})
        if rows:
            live = pd.DataFrame(rows)
            base = base.drop(columns=["price_source"], errors="ignore").merge(live, on="code", how="left", suffixes=("", "_live"))
            base["ref_price"] = base["ref_price_live"].combine_first(base["ref_price"])
            base["price_source"] = base["price_source"].fillna("cached")
            base = base.drop(columns=["ref_price_live"], errors="ignore")
    except Exception:
        pass
    return base


def generate_next_day_orders(predictions: pd.DataFrame, daily: pd.DataFrame, cfg: OrderPlanConfig, universe_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if predictions.empty or daily.empty:
        return pd.DataFrame()
    pred = attach_latest_context(predictions, daily)
    pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], format="mixed")
    score_col = "final_score_v2" if "final_score_v2" in pred.columns else "final_score"
    latest_date = pred["date"].max()
    x = pred[pred["date"] == latest_date].copy()
    risk_cfg = RiskRuleConfig(
        min_score=cfg.min_score,
        max_down_probability=cfg.max_down_probability,
        max_risk_score=cfg.max_risk_score,
        require_up_gt_down=cfg.require_up_gt_down,
    )
    x = apply_hard_risk_filters(x, risk_cfg, score_col=score_col)
    x = x[~x["risk_blocked"].fillna(False)]
    if universe_metrics is not None and not universe_metrics.empty and cfg.min_avg_amount_20 is not None:
        um = universe_metrics.copy()
        um["code"] = um["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        um = um[["code", "avg_amount_20"]].rename(columns={"avg_amount_20": "avg_amount_20_universe"})
        x = x.merge(um, on="code", how="left")
        amount_col = x["avg_amount_20_universe"].combine_first(x["avg_amount_20"]) if "avg_amount_20" in x.columns else x["avg_amount_20_universe"]
        x = x[amount_col.fillna(0) >= cfg.min_avg_amount_20]
    x = x.sort_values(score_col, ascending=False)
    # topn 只负责排序截断，硬门槛已在 apply_hard_risk_filters 中执行；不满足时允许空仓。
    if cfg.selection_mode == "quantile" and not x.empty:
        q = x[score_col].quantile(cfg.score_quantile)
        x = x[x[score_col] >= q]
    x = x.head(cfg.top_n)
    d = daily.copy(); d["code"] = d["code"].astype(str).str.zfill(6); d["date"] = pd.to_datetime(d["date"], format="mixed")
    last_close = d.sort_values("date").groupby("code", as_index=False).tail(1)[["code", "close"]].rename(columns={"close":"ref_price"})
    if cfg.use_unadjusted_ref_price:
        selected_codes = x["code"].astype(str).str.zfill(6).drop_duplicates().tolist()
        last_close = _fetch_unadjusted_last_close(selected_codes, last_close)
    else:
        last_close["price_source"] = "cached"
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
            "price_source": getattr(r, "price_source", "cached"),
            "note": "仅通过硬风控初筛；次日低开/放量下杀/跌破止损须取消或止损",
        })
    return pd.DataFrame(rows)
