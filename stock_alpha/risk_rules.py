from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskRuleConfig:
    """Hard risk gates for short-term A-share candidates.

    These rules are intentionally conservative for pre-trade plans:
    - avoid forcing weak topN names into orders;
    - block weak-trend rebound limit-up / large-positive days;
    - reject signals whose down probability is not clearly below up probability.
    """

    min_score: float = 0.45
    max_down_probability: float | None = 0.40
    max_risk_score: float | None = 0.25
    require_up_gt_down: bool = True
    weak_rebound_ret_20d: float = -0.10
    weak_rebound_pct_chg: float = 0.07
    weak_rebound_amplitude: float = 0.10
    high_atr_14: float = 0.08
    max_single_day_loss: float | None = None


def normalize_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)


def daily_risk_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build per-date daily context used by risk gates and backtests."""
    if daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["code"] = normalize_code(d["code"])
    d["date"] = pd.to_datetime(d["date"], format="mixed")
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_chg"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["code", "date"])
    parts = []
    for _, x in d.groupby("code", sort=False):
        x = x.copy().sort_values("date")
        x["pre_close"] = x["close"].shift(1)
        x["ret_20d"] = x["close"].pct_change(20)
        if "pct_chg" not in x.columns:
            x["pct_chg"] = x["close"].pct_change()
        else:
            # AkShare pct_chg is percentage points; normalize to decimal if needed.
            x["pct_chg"] = x["pct_chg"].where(x["pct_chg"].abs() <= 1, x["pct_chg"] / 100.0)
        x["amplitude"] = (x["high"] - x["low"]) / x["pre_close"].replace(0, pd.NA)
        tr = pd.concat([
            x["high"] - x["low"],
            (x["high"] - x["pre_close"]).abs(),
            (x["low"] - x["pre_close"]).abs(),
        ], axis=1).max(axis=1)
        x["atr_14"] = tr.rolling(14).mean() / x["close"].replace(0, pd.NA)
        x["avg_amount_20"] = x["amount"].rolling(20).mean() if "amount" in x.columns else pd.NA
        parts.append(x)
    cols = [
        "code", "date", "open", "high", "low", "close", "pre_close", "pct_chg",
        "ret_20d", "amplitude", "atr_14", "turnover_rate", "avg_amount_20",
    ]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out[[c for c in cols if c in out.columns]]


def latest_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build latest daily context used by risk gates and explanations."""
    out = daily_risk_features(daily)
    if out.empty:
        return out
    return out.sort_values("date").groupby("code", as_index=False).tail(1)


def attach_latest_context(predictions: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    x = predictions.copy()
    if x.empty:
        return x
    x["code"] = normalize_code(x["code"])
    x["date"] = pd.to_datetime(x["date"], format="mixed")
    if daily is None or daily.empty:
        return x
    ctx = latest_daily_features(daily)
    if ctx.empty:
        return x
    ctx = ctx.rename(columns={"date": "daily_date"})
    return x.merge(ctx, on="code", how="left", suffixes=("", "_daily"))


def apply_hard_risk_filters(
    candidates: pd.DataFrame,
    cfg: RiskRuleConfig | None = None,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Annotate candidates with risk_blocked/risk_reasons and return all rows."""
    cfg = cfg or RiskRuleConfig()
    if candidates.empty:
        return candidates.copy()
    x = candidates.copy()
    score_col = score_col or ("final_score_v2" if "final_score_v2" in x.columns else "final_score")
    reasons: list[list[str]] = [[] for _ in range(len(x))]

    def add(mask: pd.Series, reason: str) -> None:
        m = mask.fillna(False).to_numpy()
        for i, flag in enumerate(m):
            if flag:
                reasons[i].append(reason)

    if score_col in x.columns:
        add(pd.to_numeric(x[score_col], errors="coerce") < cfg.min_score, "综合分低于买入门槛")
    if cfg.max_down_probability is not None and "down_probability" in x.columns:
        add(pd.to_numeric(x["down_probability"], errors="coerce") > cfg.max_down_probability, "下跌概率过高")
    if cfg.max_risk_score is not None and "risk_score" in x.columns:
        add(pd.to_numeric(x["risk_score"], errors="coerce") > cfg.max_risk_score, "波动风险过高")
    if cfg.require_up_gt_down and {"up_probability", "down_probability"} <= set(x.columns):
        add(pd.to_numeric(x["up_probability"], errors="coerce") <= pd.to_numeric(x["down_probability"], errors="coerce"), "上涨概率不高于下跌概率")

    ret20 = pd.to_numeric(x.get("ret_20d"), errors="coerce") if "ret_20d" in x.columns else pd.Series(pd.NA, index=x.index)
    pct = pd.to_numeric(x.get("pct_chg"), errors="coerce") if "pct_chg" in x.columns else pd.Series(pd.NA, index=x.index)
    amp = pd.to_numeric(x.get("amplitude"), errors="coerce") if "amplitude" in x.columns else pd.Series(pd.NA, index=x.index)
    atr = pd.to_numeric(x.get("atr_14"), errors="coerce") if "atr_14" in x.columns else pd.Series(pd.NA, index=x.index)
    add((ret20 <= cfg.weak_rebound_ret_20d) & (pct >= cfg.weak_rebound_pct_chg), "弱势反抽/涨停后不追")
    add((ret20 <= cfg.weak_rebound_ret_20d) & (amp >= cfg.weak_rebound_amplitude), "弱趋势高振幅")
    add(atr >= cfg.high_atr_14, "ATR过高")
    if cfg.max_single_day_loss is not None:
        add(pct <= -abs(cfg.max_single_day_loss), "单日破位下跌")

    x["risk_reasons"] = ["、".join(r) for r in reasons]
    x["risk_blocked"] = x["risk_reasons"].astype(str).ne("")
    return x


def filter_trade_candidates(
    predictions: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    cfg: RiskRuleConfig | None = None,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Return candidates that pass all hard gates."""
    x = attach_latest_context(predictions, daily)
    x = apply_hard_risk_filters(x, cfg=cfg, score_col=score_col)
    return x[~x["risk_blocked"].fillna(False)].copy()
