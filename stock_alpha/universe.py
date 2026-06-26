from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UniverseFilterConfig:
    lookback_short: int = 20
    lookback_long: int = 60
    min_trade_days_60: int = 45
    min_close: float = 2.0
    max_close: float = 200.0
    min_avg_amount_20: float = 100_000_000.0
    max_avg_amount_20: float = 5_000_000_000.0  # 50亿，排除超级大盘
    min_avg_amount_60: float = 50_000_000.0
    min_turnover_20: float = 1.0
    min_amplitude_20: float = 0.02
    max_amplitude_20: float = 0.12
    min_volatility_20: float = 0.015
    max_volatility_20: float = 0.08
    max_drawdown_20: float = -0.35
    exclude_st: bool = True


def compute_universe_metrics(daily: pd.DataFrame, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    rows = []
    for code, x in df.sort_values("date").groupby("code"):
        x = x.dropna(subset=["date", "close"]).sort_values("date")
        if x.empty:
            continue
        tail20 = x.tail(20)
        tail60 = x.tail(60)
        close = x["close"]
        ret = close.pct_change()
        roll_max = tail20["close"].cummax()
        dd20 = (tail20["close"] / roll_max - 1).min() if len(tail20) else np.nan
        amp20 = ((tail20["high"] - tail20["low"]) / tail20["close"].shift(1)).replace([np.inf, -np.inf], np.nan).mean() if {"high","low"} <= set(tail20.columns) else np.nan
        rows.append({
            "code": code,
            "latest_date": x["date"].iloc[-1],
            "latest_close": x["close"].iloc[-1],
            "trade_days_60": len(tail60),
            "avg_amount_20": tail20.get("amount", pd.Series(dtype=float)).mean(),
            "avg_amount_60": tail60.get("amount", pd.Series(dtype=float)).mean(),
            "avg_turnover_20": tail20.get("turnover_rate", pd.Series(dtype=float)).mean(),
            "avg_amplitude_20": amp20,
            "volatility_20": ret.tail(20).std(),
            "max_drawdown_20": dd20,
            "zero_volume_days_60": int((tail60.get("volume", pd.Series(dtype=float)).fillna(0) <= 0).sum()),
        })
    out = pd.DataFrame(rows)
    if stock_basic is not None and not stock_basic.empty:
        b = stock_basic.copy()
        b["code"] = b["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        keep = [c for c in ["code", "name", "is_st", "list_date", "industry"] if c in b.columns]
        out = out.merge(b[keep].drop_duplicates("code"), on="code", how="left")
    if "is_st" not in out.columns:
        out["is_st"] = False
    if "name" in out.columns:
        out["is_st"] = out["is_st"].fillna(False) | out["name"].astype(str).str.contains("ST", case=False, na=False)
    return out


def filter_universe(metrics: pd.DataFrame, cfg: UniverseFilterConfig) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    m = metrics.copy()
    checks = pd.Series(True, index=m.index)
    if cfg.exclude_st and "is_st" in m.columns:
        checks &= ~m["is_st"].fillna(False)
    checks &= m["trade_days_60"].fillna(0) >= cfg.min_trade_days_60
    checks &= m["latest_close"].between(cfg.min_close, cfg.max_close)
    checks &= m["avg_amount_20"].fillna(0) >= cfg.min_avg_amount_20
    checks &= m["avg_amount_20"].fillna(float('inf')) <= cfg.max_avg_amount_20
    checks &= m["avg_amount_60"].fillna(0) >= cfg.min_avg_amount_60
    checks &= m["avg_turnover_20"].fillna(0) >= cfg.min_turnover_20
    checks &= m["avg_amplitude_20"].fillna(0).between(cfg.min_amplitude_20, cfg.max_amplitude_20)
    checks &= m["volatility_20"].fillna(0).between(cfg.min_volatility_20, cfg.max_volatility_20)
    checks &= m["max_drawdown_20"].fillna(-1) > cfg.max_drawdown_20
    checks &= m["zero_volume_days_60"].fillna(999) == 0
    m["universe_pass"] = checks
    # 可交易池排序：成交额优先，其次活跃度，扣除过高波动
    m["universe_score"] = (
        np.log1p(m["avg_amount_20"].fillna(0)) * 0.35
        + m["avg_turnover_20"].fillna(0) * 0.15
        + m["avg_amplitude_20"].fillna(0) * 10 * 0.25
        - m["volatility_20"].fillna(0) * 5 * 0.25
    )
    return m.sort_values(["universe_pass", "universe_score"], ascending=[False, False])


def build_trade_universe(daily: pd.DataFrame, stock_basic: pd.DataFrame | None = None, cfg: UniverseFilterConfig | None = None, max_size: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or UniverseFilterConfig()
    metrics = compute_universe_metrics(daily, stock_basic)
    filtered = filter_universe(metrics, cfg)
    selected = filtered[filtered["universe_pass"]].copy()
    if max_size:
        selected = selected.head(max_size)
    return selected, filtered
