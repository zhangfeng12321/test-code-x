from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IntradayConfirmConfig:
    max_open_gap_down: float = -0.02
    cancel_drawdown_from_prev_close: float = -0.02
    cancel_drawdown_from_open: float = -0.03
    confirm_above_prev_close: float = 0.0
    max_first30_volume_ratio: float = 3.0
    min_first30_volume_ratio: float = 0.8


def _norm_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    return out


def confirm_watchlist_intraday(
    watchlist: pd.DataFrame,
    realtime: pd.DataFrame,
    cfg: IntradayConfirmConfig | None = None,
) -> pd.DataFrame:
    """Confirm WATCH candidates with next-session open/early-session data.

    realtime expected columns per code:
    - open, price/current, high, low, prev_close
    - first30_volume_ratio optional: first 30min volume / recent avg first 30min volume

    Output action:
    - CONFIRM_BUY: price recovered above prev close and no early kill signal.
    - WAIT: no hard cancel, but confirmation not enough.
    - CANCEL: low-open, early breakdown, or suspicious blow-off volume.
    """
    cfg = cfg or IntradayConfirmConfig()
    if watchlist.empty or realtime.empty:
        return pd.DataFrame()
    w = _norm_code(watchlist)
    r = _norm_code(realtime)
    price_col = "price" if "price" in r.columns else "current" if "current" in r.columns else "close"
    needed = ["code", "open", price_col, "high", "low", "prev_close"]
    missing = [c for c in needed if c not in r.columns]
    if missing:
        raise ValueError(f"realtime missing columns: {missing}")
    cols = ["code", "open", price_col, "high", "low", "prev_close"] + (["first30_volume_ratio"] if "first30_volume_ratio" in r.columns else [])
    x = w.merge(r[cols], on="code", how="left")
    for c in ["open", price_col, "high", "low", "prev_close", "first30_volume_ratio"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    reasons = []
    actions = []
    for row in x.itertuples(index=False):
        rs = []
        open_p = getattr(row, "open")
        cur = getattr(row, price_col)
        low = getattr(row, "low")
        prev = getattr(row, "prev_close")
        vol_ratio = getattr(row, "first30_volume_ratio", None)
        if pd.isna(open_p) or pd.isna(cur) or pd.isna(low) or pd.isna(prev) or prev <= 0:
            actions.append("WAIT")
            reasons.append("行情不完整")
            continue
        open_gap = open_p / prev - 1
        low_vs_prev = low / prev - 1
        low_vs_open = low / open_p - 1 if open_p > 0 else 0
        cur_vs_prev = cur / prev - 1
        if open_gap <= cfg.max_open_gap_down:
            rs.append("低开超过阈值")
        if low_vs_prev <= cfg.cancel_drawdown_from_prev_close:
            rs.append("早盘跌破昨收过深")
        if low_vs_open <= cfg.cancel_drawdown_from_open:
            rs.append("开盘后放量下杀/破位")
        if vol_ratio is not None and pd.notna(vol_ratio) and vol_ratio > cfg.max_first30_volume_ratio and cur < open_p:
            rs.append("早盘异常放量但价格走弱")
        if rs:
            actions.append("CANCEL")
            reasons.append("、".join(rs))
        elif cur_vs_prev >= cfg.confirm_above_prev_close and (vol_ratio is None or pd.isna(vol_ratio) or vol_ratio >= cfg.min_first30_volume_ratio):
            actions.append("CONFIRM_BUY")
            reasons.append("站上昨收且早盘无破位")
        else:
            actions.append("WAIT")
            reasons.append("未触发取消，但强度不足")
    x["confirm_action"] = actions
    x["confirm_reason"] = reasons
    x["current_price"] = x[price_col]
    return x.drop(columns=[price_col]) if price_col != "current_price" else x
