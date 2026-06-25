from __future__ import annotations

import pandas as pd


def build_holdings_snapshots(equity: pd.DataFrame, trades: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """从交易流水还原每日持仓快照。"""
    if equity.empty or trades.empty:
        return pd.DataFrame(columns=["date", "code", "shares", "cost_price", "market_price", "market_value"])
    eq = equity.copy(); eq["date"] = pd.to_datetime(eq["date"])
    tr = trades.copy(); tr["date"] = pd.to_datetime(tr["date"])
    price_map = {}
    if daily is not None and not daily.empty:
        d = daily.copy(); d["date"] = pd.to_datetime(d["date"]); d["code"] = d["code"].astype(str).str.zfill(6)
        price_map = {(r.code, r.date): r.close for r in d.itertuples(index=False) if hasattr(r, "close")}
    positions: dict[str, dict] = {}
    rows = []
    for date in sorted(eq["date"].unique()):
        todays = tr[tr["date"] == date]
        for r in todays.itertuples(index=False):
            code = str(r.code).zfill(6)
            if r.side == "BUY":
                old = positions.get(code, {"shares": 0, "cost": 0.0})
                new_shares = old["shares"] + r.shares
                new_cost = (old["shares"] * old["cost"] + r.shares * r.price) / new_shares if new_shares else 0
                positions[code] = {"shares": new_shares, "cost": new_cost}
            elif r.side == "SELL" and code in positions:
                positions[code]["shares"] -= r.shares
                if positions[code]["shares"] <= 0:
                    positions.pop(code, None)
        for code, pos in positions.items():
            mp = price_map.get((code, pd.Timestamp(date)), pos["cost"])
            rows.append({"date": pd.Timestamp(date), "code": code, "shares": pos["shares"], "cost_price": pos["cost"], "market_price": mp, "market_value": pos["shares"] * mp})
    return pd.DataFrame(rows)
