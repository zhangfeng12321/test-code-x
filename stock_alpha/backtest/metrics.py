from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_backtest_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if equity.empty:
        return {"summary": pd.DataFrame(), "monthly": pd.DataFrame(), "yearly": pd.DataFrame(), "trade_stats": pd.DataFrame()}
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["ret"] = eq["equity"].pct_change().fillna(0)
    eq["month"] = eq["date"].dt.to_period("M").astype(str)
    eq["year"] = eq["date"].dt.year.astype(str)

    def period_ret(x):
        return x.iloc[-1] / x.iloc[0] - 1 if len(x) else 0

    monthly = eq.groupby("month")["equity"].apply(period_ret).reset_index(name="return")
    yearly = eq.groupby("year")["equity"].apply(period_ret).reset_index(name="return")

    trade_stats = pd.DataFrame([{"round_trips": 0, "win_rate": None, "avg_pnl": None, "profit_loss_ratio": None, "max_consecutive_losses": 0}])
    if not trades.empty:
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        round_pnls = []
        for code, x in t.sort_values("date").groupby("code"):
            buys = []
            for r in x.itertuples(index=False):
                if r.side == "BUY":
                    buys.append(r)
                elif r.side == "SELL" and buys:
                    b = buys.pop(0)
                    pnl = (r.price - b.price) * min(r.shares, b.shares) - getattr(r, "fee", 0) - getattr(b, "fee", 0)
                    round_pnls.append(pnl)
        if round_pnls:
            arr = np.array(round_pnls)
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            max_consec = 0; cur = 0
            for v in arr:
                if v < 0:
                    cur += 1; max_consec = max(max_consec, cur)
                else:
                    cur = 0
            trade_stats = pd.DataFrame([{
                "round_trips": len(arr),
                "win_rate": float((arr > 0).mean()),
                "avg_pnl": float(arr.mean()),
                "profit_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
                "max_consecutive_losses": max_consec,
            }])
    return {"monthly": monthly, "yearly": yearly, "trade_stats": trade_stats}
