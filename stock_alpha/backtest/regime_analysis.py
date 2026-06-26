"""市场环境分段分析：按月度和市场状态统计策略表现。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regime_analysis(equity: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """按市场环境分段统计策略表现。

    市场状态划分（基于全市场20日收益率）：
    - 牛市：市场20日收益 > 3%
    - 震荡：-3% ~ 3%
    - 熊市：< -3%
    """
    if equity.empty or daily.empty:
        return pd.DataFrame()

    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["ret"] = eq["equity"].pct_change().fillna(0)

    # 计算全市场每日平均收益
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["pct_chg"] = pd.to_numeric(d.get("pct_chg", 0), errors="coerce")
    if d["pct_chg"].abs().median() > 1:
        d["pct_chg"] = d["pct_chg"] / 100.0

    market_ret = d.groupby("date")["pct_chg"].mean().reset_index(name="market_ret_1d")
    market_ret["market_ret_20d"] = market_ret["market_ret_1d"].rolling(20).sum()

    # 划分市场状态
    def classify_regime(ret_20d):
        if pd.isna(ret_20d):
            return "unknown"
        if ret_20d > 0.03:
            return "bull"
        elif ret_20d < -0.03:
            return "bear"
        return "sideways"

    market_ret["regime"] = market_ret["market_ret_20d"].apply(classify_regime)

    # merge 到 equity
    eq = eq.merge(market_ret[["date", "regime", "market_ret_20d"]], on="date", how="left")
    eq["regime"] = eq["regime"].fillna("unknown")

    # 按 regime 分段统计
    rows = []
    for regime, x in eq.groupby("regime"):
        if len(x) < 5:
            continue
        total_ret = x["equity"].iloc[-1] / x["equity"].iloc[0] - 1 if len(x) > 1 else 0
        max_dd = (x["equity"] / x["equity"].cummax() - 1).min()
        sharpe = np.sqrt(252) * x["ret"].mean() / (x["ret"].std() + 1e-12)
        rows.append({
            "regime": regime,
            "days": len(x),
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "avg_daily_ret": x["ret"].mean(),
        })

    return pd.DataFrame(rows).sort_values("regime")


def monthly_analysis(equity: pd.DataFrame) -> pd.DataFrame:
    """按月度统计策略表现。"""
    if equity.empty:
        return pd.DataFrame()
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["month"] = eq["date"].dt.to_period("M").astype(str)
    eq["ret"] = eq["equity"].pct_change().fillna(0)

    rows = []
    for month, x in eq.groupby("month"):
        if len(x) < 2:
            continue
        month_ret = x["equity"].iloc[-1] / x["equity"].iloc[0] - 1
        max_dd = (x["equity"] / x["equity"].cummax() - 1).min()
        rows.append({
            "month": month,
            "return": month_ret,
            "max_drawdown": max_dd,
            "trade_days": len(x),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["positive"] = out["return"] > 0
    return out
