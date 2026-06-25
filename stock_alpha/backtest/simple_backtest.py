from __future__ import annotations

import pandas as pd


def backtest_topn(predictions: pd.DataFrame, labels: pd.DataFrame, top_n: int = 5, fee: float = 0.0015) -> pd.DataFrame:
    """基于 max_fwd_ret/min_fwd_ret 的粗略 TopN 验证，不替代真实撮合回测。"""
    df = predictions.merge(labels[["code", "date", "max_fwd_ret", "min_fwd_ret"]], on=["code", "date"], how="inner")
    score_col = "final_score_v2" if "final_score_v2" in df.columns else "final_score"
    rows = []
    for d, x in df.groupby("date"):
        picks = x.sort_values(score_col, ascending=False).head(top_n)
        ret = picks["max_fwd_ret"].clip(upper=0.05).fillna(0).mean() - fee if len(picks) else 0
        rows.append({"date": d, "portfolio_ret": ret, "count": len(picks)})
    out = pd.DataFrame(rows).sort_values("date")
    out["equity"] = (1 + out["portfolio_ret"]).cumprod()
    return out
