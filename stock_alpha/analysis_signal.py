from __future__ import annotations

import pandas as pd


def signal_stability(predictions: pd.DataFrame, score_col: str | None = None, top_n: int = 20) -> pd.DataFrame:
    """分析信号稳定性：入选次数、连续入选、平均排名、平均分、最近一次入选。"""
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    score_col = score_col or ("final_score_v2" if "final_score_v2" in df.columns else "final_score")
    rows = []
    selected_parts = []
    for d, x in df.groupby("date"):
        y = x.sort_values(score_col, ascending=False).head(top_n).copy()
        y["rank"] = range(1, len(y) + 1)
        selected_parts.append(y[["date", "code", score_col, "rank"]])
    if not selected_parts:
        return pd.DataFrame()
    sel = pd.concat(selected_parts, ignore_index=True).sort_values(["code", "date"])
    for code, x in sel.groupby("code"):
        dates = sorted(x["date"].unique())
        max_streak = cur = 0
        prev = None
        for d in dates:
            if prev is not None and (pd.Timestamp(d) - pd.Timestamp(prev)).days <= 4:
                cur += 1
            else:
                cur = 1
            max_streak = max(max_streak, cur)
            prev = d
        rows.append({
            "code": code,
            "selected_count": len(x),
            "max_streak": max_streak,
            "avg_rank": x["rank"].mean(),
            "avg_score": x[score_col].mean(),
            "last_selected": max(dates),
        })
    return pd.DataFrame(rows).sort_values(["selected_count", "avg_score"], ascending=False)


def turnover_by_date(predictions: pd.DataFrame, top_n: int = 20, score_col: str | None = None) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    score_col = score_col or ("final_score_v2" if "final_score_v2" in df.columns else "final_score")
    prev = set()
    rows = []
    for d, x in df.groupby("date"):
        cur = set(x.sort_values(score_col, ascending=False).head(top_n)["code"])
        if prev:
            changed = len(cur.symmetric_difference(prev)) / max(len(cur.union(prev)), 1)
        else:
            changed = 0.0
        rows.append({"date": d, "top_n": top_n, "turnover": changed, "selected": len(cur)})
        prev = cur
    return pd.DataFrame(rows)
