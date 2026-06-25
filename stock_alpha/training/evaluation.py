from __future__ import annotations

import pandas as pd


def time_split_dates(df: pd.DataFrame, train_end: str | None = None, valid_end: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(df["date"]).sort_values().dropna().unique()
    if len(dates) < 10:
        d1 = dates[int(len(dates) * 0.6)]
        d2 = dates[int(len(dates) * 0.8)]
        return pd.Timestamp(d1), pd.Timestamp(d2)
    if train_end and valid_end:
        return pd.Timestamp(train_end), pd.Timestamp(valid_end)
    d1 = dates[int(len(dates) * 0.6)]
    d2 = dates[int(len(dates) * 0.8)]
    return pd.Timestamp(d1), pd.Timestamp(d2)


def prediction_metrics(pred: pd.DataFrame, labels: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    df = pred.merge(labels[["code", "date", "label", "max_fwd_ret", "min_fwd_ret"]], on=["code", "date"], how="inner")
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for segment, x in {
        "all": df,
        "top10pct": df[df[score_col] >= df[score_col].quantile(0.90)],
        "buy": df[df.get("suggest_action", "") == "BUY"],
    }.items():
        if x.empty:
            continue
        rows.append({
            "segment": segment,
            "rows": len(x),
            "avg_max_fwd_ret": x["max_fwd_ret"].mean(),
            "avg_min_fwd_ret": x["min_fwd_ret"].mean(),
            "positive_label_rate": (x["label"] == 1).mean(),
            "negative_label_rate": (x["label"] == -1).mean(),
            "avg_score": x[score_col].mean(),
        })
    return pd.DataFrame(rows)
