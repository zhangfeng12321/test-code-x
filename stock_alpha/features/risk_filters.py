from __future__ import annotations

import pandas as pd


def infer_limit_pct(code: str, name: str | None = None, is_st: bool | None = None) -> float:
    """推断 A 股涨跌幅限制。

    - ST: 5%
    - 创业板/科创板: 20%
    - 北交所: 30%
    - 主板: 10%
    """
    code = str(code).zfill(6)[-6:]
    if is_st or (name and "ST" in str(name).upper()):
        return 0.05
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def enrich_trade_constraints(daily: pd.DataFrame, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
    """补充交易约束字段：pre_close/limit_pct/is_limit_up/is_limit_down/is_suspended/is_tradeable。"""
    df = daily.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    if stock_basic is not None and not stock_basic.empty:
        basic = stock_basic.copy()
        basic["code"] = basic["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        keep = [c for c in ["code", "name", "is_st", "list_date"] if c in basic.columns]
        df = df.merge(basic[keep].drop_duplicates("code"), on="code", how="left")
    if "name" not in df.columns:
        df["name"] = ""
    if "is_st" not in df.columns:
        df["is_st"] = df["name"].astype(str).str.contains("ST", case=False, na=False)
    df["limit_pct"] = [infer_limit_pct(c, n, s) for c, n, s in zip(df["code"], df["name"], df["is_st"])]
    df = df.sort_values(["code", "date"])
    df["pre_close"] = df.groupby("code")["close"].shift(1)
    df["is_suspended"] = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1) | (df.get("volume", 0).fillna(0) <= 0)
    up_ret = df["close"] / df["pre_close"] - 1
    df["is_limit_up"] = up_ret >= (df["limit_pct"] - 0.002)
    df["is_limit_down"] = up_ret <= (-df["limit_pct"] + 0.002)
    df["is_tradeable"] = ~df["is_suspended"] & ~df["is_st"].fillna(False)
    return df


def filter_tradeable_predictions(predictions: pd.DataFrame, daily: pd.DataFrame, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
    """过滤不可交易股票：ST/停牌/当日涨停风险等。"""
    constraints = enrich_trade_constraints(daily, stock_basic)
    cols = ["code", "date", "is_tradeable", "is_limit_up", "is_limit_down", "limit_pct"]
    pred = predictions.copy()
    pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"])
    out = pred.merge(constraints[cols], on=["code", "date"], how="left")
    out["is_tradeable"] = out["is_tradeable"].fillna(True)
    out["is_limit_up"] = out["is_limit_up"].fillna(False)
    out["is_limit_down"] = out["is_limit_down"].fillna(False)
    out["risk_blocked"] = ~out["is_tradeable"]
    out.loc[out["risk_blocked"], "suggest_action"] = "BLOCKED"
    return out
