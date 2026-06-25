from __future__ import annotations

import numpy as np
import pandas as pd


def build_intraday_features(minute: pd.DataFrame) -> pd.DataFrame:
    """V2 分钟级量能/VWAP/分时强弱特征。按 code + 日期聚合为日内特征。"""
    df = minute.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["trade_date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["pv"] = df["close"] * df["volume"]

    rows = []
    for (code, d), x in df.groupby(["code", "trade_date"]):
        x = x.sort_values("datetime")
        total_vol = x["volume"].sum()
        total_amt = x["amount"].sum() if "amount" in x else x["pv"].sum()
        vwap = total_amt / total_vol if total_vol else np.nan
        open_price = x.iloc[0]["open"]
        close_price = x.iloc[-1]["close"]
        high_idx = x["high"].idxmax()
        low_idx = x["low"].idxmin()
        early = x[x["datetime"].dt.strftime("%H:%M").between("09:30", "10:00")]
        late = x[x["datetime"].dt.strftime("%H:%M").between("14:30", "15:00")]
        vol_ma = x["volume"].rolling(20, min_periods=3).mean()
        spike_count = int((x["volume"] > vol_ma * 2.5).sum())
        above_vwap_ratio = float((x["close"] > vwap).mean()) if pd.notna(vwap) else np.nan
        rows.append({
            "code": code,
            "date": pd.to_datetime(d),
            "intraday_ret": close_price / open_price - 1 if open_price else np.nan,
            "intraday_vwap": vwap,
            "vwap_deviation": close_price / vwap - 1 if vwap else np.nan,
            "above_vwap_ratio": above_vwap_ratio,
            "open_30m_volume_ratio": early["volume"].sum() / total_vol if total_vol else np.nan,
            "late_30m_volume_ratio": late["volume"].sum() / total_vol if total_vol else np.nan,
            "minute_volume_spike_count": spike_count,
            "high_time_minutes": x.loc[high_idx, "datetime"].hour * 60 + x.loc[high_idx, "datetime"].minute,
            "low_time_minutes": x.loc[low_idx, "datetime"].hour * 60 + x.loc[low_idx, "datetime"].minute,
            "intraday_amplitude": (x["high"].max() - x["low"].min()) / open_price if open_price else np.nan,
        })
    return pd.DataFrame(rows)


V2_FEATURE_COLUMNS = [
    "intraday_ret", "vwap_deviation", "above_vwap_ratio", "open_30m_volume_ratio",
    "late_30m_volume_ratio", "minute_volume_spike_count", "high_time_minutes",
    "low_time_minutes", "intraday_amplitude",
]
