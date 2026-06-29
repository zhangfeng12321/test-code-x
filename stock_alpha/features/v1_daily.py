from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import kdj, macd, rsi
from .cross_sectional import build_cross_sectional_features, CROSS_SECTIONAL_FEATURES
from .market_env import build_market_env_features, merge_market_env, MARKET_ENV_FEATURES
from .sector_features import build_sector_features, merge_sector_features, SECTOR_FEATURES
from .northbound_features import merge_northbound_features, NORTHBOUND_FEATURES
from .dragon_tiger_features import merge_dragon_tiger_features, DRAGON_TIGER_FEATURES
from .fundamental_features import merge_fundamental_margin_features, ALL_FUNDAMENTAL_MARGIN_FEATURES


def build_daily_features(
    daily: pd.DataFrame,
    northbound_flow: pd.DataFrame | None = None,
    northbound_stock: pd.DataFrame | None = None,
    lhb_data: pd.DataFrame | None = None,
    stock_basic: pd.DataFrame | None = None,
    financial_data: pd.DataFrame | None = None,
    margin_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """V1 日线特征。输入至少包含 date/open/high/low/close/volume/amount。

    可选外部数据:
        northbound_flow: 北向资金每日净流入数据
        northbound_stock: 个股北向持股数据
        lhb_data: 龙虎榜数据
        stock_basic: 股票基本信息（含 industry 列），用于精确行业分类
        financial_data: 财务指标数据（ROE、营收增速等）
        margin_data: 融资融券数据
    """
    df = daily.copy().sort_values(["code", "date"] if "code" in daily.columns else ["date"])
    g = df.groupby("code", group_keys=False) if "code" in df.columns else [(None, df)]

    parts = []
    iterator = g if hasattr(g, "__iter__") and not isinstance(g, list) else g
    for _, x in iterator:
        x = x.copy().sort_values("date")
        for n in [1, 3, 5, 10, 20]:
            x[f"ret_{n}d"] = x["close"].pct_change(n)
            x[f"ma{n}"] = x["close"].rolling(n).mean()
            x[f"close_ma{n}_ratio"] = x["close"] / x[f"ma{n}"] - 1
        x["ma5_slope"] = x["ma5"].pct_change(3)
        x["ma10_slope"] = x["ma10"].pct_change(3)
        x["volume_ratio_5"] = x["volume"] / x["volume"].rolling(5).mean()
        x["volume_ratio_20"] = x["volume"] / x["volume"].rolling(20).mean()
        x["amount_ratio_20"] = x["amount"] / x["amount"].rolling(20).mean()
        x["amplitude"] = (x["high"] - x["low"]) / x["close"].shift(1)
        x["upper_shadow"] = (x["high"] - x[["open", "close"]].max(axis=1)) / x["close"].replace(0, np.nan)
        x["lower_shadow"] = (x[["open", "close"]].min(axis=1) - x["low"]) / x["close"].replace(0, np.nan)
        x["rsi_6"] = rsi(x["close"], 6)
        x["rsi_14"] = rsi(x["close"], 14)
        x = pd.concat([x, macd(x["close"]), kdj(x)], axis=1)
        x["atr_14"] = np.maximum.reduce([
            (x["high"] - x["low"]).to_numpy(),
            (x["high"] - x["close"].shift(1)).abs().to_numpy(),
            (x["low"] - x["close"].shift(1)).abs().to_numpy(),
        ])
        x["atr_14"] = pd.Series(x["atr_14"], index=x.index).rolling(14).mean() / x["close"]
        if "turnover_rate" not in x.columns:
            x["turnover_rate"] = np.nan
        # --- 量价结构特征 ---
        prev_close = x["close"].shift(1)
        x["gap_up"] = ((x["open"] / prev_close - 1).clip(lower=0))
        x["gap_down"] = ((x["open"] / prev_close - 1).clip(upper=0).abs())
        # 连涨/连跌天数
        up_flag = (x["close"] > prev_close).astype(int)
        down_flag = (x["close"] < prev_close).astype(int)
        consec_up = up_flag.copy()
        consec_down = down_flag.copy()
        for i in range(1, len(consec_up)):
            if up_flag.iloc[i] == 1:
                consec_up.iloc[i] = consec_up.iloc[i - 1] + 1
            if down_flag.iloc[i] == 1:
                consec_down.iloc[i] = consec_down.iloc[i - 1] + 1
        x["consecutive_up"] = consec_up
        x["consecutive_down"] = consec_down
        # 量价背离: 价涨量缩=1, 价跌量增=-1, 其他=0
        price_up = x["close"] > prev_close
        vol_shrink = x["volume"] < x["volume"].shift(1)
        vol_expand = x["volume"] > x["volume"].shift(1)
        price_down = x["close"] < prev_close
        x["vol_price_diverge"] = np.where(price_up & vol_shrink, 1, np.where(price_down & vol_expand, -1, 0))
        # 20日新高/新低
        x["new_high_20d"] = (x["close"] >= x["close"].rolling(20).max()).astype(int)
        x["new_low_20d"] = (x["close"] <= x["close"].rolling(20).min()).astype(int)
        # 收盘价在K线中的位置
        hl_range = (x["high"] - x["low"]).replace(0, np.nan)
        x["close_position"] = (x["close"] - x["low"]) / hl_range
        # ===== Path3 新增：替代缺失特征的强化量价衍生特征 =====
        # 买压指数：开收价差 / 振幅，代理开盘后主力撹盘力度
        x["buying_pressure"] = (x["close"] - x["open"]) / hl_range.replace(np.nan, 1)
        # 主力资金流代理：成交额 × 买压指数（加权驱动力）
        x["money_flow_proxy"] = x["amount"] * x["buying_pressure"]
        x["mf_proxy_ma5"] = x["money_flow_proxy"].rolling(5).mean()
        x["mf_proxy_ma10"] = x["money_flow_proxy"].rolling(10).mean()
        # 5日资金流加速度：近期就是否加速流入
        x["mf_acceleration"] = x["mf_proxy_ma5"] / x["mf_proxy_ma10"].replace(0, np.nan) - 1
        # 量能异常：当日成交量占 20日均量比例，模拟机构撹盘特征
        vol_20m = x["volume"].rolling(20).mean().replace(0, np.nan)
        x["vol_surge_ratio"] = x["volume"] / vol_20m
        # 成交量的标准差变化率：量能波动性
        x["vol_cv_10"] = x["volume"].rolling(10).std() / vol_20m
        # 短期价格相对强弱：5日相对 20日
        ma5 = x["close"].rolling(5).mean()
        ma20 = x["close"].rolling(20).mean()
        x["short_long_ratio"] = ma5 / ma20.replace(0, np.nan) - 1
        # 高低点破位强度：收盘价超过 20日最高价的程度
        high_20 = x["high"].rolling(20).max().replace(0, np.nan)
        low_20 = x["low"].rolling(20).min().replace(0, np.nan)
        x["breakout_pct"] = (x["close"] - high_20) / high_20  # 高于0表示创新高
        x["range_position_20d"] = (x["close"] - low_20) / (high_20 - low_20).replace(0, np.nan)  # 20日区间位置
        # 抓幅与放量共振：价涨且量大 = 强强上涨
        price_gain = (x["close"] - x["close"].shift(3)).clip(lower=0)
        x["gain_with_volume"] = price_gain * x["volume_ratio_5"]
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    # --- 横截面特征：当日排名分位 ---
    out = build_cross_sectional_features(out)

    # --- 市场环境特征：全市场聚合指标 ---
    market_env = build_market_env_features(daily)
    out = merge_market_env(out, market_env)

    # --- 板块强度特征 ---
    out = merge_sector_features(out, daily, stock_basic=stock_basic)

    # --- 北向资金特征 ---
    if northbound_flow is not None and not northbound_flow.empty:
        # 构建全市场每日成交额用于计算北向占比
        daily_amount = daily.groupby("date")["amount"].sum().reset_index()
        daily_amount.columns = ["date", "total_amount"]
        out = merge_northbound_features(out, northbound_flow, northbound_stock, daily_amount)
    else:
        for col in NORTHBOUND_FEATURES:
            out[col] = np.nan

    # --- 龙虎榜特征 ---
    if lhb_data is not None and not lhb_data.empty:
        out = merge_dragon_tiger_features(out, lhb_data, daily)
    else:
        for col in DRAGON_TIGER_FEATURES:
            out[col] = np.nan

    # --- 基本面 + 融资融券特征 ---
    out = merge_fundamental_margin_features(out, daily, financial_data, margin_data)

    return out


# --- 量价结构特征 ---
STRUCTURE_FEATURES = [
    "gap_up", "gap_down", "consecutive_up", "consecutive_down",
    "vol_price_diverge", "new_high_20d", "new_low_20d", "close_position",
]

# --- Path3 新增 OHLCV 衍生特征 ---
OHLCV_DERIVED_FEATURES = [
    "buying_pressure", "money_flow_proxy", "mf_acceleration",
    "vol_surge_ratio", "vol_cv_10", "short_long_ratio",
    "breakout_pct", "range_position_20d", "gain_with_volume",
]

V1_FEATURE_COLUMNS = [
    # 原有 V1 时序特征 (26个)
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
    "ma5_slope", "ma10_slope", "volume_ratio_5", "volume_ratio_20", "amount_ratio_20",
    "amplitude", "upper_shadow", "lower_shadow", "rsi_6", "rsi_14",
    "macd_dif", "macd_dea", "macd_hist", "kdj_k", "kdj_d", "kdj_j", "atr_14", "turnover_rate",
    # 量价结构特征 (8个)
    *STRUCTURE_FEATURES,
    # Path3 OHLCV 衍生特征 (9个)
    *OHLCV_DERIVED_FEATURES,
    # 横截面排名特征 (10个)
    *CROSS_SECTIONAL_FEATURES,
    # 市场环境特征 (10个)
    *MARKET_ENV_FEATURES,
    # 板块强度特征 (6个)
    *SECTOR_FEATURES,
    # 北向资金特征 (9个)
    *NORTHBOUND_FEATURES,
    # 龙虎榜特征 (9个)
    *DRAGON_TIGER_FEATURES,
    # 基本面 + 融资融券特征 (11个)
    *ALL_FUNDAMENTAL_MARGIN_FEATURES,
]
