"""基本面 + 融资融券特征模块。

基本面特征（季报数据，前向填充到日线）：
- roe: 最新净资产收益率
- net_profit_growth: 净利润同比增速
- revenue_growth: 营收同比增速
- eps: 每股收益
- bps: 每股净资产
- debt_ratio: 资产负债率
- pe_ttm_rank: PE(TTM) 在全市场的分位
- pb_rank: PB 在全市场的分位

融资融券特征（市场级，每日）：
- margin_balance_chg_5d: 融资余额5日变化率
- margin_buy_ratio: 融资买入额占全市场成交比
- short_balance_chg_5d: 融券余额5日变化率
- margin_sentiment: 融资融券情绪（融资买入额/融券余额趋势差）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ========== 基本面特征 ==========

FUNDAMENTAL_FEATURES = [
    "roe", "net_profit_growth", "revenue_growth",
    "eps", "debt_ratio",
    "pe_ttm_rank", "pb_rank",
]

MARGIN_FEATURES = [
    "margin_balance_chg_5d", "margin_buy_ratio",
    "short_balance_chg_5d", "margin_sentiment",
]

ALL_FUNDAMENTAL_MARGIN_FEATURES = FUNDAMENTAL_FEATURES + MARGIN_FEATURES


def build_fundamental_features(
    daily: pd.DataFrame,
    financial_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将季报基本面数据前向填充并合并到日线特征表。

    Args:
        daily: 日线数据（需含 code, date, close, amount）
        financial_data: 财务指标数据（report_date, code, roe, net_profit_growth 等）

    Returns:
        包含基本面特征列的 DataFrame（与 daily 同行数）
    """
    df = daily[["code", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if financial_data is None or financial_data.empty:
        for col in FUNDAMENTAL_FEATURES:
            df[col] = np.nan
        return df

    fin = financial_data.copy()
    fin["report_date"] = pd.to_datetime(fin["report_date"], errors="coerce")
    fin["code"] = fin["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)

    # 对每只股票，用最新可用季报（前向填充）
    fin = fin.sort_values(["code", "report_date"]).drop_duplicates(
        subset=["code", "report_date"], keep="last"
    )

    # merge_asof: 对每个 (code, date) 找到最近一次 <= date 的财报
    df = df.sort_values(["code", "date"])
    fin = fin.sort_values(["code", "report_date"])

    # 按股票分组做 asof merge
    result_parts = []
    for code, code_daily in df.groupby("code"):
        code_fin = fin[fin["code"] == code].copy()
        if code_fin.empty:
            part = code_daily.copy()
            for col in ["roe", "net_profit_growth", "revenue_growth", "eps", "bps", "debt_ratio"]:
                part[col] = np.nan
        else:
            fin_subset = code_fin[["report_date", "roe", "net_profit_growth", "revenue_growth", "eps", "bps", "debt_ratio"]].copy()
            fin_subset["_report_date"] = fin_subset["report_date"]
            part = pd.merge_asof(
                code_daily.sort_values("date"),
                fin_subset.rename(columns={"report_date": "date"}).sort_values("date"),
                on="date",
                direction="backward",
            )
        result_parts.append(part)

    result = pd.concat(result_parts, ignore_index=True)

    # PE_TTM 和 PB 的全市场分位排名
    if "close" in daily.columns and "eps" in result.columns:
        merged = result.merge(daily[["code", "date", "close"]].drop_duplicates(), on=["code", "date"], how="left")
        # PE_TTM 年化：根据报告期月份推断年化倍数（YTD EPS → TTM EPS）
        # Q1报告(3月) → ×4, Q2中报(6月) → ×2, Q3三季报(9月) → ×4/3, 年报(12月) → ×1
        if "_report_date" in merged.columns:
            report_month = pd.to_datetime(merged["_report_date"], errors="coerce").dt.month
            ann_factor = report_month.apply(lambda m: 4 if pd.isna(m) or m <= 3 else (2 if m <= 6 else (4 / 3 if m <= 9 else 1)))
        else:
            ann_factor = 4
        merged["pe_ttm"] = merged["close"] / (merged["eps"].replace(0, np.nan) * ann_factor)
        merged["pb"] = merged["close"] / merged["bps"].replace(0, np.nan)
        # 横截面分位排名
        merged["pe_ttm_rank"] = merged.groupby("date")["pe_ttm"].rank(pct=True)
        merged["pb_rank"] = merged.groupby("date")["pb"].rank(pct=True)
        result["pe_ttm_rank"] = merged["pe_ttm_rank"].values
        result["pb_rank"] = merged["pb_rank"].values
    else:
        result["pe_ttm_rank"] = np.nan
        result["pb_rank"] = np.nan

    # 确保所有特征列存在
    for col in FUNDAMENTAL_FEATURES:
        if col not in result.columns:
            result[col] = np.nan

    return result[["code", "date"] + FUNDAMENTAL_FEATURES]


# ========== 融资融券特征 ==========

def build_margin_features(
    daily: pd.DataFrame,
    margin_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建融资融券市场级特征，合并到日线。

    Args:
        daily: 日线数据（需含 date, amount）
        margin_data: 融资融券数据（date, margin_balance, margin_buy, short_balance）

    Returns:
        包含融资融券特征列的 DataFrame
    """
    df = daily[["code", "date"]].copy() if "code" in daily.columns else daily[["date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if margin_data is None or margin_data.empty:
        for col in MARGIN_FEATURES:
            df[col] = np.nan
        return df

    mg = margin_data.copy()
    mg["date"] = pd.to_datetime(mg["date"], errors="coerce")
    mg = mg.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # 融资余额5日变化率
    if "margin_balance" in mg.columns:
        mg["margin_balance_chg_5d"] = mg["margin_balance"].pct_change(5)
    else:
        mg["margin_balance_chg_5d"] = np.nan

    # 融资买入额占比（需要全市场每日成交额）
    daily_amount = daily.groupby("date")["amount"].sum().reset_index()
    daily_amount.columns = ["date", "total_market_amount"]
    daily_amount["date"] = pd.to_datetime(daily_amount["date"], errors="coerce")
    mg = mg.merge(daily_amount, on="date", how="left")
    if "margin_buy" in mg.columns:
        mg["margin_buy_ratio"] = mg["margin_buy"] / mg["total_market_amount"].replace(0, np.nan)
    else:
        mg["margin_buy_ratio"] = np.nan

    # 融券余额5日变化率
    if "short_balance" in mg.columns:
        mg["short_balance_chg_5d"] = mg["short_balance"].pct_change(5)
    else:
        mg["short_balance_chg_5d"] = np.nan

    # 融资融券情绪：融资买入额5日均值 vs 融券余额5日变化方向
    if "margin_buy" in mg.columns and "short_balance" in mg.columns:
        mg["margin_sentiment"] = (
            mg["margin_buy"].rolling(5).mean() / mg["margin_buy"].rolling(20).mean()
            - mg["short_balance"].pct_change(5).fillna(0)
        ).fillna(0)
    else:
        mg["margin_sentiment"] = np.nan

    # 合并到日线
    margin_cols = ["date"] + MARGIN_FEATURES
    mg_out = mg[[c for c in margin_cols if c in mg.columns]].copy()
    df = df.merge(mg_out, on="date", how="left")

    for col in MARGIN_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df


def merge_fundamental_margin_features(
    features: pd.DataFrame,
    daily: pd.DataFrame,
    financial_data: pd.DataFrame | None = None,
    margin_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将基本面和融资融券特征合并到主特征表。"""
    # 基本面
    fund_feat = build_fundamental_features(daily, financial_data)
    if not fund_feat.empty:
        fund_feat["date"] = pd.to_datetime(fund_feat["date"], errors="coerce")
        features["date"] = pd.to_datetime(features["date"], errors="coerce")
        features = features.merge(
            fund_feat[["code", "date"] + FUNDAMENTAL_FEATURES],
            on=["code", "date"], how="left", suffixes=("", "_fund")
        )
        # 去重列
        for col in FUNDAMENTAL_FEATURES:
            if f"{col}_fund" in features.columns:
                features[col] = features[col].fillna(features[f"{col}_fund"])
                features = features.drop(columns=[f"{col}_fund"])

    # 融资融券
    margin_feat = build_margin_features(daily, margin_data)
    if not margin_feat.empty and "code" in margin_feat.columns:
        margin_feat["date"] = pd.to_datetime(margin_feat["date"], errors="coerce")
        features = features.merge(
            margin_feat[["code", "date"] + MARGIN_FEATURES],
            on=["code", "date"], how="left", suffixes=("", "_mg")
        )
        for col in MARGIN_FEATURES:
            if f"{col}_mg" in features.columns:
                features[col] = features[col].fillna(features[f"{col}_mg"])
                features = features.drop(columns=[f"{col}_mg"])
    elif not margin_feat.empty:
        # 市场级特征，按 date 合并
        margin_feat["date"] = pd.to_datetime(margin_feat["date"], errors="coerce")
        features = features.merge(
            margin_feat[["date"] + MARGIN_FEATURES].drop_duplicates(subset=["date"]),
            on="date", how="left", suffixes=("", "_mg")
        )
        for col in MARGIN_FEATURES:
            if f"{col}_mg" in features.columns:
                features[col] = features[col].fillna(features[f"{col}_mg"])
                features = features.drop(columns=[f"{col}_mg"])

    # 确保所有特征列存在
    for col in ALL_FUNDAMENTAL_MARGIN_FEATURES:
        if col not in features.columns:
            features[col] = np.nan

    return features
