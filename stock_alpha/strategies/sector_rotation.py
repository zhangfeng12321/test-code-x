"""行业轮动策略：追踪板块强弱轮换，买入强势板块个股。

核心逻辑：
1. 按行业板块计算最近 N 日板块收益率排名
2. 选择 Top-K 强势板块中的个股
3. 弱势板块个股发出 AVOID 信号
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_alpha.strategies.base import BaseStrategy


@dataclass
class SectorRotationConfig:
    rotation_lookback: int = 10  # 板块动量回望天数
    top_sector_pct: float = 0.3  # 强势板块比例
    bottom_sector_pct: float = 0.3  # 弱势板块比例
    min_sector_stocks: int = 3  # 板块内最少股票数
    momentum_weights: tuple = (0.5, 0.3, 0.2)  # 5日/10日/20日动量权重


class SectorRotationStrategy(BaseStrategy):
    """行业轮动策略：板块动量 → 强势板块内选股。"""

    name = "sector_rotation"

    def __init__(self, config: SectorRotationConfig | None = None, stock_basic: pd.DataFrame | None = None):
        self.config = config or SectorRotationConfig()
        self._industry_map: dict[str, str] = {}
        if stock_basic is not None and not stock_basic.empty and "industry" in stock_basic.columns:
            sb = stock_basic[["code", "industry"]].drop_duplicates("code").copy()
            sb["code"] = sb["code"].astype(str).str.zfill(6)
            self._industry_map = dict(zip(sb["code"], sb["industry"]))

    def generate_signals(self, daily: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
        """生成行业轮动信号。"""
        if daily.empty or "code" not in daily.columns:
            return self._empty_signals()

        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        df = df.sort_values(["code", "date"])

        # 优先使用精确行业分类，回退用代码前三位粗分
        if self._industry_map:
            df["sector"] = df["code"].map(self._industry_map).fillna(df["code"].str[:3])
        else:
            df["sector"] = df["code"].str[:3]

        # 计算个股收益率
        df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
        df["ret_10d"] = df.groupby("code")["close"].pct_change(10)
        df["ret_20d"] = df.groupby("code")["close"].pct_change(20)

        # 确定信号日期范围
        dates = sorted(df["date"].dropna().unique())
        if date is not None:
            target_dates = [pd.to_datetime(date)]
        else:
            # 跳过前20天（需要数据预热）
            target_dates = dates[20:] if len(dates) > 20 else dates[-5:]

        all_signals = []
        cfg = self.config
        w5, w10, w20 = cfg.momentum_weights

        for d in target_dates:
            day_data = df[df["date"] == d].copy()
            if day_data.empty:
                continue

            # 计算板块动量（板块内个股等权收益率加权）
            sector_momentum = day_data.groupby("sector").agg(
                mom_5d=("ret_5d", "mean"),
                mom_10d=("ret_10d", "mean"),
                mom_20d=("ret_20d", "mean"),
                stock_count=("code", "count"),
            ).reset_index()

            # 过滤股票数太少的板块
            sector_momentum = sector_momentum[sector_momentum["stock_count"] >= cfg.min_sector_stocks]
            if sector_momentum.empty:
                continue

            # 综合板块动量分数
            sector_momentum["momentum_score"] = (
                sector_momentum["mom_5d"].fillna(0) * w5
                + sector_momentum["mom_10d"].fillna(0) * w10
                + sector_momentum["mom_20d"].fillna(0) * w20
            )

            # 板块排名
            sector_momentum["sector_rank"] = sector_momentum["momentum_score"].rank(pct=True)

            # 划分强势/弱势板块
            strong_sectors = sector_momentum[sector_momentum["sector_rank"] >= (1 - cfg.top_sector_pct)]["sector"].tolist()
            weak_sectors = sector_momentum[sector_momentum["sector_rank"] <= cfg.bottom_sector_pct]["sector"].tolist()

            # 为每只股票生成信号
            for _, row in day_data.iterrows():
                code = row["code"]
                sector = row["sector"]

                if sector in strong_sectors:
                    # 强势板块：个股在板块内的相对强度决定分数
                    sector_data = day_data[day_data["sector"] == sector]
                    if not sector_data.empty and "ret_5d" in sector_data.columns:
                        stock_rank = sector_data["ret_5d"].rank(pct=True)
                        idx = sector_data.index.get_loc(row.name) if row.name in sector_data.index else 0
                        relative_strength = stock_rank.iloc[idx] if idx < len(stock_rank) else 0.5
                    else:
                        relative_strength = 0.5
                    score = 0.5 + relative_strength * 0.5  # 0.5~1.0
                    action = "BUY" if score >= 0.7 else "HOLD"
                elif sector in weak_sectors:
                    score = 0.2
                    action = "AVOID"
                else:
                    score = 0.5
                    action = "HOLD"

                all_signals.append(self._make_signal(code, d, score, action, self.name))

        return pd.DataFrame(all_signals) if all_signals else self._empty_signals()
