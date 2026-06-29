"""趋势突破策略：检测价格突破 + 量能确认 + 趋势跟踪。

核心逻辑：
1. 突破 N 日新高（Donchian Channel 上轨）
2. 量能配合（成交量 > 20日均量 * volume_multiple）
3. 趋势确认（MA5 > MA10 > MA20 多头排列）
4. 信号强度 = 突破幅度/ATR * 量能倍数
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_alpha.strategies.base import BaseStrategy


@dataclass
class TrendBreakoutConfig:
    breakout_period: int = 20  # 突破回望天数
    volume_multiple: float = 1.5  # 量能倍数阈值
    atr_stop_multiple: float = 2.0  # ATR 止损倍数
    require_ma_alignment: bool = True  # 是否要求均线多头排列
    min_breakout_pct: float = 0.01  # 最小突破幅度（相对前高）


class TrendBreakoutStrategy(BaseStrategy):
    """趋势突破策略：突破新高 + 放量确认。"""

    name = "trend_breakout"

    def __init__(self, config: TrendBreakoutConfig | None = None):
        self.config = config or TrendBreakoutConfig()

    def generate_signals(self, daily: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
        """生成趋势突破信号。"""
        if daily.empty or "code" not in daily.columns:
            return self._empty_signals()

        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        df = df.sort_values(["code", "date"])

        cfg = self.config
        all_signals = []

        for code, grp in df.groupby("code"):
            grp = grp.sort_values("date").reset_index(drop=True)
            if len(grp) < cfg.breakout_period + 5:
                continue

            # 计算技术指标
            grp["highest_n"] = grp["close"].rolling(cfg.breakout_period).max()
            grp["avg_volume_20"] = grp["volume"].rolling(20).mean()
            grp["ma5"] = grp["close"].rolling(5).mean()
            grp["ma10"] = grp["close"].rolling(10).mean()
            grp["ma20"] = grp["close"].rolling(20).mean()

            # ATR
            grp["prev_close"] = grp["close"].shift(1)
            tr = pd.concat([
                grp["high"] - grp["low"],
                (grp["high"] - grp["prev_close"]).abs(),
                (grp["low"] - grp["prev_close"]).abs(),
            ], axis=1).max(axis=1)
            grp["atr_14"] = tr.rolling(14).mean()

            # 确定目标日期
            if date is not None:
                target_rows = grp[grp["date"] == pd.to_datetime(date)]
            else:
                target_rows = grp.iloc[cfg.breakout_period + 5:]

            for idx, row in target_rows.iterrows():
                if pd.isna(row["highest_n"]) or pd.isna(row["avg_volume_20"]):
                    continue

                # 条件1: 突破 N 日新高
                prev_high = grp.loc[:idx - 1, "close"].tail(cfg.breakout_period).max() if idx > 0 else row["highest_n"]
                is_breakout = row["close"] > prev_high * (1 + cfg.min_breakout_pct)

                # 条件2: 量能确认
                volume_ok = row["volume"] > row["avg_volume_20"] * cfg.volume_multiple if row["avg_volume_20"] > 0 else False

                # 条件3: 均线多头排列
                ma_aligned = True
                if cfg.require_ma_alignment:
                    ma_aligned = (
                        not pd.isna(row["ma5"]) and not pd.isna(row["ma10"]) and not pd.isna(row["ma20"])
                        and row["ma5"] > row["ma10"] > row["ma20"]
                    )

                if is_breakout and volume_ok and ma_aligned:
                    # 计算信号强度
                    breakout_strength = (row["close"] / prev_high - 1) / max(row["atr_14"] / row["close"], 0.01) if row["atr_14"] > 0 else 1.0
                    volume_ratio = row["volume"] / row["avg_volume_20"] if row["avg_volume_20"] > 0 else 1.0
                    score = min(1.0, 0.5 + breakout_strength * 0.2 + (volume_ratio - 1) * 0.1)
                    action = "BUY"
                elif is_breakout and not volume_ok:
                    # 突破但无量确认 → 观望
                    score = 0.4
                    action = "HOLD"
                else:
                    # 跌破 MA10 → 可能趋势结束
                    if not pd.isna(row["ma10"]) and row["close"] < row["ma10"]:
                        score = 0.2
                        action = "AVOID"
                    else:
                        score = 0.5
                        action = "HOLD"

                all_signals.append(self._make_signal(code, row["date"], score, action, self.name))

        return pd.DataFrame(all_signals) if all_signals else self._empty_signals()
