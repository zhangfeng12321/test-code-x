"""策略基类：定义多策略体系的统一接口。

所有策略必须继承 BaseStrategy 并实现 generate_signals 方法。
统一输出格式确保多策略编排器可以无缝合并各策略信号。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StrategyConfig:
    """策略通用配置。"""
    name: str = ""
    enabled: bool = True
    weight: float = 1.0  # 在多策略中的权重
    max_positions: int = 10  # 最大持仓数
    hold_days: int = 5  # 默认持有天数


class BaseStrategy(ABC):
    """策略基类：统一接口。

    所有策略必须输出统一格式的信号表：
    - code: 股票代码
    - date: 信号日期
    - signal_score: 信号强度 (0~1)
    - action: BUY / SELL / HOLD / AVOID
    - strategy_name: 策略名称
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, daily: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
        """生成交易信号。

        Args:
            daily: 日线数据（需含 code, date, open, high, low, close, volume, amount）
            date: 信号日期（None 表示生成所有日期的信号）

        Returns:
            DataFrame with columns: code, date, signal_score, action, strategy_name
        """

    def backtest_signals(self, daily: pd.DataFrame) -> pd.DataFrame:
        """为所有日期生成信号（用于回测）。"""
        return self.generate_signals(daily, date=None)

    @staticmethod
    def _empty_signals() -> pd.DataFrame:
        """返回空信号表。"""
        return pd.DataFrame(columns=["code", "date", "signal_score", "action", "strategy_name"])

    @staticmethod
    def _make_signal(code: str, date, score: float, action: str, strategy: str) -> dict:
        """构造单条信号记录。"""
        return {
            "code": code,
            "date": date,
            "signal_score": max(0.0, min(1.0, score)),
            "action": action,
            "strategy_name": strategy,
        }
