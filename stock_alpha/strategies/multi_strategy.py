"""多策略编排器：信号合并 + 资金分配 + 冲突解决。

核心功能：
1. 收集各策略信号，按策略权重加权
2. 等权分配资金
3. 信号矛盾时多数投票决定
4. 支持各策略独立回测 + 组合回测对比
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from stock_alpha.strategies.base import BaseStrategy


def _ts() -> str:
    """返回当前时间戳，用于日志打印。"""
    return datetime.now().strftime('%H:%M:%S')


@dataclass
class MultiStrategyConfig:
    strategy_names: list = field(default_factory=lambda: ["factor_alpha", "sector_rotation", "trend_breakout"])
    weights: list = field(default_factory=lambda: [0.33, 0.33, 0.34])
    capital_allocation: str = "equal"  # equal / proportional
    conflict_resolution: str = "majority"  # majority / weighted / conservative
    min_agreement: int = 2  # 最少几个策略同意才 BUY
    combined_min_score: float = 0.5  # 合并后最低分数门槛


class MultiStrategyOrchestrator:
    """多策略编排器。"""

    def __init__(self, strategies: list[BaseStrategy], config: MultiStrategyConfig | None = None):
        self.strategies = strategies
        self.config = config or MultiStrategyConfig()
        # 权重归一化
        weights = self.config.weights[:len(strategies)]
        total = sum(weights)
        self.weights = [w / total for w in weights] if total > 0 else [1.0 / len(strategies)] * len(strategies)

    def generate_combined_signals(self, daily: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
        """收集各策略信号并加权合并。

        Returns:
            合并后的信号表，包含:
            code, date, final_score, action, contributing_strategies,
            factor_alpha_score, sector_rotation_score, trend_breakout_score
        """
        all_strategy_signals = {}

        for i, strategy in enumerate(self.strategies):
            try:
                signals = strategy.generate_signals(daily, date)
                if not signals.empty:
                    all_strategy_signals[strategy.name] = signals
            except Exception as e:
                print(f"[{_ts()}] Strategy {strategy.name} failed: {e}")
                continue

        if not all_strategy_signals:
            return pd.DataFrame()

        # 合并各策略信号
        return self._merge_signals(all_strategy_signals)

    def _merge_signals(self, all_signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """按权重合并各策略信号（向量化实现）。"""
        cfg = self.config

        # 向量化收集所有策略信号
        parts = []
        for name, signals in all_signals.items():
            df = signals[["code", "date", "signal_score", "action"]].copy()
            df["strategy"] = name
            parts.append(df)

        if not parts:
            return pd.DataFrame()

        records_df = pd.concat(parts, ignore_index=True)
        records_df["date"] = pd.to_datetime(records_df["date"], errors="coerce")

        # 透视表：每个 (code, date) 一行，每列是一个策略的分数/action
        score_pivot = records_df.pivot_table(
            index=["code", "date"], columns="strategy", values="signal_score", aggfunc="first"
        ).reset_index()
        action_pivot = records_df.pivot_table(
            index=["code", "date"], columns="strategy", values="action", aggfunc="first"
        ).reset_index()

        # 计算加权得分（缺失策略用 0.5 中性分）
        weighted_score = pd.Series(0.0, index=score_pivot.index)
        for i, strategy in enumerate(self.strategies):
            sname = strategy.name
            if sname in score_pivot.columns:
                weighted_score += score_pivot[sname].fillna(0.5) * self.weights[i]
            else:
                weighted_score += 0.5 * self.weights[i]

        result = score_pivot[["code", "date"]].copy()
        result["final_score"] = weighted_score.values

        # 各策略分数列
        for strategy in self.strategies:
            sname = strategy.name
            result[f"{sname}_score"] = score_pivot[sname].values if sname in score_pivot.columns else np.nan

        # 冲突解决（仅需遍历唯一 code-date 对，而非全量行）
        actions = []
        contributing = []
        agreement = []
        for idx in range(len(result)):
            strategy_actions = {}
            for strategy in self.strategies:
                sname = strategy.name
                if sname in action_pivot.columns:
                    val = action_pivot.iloc[idx][sname]
                    if pd.notna(val):
                        strategy_actions[sname] = val
            ws = result.loc[idx, "final_score"]
            actions.append(self._resolve_conflict(strategy_actions, ws))
            buy_strategies = [s for s, a in strategy_actions.items() if a == "BUY"]
            contributing.append(",".join(buy_strategies) if buy_strategies else "")
            agreement.append(len(buy_strategies))

        result["action"] = actions
        result["contributing_strategies"] = contributing
        result["strategy_agreement"] = agreement

        # 兼容 pipeline 的字段名
        result["up_probability"] = result["final_score"]
        result["down_probability"] = 1 - result["final_score"]
        result["neutral_probability"] = 0.0
        result["risk_score"] = 0.0
        result["suggest_action"] = result["action"]

        return result.sort_values("final_score", ascending=False)

    def _resolve_conflict(self, actions: dict[str, str], weighted_score: float) -> str:
        """解决策略信号冲突。"""
        cfg = self.config
        buy_count = sum(1 for a in actions.values() if a == "BUY")
        avoid_count = sum(1 for a in actions.values() if a == "AVOID")
        total = len(actions)

        if cfg.conflict_resolution == "majority":
            # 多数投票
            if buy_count >= cfg.min_agreement:
                return "BUY"
            elif avoid_count > total // 2:
                return "AVOID"
            else:
                return "WATCH"
        elif cfg.conflict_resolution == "conservative":
            # 保守：任何一个 AVOID 就不买
            if avoid_count > 0:
                return "AVOID" if weighted_score < 0.5 else "WATCH"
            elif buy_count >= cfg.min_agreement:
                return "BUY"
            else:
                return "WATCH"
        else:
            # weighted: 纯看加权分数
            if weighted_score >= 0.7:
                return "BUY"
            elif weighted_score <= 0.3:
                return "AVOID"
            else:
                return "WATCH"

    def allocate_capital(self, total_capital: float) -> dict[str, float]:
        """资金分配。"""
        allocation = {}
        for i, strategy in enumerate(self.strategies):
            allocation[strategy.name] = total_capital * self.weights[i]
        return allocation

    def backtest_all(self, daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """各策略独立回测 + 组合回测。"""
        results = {}

        # 各策略独立信号
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(daily, date=None)
                results[f"{strategy.name}_signals"] = signals
            except Exception as e:
                print(f"[{_ts()}] Backtest {strategy.name} failed: {e}")
                results[f"{strategy.name}_signals"] = pd.DataFrame()

        # 组合信号
        combined = self.generate_combined_signals(daily, date=None)
        results["combined_signals"] = combined

        # 统计
        summary_rows = []
        for strategy in self.strategies:
            key = f"{strategy.name}_signals"
            signals = results.get(key, pd.DataFrame())
            if not signals.empty:
                buy_count = (signals["action"] == "BUY").sum()
                total = len(signals)
                summary_rows.append({
                    "strategy": strategy.name,
                    "total_signals": total,
                    "buy_signals": buy_count,
                    "buy_rate": buy_count / total if total > 0 else 0,
                })
        if not combined.empty:
            buy_count = (combined["action"] == "BUY").sum()
            summary_rows.append({
                "strategy": "combined",
                "total_signals": len(combined),
                "buy_signals": buy_count,
                "buy_rate": buy_count / len(combined) if len(combined) > 0 else 0,
            })
        results["summary"] = pd.DataFrame(summary_rows)

        return results
