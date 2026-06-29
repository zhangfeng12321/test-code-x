"""策略注册表：多策略体系统一入口。"""
from stock_alpha.strategies.base import BaseStrategy, StrategyConfig
from stock_alpha.strategies.sector_rotation import SectorRotationStrategy, SectorRotationConfig
from stock_alpha.strategies.trend_breakout import TrendBreakoutStrategy, TrendBreakoutConfig
from stock_alpha.strategies.multi_strategy import MultiStrategyOrchestrator, MultiStrategyConfig


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """根据名称获取策略实例。"""
    if name == "sector_rotation":
        stock_basic = kwargs.get("stock_basic")
        return SectorRotationStrategy(
            config=SectorRotationConfig(**{k: v for k, v in kwargs.items() if hasattr(SectorRotationConfig, k)}),
            stock_basic=stock_basic,
        )
    elif name == "trend_breakout":
        return TrendBreakoutStrategy(config=TrendBreakoutConfig(**{k: v for k, v in kwargs.items() if hasattr(TrendBreakoutConfig, k)}))
    elif name == "factor_alpha":
        from stock_alpha.strategies.factor_alpha import FactorAlphaStrategy
        from stock_alpha.storage.cache import DataLake
        lake = kwargs.get("lake") or DataLake(kwargs.get("data_root", "data"))
        return FactorAlphaStrategy(
            lake=lake,
            model_type=kwargs.get("model_type", "ensemble"),
            ensemble_alpha=kwargs.get("ensemble_alpha", 0.6),
        )
    else:
        raise ValueError(f"Unknown strategy: {name}. Available: factor_alpha, sector_rotation, trend_breakout")


STRATEGY_REGISTRY = {
    "factor_alpha": "stock_alpha.strategies.factor_alpha.FactorAlphaStrategy",
    "sector_rotation": "stock_alpha.strategies.sector_rotation.SectorRotationStrategy",
    "trend_breakout": "stock_alpha.strategies.trend_breakout.TrendBreakoutStrategy",
}
