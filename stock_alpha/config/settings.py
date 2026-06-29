from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class PipelineConfig:
    provider: str = "fallback"
    data_root: str = "data"
    start: str = "20240101"
    end: str = "20240630"
    codes: list[str] = field(default_factory=list)
    limit: int | None = None
    batch_size: int = 50
    top_n: int = 5
    hold_days: int = 3
    min_score: float = 0.45
    buy_fee: float = 0.0003
    sell_fee: float = 0.0013
    slippage: float = 0.001
    max_position_pct: float = 0.2
    take_profit: float | None = None
    stop_loss: float | None = None
    train_end: str | None = None
    valid_end: str | None = None
    train_days: int = 120
    test_days: int = 30
    step_days: int = 30
    force: bool = False
    run_minute: bool = False
    minute_period: str = "5"
    run_grid_search: bool = False
    archive_outputs: bool = True
    use_universe_filter: bool = True
    universe_max_size: int | None = 800
    min_avg_amount_20: float = 100000000.0
    min_avg_amount_60: float = 50000000.0
    min_turnover_20: float = 1.0
    min_amplitude_20: float = 0.02
    max_amplitude_20: float = 0.12
    min_volatility_20: float = 0.015
    max_volatility_20: float = 0.08
    selection_mode: str = "topn"
    score_quantile: float = 0.95
    skip_download: bool = False
    max_down_probability: float | None = 0.40
    max_risk_score: float | None = 0.25
    min_order_avg_amount_20: float | None = 100000000.0
    max_daily_buys: int | None = 5
    label_profit_take: float = 0.03
    label_stop_loss: float = 0.02
    label_horizon: int = 3
    min_train_days: int = 250  # 参与训练至少需要的交易天数（约1年）
    model_type: str = "ranker"  # "classifier" / "ranker" / "ensemble"
    ensemble_alpha: float = 0.6  # ensemble 模式下 Ranker 权重（0~1），Classifier 权重为 1-alpha
    label_horizons: list = None  # 双时间框架，如 [5, 10]，默认只用 label_horizon
    # --- 多策略体系 ---
    use_multi_strategy: bool = False  # 是否启用多策略模式
    strategies: list = None  # 策略列表，如 ["factor_alpha", "sector_rotation", "trend_breakout"]
    strategy_weights: list = None  # 策略权重，如 [0.33, 0.33, 0.34]
    # --- 市场状态过滤 (Bull Market Filter) ---
    use_bull_filter: bool = True       # 开启市场状态过滤（建议开启）
    bull_breadth_days: int = 20        # 计算全市场近 N 日等权涨幅
    bull_breadth_pause: float = 0.12   # 近 N 日市场涨幅 > 此阈值时暂停新买入（12%防止误报）
    bull_ma_days: int = 60             # 全市场均价 MA窗口天数

    @staticmethod
    def from_file(path: str | Path) -> "PipelineConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return PipelineConfig(**data)

    def to_file(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
