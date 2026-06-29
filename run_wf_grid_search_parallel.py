#!/usr/bin/env python3
"""独立重跑 WF Grid Search（并行版），复用已缓存的 daily + walk_forward predictions。

用法：
    cd /Users/admin/.openclaw/workspaces/workspace-engineer-jay/projects/stock-alpha-model
    .venv/bin/python run_wf_grid_search_parallel.py          # 默认 n_jobs=4
    .venv/bin/python run_wf_grid_search_parallel.py 6        # 指定 n_jobs
"""
import sys
import time
from pathlib import Path

import pandas as pd

from stock_alpha.storage.cache import DataLake
from stock_alpha.training.train_v1 import V1Trainer
from stock_alpha.backtest.ashare_backtest import AShareBacktestConfig
from stock_alpha.optimization.grid_search import GridSearchRunner

DATA_ROOT = "data_real_2000_10pct"

def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"=== WF Grid Search (parallel, n_jobs={n_jobs}) ===", flush=True)

    lake = DataLake(root=DATA_ROOT)
    trainer = V1Trainer(lake)

    # 1. 加载 universe 选中的 codes（避免加载全部 5000 只）
    selected = lake.read_parquet("universe", "selected")
    if selected.empty:
        print("ERROR: universe/selected.csv not found, run full pipeline first", flush=True)
        sys.exit(1)
    codes = selected["code"].astype(str).str.zfill(6).tolist()
    print(f"[1/4] universe selected: {len(codes)} codes", flush=True)

    # 2. 只加载这些 codes 的 daily 数据
    t0 = time.time()
    daily = trainer.load_daily(codes)
    print(f"[2/4] daily loaded: {len(daily)} rows, {daily['code'].nunique()} codes, {time.time()-t0:.1f}s", flush=True)

    # 3. 验证 walk_forward predictions 存在
    wf_pred = lake.read_parquet("predictions", "walk_forward")
    if wf_pred.empty:
        print("ERROR: predictions/walk_forward.csv not found", flush=True)
        sys.exit(1)
    wf_pred["date"] = pd.to_datetime(wf_pred["date"])
    print(f"[3/4] wf predictions: {len(wf_pred)} rows, {wf_pred['code'].nunique()} codes", flush=True)

    # 4. 构造 base config（与 pipeline 一致）
    base_cfg = AShareBacktestConfig(
        top_n=5, hold_days=10, min_score=0.0,
        buy_fee=0.0003, sell_fee=0.0013, slippage=0.002,
        max_position_pct=0.1, take_profit=0.2, stop_loss=0.12,
        selection_mode="topn", score_quantile=0.95,
        max_daily_buys=5,
    )

    # 5. 跑并行 grid search
    print(f"[4/4] starting grid search (n_jobs={n_jobs})...", flush=True)
    gs = GridSearchRunner(lake)
    result = gs.run_on_wf_predictions(daily, base_cfg=base_cfg, n_jobs=n_jobs)

    # 6. 输出 top 10
    print(f"\n=== Top 10 by Sharpe ===", flush=True)
    cols = ["sharpe", "total_return", "max_drawdown", "hold_days", "top_n",
            "stop_loss", "take_profit", "use_atr_stop", "atr_stop_multiplier"]
    show_cols = [c for c in cols if c in result.columns]
    print(result[show_cols].head(10).to_string(index=False), flush=True)

    # 7. 写推荐配置
    import json
    recommended = gs.write_recommended_config(
        {"top_n": 5, "hold_days": 10, "take_profit": 0.2, "stop_loss": 0.12},
        Path(DATA_ROOT) / "optimization" / "recommended_config.json",
        use_wf=True,
    )
    print(f"\nrecommended config saved to: {recommended}", flush=True)

if __name__ == "__main__":
    main()
