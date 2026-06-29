from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import itertools
import time

import pandas as pd
from joblib import Parallel, delayed

from stock_alpha.backtest.ashare_backtest import AShareBacktestConfig, AShareBacktester
from stock_alpha.storage.cache import DataLake


def _run_single_wf_backtest(cfg: AShareBacktestConfig, daily: pd.DataFrame, wf_pred: pd.DataFrame) -> dict:
    """单个参数组合的回测（模块级函数，供 joblib pickle 并行调用）。"""
    bt = AShareBacktester(cfg).run(daily, wf_pred)
    m = bt["metrics"].iloc[0].to_dict() if not bt["metrics"].empty else {}
    ts = GridSearchRunner._trade_stats(bt.get("trades", pd.DataFrame()))
    return {"metrics": m, "trade_stats": ts}


@dataclass
class GridSearchRunner:
    lake: DataLake
    top_n_values: list[int] = field(default_factory=lambda: [3, 5, 10])
    hold_days_values: list[int] = field(default_factory=lambda: [2, 3, 5])
    min_score_values: list[float] = field(default_factory=lambda: [0.35, 0.45, 0.55])
    take_profit_values: list[float | None] = field(default_factory=lambda: [None, 0.03, 0.05])
    stop_loss_values: list[float | None] = field(default_factory=lambda: [None, 0.02, 0.03])

    def run(self, daily: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for top_n in self.top_n_values:
            for hold_days in self.hold_days_values:
                for min_score in self.min_score_values:
                    for take_profit in self.take_profit_values:
                        for stop_loss in self.stop_loss_values:
                            cfg = AShareBacktestConfig(top_n=top_n, hold_days=hold_days, min_score=min_score, take_profit=take_profit, stop_loss=stop_loss)
                            bt = AShareBacktester(cfg).run(daily, predictions)
                            m = bt["metrics"].iloc[0].to_dict() if not bt["metrics"].empty else {}
                            rows.append({"top_n": top_n, "hold_days": hold_days, "min_score": min_score, "take_profit": take_profit, "stop_loss": stop_loss, **m})
        out = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)
        self.lake.write_parquet("optimization", "grid_search", out)
        if not out.empty:
            self.lake.write_parquet("optimization", "best_params", pd.DataFrame([out.iloc[0].to_dict()]))
        return out

    def run_on_wf_predictions(self, daily: pd.DataFrame, base_cfg: AShareBacktestConfig | None = None, n_jobs: int = 1) -> pd.DataFrame:
        """在 Walk-Forward OOS 预测上搜索最优执行参数。

        与 run() 不同，这里：
        1. 使用 WF 的 OOS predictions（样本外，无前视偏差）
        2. 搜索空间针对执行参数（stop_loss/take_profit/hold_days），不搜 min_score
        3. 包含 ATR 动态止损选项
        4. 关闭严格风控过滤（与 WF 回测一致）
        5. n_jobs > 1 时使用 joblib 并行回测
        """
        wf_pred = self.lake.read_parquet("predictions", "walk_forward")
        if wf_pred.empty:
            raise RuntimeError("walk_forward predictions not found; run walk-forward first")

        # 搜索空间：针对执行参数
        search_space = {
            "top_n": [3, 5],
            "hold_days": [3, 5, 7, 10, 15],
            "stop_loss": [None, 0.05, 0.07, 0.10, 0.12, 0.15],
            "take_profit": [None, 0.06, 0.08, 0.10, 0.15, 0.20],
            "use_atr_stop": [False, True],
            "atr_stop_multiplier": [2.0, 2.5, 3.0],
            "atr_profit_multiplier": [3.0, 4.0],
        }

        # 基础配置
        base = base_cfg or AShareBacktestConfig()

        # === 生成全部参数组合 + 元数据 ===
        tasks: list[tuple[AShareBacktestConfig, dict]] = []
        for top_n in search_space["top_n"]:
            for hold_days in search_space["hold_days"]:
                # --- 固定止损模式 ---
                for stop_loss in search_space["stop_loss"]:
                    for take_profit in search_space["take_profit"]:
                        cfg = AShareBacktestConfig(
                            top_n=top_n,
                            hold_days=hold_days,
                            initial_cash=base.initial_cash,
                            buy_fee=base.buy_fee,
                            sell_fee=base.sell_fee,
                            slippage=base.slippage,
                            lot_size=base.lot_size,
                            max_position_pct=base.max_position_pct,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            min_score=0.0,
                            selection_mode="topn",
                            max_daily_buys=base.max_daily_buys,
                            require_up_gt_down=False,
                            enable_atr_filter=False,
                            enable_weak_rebound_filter=False,
                            use_atr_stop=False,
                        )
                        meta = {
                            "top_n": top_n, "hold_days": hold_days,
                            "stop_loss": stop_loss, "take_profit": take_profit,
                            "use_atr_stop": False, "atr_stop_multiplier": None,
                            "atr_profit_multiplier": None,
                        }
                        tasks.append((cfg, meta))

                # --- ATR 动态止损模式 ---
                for atr_stop_mult in search_space["atr_stop_multiplier"]:
                    for atr_profit_mult in search_space["atr_profit_multiplier"]:
                        cfg = AShareBacktestConfig(
                            top_n=top_n,
                            hold_days=hold_days,
                            initial_cash=base.initial_cash,
                            buy_fee=base.buy_fee,
                            sell_fee=base.sell_fee,
                            slippage=base.slippage,
                            lot_size=base.lot_size,
                            max_position_pct=base.max_position_pct,
                            stop_loss=None,
                            take_profit=None,
                            min_score=0.0,
                            selection_mode="topn",
                            max_daily_buys=base.max_daily_buys,
                            require_up_gt_down=False,
                            enable_atr_filter=False,
                            enable_weak_rebound_filter=False,
                            use_atr_stop=True,
                            atr_stop_multiplier=atr_stop_mult,
                            use_atr_profit=True,
                            atr_profit_multiplier=atr_profit_mult,
                        )
                        meta = {
                            "top_n": top_n, "hold_days": hold_days,
                            "stop_loss": None, "take_profit": None,
                            "use_atr_stop": True,
                            "atr_stop_multiplier": atr_stop_mult,
                            "atr_profit_multiplier": atr_profit_mult,
                        }
                        tasks.append((cfg, meta))

        total = len(tasks)
        print(f"[wf_grid_search] {total} combos, n_jobs={n_jobs}, starting...", flush=True)
        t0 = time.time()

        # === 执行回测（串行 or 并行）===
        if n_jobs > 1:
            # 并行：joblib loky 后端，daily 和 wf_pred 会自动去重传输
            raw_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_run_single_wf_backtest)(cfg, daily, wf_pred)
                for cfg, _ in tasks
            )
        else:
            raw_results = []
            for i, (cfg, _) in enumerate(tasks):
                raw_results.append(_run_single_wf_backtest(cfg, daily, wf_pred))
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (total - i - 1)
                    print(f"[wf_grid_search] {i+1}/{total} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

        # === 汇总结果 ===
        rows = []
        for (_cfg, meta), res in zip(tasks, raw_results):
            rows.append({**meta, **res["metrics"], **res["trade_stats"]})

        out = pd.DataFrame(rows)
        elapsed_total = time.time() - t0
        print(f"[wf_grid_search] {total} combos done in {elapsed_total:.1f}s", flush=True)
        if out.empty:
            return out
        # 按 Sharpe 排序（次要：总收益）
        out = out.sort_values(["sharpe", "total_return"], ascending=False).reset_index(drop=True)
        self.lake.write_parquet("optimization", "wf_grid_search", out)
        if not out.empty:
            self.lake.write_parquet("optimization", "wf_best_params", pd.DataFrame([out.iloc[0].to_dict()]))
            best = out.iloc[0]
            print(f"[wf_grid_search] best: sharpe={best.get('sharpe', 0):.3f} "
                  f"hold={int(best.get('hold_days', 0))} top={int(best.get('top_n', 0))} "
                  f"sl={best.get('stop_loss')} tp={best.get('take_profit')} "
                  f"atr={best.get('use_atr_stop')}", flush=True)
        return out

    @staticmethod
    def _trade_stats(trades: pd.DataFrame) -> dict:
        """从交易记录计算胜率、盈亏比等统计。"""
        if trades.empty or "side" not in trades.columns:
            return {"win_rate": None, "profit_loss_ratio": None, "stop_loss_pct": None}
        sells = trades[trades["side"] == "SELL"]
        if sells.empty:
            return {"win_rate": None, "profit_loss_ratio": None, "stop_loss_pct": None}
        # 止损比例
        total_sells = len(sells)
        sl_count = len(sells[sells["reason"] == "stop_loss"]) if "reason" in sells.columns else 0
        tp_count = len(sells[sells["reason"] == "take_profit"]) if "reason" in sells.columns else 0
        return {
            "win_rate_approx": tp_count / total_sells if total_sells > 0 else None,
            "stop_loss_pct": sl_count / total_sells if total_sells > 0 else None,
            "take_profit_pct": tp_count / total_sells if total_sells > 0 else None,
        }

    def write_recommended_config(self, base_config: dict, path: str | Path, use_wf: bool = False) -> Path:
        """写入推荐配置。use_wf=True 时优先使用 WF 最优参数。"""
        key = "wf_best_params" if use_wf else "best_params"
        best = self.lake.read_parquet("optimization", key)
        if best.empty:
            # fallback 到另一个
            alt_key = "best_params" if use_wf else "wf_best_params"
            best = self.lake.read_parquet("optimization", alt_key)
        if best.empty:
            raise RuntimeError("best_params not found; run grid search first")
        row = best.iloc[0]
        cfg = dict(base_config)
        def none_or_float(v):
            return None if pd.isna(v) else float(v)
        cfg.update({
            "top_n": int(row["top_n"]) if "top_n" in row else cfg.get("top_n", 5),
            "hold_days": int(row["hold_days"]) if "hold_days" in row else cfg.get("hold_days", 5),
            "take_profit": none_or_float(row.get("take_profit")),
            "stop_loss": none_or_float(row.get("stop_loss")),
        })
        # ATR 参数
        if row.get("use_atr_stop"):
            cfg["use_atr_stop"] = True
            cfg["atr_stop_multiplier"] = float(row.get("atr_stop_multiplier", 2.0))
            cfg["atr_profit_multiplier"] = float(row.get("atr_profit_multiplier", 3.0))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p
