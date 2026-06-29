from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stock_alpha.config.settings import PipelineConfig
from stock_alpha.data.batch import BatchDownloadRunner
from stock_alpha.data.downloader import MarketDataDownloader
from stock_alpha.data.pre_filter import PreFilterConfig, pre_filter_stock_basic, pre_filter_summary
from stock_alpha.data.providers.akshare_provider import AkShareProvider
from stock_alpha.data.providers.baostock_provider import BaoStockProvider
from stock_alpha.data.providers.fallback_provider import FallbackMarketDataProvider
from stock_alpha.data.quality import check_daily_quality, summarize_quality
from stock_alpha.data.trade_calendar import build_trade_calendar
from stock_alpha.reports.daily_report import DailyReportGenerator
from stock_alpha.reports.html_report import HtmlReportGenerator
from stock_alpha.runtime.runlog import RunLogger
from stock_alpha.storage.cache import DataLake
from stock_alpha.training.train_v1 import V1Trainer
from stock_alpha.backtest.ashare_backtest import AShareBacktester, AShareBacktestConfig
from stock_alpha.training.walk_forward import WalkForwardRunner
from stock_alpha.analysis_signal import signal_stability, turnover_by_date
from stock_alpha.runtime.archive import copy_if_exists
from stock_alpha.optimization.grid_search import GridSearchRunner
from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.reports.candidate_analysis import candidate_risk_tags, explain_candidates
from stock_alpha.trading_plan import OrderPlanConfig, generate_next_day_orders
from stock_alpha.universe import UniverseFilterConfig, build_trade_universe
from stock_alpha.risk_rules import RiskRuleConfig, attach_latest_context, apply_hard_risk_filters
from stock_alpha.watchlist import WatchlistConfig, generate_watchlist


def make_provider(name: str, data_root: str | None = None):
    if name == "akshare":
        return AkShareProvider(data_root=data_root)
    if name == "baostock":
        return BaoStockProvider()
    if name == "fallback":
        return FallbackMarketDataProvider([AkShareProvider(data_root=data_root), BaoStockProvider()])
    raise ValueError(name)


class FullPipeline:
    def __init__(self, cfg: PipelineConfig, logger: RunLogger | None = None):
        self.cfg = cfg
        self.lake = DataLake(cfg.data_root)
        self.logger = logger or RunLogger(Path(cfg.data_root) / "runs")

    def run(self) -> dict:
        cfg = self.cfg
        self.logger.event("pipeline", "start", config=cfg.__dict__)
        codes = cfg.codes
        if not cfg.skip_download:
            pr = make_provider(cfg.provider, data_root=cfg.data_root)
            try:
                downloader = MarketDataDownloader(pr, self.lake)
                if not codes:
                    basic = downloader.get_stock_universe(limit=cfg.limit)
                    # 首轮预过滤：下载前排除 ST/北交所/次新
                    pre_cfg = PreFilterConfig(as_of_date=cfg.end)
                    passed, excluded = pre_filter_stock_basic(basic, pre_cfg)
                    self.lake.write_parquet("meta", "pre_filter_excluded", excluded)
                    codes = passed["code"].tolist()
                    self.logger.event("pre_filter", "done", total=len(basic), passed=len(passed), excluded=len(excluded))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {pre_filter_summary(passed, excluded)}")
                self.logger.event("download", "start", codes=len(codes))
                BatchDownloadRunner(downloader, cfg.batch_size).download_daily_batches(codes, cfg.start, cfg.end, force=cfg.force)
                self.logger.event("download", "done")
                if cfg.run_minute:
                    downloader.download_minute(codes, cfg.start, cfg.end, period=cfg.minute_period, force=cfg.force)
                # 下载北向资金、龙虎榜、财务、融资融券、个股北向持股
                self.logger.event("download_extra", "start")
                downloader.download_northbound_flow(cfg.start, cfg.end, force=cfg.force)
                downloader.download_dragon_tiger(cfg.start, cfg.end, force=cfg.force)
                downloader.download_fundamentals(codes, force=cfg.force)
                downloader.download_margin_data(cfg.start, cfg.end, force=cfg.force)
                downloader.download_northbound_stock(codes, cfg.start, cfg.end, force=cfg.force)
                self.logger.event("download_extra", "done")
            finally:
                if hasattr(pr, "close"):
                    pr.close()
        else:
            self.logger.event("download", "skipped", reason="handled_by_runner")

        trainer = V1Trainer(self.lake)
        daily_all = trainer.load_daily(codes)

        # === 训练前数据充分性门控 ===
        # 排除交易天数不足的股票，避免噪音样本污染模型
        if not daily_all.empty and cfg.min_train_days > 0:
            day_counts = daily_all.groupby("code")["date"].nunique().reset_index(name="trade_days")
            sufficient = day_counts[day_counts["trade_days"] >= cfg.min_train_days]["code"].tolist()
            excluded_codes = day_counts[day_counts["trade_days"] < cfg.min_train_days].copy()
            excluded_codes["reason"] = "trade_days < " + str(cfg.min_train_days)
            if not excluded_codes.empty:
                self.lake.write_parquet("meta", "train_pool_excluded", excluded_codes)
            if sufficient:
                daily_all = daily_all[daily_all["code"].isin(sufficient)].copy()
                codes = sufficient
            self.logger.event("train_gate", "done", sufficient=len(sufficient), excluded=len(excluded_codes), min_train_days=cfg.min_train_days)

        # === 训练池 vs 交易池分离 ===
        # 训练池：用全量可用数据训练模型（学规律）
        # 交易池：只从流动性好的子集中选标的交易（安全执行）
        trade_daily = daily_all  # 默认训练池=全量
        trade_codes = codes
        if cfg.use_universe_filter:
            ucfg = UniverseFilterConfig(
                min_avg_amount_20=cfg.min_avg_amount_20, min_avg_amount_60=cfg.min_avg_amount_60,
                min_turnover_20=cfg.min_turnover_20, min_amplitude_20=cfg.min_amplitude_20,
                max_amplitude_20=cfg.max_amplitude_20, min_volatility_20=cfg.min_volatility_20,
                max_volatility_20=cfg.max_volatility_20,
            )
            selected, metrics_all = build_trade_universe(daily_all, cfg=ucfg, max_size=cfg.universe_max_size)
            self.lake.write_parquet("universe", "metrics", metrics_all)
            self.lake.write_parquet("universe", "selected", selected)
            selected_codes = selected["code"].astype(str).tolist()
            if selected_codes:
                trade_daily = daily_all[daily_all["code"].astype(str).str.zfill(6).isin(selected_codes)].copy()
                trade_codes = selected_codes
            self.logger.event("universe", "done", selected=len(selected_codes), total=metrics_all["code"].nunique() if not metrics_all.empty else 0, train_codes=len(codes))

        # 质量检查在全量数据上做，使用交易日历精确判断
        trade_cal = build_trade_calendar(daily_all, cache_path=self.lake.data_path("meta", "trade_calendar"))
        self.logger.event("quality", "start", rows=len(daily_all), trade_cal_days=trade_cal.total_days)
        issues = check_daily_quality(daily_all, cfg.start, cfg.end, trade_calendar=trade_cal)
        summary = summarize_quality(issues)
        self.lake.write_parquet("quality", "daily_issues", issues)
        self.lake.write_parquet("quality", "daily_summary", summary)
        self.logger.event("quality", "done", issues=len(issues))

        # 训练：用全量数据（训练池），支持双时间框架
        horizons = cfg.label_horizons or [cfg.label_horizon]

        # === 多策略模式 ===
        if cfg.use_multi_strategy:
            self.logger.event("multi_strategy", "start", strategies=cfg.strategies or ["sector_rotation", "trend_breakout"])
            from stock_alpha.strategies import get_strategy
            from stock_alpha.strategies.multi_strategy import MultiStrategyOrchestrator, MultiStrategyConfig

            strategy_names = cfg.strategies or ["sector_rotation", "trend_breakout"]
            strategy_weights = cfg.strategy_weights or [1.0 / len(strategy_names)] * len(strategy_names)

            strategies = []
            stock_basic = self.lake.read_parquet("meta", "stock_basic")
            for sname in strategy_names:
                if sname == "factor_alpha":
                    strategies.append(get_strategy(sname, lake=self.lake, model_type=cfg.model_type, ensemble_alpha=cfg.ensemble_alpha))
                else:
                    strategies.append(get_strategy(sname, stock_basic=stock_basic if not stock_basic.empty else None))

            orchestrator = MultiStrategyOrchestrator(
                strategies=strategies,
                config=MultiStrategyConfig(strategy_names=strategy_names, weights=strategy_weights),
            )
            combined = orchestrator.generate_combined_signals(daily_all)
            if not combined.empty:
                self.lake.write_parquet("predictions", "v1_latest", combined)
                # 保存各策略独立信号
                bt_results = orchestrator.backtest_all(daily_all)
                if "summary" in bt_results and not bt_results["summary"].empty:
                    self.lake.write_parquet("predictions", "strategy_summary", bt_results["summary"])
            self.logger.event("multi_strategy", "done", signals=len(combined) if not combined.empty else 0)
            train_res = None  # 多策略模式无单一模型训练结果
        else:
            # === 单策略模式（原有逻辑） ===
            self.logger.event("train", "start", train_pool=len(codes), model_type=cfg.model_type, horizons=horizons)

            # 主模型（第一个 horizon，用于交易计划）
            model_path = self.logger.dir / "models" / "v1_daily_lgb.pkl" if cfg.archive_outputs else None
            train_res = trainer.train(codes=codes, train_end=cfg.train_end, valid_end=cfg.valid_end, model_path=model_path, label_profit_take=cfg.label_profit_take, label_stop_loss=cfg.label_stop_loss, label_horizon=horizons[0], model_type=cfg.model_type, ensemble_alpha=cfg.ensemble_alpha)
            self.logger.event("train", "done", backend=train_res.backend, rows=train_res.rows, horizon=horizons[0])

            # 如果有第二个 horizon，串行训练对比版本
            if len(horizons) > 1:
                for h in horizons[1:]:
                    self.logger.event("train_alt", "start", horizon=h)
                    alt_res = trainer.train(codes=codes, train_end=cfg.train_end, valid_end=cfg.valid_end, model_path=None, label_profit_take=cfg.label_profit_take, label_stop_loss=cfg.label_stop_loss, label_horizon=h, model_type=cfg.model_type, ensemble_alpha=cfg.ensemble_alpha)
                    alt_pred = self.lake.read_parquet("predictions", "v1_latest")
                    self.lake.write_parquet("predictions", f"v1_latest_{h}d", alt_pred)
                    self.logger.event("train_alt", "done", horizon=h, backend=alt_res.backend, rows=alt_res.rows)

            # 恢复主模型预测（第一个 horizon 的）
            if len(horizons) > 1:
                trainer.train(codes=codes, train_end=cfg.train_end, valid_end=cfg.valid_end, model_path=model_path, label_profit_take=cfg.label_profit_take, label_stop_loss=cfg.label_stop_loss, label_horizon=horizons[0], model_type=cfg.model_type, ensemble_alpha=cfg.ensemble_alpha)
                self.lake.write_parquet("predictions", f"v1_latest_{horizons[0]}d", self.lake.read_parquet("predictions", "v1_latest"))

        pred = self.lake.read_parquet("predictions", "v1_latest")
        # 回测和交易计划只用交易池数据（universe 过滤后的流动性好的股票）
        # 但预测是全量的，需要过滤到交易池范围
        trade_pred = pred[pred["code"].astype(str).str.zfill(6).isin(
            trade_daily["code"].astype(str).str.zfill(6).unique().tolist()
        )].copy() if cfg.use_universe_filter and not trade_daily.empty else pred
        self.logger.event("backtest", "start", trade_pool=len(trade_pred["code"].unique()) if not trade_pred.empty else 0)
        bt_cfg = AShareBacktestConfig(
            top_n=cfg.top_n, hold_days=cfg.hold_days, min_score=cfg.min_score,
            buy_fee=cfg.buy_fee, sell_fee=cfg.sell_fee, slippage=cfg.slippage,
            max_position_pct=cfg.max_position_pct, take_profit=cfg.take_profit, stop_loss=cfg.stop_loss,
            selection_mode=cfg.selection_mode, score_quantile=cfg.score_quantile,
            max_down_probability=cfg.max_down_probability, max_risk_score=cfg.max_risk_score,
            max_daily_buys=cfg.max_daily_buys,
            # 市场状态过滤
            use_bull_filter=getattr(cfg, 'use_bull_filter', True),
            bull_breadth_days=getattr(cfg, 'bull_breadth_days', 20),
            bull_breadth_pause=getattr(cfg, 'bull_breadth_pause', 0.08),
            bull_ma_days=getattr(cfg, 'bull_ma_days', 60),
        )
        # === 数据泄漏修复：回测只用样本外（OOS）预测 ===
        # 模型在 train_end 之前的数据上训练，对该期间的预测属于 in-sample，回测不可信
        import pandas as pd
        oos_start = pd.Timestamp(cfg.train_end) if cfg.train_end else None
        if oos_start is not None and not trade_pred.empty:
            date_col = pd.to_datetime(trade_pred["date"])
            trade_pred_oos = trade_pred[date_col > oos_start].copy()
        else:
            trade_pred_oos = trade_pred
        bt = AShareBacktester(bt_cfg).run(trade_daily, trade_pred_oos)
        for k, v in bt.items():
            self.lake.write_parquet("backtest", f"{k}_latest", v)
        self.logger.event("backtest", "done", metrics=bt["metrics"].to_dict("records"))

        stability = signal_stability(trade_pred, top_n=max(cfg.top_n, 20))
        turnover = turnover_by_date(trade_pred, top_n=max(cfg.top_n, 20))
        # 加载全量外部数据用于特征构建（与训练时保持一致）
        northbound_flow = self.lake.read_parquet("northbound", "daily_flow")
        northbound_stock = self.lake.read_parquet("northbound", "stock_holdings")
        lhb_data = self.lake.read_parquet("dragon_tiger", "lhb_detail")
        stock_basic = self.lake.read_parquet("meta", "stock_basic")
        financial_data = self.lake.read_parquet("fundamental", "financial_indicators")
        margin_data = self.lake.read_parquet("margin", "daily_summary")
        features_latest = build_daily_features(
            trade_daily,
            northbound_flow=northbound_flow if not northbound_flow.empty else None,
            northbound_stock=northbound_stock if not northbound_stock.empty else None,
            lhb_data=lhb_data if not lhb_data.empty else None,
            stock_basic=stock_basic if not stock_basic.empty else None,
            financial_data=financial_data if not financial_data.empty else None,
            margin_data=margin_data if not margin_data.empty else None,
        )
        pred_for_analysis = attach_latest_context(trade_pred, trade_daily)
        pred_for_analysis = apply_hard_risk_filters(
            pred_for_analysis,
            RiskRuleConfig(min_score=cfg.min_score, max_down_probability=cfg.max_down_probability, max_risk_score=cfg.max_risk_score),
        )
        risk_tags = candidate_risk_tags(pred_for_analysis, trade_daily)
        explanations = explain_candidates(features_latest, trade_pred, self.lake.read_parquet("evaluation", "feature_importance"))
        universe_metrics = self.lake.read_parquet("universe", "metrics")
        orders = generate_next_day_orders(
            trade_pred, trade_daily,
            OrderPlanConfig(top_n=cfg.top_n, min_score=cfg.min_score, max_position_pct=cfg.max_position_pct,
                            take_profit=cfg.take_profit, stop_loss=cfg.stop_loss, selection_mode=cfg.selection_mode,
                            score_quantile=cfg.score_quantile, max_down_probability=cfg.max_down_probability,
                            max_risk_score=cfg.max_risk_score, min_avg_amount_20=cfg.min_order_avg_amount_20),
            universe_metrics=universe_metrics,
            stock_basic=stock_basic if not stock_basic.empty else None,
        )
        watchlist = generate_watchlist(
            trade_pred, trade_daily,
            WatchlistConfig(top_n=max(cfg.top_n * 2, 20), min_avg_amount_20=cfg.min_order_avg_amount_20),
            universe_metrics=universe_metrics,
        )
        self.lake.write_parquet("analysis", "signal_stability", stability)
        self.lake.write_parquet("analysis", "turnover", turnover)
        self.lake.write_parquet("analysis", "candidate_risk_tags", risk_tags)
        self.lake.write_parquet("analysis", "candidate_explanations", explanations)
        self.lake.write_parquet("orders", "next_day_orders", orders)
        self.lake.write_parquet("orders", "watchlist", watchlist)
        self.logger.event("analysis", "done", stability_rows=len(stability), turnover_rows=len(turnover), orders=len(orders), watchlist=len(watchlist))

        try:
            self.logger.event("walk_forward", "start")
            # Walk-Forward 使用基础配置作为参考，内部会自动调整 min_score 和 selection_mode
            wf = WalkForwardRunner(
                self.lake, cfg.train_days, cfg.test_days, cfg.step_days,
                backtest_config=bt_cfg,
                label_profit_take=cfg.label_profit_take,
                label_stop_loss=cfg.label_stop_loss,
                label_horizon=cfg.label_horizon,
                model_type=cfg.model_type,
                ensemble_alpha=cfg.ensemble_alpha,
            ).run(trade_daily)
            self.logger.event("walk_forward", "done", windows=len(wf["windows"]), predictions=len(wf["predictions"]))

            # === WF 完成后：OOS Grid Search + OOS 概率校准 ===
            try:
                self.logger.event("wf_grid_search", "start")
                gs_runner = GridSearchRunner(self.lake)
                wf_gs = gs_runner.run_on_wf_predictions(trade_daily, base_cfg=bt_cfg, n_jobs=4)
                # 写入推荐配置
                recommended_config = gs_runner.write_recommended_config(
                    cfg.__dict__, self.logger.dir / "recommended_config.json", use_wf=True
                )
                best_row = wf_gs.iloc[0] if not wf_gs.empty else {}
                self.logger.event("wf_grid_search", "done",
                    combos=len(wf_gs),
                    best_sharpe=float(best_row.get("sharpe", 0)),
                    best_stop_loss=best_row.get("stop_loss"),
                    best_take_profit=best_row.get("take_profit"),
                    best_hold_days=int(best_row.get("hold_days", 0)),
                    recommended_config=str(recommended_config),
                )
            except Exception as e:
                self.logger.event("wf_grid_search", "skipped", error=repr(e))

            # === OOS 概率校准 ===
            try:
                from stock_alpha.models.v2_ranker_model import V2RankerModel
                wf_pred_data = self.lake.read_parquet("predictions", "walk_forward")
                if not wf_pred_data.empty:
                    calibrator = V2RankerModel()
                    oos_map = calibrator.calibrate_from_oos(wf_pred_data, trade_daily, horizon=cfg.label_horizon)
                    if oos_map is not None:
                        import numpy as np
                        oos_df = pd.DataFrame(oos_map, columns=["score_bin", "actual_win_rate"])
                        self.lake.write_parquet("evaluation", "oos_calibration", oos_df)
                        self.logger.event("oos_calibration", "done", bins=len(oos_map),
                            top_bin_winrate=float(oos_map[-1, 1]),
                            bottom_bin_winrate=float(oos_map[0, 1]),
                        )
                    else:
                        self.logger.event("oos_calibration", "skipped", reason="insufficient_data")
            except Exception as e:
                self.logger.event("oos_calibration", "skipped", error=repr(e))

        except Exception as e:
            self.logger.event("walk_forward", "skipped", error=repr(e))

        recommended_config = None
        if cfg.run_grid_search:
            self.logger.event("grid_search", "start")
            gs = GridSearchRunner(self.lake).run(trade_daily, pred)
            recommended_config = GridSearchRunner(self.lake).write_recommended_config(cfg.__dict__, self.logger.dir / "recommended_config.json")
            self.logger.event("grid_search", "done", rows=len(gs), recommended_config=str(recommended_config))

        self.logger.event("report", "start")
        md = DailyReportGenerator().generate(
            pred,
            top_n=max(cfg.top_n, 20),
            backtest_metrics=self.lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=self.lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=self.lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=self.lake.read_parquet("quality", "daily_summary"),
            risk_tags=self.lake.read_parquet("analysis", "candidate_risk_tags"),
            explanations=self.lake.read_parquet("analysis", "candidate_explanations"),
            orders=self.lake.read_parquet("orders", "next_day_orders"),
            watchlist=self.lake.read_parquet("orders", "watchlist"),
        )
        html = HtmlReportGenerator().generate(
            pred,
            top_n=max(cfg.top_n, 20),
            backtest_metrics=self.lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=self.lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=self.lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=self.lake.read_parquet("quality", "daily_summary"),
            monthly=self.lake.read_parquet("backtest", "monthly_latest"),
            trades=self.lake.read_parquet("backtest", "trades_latest"),
            equity=self.lake.read_parquet("backtest", "equity_latest"),
            holdings=self.lake.read_parquet("backtest", "holdings_latest"),
            signal_stability=self.lake.read_parquet("analysis", "signal_stability"),
            turnover=self.lake.read_parquet("analysis", "turnover"),
            risk_tags=self.lake.read_parquet("analysis", "candidate_risk_tags"),
            explanations=self.lake.read_parquet("analysis", "candidate_explanations"),
            orders=self.lake.read_parquet("orders", "next_day_orders"),
            watchlist=self.lake.read_parquet("orders", "watchlist"),
        )
        archived_md = copy_if_exists(md, self.logger.dir / "reports" / Path(md).name) if cfg.archive_outputs else None
        archived_html = copy_if_exists(html, self.logger.dir / "reports" / Path(html).name) if cfg.archive_outputs else None
        archived_orders = copy_if_exists(self.lake.data_path("orders", "next_day_orders"), self.logger.dir / "orders" / "next_day_orders.csv") if cfg.archive_outputs else None
        archived_watchlist = copy_if_exists(self.lake.data_path("orders", "watchlist"), self.logger.dir / "orders" / "watchlist.csv") if cfg.archive_outputs else None
        archived_risk = copy_if_exists(self.lake.data_path("analysis", "candidate_risk_tags"), self.logger.dir / "analysis" / "candidate_risk_tags.csv") if cfg.archive_outputs else None
        archived_explain = copy_if_exists(self.lake.data_path("analysis", "candidate_explanations"), self.logger.dir / "analysis" / "candidate_explanations.csv") if cfg.archive_outputs else None
        self.logger.event("report", "done", markdown=str(md), html=str(html), archived_markdown=str(archived_md), archived_html=str(archived_html), archived_orders=str(archived_orders))
        summary_path = self.logger.write_summary(markdown=str(md), html=str(html), archived_markdown=str(archived_md), archived_html=str(archived_html), archived_orders=str(archived_orders), archived_watchlist=str(archived_watchlist), archived_risk=str(archived_risk), archived_explain=str(archived_explain), model=str(train_res.model_path) if train_res else "multi_strategy", recommended_config=str(recommended_config) if recommended_config else None)
        self.logger.event("pipeline", "done", summary=str(summary_path))
        return {"markdown": md, "html": html, "summary": summary_path, "run_id": self.logger.run_id}
