from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_alpha.backtest.ashare_backtest import AShareBacktester, AShareBacktestConfig
from stock_alpha.data.downloader import MarketDataDownloader
from stock_alpha.data.providers.akshare_provider import AkShareProvider
from stock_alpha.data.providers.baostock_provider import BaoStockProvider
from stock_alpha.data.providers.fallback_provider import FallbackMarketDataProvider
from stock_alpha.reports.daily_report import DailyReportGenerator
from stock_alpha.storage.cache import DataLake
from stock_alpha.training.train_v1 import V1Trainer
from stock_alpha.analysis_signal import signal_stability, turnover_by_date


def _ts() -> str:
    """返回当前时间戳，用于日志打印。"""
    return datetime.now().strftime('%H:%M:%S')


def provider(name: str):
    if name == "akshare":
        return AkShareProvider()
    if name == "baostock":
        return BaoStockProvider()
    if name == "fallback":
        return FallbackMarketDataProvider([AkShareProvider(), BaoStockProvider()])
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["init-config", "pipeline", "runs-index", "grid-search", "download-daily", "download-minute", "download-extra", "batch-download", "retry-failed", "quality-check", "train-v1", "backtest", "daily-report", "walk-forward", "real-smoke", "all-demo"])
    p.add_argument("--provider", default="akshare", choices=["akshare", "baostock", "fallback"])
    p.add_argument("--start", default="20240101")
    p.add_argument("--end", default="20241231")
    p.add_argument("--codes", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--period", default="5")
    p.add_argument("--data-root", default="data")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--hold-days", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.45)
    p.add_argument("--take-profit", type=float)
    p.add_argument("--stop-loss", type=float)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-end")
    p.add_argument("--valid-end")
    p.add_argument("--train-days", type=int, default=120)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--config")
    args = p.parse_args()

    if args.command == "init-config":
        from stock_alpha.config.settings import PipelineConfig
        path = args.config or "config/pipeline.example.json"
        print(PipelineConfig().to_file(path))
        return

    if args.command == "pipeline":
        from stock_alpha.config.settings import PipelineConfig
        from stock_alpha.runtime.pipeline import FullPipeline
        cfg = PipelineConfig.from_file(args.config) if args.config else PipelineConfig(
            provider=args.provider, data_root=args.data_root, start=args.start, end=args.end,
            codes=args.codes or [], limit=args.limit, batch_size=args.batch_size, top_n=args.top_n,
            train_end=args.train_end, valid_end=args.valid_end, train_days=args.train_days,
            test_days=args.test_days, step_days=args.step_days, force=args.force, minute_period=args.period
        )
        res = FullPipeline(cfg).run()
        print(res)
        return

    if args.command == "runs-index":
        from stock_alpha.reports.runs_index import write_runs_index
        print(write_runs_index(args.data_root))
        return

    lake = DataLake(args.data_root, Path(args.data_root) / "stock_alpha.duckdb")

    if args.command in ["batch-download", "retry-failed"]:
        from stock_alpha.data.batch import BatchDownloadRunner
        pr = provider(args.provider)
        dl = MarketDataDownloader(pr, lake)
        runner = BatchDownloadRunner(dl, batch_size=args.batch_size)
        codes = args.codes
        if args.command == "batch-download":
            if not codes:
                basic = dl.get_stock_universe(limit=args.limit)
                codes = basic["code"].tolist()
            # 排除北交所（8/4开头）+ ST股
            codes = [c for c in codes if not str(c).zfill(6).startswith(("8", "4"))]
            # 排除ST股（如果stock_basic可用）
            if not codes:
                pass
            elif 'basic' in dir() and not basic.empty and 'name' in basic.columns:
                st_codes = set(basic[basic['name'].astype(str).str.contains('ST', case=False, na=False)]['code'].astype(str).str.zfill(6))
                codes = [c for c in codes if str(c).zfill(6) not in st_codes]
            print(f"[{_ts()}] 预过滤后下载数量: {len(codes)}（已排除北交所+ST）")
            runner.download_daily_batches(codes, args.start, args.end, force=args.force)
        else:
            runner.retry_failed_daily(args.start, args.end, force=True)
        if hasattr(pr, "close"):
            pr.close()
        return

    if args.command == "download-extra":
        # 下载北向资金 + 龙虎榜 + 财务 + 融资融券 + 个股北向持股
        pr = provider(args.provider)
        dl = MarketDataDownloader(pr, lake)
        # 先获取股票列表（用于个股级数据下载）
        try:
            basic = dl.get_stock_universe(limit=args.limit)
            codes = basic["code"].tolist() if not basic.empty else []
        except Exception:
            codes = args.codes or []
        print(f"[{_ts()}] 下载北向资金数据: {args.start} ~ {args.end}")
        nb = dl.download_northbound_flow(args.start, args.end, force=args.force)
        print(f"[{_ts()}]   北向资金: {len(nb)} 行")
        print(f"[{_ts()}] 下载龙虎榜数据: {args.start} ~ {args.end}")
        lhb = dl.download_dragon_tiger(args.start, args.end, force=args.force)
        print(f"[{_ts()}]   龙虎榜: {len(lhb)} 行")
        if codes:
            print(f"[{_ts()}] 下载个股北向持股数据: {len(codes)} 只股票")
            nbs = dl.download_northbound_stock(codes, args.start, args.end, force=args.force)
            print(f"[{_ts()}]   个股北向持股: {len(nbs)} 行")
            print(f"[{_ts()}] 下载财务指标数据: {len(codes)} 只股票")
            fund = dl.download_fundamentals(codes, force=args.force)
            print(f"[{_ts()}]   财务指标: {len(fund)} 行")
        print(f"[{_ts()}] 下载融资融券数据: {args.start} ~ {args.end}")
        margin = dl.download_margin_data(args.start, args.end, force=args.force)
        print(f"[{_ts()}]   融资融券: {len(margin)} 行")
        if hasattr(pr, "close"):
            pr.close()
        return

    if args.command == "quality-check":
        from stock_alpha.data.quality import check_daily_quality, summarize_quality
        daily = V1Trainer(lake).load_daily(args.codes)
        issues = check_daily_quality(daily, args.start, args.end)
        lake.write_parquet("quality", "daily_issues", issues)
        lake.write_parquet("quality", "daily_summary", summarize_quality(issues))
        print(summarize_quality(issues).to_string(index=False))
        return

    if args.command == "real-smoke":
        args.codes = args.codes or ["600000", "000001", "300750"]
        args.start = args.start or "20240101"
        args.end = args.end or "20241231"
        pr = provider(args.provider)
        dl = MarketDataDownloader(pr, lake)
        dl.download_daily(args.codes, args.start, args.end, limit=args.limit, force=args.force)
        if hasattr(pr, "close"):
            pr.close()
        result = V1Trainer(lake).train(codes=args.codes, train_end=args.train_end, valid_end=args.valid_end)
        print(f"[{_ts()}] trained backend={result.backend} rows={result.rows} model={result.model_path}")
        daily = V1Trainer(lake).load_daily(args.codes)
        pred = lake.read_parquet("predictions", "v1_latest")
        bt = AShareBacktester(AShareBacktestConfig(top_n=args.top_n, hold_days=args.hold_days, min_score=args.min_score, take_profit=args.take_profit, stop_loss=args.stop_loss)).run(daily, pred)
        lake.write_parquet("backtest", "equity_latest", bt["equity"])
        lake.write_parquet("backtest", "trades_latest", bt["trades"])
        for k in ["metrics", "monthly", "yearly", "trade_stats"]:
            if k in bt:
                lake.write_parquet("backtest", f"{k}_latest", bt[k])
        print(bt["metrics"].to_string(index=False))
        pred_for_analysis = pred
        if lake.read_parquet("analysis", "signal_stability").empty and not pred_for_analysis.empty:
            lake.write_parquet("analysis", "signal_stability", signal_stability(pred_for_analysis, top_n=max(args.top_n, 20)))
            lake.write_parquet("analysis", "turnover", turnover_by_date(pred_for_analysis, top_n=max(args.top_n, 20)))
        path = DailyReportGenerator().generate(
            pred,
            top_n=max(args.top_n, 20),
            backtest_metrics=lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=lake.read_parquet("quality", "daily_summary"),
            risk_tags=lake.read_parquet("analysis", "candidate_risk_tags"),
            explanations=lake.read_parquet("analysis", "candidate_explanations"),
            orders=lake.read_parquet("orders", "next_day_orders"),
        )
        from stock_alpha.reports.html_report import HtmlReportGenerator
        html = HtmlReportGenerator().generate(
            pred,
            top_n=max(args.top_n, 20),
            backtest_metrics=lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=lake.read_parquet("quality", "daily_summary"),
            monthly=lake.read_parquet("backtest", "monthly_latest"),
            trades=lake.read_parquet("backtest", "trades_latest"),
            equity=lake.read_parquet("backtest", "equity_latest"),
            holdings=lake.read_parquet("backtest", "holdings_latest"),
            signal_stability=lake.read_parquet("analysis", "signal_stability"),
            turnover=lake.read_parquet("analysis", "turnover"),
        )
        print(f"[{_ts()}] report={path}")
        print(f"[{_ts()}] html={html}")
        return

    if args.command in ["download-daily", "download-minute", "all-demo"]:
        pr = provider(args.provider)
        dl = MarketDataDownloader(pr, lake)
        codes = args.codes
        if not codes:
            try:
                basic = dl.get_stock_universe(limit=args.limit)
                codes = basic["code"].tolist()
            except Exception:
                codes = ["600000", "000001"]
        if args.command in ["download-daily", "all-demo"]:
            dl.download_daily(codes, args.start, args.end, limit=args.limit, force=args.force)
        if args.command in ["download-minute", "all-demo"]:
            dl.download_minute(codes, args.start, args.end, period=args.period, limit=args.limit, force=args.force)
        if hasattr(pr, "close"):
            pr.close()
        if args.command != "all-demo":
            return

    trainer = V1Trainer(lake)
    if args.command in ["train-v1", "all-demo"]:
        result = trainer.train(codes=args.codes, train_end=args.train_end, valid_end=args.valid_end)
        print(f"[{_ts()}] trained backend={result.backend} rows={result.rows} model={result.model_path} predictions={result.predictions_path}")
        if args.command != "all-demo":
            return

    if args.command == "grid-search":
        from stock_alpha.optimization.grid_search import GridSearchRunner
        daily = trainer.load_daily(args.codes)
        pred = lake.read_parquet("predictions", "v1_latest")
        runner = GridSearchRunner(lake)
        out = runner.run(daily, pred)
        rec = runner.write_recommended_config({"provider": args.provider, "data_root": args.data_root, "start": args.start, "end": args.end, "codes": args.codes or [], "limit": args.limit, "batch_size": args.batch_size}, Path(args.data_root) / "optimization" / "recommended_config.json")
        print(out.head(20).to_string(index=False))
        print(f"[{_ts()}] recommended_config={rec}")
        return

    if args.command == "walk-forward":
        from stock_alpha.training.walk_forward import WalkForwardRunner
        daily = trainer.load_daily(args.codes)
        res = WalkForwardRunner(lake, train_days=args.train_days, test_days=args.test_days, step_days=args.step_days).run(daily)
        print(f"[{_ts()}] walk_forward windows={len(res['windows'])} predictions={len(res['predictions'])}")
        return

    if args.command in ["backtest", "all-demo"]:
        daily = trainer.load_daily(args.codes)
        pred = lake.read_parquet("predictions", "v1_latest")
        bt = AShareBacktester(AShareBacktestConfig(top_n=args.top_n, hold_days=args.hold_days, min_score=args.min_score, take_profit=args.take_profit, stop_loss=args.stop_loss)).run(daily, pred)
        lake.write_parquet("backtest", "equity_latest", bt["equity"])
        lake.write_parquet("backtest", "trades_latest", bt["trades"])
        for k in ["metrics", "monthly", "yearly", "trade_stats"]:
            if k in bt:
                lake.write_parquet("backtest", f"{k}_latest", bt[k])
        print(bt["metrics"].to_string(index=False))
        if args.command != "all-demo":
            return

    if args.command in ["daily-report", "all-demo"]:
        pred = lake.read_parquet("predictions", "v1_latest")
        pred_for_analysis = pred
        if lake.read_parquet("analysis", "signal_stability").empty and not pred_for_analysis.empty:
            lake.write_parquet("analysis", "signal_stability", signal_stability(pred_for_analysis, top_n=max(args.top_n, 20)))
            lake.write_parquet("analysis", "turnover", turnover_by_date(pred_for_analysis, top_n=max(args.top_n, 20)))
        path = DailyReportGenerator().generate(
            pred,
            top_n=max(args.top_n, 20),
            backtest_metrics=lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=lake.read_parquet("quality", "daily_summary"),
            risk_tags=lake.read_parquet("analysis", "candidate_risk_tags"),
            explanations=lake.read_parquet("analysis", "candidate_explanations"),
            orders=lake.read_parquet("orders", "next_day_orders"),
        )
        from stock_alpha.reports.html_report import HtmlReportGenerator
        html = HtmlReportGenerator().generate(
            pred,
            top_n=max(args.top_n, 20),
            backtest_metrics=lake.read_parquet("backtest", "metrics_latest"),
            trade_stats=lake.read_parquet("backtest", "trade_stats_latest"),
            feature_importance=lake.read_parquet("evaluation", "feature_importance"),
            quality_summary=lake.read_parquet("quality", "daily_summary"),
            monthly=lake.read_parquet("backtest", "monthly_latest"),
            trades=lake.read_parquet("backtest", "trades_latest"),
            equity=lake.read_parquet("backtest", "equity_latest"),
            holdings=lake.read_parquet("backtest", "holdings_latest"),
            signal_stability=lake.read_parquet("analysis", "signal_stability"),
            turnover=lake.read_parquet("analysis", "turnover"),
        )
        print(f"[{_ts()}] report={path}")
        print(f"[{_ts()}] html={html}")


if __name__ == "__main__":
    main()
