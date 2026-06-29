from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from stock_alpha.backtest.ashare_backtest import AShareBacktester, AShareBacktestConfig
from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.labels.triple_barrier import make_triple_barrier_labels
from stock_alpha.labels.ranking_label import make_ranking_labels
from stock_alpha.models.v1_daily_model import V1DailyAlphaModel
from stock_alpha.models.v2_ranker_model import V2RankerModel
from stock_alpha.models.ensemble_model import EnsembleModel
from stock_alpha.storage.cache import DataLake


@dataclass
class WalkForwardRunner:
    lake: DataLake
    train_days: int = 120
    test_days: int = 30
    step_days: int = 30
    purge_days: int = 5  # 训练集和测试集之间的“净化间隔”，避免标签泄漏
    backtest_config: AShareBacktestConfig | None = None
    label_profit_take: float = 0.03
    label_stop_loss: float = 0.02
    label_horizon: int = 3
    model_type: str = "ranker"  # "classifier" / "ranker" / "ensemble"
    ensemble_alpha: float = 0.6
    wf_min_score: float | None = None  # Walk-Forward 专用 min_score，None=使用自适应 topn 模式

    def run(self, daily: pd.DataFrame, score_col: str = "final_score") -> dict[str, pd.DataFrame]:
        """Walk-Forward 滚动窗口验证。

        严格性保证：
        1. Purge Gap: 训练集和测试集之间留 purge_days 间隔，避免标签前视泄漏
        2. 严格时间切分：模型只能看到训练集内数据
        3. 多模型支持：支持 classifier/ranker/ensemble
        4. 窗口级指标：每个窗口单独统计预测准确度
        """
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], format="mixed")
        daily["code"] = daily["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        dates = sorted(daily["date"].dropna().unique())

        min_required = self.train_days + self.purge_days + self.test_days
        if len(dates) < min_required:
            raise RuntimeError(f"not enough dates for walk-forward: need {min_required}, got {len(dates)}")

        # 加载外部数据
        northbound_flow = self.lake.read_parquet("northbound", "daily_flow")
        northbound_stock = self.lake.read_parquet("northbound", "stock_holdings")
        lhb_data = self.lake.read_parquet("dragon_tiger", "lhb_detail")
        stock_basic = self.lake.read_parquet("meta", "stock_basic")
        financial_data = self.lake.read_parquet("fundamental", "financial_indicators")
        margin_data = self.lake.read_parquet("margin", "daily_summary")

        # 构建全量特征（只做特征计算，不涉及标签）
        features = build_daily_features(
            daily,
            northbound_flow=northbound_flow if not northbound_flow.empty else None,
            northbound_stock=northbound_stock if not northbound_stock.empty else None,
            lhb_data=lhb_data if not lhb_data.empty else None,
            stock_basic=stock_basic if not stock_basic.empty else None,
            financial_data=financial_data if not financial_data.empty else None,
            margin_data=margin_data if not margin_data.empty else None,
        )

        # 生成标签
        print(f"[walk_forward] features built: {len(features)} rows, {features['code'].nunique()} codes, "
              f"{len([c for c in features.columns if c not in ('code','date')])} feature cols", flush=True)
        ranking_labels = make_ranking_labels(daily, horizon=self.label_horizon)
        classification_labels = make_triple_barrier_labels(
            daily, profit_take=self.label_profit_take, stop_loss=self.label_stop_loss, horizon=self.label_horizon
        )

        all_preds = []
        windows = []
        start_idx = 0

        # 预计算总窗口数
        _total_windows = 0
        _tmp_idx = 0
        while _tmp_idx + min_required <= len(dates):
            _total_windows += 1
            _tmp_idx += self.step_days
        print(f"[walk_forward] {_total_windows} windows, "
              f"train={self.train_days}d test={self.test_days}d step={self.step_days}d "
              f"purge={self.purge_days}d model={self.model_type} "
              f"threads=1 (sequential)", flush=True)
        _wf_t0 = time.time()

        while start_idx + min_required <= len(dates):
            train_start = pd.Timestamp(dates[start_idx])
            train_end = pd.Timestamp(dates[start_idx + self.train_days - 1])
            # Purge gap: 跳过 purge_days
            test_start_idx = start_idx + self.train_days + self.purge_days
            test_end_idx = test_start_idx + self.test_days - 1
            if test_end_idx >= len(dates):
                break
            test_start = pd.Timestamp(dates[test_start_idx])
            test_end = pd.Timestamp(dates[test_end_idx])

            # 严格切分：训练集只包含 train_start ~ train_end
            train_f = features[(features["date"] >= train_start) & (features["date"] <= train_end)]
            test_f = features[(features["date"] >= test_start) & (features["date"] <= test_end)]

            if train_f.empty or test_f.empty:
                start_idx += self.step_days
                continue

            # 根据 model_type 训练
            model = self._train_model(train_f, ranking_labels, classification_labels, train_start, train_end)
            pred = model.predict(test_f)

            # 记录窗口元数据
            pred["wf_train_start"] = train_start
            pred["wf_train_end"] = train_end
            pred["wf_test_start"] = test_start
            pred["wf_test_end"] = test_end
            all_preds.append(pred)

            _wf_elapsed = time.time() - _wf_t0
            _wf_done = len(windows) + 1
            print(f"[walk_forward] window {_wf_done}/{_total_windows} "
                  f"train={train_start.strftime('%Y%m%d')}~{train_end.strftime('%Y%m%d')} "
                  f"test={test_start.strftime('%Y%m%d')}~{test_end.strftime('%Y%m%d')} "
                  f"pred={len(pred)} rows, {_wf_elapsed:.1f}s elapsed", flush=True)

            # 窗口级指标
            window_info = {
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
                "purge_days": self.purge_days,
                "train_rows": len(train_f), "test_rows": len(test_f),
                "pred_rows": len(pred),
                "backend": model.backend,
                "model_type": self.model_type,
            }
            # 计算窗口级预测准确度（与实际收益对比）
            window_info.update(self._window_metrics(pred, test_f, daily, test_start, test_end))
            windows.append(window_info)

            start_idx += self.step_days

        _wf_total_time = time.time() - _wf_t0
        print(f"[walk_forward] done: {len(all_preds)} windows, "
              f"{_wf_total_time:.1f}s total, "
              f"{_wf_total_time/max(len(all_preds),1):.1f}s/window", flush=True)

        predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
        windows_df = pd.DataFrame(windows)

        if not predictions.empty:
            self.lake.write_parquet("predictions", "walk_forward", predictions)
            base_cfg = self.backtest_config or AShareBacktestConfig(score_col=score_col)
            # Walk-Forward 使用自适应阈值：降低 min_score，依赖 topn 选股
            # 小窗口模型的分数分布远低于全量模型，不能用相同的绝对门槛
            # 关闭严格风控过滤，仅依赖 topn 排序选股，确保足够交易量
            wf_bt_cfg = AShareBacktestConfig(
                top_n=base_cfg.top_n,
                hold_days=base_cfg.hold_days,
                initial_cash=base_cfg.initial_cash,
                buy_fee=base_cfg.buy_fee,
                sell_fee=base_cfg.sell_fee,
                slippage=base_cfg.slippage,
                lot_size=base_cfg.lot_size,
                max_position_pct=base_cfg.max_position_pct,
                take_profit=base_cfg.take_profit,
                stop_loss=base_cfg.stop_loss,
                score_col=score_col,
                selection_mode="topn",
                min_score=self.wf_min_score if self.wf_min_score is not None else 0.0,
                max_down_probability=None,  # Walk-Forward 不用绝对概率门槛
                max_risk_score=None,
                max_daily_buys=base_cfg.max_daily_buys,
                require_up_gt_down=False,  # WF不要求上涨>下跌，仅看topn排序
                enable_atr_filter=False,    # WF不用ATR过滤，避免过度筛选
                enable_weak_rebound_filter=False,  # WF不用弱势反抽过滤
            )
            bt = AShareBacktester(wf_bt_cfg).run(daily, predictions)
            for k, v in bt.items():
                self.lake.write_parquet("walk_forward", f"{k}", v)
        self.lake.write_parquet("walk_forward", "windows", windows_df)

        # 汇总指标
        summary = self._overall_summary(windows_df, predictions)
        self.lake.write_parquet("walk_forward", "oos_summary", summary)

        return {"predictions": predictions, "windows": windows_df, "summary": summary}

    def _train_model(self, train_f, ranking_labels, classification_labels, train_start, train_end):
        """根据 model_type 训练模型，自动排除无效特征。"""
        from stock_alpha.features.v1_daily import V1_FEATURE_COLUMNS

        # 特征预筛选：排除全 NaN / 零方差特征
        available_cols = [c for c in V1_FEATURE_COLUMNS if c in train_f.columns]
        train_subset = train_f[available_cols].apply(pd.to_numeric, errors="coerce")
        valid_cols = []
        for c in available_cols:
            col = train_subset[c]
            if col.notna().sum() > len(col) * 0.1 and col.nunique() > 1:
                valid_cols.append(c)

        train_rank_l = ranking_labels[
            (ranking_labels["date"] >= train_start) & (ranking_labels["date"] <= train_end)
        ]
        train_cls_l = classification_labels[
            (classification_labels["date"] >= train_start) & (classification_labels["date"] <= train_end)
        ]

        if self.model_type == "ensemble":
            model = EnsembleModel(alpha=self.ensemble_alpha)
            model.feature_columns = valid_cols
            model.ranker.feature_columns = valid_cols
            model.classifier.feature_columns = valid_cols
            model.fit(train_f, train_rank_l, train_cls_l)
        elif self.model_type == "ranker":
            model = V2RankerModel()
            model.feature_columns = valid_cols
            model.fit(train_f, train_rank_l)
        else:
            model = V1DailyAlphaModel()
            model.feature_columns = valid_cols
            model.fit(train_f, train_cls_l)
        return model

    def _window_metrics(self, pred, test_f, daily, test_start, test_end) -> dict:
        """计算单窗口预测准确度指标。"""
        metrics = {}
        if pred.empty:
            return metrics

        # Top-N 股票的实际未来收益
        top_n = min(5, len(pred["code"].unique()))
        # 每天的 top 股票
        pred_sorted = pred.sort_values(["date", "final_score"], ascending=[True, False])
        top_per_day = pred_sorted.groupby("date").head(top_n)

        # 计算这些股票在测试期内的实际收益
        test_daily = daily[(daily["date"] >= test_start) & (daily["date"] <= test_end)].copy()
        if not test_daily.empty:
            # 计算每只股票 horizon 日后的收益
            fwd_rets = []
            for code, grp in test_daily.groupby("code"):
                grp = grp.sort_values("date")
                grp["fwd_ret"] = grp["close"].pct_change(self.label_horizon).shift(-self.label_horizon)
                fwd_rets.append(grp[["code", "date", "fwd_ret"]])
            fwd_df = pd.concat(fwd_rets, ignore_index=True) if fwd_rets else pd.DataFrame()

            if not fwd_df.empty:
                top_with_ret = top_per_day.merge(fwd_df, on=["code", "date"], how="left")
                metrics["top_n_avg_ret"] = top_with_ret["fwd_ret"].mean()
                metrics["top_n_win_rate"] = (top_with_ret["fwd_ret"] > 0).mean()
                # 全样本平均收益（benchmark）
                all_with_ret = pred.merge(fwd_df, on=["code", "date"], how="left")
                metrics["all_avg_ret"] = all_with_ret["fwd_ret"].mean()
                # Top vs All 超额
                metrics["excess_ret"] = metrics.get("top_n_avg_ret", 0) - metrics.get("all_avg_ret", 0)

        return metrics

    def _overall_summary(self, windows_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """汇总所有窗口的样本外指标。"""
        if windows_df.empty:
            return pd.DataFrame()

        summary = {
            "total_windows": len(windows_df),
            "total_test_rows": int(windows_df["pred_rows"].sum()) if "pred_rows" in windows_df.columns else 0,
            "model_type": self.model_type,
            "purge_days": self.purge_days,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
        }

        # 汇总窗口级指标
        for metric in ["top_n_avg_ret", "top_n_win_rate", "excess_ret", "all_avg_ret"]:
            if metric in windows_df.columns:
                values = windows_df[metric].dropna()
                if not values.empty:
                    summary[f"{metric}_mean"] = values.mean()
                    summary[f"{metric}_std"] = values.std()
                    summary[f"{metric}_min"] = values.min()
                    summary[f"{metric}_max"] = values.max()

        return pd.DataFrame([summary])
