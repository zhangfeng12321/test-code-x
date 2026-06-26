from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from stock_alpha.risk_rules import RiskRuleConfig, apply_hard_risk_filters, daily_risk_features


@dataclass
class AShareBacktestConfig:
    top_n: int = 5
    hold_days: int = 3
    initial_cash: float = 1_000_000.0
    buy_fee: float = 0.0003
    sell_fee: float = 0.0013  # commission + stamp duty rough estimate
    slippage: float = 0.001
    lot_size: int = 100
    max_position_pct: float = 0.2
    take_profit: float | None = None
    stop_loss: float | None = None
    score_col: str = "final_score"
    selection_mode: str = "threshold"  # threshold/topn/quantile
    score_quantile: float = 0.95
    max_down_probability: float | None = None
    max_risk_score: float | None = None
    max_daily_buys: int | None = None
    min_score: float = 0.45
    # --- 市场级风控 ---
    market_crash_threshold: float = -0.02   # 全市场均跌超此值暂停买入
    max_down_limit_count: int = 100         # 跌停家数超此值暂停
    portfolio_drawdown_pause: float = 0.10  # 组合5日回撤超此值暂停
    # --- 行业集中度 ---
    max_sector_pct: float = 0.40  # 同板块最大仓位占比
    # --- 连亏熔断 ---
    max_consecutive_loss_pause: int = 6    # 连续亏损N次暂停
    pause_days_after_streak: int = 5       # 暂停天数


def _is_limit_up(open_price: float, pre_close: float, limit_pct: float = 0.10) -> bool:
    return pre_close > 0 and open_price / pre_close - 1 >= (limit_pct - 0.002)


def _is_limit_down(open_price: float, pre_close: float, limit_pct: float = 0.10) -> bool:
    return pre_close > 0 and open_price / pre_close - 1 <= (-limit_pct + 0.002)


class AShareBacktester:
    """A 股简化撮合回测：T+1 买入、固定持有、涨跌停过滤、手续费/滑点/100股。"""

    def __init__(self, config: Optional[AShareBacktestConfig] = None):
        self.config = config or AShareBacktestConfig()

    def run(self, daily: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        cfg = self.config
        bars = daily.copy()
        bars["date"] = pd.to_datetime(bars["date"], format="mixed")
        bars = bars.sort_values(["code", "date"])
        if "limit_pct" not in bars.columns:
            from stock_alpha.features.risk_filters import enrich_trade_constraints
            bars = enrich_trade_constraints(bars)
        elif "pre_close" not in bars.columns:
            bars["pre_close"] = bars.groupby("code")["close"].shift(1)
        calendar = sorted(bars["date"].dropna().unique())
        pred = predictions.copy()
        pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        pred["date"] = pd.to_datetime(pred["date"], format="mixed")
        risk_ctx = daily_risk_features(bars)
        if not risk_ctx.empty:
            pred = pred.merge(risk_ctx, on=["code", "date"], how="left", suffixes=("", "_daily"))

        cash = cfg.initial_cash
        positions: dict[str, dict] = {}
        trades = []
        equity_rows = []
        # 风控状态
        consecutive_losses = 0
        pause_until_idx = -1  # 熔断暂停到第N个 calendar idx

        by_date_pred = {d: x for d, x in pred.groupby("date")}
        bar_idx = {(r.code, r.date): r for r in bars.itertuples(index=False)}

        # 预计算每日市场状态（用于市场熔断判断）
        market_daily = bars.groupby("date").agg(
            market_ret=("pct_chg", "mean") if "pct_chg" in bars.columns else ("close", "count"),
            down_limit_count=("is_limit_down", "sum") if "is_limit_down" in bars.columns else ("close", "count"),
        ).reset_index() if "pct_chg" in bars.columns else pd.DataFrame()
        if not market_daily.empty and "market_ret" in market_daily.columns:
            # pct_chg 可能是百分比形式
            if market_daily["market_ret"].abs().median() > 1:
                market_daily["market_ret"] = market_daily["market_ret"] / 100.0
            market_state = {r.date: r for r in market_daily.itertuples(index=False)}
        else:
            market_state = {}

        for idx, d in enumerate(calendar):
            # 先卖：止盈/止损/持有期到期。保守假设同日止盈止损均触发时先止损。
            for code in list(positions.keys()):
                pos = positions[code]
                bar = bar_idx.get((code, d))
                if bar is None:
                    continue
                sell_reason = None
                target_price = None
                if cfg.stop_loss is not None and bar.low <= pos["buy_price"] * (1 - cfg.stop_loss):
                    sell_reason = "stop_loss"
                    target_price = pos["buy_price"] * (1 - cfg.stop_loss)
                elif cfg.take_profit is not None and bar.high >= pos["buy_price"] * (1 + cfg.take_profit):
                    sell_reason = "take_profit"
                    target_price = pos["buy_price"] * (1 + cfg.take_profit)
                elif idx - pos["buy_idx"] >= cfg.hold_days:
                    sell_reason = "hold_days"
                    target_price = bar.open
                if sell_reason:
                    if _is_limit_down(bar.open, bar.pre_close, getattr(bar, "limit_pct", 0.10)):
                        continue
                    price = target_price * (1 - cfg.slippage)
                    amount = pos["shares"] * price
                    fee = amount * cfg.sell_fee
                    cash += amount - fee
                    trades.append({"date": d, "code": code, "side": "SELL", "price": price, "shares": pos["shares"], "amount": amount, "fee": fee, "reason": sell_reason})
                    del positions[code]

            # 再买：使用前一交易日收盘后信号，下一交易日开盘买
            if idx > 0:
                # === 市场级风控检查 ===
                buy_paused = False
                # 1. 连亏熔断
                if idx <= pause_until_idx:
                    buy_paused = True
                # 2. 市场熔断
                if not buy_paused and market_state:
                    ms = market_state.get(d)
                    if ms is not None:
                        if hasattr(ms, "market_ret") and ms.market_ret < cfg.market_crash_threshold:
                            buy_paused = True
                        if hasattr(ms, "down_limit_count") and ms.down_limit_count > cfg.max_down_limit_count:
                            buy_paused = True
                # 3. 组合回撤暂停
                if not buy_paused and len(equity_rows) >= 5:
                    recent_eq = [r["equity"] for r in equity_rows[-5:]]
                    peak = max(recent_eq)
                    if peak > 0 and (recent_eq[-1] / peak - 1) < -cfg.portfolio_drawdown_pause:
                        buy_paused = True

                if not buy_paused:
                    signal_date = calendar[idx - 1]
                    signals = by_date_pred.get(signal_date)
                    if signals is not None:
                        signals = signals.sort_values(cfg.score_col, ascending=False)
                        risk_cfg = RiskRuleConfig(
                            min_score=cfg.min_score,
                            max_down_probability=cfg.max_down_probability,
                            max_risk_score=cfg.max_risk_score,
                            require_up_gt_down=True,
                        )
                        signals = apply_hard_risk_filters(signals, risk_cfg, score_col=cfg.score_col)
                        signals = signals[~signals["risk_blocked"].fillna(False)]
                        # topn 只截断已通过硬门槛的信号
                        if cfg.selection_mode == "quantile" and not signals.empty:
                            q = signals[cfg.score_col].quantile(cfg.score_quantile)
                            signals = signals[signals[cfg.score_col] >= q]
                        signals = signals.head(cfg.top_n)
                        slots = max(cfg.top_n - len(positions), 0)
                        if cfg.max_daily_buys is not None:
                            slots = min(slots, cfg.max_daily_buys)
                        for r in signals.itertuples(index=False):
                            if slots <= 0 or r.code in positions:
                                continue
                            bar = bar_idx.get((r.code, d))
                            if bar is None or pd.isna(bar.open) or pd.isna(bar.pre_close):
                                continue
                            if _is_limit_up(bar.open, bar.pre_close, getattr(bar, "limit_pct", 0.10)):
                                continue
                            # === 行业集中度控制 ===
                            sector = r.code[:3]
                            if positions and cfg.max_sector_pct < 1.0:
                                same_sector = sum(1 for c in positions if c[:3] == sector)
                                if same_sector / max(len(positions) + 1, 1) >= cfg.max_sector_pct:
                                    continue
                            budget = min(cash * cfg.max_position_pct, cfg.initial_cash * cfg.max_position_pct)
                            price = bar.open * (1 + cfg.slippage)
                            shares = int((budget / price) // cfg.lot_size * cfg.lot_size)
                            if shares <= 0:
                                continue
                            amount = shares * price
                            fee = amount * cfg.buy_fee
                            if amount + fee > cash:
                                continue
                            cash -= amount + fee
                            positions[r.code] = {"shares": shares, "buy_price": price, "buy_idx": idx, "buy_date": d}
                            trades.append({"date": d, "code": r.code, "side": "BUY", "price": price, "shares": shares, "amount": amount, "fee": fee, "reason": "signal"})
                            slots -= 1

            market_value = 0.0
            for code, pos in positions.items():
                bar = bar_idx.get((code, d))
                if bar is not None and not pd.isna(bar.close):
                    market_value += pos["shares"] * bar.close
            equity = cash + market_value
            equity_rows.append({"date": d, "cash": cash, "market_value": market_value, "equity": equity, "positions": len(positions)})

        equity = pd.DataFrame(equity_rows)
        trades_df = pd.DataFrame(trades)
        metrics = self.metrics(equity, trades_df)
        try:
            from stock_alpha.backtest.metrics import enrich_backtest_metrics
            extra = enrich_backtest_metrics(equity, trades_df)
        except Exception:
            extra = {}
        try:
            from stock_alpha.backtest.holdings import build_holdings_snapshots
            holdings = build_holdings_snapshots(equity, trades_df, daily)
        except Exception:
            holdings = pd.DataFrame()
        return {"equity": equity, "trades": trades_df, "holdings": holdings, "metrics": metrics, **extra}

    @staticmethod
    def metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame()
        eq = equity.copy()
        eq["ret"] = eq["equity"].pct_change().fillna(0)
        total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        roll_max = eq["equity"].cummax()
        max_dd = (eq["equity"] / roll_max - 1).min()
        ann_ret = (1 + total_return) ** (252 / max(len(eq), 1)) - 1
        sharpe = np.sqrt(252) * eq["ret"].mean() / (eq["ret"].std() + 1e-12)
        return pd.DataFrame([{
            "total_return": total_return,
            "annual_return": ann_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "trade_count": len(trades),
            "final_equity": eq["equity"].iloc[-1],
        }])
