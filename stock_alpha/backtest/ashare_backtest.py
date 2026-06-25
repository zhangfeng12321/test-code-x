from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


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
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.sort_values(["code", "date"])
        if "limit_pct" not in bars.columns:
            from stock_alpha.features.risk_filters import enrich_trade_constraints
            bars = enrich_trade_constraints(bars)
        elif "pre_close" not in bars.columns:
            bars["pre_close"] = bars.groupby("code")["close"].shift(1)
        calendar = sorted(bars["date"].dropna().unique())
        pred = predictions.copy()
        pred["date"] = pd.to_datetime(pred["date"])

        cash = cfg.initial_cash
        positions: dict[str, dict] = {}
        trades = []
        equity_rows = []

        by_date_pred = {d: x for d, x in pred.groupby("date")}
        bar_idx = {(r.code, r.date): r for r in bars.itertuples(index=False)}

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
                signal_date = calendar[idx - 1]
                signals = by_date_pred.get(signal_date)
                if signals is not None:
                    signals = signals.sort_values(cfg.score_col, ascending=False)
                    if cfg.max_down_probability is not None and "down_probability" in signals.columns:
                        signals = signals[signals["down_probability"] <= cfg.max_down_probability]
                    if cfg.max_risk_score is not None and "risk_score" in signals.columns:
                        signals = signals[signals["risk_score"] <= cfg.max_risk_score]
                    if cfg.selection_mode == "topn":
                        signals = signals.head(cfg.top_n)
                    elif cfg.selection_mode == "quantile":
                        q = signals[cfg.score_col].quantile(cfg.score_quantile)
                        signals = signals[signals[cfg.score_col] >= q].head(cfg.top_n)
                    else:
                        signals = signals[signals[cfg.score_col] >= cfg.min_score].head(cfg.top_n)
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
