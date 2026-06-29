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
    # --- 真实交易成本建模 ---
    volume_participation_limit: float = 0.10  # 买入股数不超过20日均量的此比例
    dynamic_slippage: bool = True  # 是否启用动态滑点（基于成交量比例）
    base_slippage: float = 0.001  # 基础滑点（A股短线实际基础滑点）
    impact_coefficient: float = 0.3  # 冲击成本系数（participation_rate * coeff = 额外滑点）
    # --- 信号集中度衰减 ---
    max_consecutive_select: int = 20  # 连续入选超过 N 天不再买入，防止模型偏好固化
    # --- 风控规则开关（Walk-Forward 可关闭严格过滤）---
    require_up_gt_down: bool = True  # 是否要求 up_prob > down_prob
    enable_atr_filter: bool = True   # 是否启用 ATR 过高过滤
    enable_weak_rebound_filter: bool = True  # 是否启用弱势反抽过滤
    # --- ATR 动态止损/止盈 ---
    use_atr_stop: bool = False       # 是否启用 ATR 动态止损
    atr_stop_multiplier: float = 2.0  # 止损 = 买入价 × (1 - ATR_14 × multiplier)
    use_atr_profit: bool = False      # 是否启用 ATR 动态止盈
    atr_profit_multiplier: float = 3.0  # 止盈 = 买入价 × (1 + ATR_14 × multiplier)
    # --- 移动止损（Trailing Stop）---
    use_trailing_stop: bool = False   # 是否启用移动止损
    trailing_stop_pct: float = 0.05  # 止损 = 持仓期间最高价 × (1 - pct)
    # --- 分数动态仓位 ---
    score_based_sizing: bool = False  # 是否根据模型分数动态调整仓位
    score_high_threshold: float = 0.98  # 分数 >= 此值时，仓位乘以 1.5
    score_mid_threshold: float = 0.90   # 分数 >= 此值时，仓位 x 1.0；低于此值，仓位 x 0.5
    rank_based_sizing: bool = False  # 按当日选中内排名调整仓位：第1名多仓，后几名少仓
    # --- 市场状态过滤（Bull Market Filter）---
    use_bull_filter: bool = False       # 开启市场状态过滤
    bull_breadth_days: int = 20         # 计算全市场近 N 日等权涨幅
    bull_breadth_pause: float = 0.12    # 近 N 日市场涨幅 > 此阈值时暂停新开仓（12%防止误报）
    bull_ma_days: int = 60              # MA过滤：全市场均价 > MA_N，认为市场备战 bullish


def _calc_dynamic_slippage(shares: int, avg_volume_20: float, cfg: AShareBacktestConfig) -> float:
    """动态滑点：基于成交量占比计算实际滑点。

    原理：买入股数占日均成交量比例越大，冲击成本越高。
    slippage = base_slippage + participation_rate * impact_coefficient
    """
    if not cfg.dynamic_slippage or avg_volume_20 <= 0:
        return cfg.slippage
    participation_rate = shares / avg_volume_20
    return cfg.base_slippage + participation_rate * cfg.impact_coefficient


def _volume_constrained_shares(budget: float, price: float, avg_volume_20: float, cfg: AShareBacktestConfig) -> int:
    """成交量约束：买入股数不超过日均量的一定比例。

    避免小盘股买入量超过其日常成交量，导致实际无法成交。
    """
    import math
    # 健壮性防护：NaN/inf/非正值一律返回 0
    try:
        if not (price > 0) or not (budget > 0):
            return 0
        if math.isnan(price) or math.isnan(budget) or math.isinf(price) or math.isinf(budget):
            return 0
    except (TypeError, ValueError):
        return 0
    # 基于资金的理论股数
    raw = (budget / price) // cfg.lot_size * cfg.lot_size
    if math.isnan(raw) or math.isinf(raw) or raw <= 0:
        return 0
    max_by_budget = int(raw)
    # 基于成交量的上限
    if avg_volume_20 > 0 and cfg.volume_participation_limit > 0:
        vol_raw = (avg_volume_20 * cfg.volume_participation_limit) // cfg.lot_size * cfg.lot_size
        if math.isnan(vol_raw) or math.isinf(vol_raw) or vol_raw <= 0:
            return max_by_budget
        max_by_volume = int(vol_raw)
        return min(max_by_budget, max_by_volume)
    return max_by_budget


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
        # 构建行业映射：优先用 industry 字段，fallback 用代码前缀区分板块
        industry_map: dict[str, str] = {}
        if "industry" in bars.columns:
            _ind = bars.dropna(subset=["industry"]).drop_duplicates("code")[["code", "industry"]]
            industry_map = dict(zip(_ind["code"], _ind["industry"]))
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
        # 信号集中度跟踪：记录每只股票连续入选天数
        consecutive_select: dict[str, int] = {}

        by_date_pred = {d: x for d, x in pred.groupby("date")}
        bar_idx = {(r.code, r.date): r for r in bars.itertuples(index=False)}

        # 预计算每日市场状态（用于市场熔断判断）
        if "pct_chg" in bars.columns:
            agg_dict: dict = {"market_ret": ("pct_chg", "mean")}
            if "is_limit_down" in bars.columns:
                agg_dict["down_limit_count"] = ("is_limit_down", "sum")
            else:
                bars["_zero"] = 0
                agg_dict["down_limit_count"] = ("_zero", "sum")
            market_daily = bars.groupby("date").agg(**agg_dict).reset_index()
        else:
            market_daily = pd.DataFrame()
        if not market_daily.empty and "market_ret" in market_daily.columns:
            # pct_chg 可能是百分比形式
            if market_daily["market_ret"].abs().median() > 1:
                market_daily["market_ret"] = market_daily["market_ret"] / 100.0
            market_state = {r.date: r for r in market_daily.itertuples(index=False)}
        else:
            market_state = {}

        # === 预计算 Bull Market Filter 所需的滚动指标 ===
        bull_regime: dict = {}  # date -> bool（True=牛市状态，暂停新买入）
        if cfg.use_bull_filter and not market_daily.empty:
            md = market_daily.sort_values("date").copy()
            # 1. 滚动 N 日等权累计涨幅
            md["rolling_ret"] = (1 + md["market_ret"]).rolling(cfg.bull_breadth_days).apply(
                lambda x: x.prod() - 1, raw=True
            )
            # 2. 全市场均价 MA(N)
            if "close" in bars.columns:
                avg_close = bars.groupby("date")["close"].mean().reset_index()
                avg_close.columns = ["date", "avg_close"]
                avg_close = avg_close.sort_values("date")
                avg_close["avg_close_ma"] = avg_close["avg_close"].rolling(cfg.bull_ma_days).mean()
                md = md.merge(avg_close, on="date", how="left")
            else:
                md["avg_close"] = None; md["avg_close_ma"] = None
            # 3. 合并两个条件
            for r in md.itertuples(index=False):
                breadth_bull = (hasattr(r, "rolling_ret") and r.rolling_ret is not None
                                and not pd.isna(r.rolling_ret)
                                and r.rolling_ret > cfg.bull_breadth_pause)
                ma_bull = (hasattr(r, "avg_close") and r.avg_close is not None
                           and hasattr(r, "avg_close_ma") and r.avg_close_ma is not None
                           and not pd.isna(r.avg_close) and not pd.isna(r.avg_close_ma)
                           and r.avg_close > r.avg_close_ma)
                # 两个条件同时满足才认为是牛市（更严格，减少误报）
                bull_regime[r.date] = breadth_bull and ma_bull

        for idx, d in enumerate(calendar):
            # 先卖：止盈/止损/持有期到期。保守假设同日止盈止损均触发时先止损。
            for code in list(positions.keys()):
                pos = positions[code]
                bar = bar_idx.get((code, d))
                if bar is None:
                    continue
                sell_reason = None
                target_price = None
                # === 移动止损：更新持仓期间最高价（每日更新）===
                if cfg.use_trailing_stop and not pd.isna(bar.high):
                    pos["high_watermark"] = max(pos.get("high_watermark", pos["buy_price"]), bar.high)
                    trailing_stop_price = pos["high_watermark"] * (1 - cfg.trailing_stop_pct)
                else:
                    trailing_stop_price = None
                # === 计算止损价：移动止损 > ATR 动态 > 固定比例（优先级递减）===
                if cfg.use_trailing_stop and trailing_stop_price is not None:
                    stop_price = trailing_stop_price
                elif cfg.use_atr_stop:
                    bar_atr = getattr(bar, "atr_14", None)
                    if bar_atr and bar_atr > 0:
                        stop_price = pos["buy_price"] * (1 - bar_atr * cfg.atr_stop_multiplier)
                    elif cfg.stop_loss is not None:
                        stop_price = pos["buy_price"] * (1 - cfg.stop_loss)
                    else:
                        stop_price = None
                else:
                    stop_price = pos["buy_price"] * (1 - cfg.stop_loss) if cfg.stop_loss is not None else None
                # === 计算止盈价：优先 ATR 动态，fallback 固定比例 ===
                if cfg.use_atr_profit:
                    bar_atr = getattr(bar, "atr_14", None)
                    if bar_atr and bar_atr > 0:
                        tp_price = pos["buy_price"] * (1 + bar_atr * cfg.atr_profit_multiplier)
                    elif cfg.take_profit is not None:
                        tp_price = pos["buy_price"] * (1 + cfg.take_profit)
                    else:
                        tp_price = None
                else:
                    tp_price = pos["buy_price"] * (1 + cfg.take_profit) if cfg.take_profit is not None else None
                if stop_price is not None and bar.low <= stop_price:
                    sell_reason = "stop_loss"
                    # 跳空低开穿越止损线时，以开盘价成交（更保守）
                    target_price = min(stop_price, bar.open)
                elif tp_price is not None and bar.high >= tp_price:
                    sell_reason = "take_profit"
                    # 跳空高开穿越止盈线时，以开盘价成交（更保守）
                    target_price = max(tp_price, bar.open)
                elif idx - pos["buy_idx"] >= cfg.hold_days:
                    sell_reason = "hold_days"
                    target_price = bar.open
                if sell_reason:
                    if _is_limit_down(bar.open, bar.pre_close, getattr(bar, "limit_pct", 0.10)):
                        continue
                    # 卖出端也使用动态滑点（与买入端一致）
                    sell_avg_vol = getattr(bar, "avg_volume_20", 0) or 0
                    if sell_avg_vol <= 0 and hasattr(bar, "volume"):
                        sell_avg_vol = bar.volume
                    sell_slippage = _calc_dynamic_slippage(pos["shares"], sell_avg_vol, cfg)
                    price = target_price * (1 - sell_slippage)
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
                # 4. 牛市状态过滤：近 N 日市场大涨 + 均价在 MA 之上，暂停新开仓
                if not buy_paused and cfg.use_bull_filter and bull_regime.get(d, False):
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
                            require_up_gt_down=cfg.require_up_gt_down,
                            # 通过极宽松阈值禁用 ATR/弱势反抽过滤
                            high_atr_14=0.08 if cfg.enable_atr_filter else 99.0,
                            weak_rebound_pct_chg=0.07 if cfg.enable_weak_rebound_filter else 99.0,
                        )
                        signals = apply_hard_risk_filters(signals, risk_cfg, score_col=cfg.score_col)
                        signals = signals[~signals["risk_blocked"].fillna(False)]
                        # topn 只截断已通过硬门槛的信号
                        if cfg.selection_mode == "quantile" and not signals.empty:
                            q = signals[cfg.score_col].quantile(cfg.score_quantile)
                            signals = signals[signals[cfg.score_col] >= q]
                        signals = signals.head(cfg.top_n)
                        # 加入当日排名列，给 rank_based_sizing 使用
                        signals = signals.copy()
                        signals["_day_rank"] = range(1, len(signals) + 1)
                        # 信号集中度跟踪：更新连续入选计数
                        today_codes = set(signals["code"].tolist()) if not signals.empty else set()
                        new_consecutive = {}
                        for code in today_codes:
                            new_consecutive[code] = consecutive_select.get(code, 0) + 1
                        consecutive_select = new_consecutive
                        slots = max(cfg.top_n - len(positions), 0)
                        if cfg.max_daily_buys is not None:
                            slots = min(slots, cfg.max_daily_buys)
                        for r in signals.itertuples(index=False):
                            if slots <= 0 or r.code in positions:
                                continue
                            # 信号集中度衰减：连续入选超过 N 天不再买入
                            if cfg.max_consecutive_select > 0 and consecutive_select.get(r.code, 0) > cfg.max_consecutive_select:
                                continue
                            bar = bar_idx.get((r.code, d))
                            if bar is None or pd.isna(bar.open) or pd.isna(bar.pre_close):
                                continue
                            if _is_limit_up(bar.open, bar.pre_close, getattr(bar, "limit_pct", 0.10)):
                                continue
                            # === 行业集中度控制 ===
                            sector = industry_map.get(r.code, r.code[:3])  # 优先用真实行业，fallback用代码前缀
                            if positions and cfg.max_sector_pct < 1.0:
                                same_sector = sum(1 for c in positions if industry_map.get(c, c[:3]) == sector)
                                if same_sector / max(len(positions) + 1, 1) >= cfg.max_sector_pct:
                                    continue
                            # === 分数动态仓位：根据模型分数调整仓位乗数 ===
                            base_budget = min(cash * cfg.max_position_pct, cfg.initial_cash * cfg.max_position_pct)
                            if cfg.rank_based_sizing:
                                # 按当日选中内排名：第1名 1.5x，2-3名 1.0x，4+名 0.6x
                                day_rank = getattr(r, "_day_rank", 1)
                                if day_rank == 1:
                                    size_mult = 1.5
                                elif day_rank <= 3:
                                    size_mult = 1.0
                                else:
                                    size_mult = 0.6
                                budget = min(base_budget * size_mult, cfg.initial_cash * 0.20)
                            elif cfg.score_based_sizing:
                                score_val = getattr(r, cfg.score_col, None) or 0
                                if score_val >= cfg.score_high_threshold:
                                    size_mult = 1.5
                                elif score_val >= cfg.score_mid_threshold:
                                    size_mult = 1.0
                                else:
                                    size_mult = 0.5
                                budget = min(base_budget * size_mult, cfg.initial_cash * 0.20)  # 单仓不超过 20%
                            else:
                                budget = base_budget
                            # NaN 防护：budget 或 cash 为 NaN 时跳过
                            if pd.isna(budget) or pd.isna(cash) or budget <= 0:
                                continue
                            price = bar.open * (1 + cfg.slippage)  # 初始估算价
                            if pd.isna(price) or price <= 0:
                                continue
                            # 成交量约束：买入股数不超过日均量比例
                            avg_vol = getattr(bar, "avg_volume_20", 0) or 0
                            if avg_vol <= 0 and hasattr(bar, "volume"):
                                avg_vol = bar.volume  # fallback: 当日成交量
                            shares = _volume_constrained_shares(budget, bar.open, avg_vol, cfg)
                            if shares <= 0:
                                continue
                            # 动态滑点：基于实际买入量/日均量计算冲击成本
                            actual_slippage = _calc_dynamic_slippage(shares, avg_vol, cfg)
                            price = bar.open * (1 + actual_slippage)
                            amount = shares * price
                            fee = amount * cfg.buy_fee
                            # NaN 防护：确保金额有效后才扣减 cash
                            if pd.isna(amount) or pd.isna(fee) or amount <= 0:
                                continue
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
