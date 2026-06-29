from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from stock_alpha.risk_rules import RiskRuleConfig, attach_latest_context, apply_hard_risk_filters


@dataclass
class OrderPlanConfig:
    capital: float = 500_000.0
    top_n: int = 10
    min_score: float = 0.45
    max_position_pct: float = 0.1
    lot_size: int = 100
    take_profit: float | None = None
    stop_loss: float | None = None
    selection_mode: str = "threshold"
    score_quantile: float = 0.95
    max_down_probability: float | None = None
    max_risk_score: float | None = None
    min_avg_amount_20: float | None = None
    require_up_gt_down: bool = True
    use_unadjusted_ref_price: bool = True
    # --- 组合优化 ---
    position_sizing: str = "risk_parity"  # equal / risk_parity / score_weighted
    max_sector_pct: float = 0.4  # 同行业最大总仓位


def _fetch_unadjusted_last_close(codes: list[str], fallback: pd.DataFrame) -> pd.DataFrame:
    """Best-effort latest unadjusted close for executable order prices.

    Model features may use adjusted historical prices, but order sizing/stop lines must use
    real trade prices. If AkShare/network is unavailable, fall back to cached prices and mark it.
    """
    base = fallback.copy()
    if base.empty or not codes:
        return base
    base["price_source"] = "cached"
    try:
        from stock_alpha.data.providers.akshare_provider import AkShareProvider
        provider = AkShareProvider()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")
        rows = []
        for code in codes:
            df = provider.get_daily_bars(code, start, end, adjust="")
            if df.empty:
                continue
            df = df.sort_values("date").tail(1)
            close = pd.to_numeric(df["close"].iloc[0], errors="coerce")
            if pd.notna(close) and close > 0:
                rows.append({"code": code, "ref_price": float(close), "price_source": "unadjusted_akshare"})
        if rows:
            live = pd.DataFrame(rows)
            base = base.drop(columns=["price_source"], errors="ignore").merge(live, on="code", how="left", suffixes=("", "_live"))
            base["ref_price"] = base["ref_price_live"].combine_first(base["ref_price"])
            base["price_source"] = base["price_source"].fillna("cached")
            base = base.drop(columns=["ref_price_live"], errors="ignore")
    except Exception:
        pass
    return base


def _get_stock_name_map(stock_basic: pd.DataFrame | None = None) -> dict:
    """Best-effort fetch stock code -> name mapping, preferring local cache."""
    if stock_basic is not None and not stock_basic.empty and "name" in stock_basic.columns:
        basics = stock_basic.copy()
        basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        return dict(zip(basics["code"], basics["name"]))
    try:
        from stock_alpha.data.providers.akshare_provider import AkShareProvider
        basics = AkShareProvider().get_stock_basic()
        if not basics.empty:
            basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            return dict(zip(basics["code"], basics["name"]))
    except Exception:
        pass
    return {}


def generate_next_day_orders(predictions: pd.DataFrame, daily: pd.DataFrame, cfg: OrderPlanConfig, universe_metrics: pd.DataFrame | None = None, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
    if predictions.empty or daily.empty:
        return pd.DataFrame()
    pred = attach_latest_context(predictions, daily)
    pred["code"] = pred["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], format="mixed")
    score_col = "final_score_v2" if "final_score_v2" in pred.columns else "final_score"
    # 获取股票名称映射（优先本地缓存）
    name_map = _get_stock_name_map(stock_basic=stock_basic)
    latest_date = pred["date"].max()
    x = pred[pred["date"] == latest_date].copy()
    risk_cfg = RiskRuleConfig(
        min_score=cfg.min_score,
        max_down_probability=cfg.max_down_probability,
        max_risk_score=cfg.max_risk_score,
        require_up_gt_down=cfg.require_up_gt_down,
    )
    x = apply_hard_risk_filters(x, risk_cfg, score_col=score_col)
    x = x[~x["risk_blocked"].fillna(False)]
    if universe_metrics is not None and not universe_metrics.empty and cfg.min_avg_amount_20 is not None:
        um = universe_metrics.copy()
        um["code"] = um["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        um = um[["code", "avg_amount_20"]].rename(columns={"avg_amount_20": "avg_amount_20_universe"})
        x = x.merge(um, on="code", how="left")
        amount_col = x["avg_amount_20_universe"].combine_first(x["avg_amount_20"]) if "avg_amount_20" in x.columns else x["avg_amount_20_universe"]
        x = x[amount_col.fillna(0) >= cfg.min_avg_amount_20]
    x = x.sort_values(score_col, ascending=False)
    # topn 只负责排序截断，硬门槛已在 apply_hard_risk_filters 中执行；不满足时允许空仓。
    if cfg.selection_mode == "quantile" and not x.empty:
        q = x[score_col].quantile(cfg.score_quantile)
        x = x[x[score_col] >= q]
    x = x.head(cfg.top_n)
    d = daily.copy(); d["code"] = d["code"].astype(str).str.zfill(6); d["date"] = pd.to_datetime(d["date"], format="mixed")
    last_close = d.sort_values("date").groupby("code", as_index=False).tail(1)[["code", "close"]].rename(columns={"close":"ref_price"})
    if cfg.use_unadjusted_ref_price:
        selected_codes = x["code"].astype(str).str.zfill(6).drop_duplicates().tolist()
        last_close = _fetch_unadjusted_last_close(selected_codes, last_close)
    else:
        last_close["price_source"] = "cached"
    x = x.merge(last_close, on="code", how="left")
    rows = []
    budget = cfg.capital * cfg.max_position_pct
    # --- 组合优化：根据方法计算仓位权重 ---
    if cfg.position_sizing != "equal":
        try:
            from stock_alpha.optimization.portfolio import compute_position_weights, allocate_shares
            # 注入 sector 列：优先用 stock_basic 的 industry，回退用代码前缀
            if "sector" not in x.columns:
                if stock_basic is not None and not stock_basic.empty and "industry" in stock_basic.columns:
                    sb = stock_basic[["code", "industry"]].drop_duplicates("code").copy()
                    sb["code"] = sb["code"].astype(str).str.zfill(6)
                    x = x.merge(sb, on="code", how="left")
                    x["sector"] = x["industry"].fillna(x["code"].str[:3])
                else:
                    x["sector"] = x["code"].str[:3]
            weights = compute_position_weights(
                x, d, method=cfg.position_sizing,
                max_position_pct=cfg.max_position_pct,
                max_sector_pct=cfg.max_sector_pct,
                score_col=score_col,
            )
            if not weights.empty:
                # 获取参考价格（ref_price 已通过 merge 在 x 中）
                price_map = dict(zip(x["code"], pd.to_numeric(x.get("ref_price", pd.Series()), errors="coerce").fillna(0)))
                alloc = allocate_shares(weights, cfg.capital, price_map, cfg.lot_size)
                for _, ar in alloc.iterrows():
                    ref = ar["ref_price"]
                    if pd.isna(ref) or ref <= 0:
                        continue
                    code_row = x[x["code"] == ar["code"]]
                    rows.append({
                        "signal_date": latest_date,
                        "code": ar["code"],
                        "name": name_map.get(ar["code"], ""),
                        "action": "BUY",
                        "ref_price": ref,
                        "shares": int(ar["shares"]),
                        "planned_amount": ar["actual_amount"],
                        "weight": ar["weight"],
                        "score": float(code_row[score_col].iloc[0]) if not code_row.empty else 0,
                        "up_probability": float(code_row["up_probability"].iloc[0]) if not code_row.empty and "up_probability" in code_row.columns else None,
                        "down_probability": float(code_row["down_probability"].iloc[0]) if not code_row.empty and "down_probability" in code_row.columns else None,
                        "take_profit_price": ref * (1 + cfg.take_profit) if cfg.take_profit else None,
                        "stop_loss_price": ref * (1 - cfg.stop_loss) if cfg.stop_loss else None,
                        "price_source": "cached",
                        "sizing_method": cfg.position_sizing,
                        "note": f"组合优化({cfg.position_sizing})；次日低开/放量下杀/跌破止损须取消或止损",
                    })
                return pd.DataFrame(rows)
        except Exception:
            pass  # fallback 到等权分配

    # 等权分配 fallback
    for r in x.itertuples(index=False):
        ref = getattr(r, "ref_price", None)
        if pd.isna(ref) or ref <= 0:
            continue
        shares = int((budget / ref) // cfg.lot_size * cfg.lot_size)
        if shares <= 0:
            continue
        rows.append({
            "signal_date": latest_date,
            "code": r.code,
            "name": name_map.get(r.code, ""),
            "action": "BUY",
            "ref_price": ref,
            "shares": shares,
            "planned_amount": shares * ref,
            "score": getattr(r, score_col),
            "up_probability": getattr(r, "up_probability", None),
            "down_probability": getattr(r, "down_probability", None),
            "take_profit_price": ref * (1 + cfg.take_profit) if cfg.take_profit else None,
            "stop_loss_price": ref * (1 - cfg.stop_loss) if cfg.stop_loss else None,
            "price_source": getattr(r, "price_source", "cached"),
            "note": "仅通过硬风控初筛；次日低开/放量下杀/跌破止损须取消或止损",
        })
    return pd.DataFrame(rows)


def detect_market_regime(
    daily: pd.DataFrame,
    breadth_days: int = 20,
    breadth_threshold: float = 0.12,
    ma_days: int = 60,
) -> dict:
    """实时市场状态检测（完全无前视）。

    状态判断逻辑：
    - 牛市警告：近 breadth_days 日全市场等权涨幅 > breadth_threshold
              AND 全市场均价 > MA(ma_days)
    - 正常状态：其中一个或两个条件不满足
    """
    if daily.empty:
        return {"is_bull_market": False, "status_text": "数据不足", "action": "正常运行"}

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    avg_close = df.groupby("date")["close"].mean().sort_index()
    if len(avg_close) < max(breadth_days, ma_days):
        return {"is_bull_market": False, "status_text": "数据不足", "action": "正常运行"}

    pct_chg = avg_close.pct_change()
    rolling_ret = float((1 + pct_chg.tail(breadth_days)).prod() - 1)
    ma_n = float(avg_close.tail(ma_days).mean())
    current_avg = float(avg_close.iloc[-1])
    avg_vs_ma = (current_avg / ma_n - 1) if ma_n > 0 else 0.0

    breadth_bull = rolling_ret > breadth_threshold
    ma_bull = current_avg > ma_n
    is_bull = breadth_bull and ma_bull

    if is_bull:
        status_text = f"▲ 牛市模式  近{breadth_days}日全市场+{rolling_ret*100:.1f}%，均价在MA{ma_days}上方+{avg_vs_ma*100:.1f}%"
        action = "暂停新开仓，已持仓位按止盈/止损/到期正常管理"
    elif rolling_ret > 0 and ma_bull:
        status_text = f"▶ 稳健市场  近{breadth_days}日+{rolling_ret*100:.1f}%，均价在MA{ma_days}上方"
        action = "正常运行策略"
    elif rolling_ret < -0.05:
        status_text = f"↓ 弱势市场  近{breadth_days}日{rolling_ret*100:.1f}%，谨慎"
        action = "正常运行策略（止损保护生效）"
    else:
        status_text = f"─ 震荡市场  近{breadth_days}日{rolling_ret*100:.1f}%，均价偏差{avg_vs_ma*100:.1f}%"
        action = "正常运行策略"

    return {
        "is_bull_market": is_bull,
        "rolling_ret_20d": rolling_ret,
        "avg_vs_ma60": avg_vs_ma,
        "status_text": status_text,
        "action": action,
    }
