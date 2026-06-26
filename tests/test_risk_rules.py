import pandas as pd

from stock_alpha.backtest.ashare_backtest import AShareBacktester, AShareBacktestConfig
from stock_alpha.risk_rules import RiskRuleConfig, apply_hard_risk_filters, attach_latest_context
from stock_alpha.trading_plan import OrderPlanConfig, generate_next_day_orders


def test_topn_does_not_force_low_score_orders():
    pred = pd.DataFrame([
        {"code": "001330", "date": "2026-06-25", "final_score": 0.35, "up_probability": 0.63, "down_probability": 0.23, "risk_score": 0.2},
        {"code": "002235", "date": "2026-06-25", "final_score": 0.22, "up_probability": 0.49, "down_probability": 0.39, "risk_score": 0.2},
    ])
    daily = pd.DataFrame([
        {"code": "001330", "date": "2026-06-24", "open": 6, "high": 6.2, "low": 5.9, "close": 6.0, "amount": 1e8, "volume": 1e7},
        {"code": "001330", "date": "2026-06-25", "open": 6.23, "high": 6.93, "low": 6.19, "close": 6.93, "amount": 7e8, "volume": 1e8, "pct_chg": 10.0},
        {"code": "002235", "date": "2026-06-24", "open": 8, "high": 8.2, "low": 7.8, "close": 8.1, "amount": 1e8, "volume": 1e7},
        {"code": "002235", "date": "2026-06-25", "open": 8.1, "high": 8.8, "low": 8.0, "close": 8.2, "amount": 3e8, "volume": 4e7, "pct_chg": 1.2},
    ])
    orders = generate_next_day_orders(pred, daily, OrderPlanConfig(top_n=10, selection_mode="topn", min_score=0.45, use_unadjusted_ref_price=False))
    assert orders.empty


def test_weak_rebound_limit_up_is_blocked():
    rows = []
    for i in range(22):
        close = 9.5 - i * 0.15
        rows.append({"code": "001330", "date": f"2026-05-{i+1:02d}", "open": close, "high": close * 1.02, "low": close * 0.98, "close": close, "amount": 1e8, "volume": 1e7, "pct_chg": -1.0})
    rows[-1].update({"date": "2026-06-25", "open": 6.23, "high": 6.93, "low": 6.19, "close": 6.93, "pct_chg": 10.0, "amount": 7e8})
    daily = pd.DataFrame(rows)
    pred = pd.DataFrame([{"code": "001330", "date": "2026-06-25", "final_score": 0.7, "up_probability": 0.8, "down_probability": 0.1, "risk_score": 0.1}])
    x = attach_latest_context(pred, daily)
    checked = apply_hard_risk_filters(x, RiskRuleConfig(min_score=0.45, max_down_probability=0.4, max_risk_score=0.25))
    assert checked["risk_blocked"].iloc[0]
    assert "弱势反抽" in checked["risk_reasons"].iloc[0]


def test_backtest_topn_respects_min_score_gate():
    dates = pd.date_range("2026-06-24", periods=3, freq="D")
    daily = pd.DataFrame([
        {"code": "001330", "date": dates[0], "open": 6.0, "high": 6.1, "low": 5.9, "close": 6.0, "volume": 1_000_000},
        {"code": "001330", "date": dates[1], "open": 6.0, "high": 6.1, "low": 5.9, "close": 6.0, "volume": 1_000_000},
        {"code": "001330", "date": dates[2], "open": 6.0, "high": 6.1, "low": 5.9, "close": 6.0, "volume": 1_000_000},
    ])
    pred = pd.DataFrame([{"code": "001330", "date": dates[0], "final_score": 0.35, "up_probability": 0.6, "down_probability": 0.2, "risk_score": 0.1}])
    bt = AShareBacktester(AShareBacktestConfig(selection_mode="topn", top_n=1, min_score=0.45, max_down_probability=0.4, max_risk_score=0.25)).run(daily, pred)
    assert bt["trades"].empty
