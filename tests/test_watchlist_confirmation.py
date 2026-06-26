import pandas as pd

from stock_alpha.intraday_confirmation import confirm_watchlist_intraday
from stock_alpha.watchlist import WatchlistConfig, generate_watchlist


def test_watchlist_is_softer_than_buy_but_blocks_weak_rebound():
    rows = []
    for i in range(22):
        close = 10 + i * 0.03
        rows.append({"code": "601958", "date": f"2026-05-{i+1:02d}", "open": close, "high": close * 1.02, "low": close * 0.98, "close": close, "amount": 3e8, "volume": 1e7, "pct_chg": 0.3})
    rows[-1].update({"date": "2026-06-25", "open": 10.5, "high": 10.8, "low": 10.2, "close": 10.6, "amount": 4e8, "pct_chg": -1.0})
    daily = pd.DataFrame(rows)
    pred = pd.DataFrame([{"code": "601958", "date": "2026-06-25", "final_score": 0.28, "up_probability": 0.54, "down_probability": 0.26, "risk_score": 0.38}])
    wl = generate_watchlist(pred, daily, WatchlistConfig(min_score=0.18, max_down_probability=0.55, max_risk_score=0.45, min_avg_amount_20=None))
    assert len(wl) == 1
    assert wl.iloc[0]["action"] == "WATCH"


def test_intraday_confirmation_cancel_and_confirm():
    watch = pd.DataFrame([
        {"code": "601958", "action": "WATCH", "score": 0.28},
        {"code": "002235", "action": "WATCH", "score": 0.25},
    ])
    rt = pd.DataFrame([
        {"code": "601958", "open": 10.0, "price": 10.25, "high": 10.3, "low": 9.95, "prev_close": 10.1, "first30_volume_ratio": 1.2},
        {"code": "002235", "open": 8.0, "price": 7.7, "high": 8.05, "low": 7.65, "prev_close": 8.3, "first30_volume_ratio": 3.5},
    ])
    out = confirm_watchlist_intraday(watch, rt)
    actions = dict(zip(out["code"], out["confirm_action"]))
    assert actions["601958"] == "CONFIRM_BUY"
    assert actions["002235"] == "CANCEL"
