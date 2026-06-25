from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from stock_alpha.backtest.simple_backtest import backtest_topn
from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.features.v2_intraday import build_intraday_features
from stock_alpha.features.v4_level2 import build_level2_features
from stock_alpha.framework.pipeline import V1Pipeline, V2Pipeline, V4Pipeline
from stock_alpha.labels.triple_barrier import make_triple_barrier_labels


def synthetic_daily(codes=("600001", "000001"), days=180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for code in codes:
        price = 10 + rng.normal(0, 0.2)
        for i in range(days):
            d = start + pd.Timedelta(days=i)
            if d.weekday() >= 5:
                continue
            ret = rng.normal(0.0008, 0.025)
            open_ = price * (1 + rng.normal(0, 0.005))
            close = price * (1 + ret)
            high = max(open_, close) * (1 + abs(rng.normal(0.01, 0.01)))
            low = min(open_, close) * (1 - abs(rng.normal(0.01, 0.01)))
            volume = max(1, rng.normal(800000, 200000))
            amount = volume * close
            rows.append({"code": code, "date": d, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "amount": amount, "turnover_rate": rng.uniform(1, 8)})
            price = close
    return pd.DataFrame(rows)


def synthetic_minute(codes=("600001", "000001")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    base_day = pd.Timestamp("2025-06-24")
    minutes = list(pd.date_range(base_day.replace(hour=9, minute=30), base_day.replace(hour=11, minute=30), freq="5min")) + list(pd.date_range(base_day.replace(hour=13), base_day.replace(hour=15), freq="5min"))
    for code in codes:
        price = 10
        for ts in minutes:
            open_ = price
            close = price * (1 + rng.normal(0, 0.002))
            high = max(open_, close) * (1 + rng.uniform(0, 0.002))
            low = min(open_, close) * (1 - rng.uniform(0, 0.002))
            vol = max(1000, rng.normal(30000, 10000))
            if ts.strftime("%H:%M") <= "10:00":
                vol *= 2
            rows.append({"code": code, "datetime": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol, "amount": vol * close})
            price = close
    return pd.DataFrame(rows)


def synthetic_level2(code="600001", n=20) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    base = pd.Timestamp("2025-06-24 09:30:00")
    for i in range(n):
        mid = 10 + rng.normal(0, 0.02)
        row = {"code": code, "datetime": base + pd.Timedelta(seconds=i * 3), "last_price": mid + rng.normal(0, 0.005)}
        for lv in range(1, 11):
            row[f"bid_price_{lv}"] = mid - 0.01 * lv
            row[f"ask_price_{lv}"] = mid + 0.01 * lv
            row[f"bid_volume_{lv}"] = max(100, rng.normal(8000, 2000))
            row[f"ask_volume_{lv}"] = max(100, rng.normal(7600, 2000))
        rows.append(row)
    return pd.DataFrame(rows)


def smoke() -> None:
    daily = synthetic_daily()
    features = build_daily_features(daily)
    labels = make_triple_barrier_labels(daily)
    v1_pred = V1Pipeline().fit_predict(daily)
    minute = synthetic_minute()
    v2_pred = V2Pipeline().score(v1_pred, minute)
    l2 = synthetic_level2()
    l2_features = build_level2_features(l2)
    l2_score = V4Pipeline().score_level2(l2)
    bt = backtest_topn(v1_pred, labels, top_n=2)
    print("V1 features:", features.shape)
    print("V1 predictions:", v1_pred.head(3).to_dict("records"))
    print("V1 backend:", v1_pred.attrs.get("backend"))
    print("V2 predictions:", v2_pred.head(3).to_dict("records"))
    print("V4 level2 features:", l2_features[["spread", "depth_imbalance", "level1_imbalance"]].head(3).to_dict("records"))
    print("V4 level2 scores:", l2_score.head(3).to_dict("records"))
    print("Backtest tail:", bt.tail(3).to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke"])
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()


if __name__ == "__main__":
    main()
