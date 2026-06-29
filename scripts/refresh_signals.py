#!/usr/bin/env python3
"""
一键刷新实时信号 — 无需服务器

用法:
    .venv/bin/python scripts/refresh_signals.py
    .venv/bin/python scripts/refresh_signals.py --open   # 刷新后自动打开浏览器

原理: 获取实时行情 → 计算信号 → 将数据注入 HTML → 浏览器打开即看到结果
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from stock_alpha.storage.cache import DataLake


def fetch_realtime_quotes(codes: list[str]) -> dict:
    """获取个股实时行情"""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    target = df[df["代码"].isin(codes)].copy()
    result = {}
    for _, row in target.iterrows():
        code = row["代码"]
        result[code] = {
            "name": str(row.get("名称", "---")),
            "current": float(row.get("最新价", 0) or 0),
            "open": float(row.get("今开", 0) or 0),
            "prev_close": float(row.get("昨收", 0) or 0),
            "high": float(row.get("最高", 0) or 0),
            "low": float(row.get("最低", 0) or 0),
            "pct_chg": float(row.get("涨跌幅", 0) or 0),
            "volume_ratio": float(row.get("量比", 0) or 0),
            "amount": float(row.get("成交额", 0) or 0),
        }
    return result


def fetch_sector_realtime() -> dict:
    """获取板块实时行情"""
    import akshare as ak
    df = ak.stock_board_industry_name_em()
    result = {}
    for _, row in df.iterrows():
        name = str(row.get("板块名称", ""))
        result[name] = {
            "pct_chg": float(row.get("涨跌幅", 0) or 0),
            "up_count": int(row.get("上涨家数", 0) or 0),
            "down_count": int(row.get("下跌家数", 0) or 0),
        }
    return result


def match_sector(sector_name: str, board_names: list[str]) -> str | None:
    """模糊匹配板块名"""
    if not sector_name or sector_name == "未知":
        return None
    if sector_name in board_names:
        return sector_name
    clean = sector_name.rstrip("业").rstrip("行")
    for b in board_names:
        if b == clean:
            return b
    keywords = [clean]
    if len(clean) > 2:
        keywords.append(clean[:2])
    candidates = []
    for kw in keywords:
        if len(kw) < 2:
            continue
        for b in board_names:
            if kw in b:
                candidates.append(b)
    return min(candidates, key=len) if candidates else None


def compute_signals(codes: list[str], orders_data: dict, sector_map: dict) -> dict:
    """计算信号数据"""
    quotes = fetch_realtime_quotes(codes)
    sectors = fetch_sector_realtime()
    board_names = list(sectors.keys())
    signals = {}

    for code in codes:
        q = quotes.get(code)
        order = orders_data.get(code, {})
        ref_price = order.get("ref_price", 0)
        sector_name = sector_map.get(code, "未知")

        if not q or not ref_price:
            signals[code] = {"status": "no_data", "indicators": [], "overall": "unknown"}
            continue

        prev_close = q["prev_close"] if q["prev_close"] > 0 else ref_price
        open_price = q["open"]
        current = q["current"]
        volume_ratio = q["volume_ratio"]
        min_open = prev_close * 0.98
        indicators = []

        # 指标1: 开盘价
        if open_price > 0:
            open_gap = (open_price / prev_close - 1) * 100
            c1 = open_price >= min_open
            indicators.append({
                "name": "开盘价",
                "value": f"¥{open_price:.2f} ({open_gap:+.1f}%)",
                "threshold": f"≥ ¥{min_open:.2f}",
                "pass": c1,
                "detail": "不低开超2%" if c1 else f"低开{open_gap:.1f}%，超过阈值",
            })
        else:
            indicators.append({"name": "开盘价", "value": "未开盘", "threshold": f"≥ ¥{min_open:.2f}", "pass": None, "detail": "等待开盘"})

        # 指标2: 量价关系
        is_kill = volume_ratio > 2.0 and current < open_price
        c2 = not is_kill
        indicators.append({
            "name": "量价关系",
            "value": f"量比 {volume_ratio:.1f} / 现价{'↑' if current >= open_price else '↓'}",
            "threshold": "量比≤2 或 价格不跌破开盘",
            "pass": c2,
            "detail": "正常" if c2 else "放量下杀（量比>2且破开盘价）",
        })

        # 指标3: 站上昨收
        c3 = current >= prev_close
        pct = (current / prev_close - 1) * 100 if prev_close > 0 else 0
        indicators.append({
            "name": "站上昨收",
            "value": f"¥{current:.2f} ({pct:+.1f}%)",
            "threshold": f"≥ ¥{prev_close:.2f}",
            "pass": c3,
            "detail": "已站上昨收" if c3 else f"低于昨收{abs(pct):.1f}%",
        })

        # 指标4: 板块
        matched = match_sector(sector_name, board_names)
        if matched and matched in sectors:
            s = sectors[matched]
            s_pct = s["pct_chg"]
            s_up, s_down = s["up_count"], s["down_count"]
            weak = s_pct < -1.0
            ratio_bad = s_up > 0 and s_down / s_up > 2.0
            c4 = not (weak or ratio_bad)
            parts = []
            if weak: parts.append(f"跌{s_pct:.1f}%")
            if ratio_bad: parts.append(f"跌{s_down}/涨{s_up}")
            indicators.append({
                "name": "板块强弱",
                "value": f"[{matched}] {s_pct:+.1f}% (涨{s_up}/跌{s_down})",
                "threshold": "涨跌幅≥-1% 且 跌/涨<2",
                "pass": c4,
                "detail": "板块正常" if c4 else f"板块弱势: {', '.join(parts)}",
            })
        else:
            indicators.append({"name": "板块强弱", "value": f"「{sector_name}」未匹配", "threshold": "需手动确认", "pass": None, "detail": "未匹配到东财板块"})

        # 总信号
        passed = [i["pass"] for i in indicators if i["pass"] is not None]
        all_pass = all(passed) if passed else False
        any_fail = any(p is False for p in passed)

        overall = "buy" if all_pass and len(passed) == 4 else "no_buy" if any_fail else "waiting"
        signals[code] = {
            "status": "ok",
            "name": q.get("name", "---"),
            "current": current,
            "pct_chg": q["pct_chg"],
            "indicators": indicators,
            "overall": overall,
            "overall_text": "✅ 满足买入条件" if overall == "buy" else "❌ 不满足买入条件" if overall == "no_buy" else "⏳ 等待确认",
        }

    return {"error": None, "signals": signals, "timestamp": time.strftime("%H:%M:%S")}


def inject_signal_data(html_path: Path, signal_data: dict) -> None:
    """将信号数据注入到 HTML 文件中"""
    content = html_path.read_text(encoding="utf-8")
    data_json = json.dumps(signal_data, ensure_ascii=False)
    # 替换占位符
    replacement = f"SIGNAL_DATA = {data_json};"
    content = content.replace("// __SIGNAL_DATA_PLACEHOLDER__", replacement)
    html_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="一键刷新实时信号（无需服务器）")
    parser.add_argument("--data-root", default="data_real_2000_10pct")
    parser.add_argument("--open", action="store_true", default=True, help="刷新后自动打开浏览器")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    lake = DataLake(args.data_root)

    # 加载订单
    orders = lake.read_parquet("orders", "next_day_orders")
    if orders.empty:
        print("  ⚠️ 无待确认订单")
        return

    orders["code"] = orders["code"].astype(str).str.zfill(6)
    codes = orders["code"].tolist()
    orders_data = {}
    for _, r in orders.iterrows():
        orders_data[r["code"]] = {"ref_price": float(r.get("ref_price", 0))}

    # 板块映射
    sector_map = {}
    try:
        basics = lake.read_parquet("meta", "stock_basic")
        if not basics.empty and "industry" in basics.columns:
            basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            sector_map = dict(zip(basics["code"], basics["industry"].fillna("未知")))
    except Exception:
        pass

    print()
    print("━" * 60)
    print("  📡 获取实时信号数据...")
    print("━" * 60)
    print(f"  待查股票: {', '.join(codes)}")

    # 获取信号
    signal_data = compute_signals(codes, orders_data, sector_map)

    if signal_data.get("error"):
        print(f"  ✗ {signal_data['error']}")
        return

    # 汇总
    signals = signal_data.get("signals", {})
    buy_count = sum(1 for s in signals.values() if s.get("overall") == "buy")
    fail_count = sum(1 for s in signals.values() if s.get("overall") == "no_buy")
    wait_count = sum(1 for s in signals.values() if s.get("overall") == "waiting")
    print(f"  ✓ 数据获取完成 ({signal_data['timestamp']})")
    print(f"  📊 结果: ✅可买{buy_count} | ❌不买{fail_count} | ⏳等待{wait_count}")

    # 找到最新报告HTML
    report_dir = Path(args.data_root).parent / "reports"
    if not report_dir.exists():
        report_dir = Path("reports")
    html_files = sorted(report_dir.glob("daily_report_*.html"), reverse=True)
    if not html_files:
        print("  ✗ 未找到报告HTML文件")
        return

    html_path = html_files[0]

    # 先重新生成干净的HTML（避免重复注入）
    print(f"  📝 重新生成报告...")
    try:
        from stock_alpha.reports.html_report import HtmlReportGenerator
        predictions = lake.read_parquet("predictions", "v1_latest")
        watchlist = lake.read_parquet("orders", "watchlist")
        metrics = lake.read_parquet("backtest", "metrics_latest")
        stats = lake.read_parquet("backtest", "trade_stats_latest")
        gen = HtmlReportGenerator(out_dir=str(report_dir))
        html_path = gen.generate(predictions=predictions, backtest_metrics=metrics, trade_stats=stats, orders=orders, watchlist=watchlist)
    except Exception as e:
        print(f"  ⚠️ 报告重新生成失败({e})，使用已有文件")

    # 注入信号数据
    inject_signal_data(html_path, signal_data)
    print(f"  ✓ 信号数据已注入: {html_path}")
    print()

    # 打印简要结果
    for code, sig in signals.items():
        if sig.get("status") != "ok":
            continue
        icon = "✅" if sig["overall"] == "buy" else "❌" if sig["overall"] == "no_buy" else "⏳"
        print(f"  {icon} {code} {sig['name']}  {sig['overall_text']}")
        for ind in sig.get("indicators", []):
            i_icon = "🟢" if ind["pass"] is True else "🔴" if ind["pass"] is False else "🟡"
            print(f"     {i_icon} {ind['name']}: {ind['value']}")
    print()

    # 打开浏览器
    if args.open and not args.no_open:
        abs_path = str(html_path.resolve())
        print(f"  🌐 打开浏览器: {abs_path}")
        if platform.system() == "Darwin":
            subprocess.run(["open", abs_path], check=False)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", abs_path], check=False)
        else:
            os.startfile(abs_path)


if __name__ == "__main__":
    main()
