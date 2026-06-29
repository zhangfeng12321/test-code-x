#!/usr/bin/env python3
"""
实时信号服务器 — 为 HTML 报告提供实时行情 API

用法:
    .venv/bin/python scripts/signal_server.py
    .venv/bin/python scripts/signal_server.py --port 8888

功能:
    - 提供 /api/signals?codes=002803,603698 实时信号接口
    - 静态文件服务（报告 HTML）
    - 页面刷新即获取最新数据
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


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
    if candidates:
        return min(candidates, key=len)
    return None


def compute_signals(codes: list[str], orders_data: dict, sector_map: dict) -> dict:
    """计算每只股票的实时信号"""
    try:
        quotes = fetch_realtime_quotes(codes)
    except Exception as e:
        return {"error": f"行情获取失败: {str(e)}", "signals": {}}

    try:
        sectors = fetch_sector_realtime()
    except Exception:
        sectors = {}

    board_names = list(sectors.keys())
    signals = {}

    for code in codes:
        q = quotes.get(code)
        order = orders_data.get(code, {})
        ref_price = order.get("ref_price", 0)
        sector_name = sector_map.get(code, "未知")

        if not q or not ref_price:
            signals[code] = {
                "status": "no_data",
                "message": "无实时数据",
                "indicators": [],
                "overall": "unknown",
            }
            continue

        prev_close = q["prev_close"] if q["prev_close"] > 0 else ref_price
        open_price = q["open"]
        current = q["current"]
        volume_ratio = q["volume_ratio"]
        min_open = prev_close * 0.98

        indicators = []

        # 指标1: 开盘价不低开超2%
        if open_price > 0:
            open_gap = (open_price / prev_close - 1) * 100
            c1_pass = open_price >= min_open
            indicators.append({
                "name": "开盘价",
                "value": f"¥{open_price:.2f} ({open_gap:+.1f}%)",
                "threshold": f"≥ ¥{min_open:.2f}",
                "pass": c1_pass,
                "detail": "不低开超2%" if c1_pass else f"低开{open_gap:.1f}%，超过阈值",
            })
        else:
            indicators.append({
                "name": "开盘价",
                "value": "未开盘",
                "threshold": f"≥ ¥{min_open:.2f}",
                "pass": None,
                "detail": "等待开盘",
            })

        # 指标2: 不放量下杀
        is_volume_kill = volume_ratio > 2.0 and current < open_price
        c2_pass = not is_volume_kill
        if volume_ratio > 0:
            indicators.append({
                "name": "量价关系",
                "value": f"量比 {volume_ratio:.1f} / 现价{'↑' if current >= open_price else '↓'}",
                "threshold": "量比≤2 或 价格不跌破开盘",
                "pass": c2_pass,
                "detail": "正常" if c2_pass else "放量下杀（量比>2且破开盘价）",
            })
        else:
            indicators.append({
                "name": "量价关系",
                "value": "等待数据",
                "threshold": "量比≤2 或 价格不跌破开盘",
                "pass": None,
                "detail": "等待交易数据",
            })

        # 指标3: 站上昨收
        c3_pass = current >= prev_close
        cur_vs_prev = (current / prev_close - 1) * 100 if prev_close > 0 else 0
        indicators.append({
            "name": "站上昨收",
            "value": f"¥{current:.2f} ({cur_vs_prev:+.1f}%)",
            "threshold": f"≥ ¥{prev_close:.2f}",
            "pass": c3_pass,
            "detail": "已站上昨收" if c3_pass else f"低于昨收{abs(cur_vs_prev):.1f}%",
        })

        # 指标4: 板块不弱势
        matched_board = match_sector(sector_name, board_names)
        if matched_board and matched_board in sectors:
            s = sectors[matched_board]
            s_pct = s["pct_chg"]
            s_up = s["up_count"]
            s_down = s["down_count"]
            sector_weak = s_pct < -1.0
            sector_ratio_bad = s_up > 0 and s_down / s_up > 2.0
            c4_pass = not (sector_weak or sector_ratio_bad)
            detail_parts = []
            if sector_weak:
                detail_parts.append(f"跌{s_pct:.1f}%")
            if sector_ratio_bad:
                detail_parts.append(f"跌{s_down}/涨{s_up}")
            indicators.append({
                "name": "板块强弱",
                "value": f"[{matched_board}] {s_pct:+.1f}% (涨{s_up}/跌{s_down})",
                "threshold": "涨跌幅≥-1% 且 跌/涨<2",
                "pass": c4_pass,
                "detail": "板块正常" if c4_pass else f"板块弱势: {', '.join(detail_parts)}",
            })
        else:
            indicators.append({
                "name": "板块强弱",
                "value": f"「{sector_name}」未匹配",
                "threshold": "涨跌幅≥-1% 且 跌/涨<2",
                "pass": None,
                "detail": "需手动确认板块情况",
            })

        # 总信号
        passed_list = [i["pass"] for i in indicators if i["pass"] is not None]
        all_pass = all(passed_list) if passed_list else False
        any_fail = any(p is False for p in passed_list)

        if all_pass and len(passed_list) == 4:
            overall = "buy"
        elif any_fail:
            overall = "no_buy"
        else:
            overall = "waiting"

        signals[code] = {
            "status": "ok",
            "name": q.get("name", "---"),
            "current": current,
            "pct_chg": q["pct_chg"],
            "indicators": indicators,
            "overall": overall,
            "overall_text": "✅ 满足买入" if overall == "buy" else "❌ 不满足" if overall == "no_buy" else "⏳ 等待确认",
        }

    return {"error": None, "signals": signals, "timestamp": time.strftime("%H:%M:%S")}


class SignalHandler(SimpleHTTPRequestHandler):
    """HTTP handler: static files + /api/signals endpoint"""

    orders_data: dict = {}
    sector_map: dict = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/signals":
            self._handle_signals(parsed)
        else:
            super().do_GET()

    def _handle_signals(self, parsed):
        params = parse_qs(parsed.query)
        codes = params.get("codes", [""])[0].split(",")
        codes = [c.strip().zfill(6) for c in codes if c.strip()]

        if not codes:
            self._json_response({"error": "missing codes param"})
            return

        result = compute_signals(codes, self.orders_data, self.sector_map)
        self._json_response(result)

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if "/api/" in str(args[0]):
            print(f"  [API] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="实时信号服务器")
    parser.add_argument("--port", type=int, default=8877, help="端口号")
    parser.add_argument("--data-root", default="data_real_2000_10pct", help="数据目录")
    args = parser.parse_args()

    from stock_alpha.storage.cache import DataLake
    lake = DataLake(args.data_root)

    # 加载订单数据
    orders = lake.read_parquet("orders", "next_day_orders")
    orders_data = {}
    if not orders.empty:
        orders["code"] = orders["code"].astype(str).str.zfill(6)
        for _, r in orders.iterrows():
            orders_data[r["code"]] = {
                "ref_price": float(r.get("ref_price", 0)),
                "shares": int(r.get("shares", 0)),
                "take_profit_price": float(r["take_profit_price"]) if pd.notna(r.get("take_profit_price")) else None,
                "stop_loss_price": float(r["stop_loss_price"]) if pd.notna(r.get("stop_loss_price")) else None,
            }

    # 加载板块映射
    sector_map = {}
    try:
        basics = lake.read_parquet("meta", "stock_basic")
        if not basics.empty and "industry" in basics.columns:
            basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            sector_map = dict(zip(basics["code"], basics["industry"].fillna("未知")))
    except Exception:
        pass

    SignalHandler.orders_data = orders_data
    SignalHandler.sector_map = sector_map

    # 切换到 reports 目录作为静态文件根
    import os
    os.chdir(Path(__file__).resolve().parent.parent / "reports")

    server = HTTPServer(("0.0.0.0", args.port), SignalHandler)
    print()
    print("━" * 60)
    print(f"  📡 实时信号服务器已启动")
    print(f"  📊 报告地址: http://localhost:{args.port}/daily_report_20260626.html")
    print(f"  🔌 API地址:  http://localhost:{args.port}/api/signals?codes=...")
    print(f"  ⏹  按 Ctrl+C 停止")
    print("━" * 60)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
