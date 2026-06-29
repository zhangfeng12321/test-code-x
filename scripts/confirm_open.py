#!/usr/bin/env python3
"""
盘中开盘确认脚本 — 9:45 运行，自动判断4个买入条件

用法:
    .venv/bin/python scripts/confirm_open.py
    .venv/bin/python scripts/confirm_open.py --data-root data_real_2000_10pct

4个确认条件:
    1. 开盘价 >= 昨收 × 0.98（不低开超2%）
    2. 不出现放量下杀（量比>2 且股价低于开盘价 → 放弃）
    3. 当前价站上昨收
    4. 所属板块不弱势（板块涨幅 >= -1% 且 下跌家数/上涨家数 < 2）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_alpha.storage.cache import DataLake


def fetch_realtime_quotes(codes: list[str]) -> pd.DataFrame:
    """获取个股实时行情（东方财富）"""
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    target = df[df["代码"].isin(codes)].copy()
    target = target.rename(columns={
        "代码": "code",
        "名称": "name",
        "最新价": "current",
        "今开": "open",
        "昨收": "prev_close",
        "最高": "high",
        "最低": "low",
        "涨跌幅": "pct_chg",
        "量比": "volume_ratio",
        "成交额": "amount",
    })
    for col in ["current", "open", "prev_close", "high", "low", "pct_chg", "volume_ratio", "amount"]:
        if col in target.columns:
            target[col] = pd.to_numeric(target[col], errors="coerce")
    return target[["code", "name", "current", "open", "prev_close", "high", "low",
                   "pct_chg", "volume_ratio", "amount"]].copy()


def fetch_sector_realtime() -> pd.DataFrame:
    """获取板块实时行情（东方财富行业板块）"""
    import akshare as ak

    df = ak.stock_board_industry_name_em()
    df = df.rename(columns={
        "板块名称": "sector_name",
        "涨跌幅": "sector_pct_chg",
        "上涨家数": "up_count",
        "下跌家数": "down_count",
    })
    for col in ["sector_pct_chg", "up_count", "down_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["sector_name", "sector_pct_chg", "up_count", "down_count"]].copy()


def get_stock_sector_map(lake: DataLake) -> dict[str, str]:
    """获取股票->板块映射（优先本地缓存）"""
    try:
        basics = lake.read_parquet("meta", "stock_basic")
        if not basics.empty and "industry" in basics.columns:
            basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            return dict(zip(basics["code"], basics["industry"].fillna("未知")))
    except Exception:
        pass
    return {}


def match_sector_in_boards(sector_name: str, board_names: list[str]) -> str | None:
    """模糊匹配本地行业名到东方财富板块名

    策略：
    1. 精确匹配
    2. 去掉后缀"业""行业"后匹配
    3. 关键词包含匹配（取最短匹配结果，避免过宽）
    """
    if not sector_name or sector_name == "未知":
        return None

    # 精确匹配
    if sector_name in board_names:
        return sector_name

    # 去掉常见后缀再匹配
    clean = sector_name.rstrip("业").rstrip("行")
    for b in board_names:
        if b == clean:
            return b

    # 关键词包含匹配：本地名称的核心词在板块名中
    # 例如 "医药制造业" -> 核心词 "医药" -> 匹配 "医药商业"、"中药"
    keywords = [clean]
    if len(clean) > 2:
        keywords.append(clean[:2])  # 取前2字作为核心关键词

    candidates = []
    for kw in keywords:
        if len(kw) < 2:
            continue
        for b in board_names:
            if kw in b:
                candidates.append(b)

    if candidates:
        # 返回最短的匹配（最精确）
        return min(candidates, key=len)

    return None


def check_conditions(
    orders: pd.DataFrame,
    quotes: pd.DataFrame,
    sectors: pd.DataFrame,
    sector_map: dict[str, str],
    max_open_gap: float = -0.02,
    max_volume_ratio_for_kill: float = 2.0,
    sector_weak_threshold: float = -1.0,
    sector_down_ratio: float = 2.0,
) -> pd.DataFrame:
    """检查4个开盘确认条件，返回每只股票的判定结果"""

    results = []
    sector_lookup = dict(zip(sectors["sector_name"], sectors.itertuples(index=False)))

    for _, order in orders.iterrows():
        code = str(order["code"]).zfill(6)
        ref_price = float(order.get("ref_price", 0))
        score = float(order.get("score", 0))
        sector_name = sector_map.get(code, "未知")

        # 获取实时行情
        q = quotes[quotes["code"] == code]
        if q.empty:
            results.append({
                "code": code,
                "name": "---",
                "sector": sector_name,
                "score": score,
                "verdict": "⚠️ 无行情",
                "reason": "未获取到实时数据",
                "open": None,
                "current": None,
                "min_open": ref_price * 0.98,
            })
            continue

        row = q.iloc[0]
        name = row.get("name", "---")
        open_price = float(row["open"])
        current = float(row["current"])
        prev_close = float(row["prev_close"])
        volume_ratio = float(row.get("volume_ratio", 0))

        min_open = prev_close * (1 + max_open_gap)  # 不低开超2%
        conditions = []
        passed = []

        # 条件1: 不低开超2%
        c1 = open_price >= min_open
        passed.append(c1)
        if not c1:
            conditions.append(f"❌ 低开{(open_price/prev_close - 1)*100:.1f}%（阈值-2%）")
        else:
            conditions.append(f"✅ 开盘{(open_price/prev_close - 1)*100:+.1f}%")

        # 条件2: 不放量下杀
        is_volume_kill = volume_ratio > max_volume_ratio_for_kill and current < open_price
        c2 = not is_volume_kill
        passed.append(c2)
        if not c2:
            conditions.append(f"❌ 放量下杀（量比{volume_ratio:.1f}，价格跌破开盘）")
        else:
            conditions.append(f"✅ 量比{volume_ratio:.1f}，无放量下杀")

        # 条件3: 站上昨收
        c3 = current >= prev_close
        passed.append(c3)
        if not c3:
            conditions.append(f"❌ 现价{current:.2f} < 昨收{prev_close:.2f}")
        else:
            conditions.append(f"✅ 现价{current:.2f} >= 昨收{prev_close:.2f}")

        # 条件4: 板块不弱势
        c4 = True
        sector_info = ""
        board_names = sectors["sector_name"].tolist() if not sectors.empty else []
        matched_board = match_sector_in_boards(sector_name, board_names) if sector_name != "未知" else None

        if matched_board:
            s_match = sectors[sectors["sector_name"] == matched_board]
            s = s_match.iloc[0]
            s_pct = float(s["sector_pct_chg"])
            s_up = int(s.get("up_count", 0))
            s_down = int(s.get("down_count", 0))
            sector_weak = s_pct < sector_weak_threshold
            sector_ratio_bad = s_up > 0 and s_down / s_up > sector_down_ratio
            if sector_weak or sector_ratio_bad:
                c4 = False
                reasons = []
                if sector_weak:
                    reasons.append(f"板块跌{s_pct:.1f}%")
                if sector_ratio_bad:
                    reasons.append(f"跌{s_down}/涨{s_up}")
                conditions.append(f"❌ 板块弱势[{matched_board}]（{', '.join(reasons)}）")
            else:
                conditions.append(f"✅ 板块[{matched_board}] {s_pct:+.1f}%（涨{s_up}/跌{s_down}）")
            sector_info = f"{s_pct:+.1f}%"
        elif sector_name != "未知":
            conditions.append(f"⚠️ 板块「{sector_name}」未匹配到东财板块，需手动确认")
        else:
            conditions.append("⚠️ 板块未知，需手动确认")

        passed.append(c4)
        all_pass = all(passed)

        if all_pass:
            verdict = "✅ 可买入"
        elif sum(passed) >= 3:
            verdict = "⚠️ 谨慎"
        else:
            verdict = "❌ 放弃"

        results.append({
            "code": code,
            "name": name,
            "sector": sector_name,
            "sector_chg": sector_info,
            "score": score,
            "open": open_price,
            "current": current,
            "prev_close": prev_close,
            "volume_ratio": volume_ratio,
            "min_open": min_open,
            "verdict": verdict,
            "conditions": conditions,
            "reason": " | ".join(conditions),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="盘中开盘确认：自动检查4个买入条件")
    parser.add_argument("--data-root", default="data_real_2000_10pct", help="数据目录")
    args = parser.parse_args()

    lake = DataLake(args.data_root)

    # 读取昨日生成的交易计划
    orders = lake.read_parquet("orders", "next_day_orders")
    if orders.empty:
        print("  ⚠️  无待确认的交易计划（orders/next_day_orders 为空）")
        return

    orders["code"] = orders["code"].astype(str).str.zfill(6)
    codes = orders["code"].tolist()

    print()
    print("━" * 70)
    print("  🔍 盘中开盘确认（自动检查4个买入条件）")
    print("━" * 70)
    print()
    print(f"  待确认标的: {len(codes)} 只")
    print(f"  正在获取实时行情...")
    print()

    # 获取实时数据
    try:
        quotes = fetch_realtime_quotes(codes)
        print(f"  ✓ 个股行情获取成功（{len(quotes)}/{len(codes)} 只）")
    except Exception as e:
        print(f"  ✗ 个股行情获取失败: {e}")
        return

    try:
        sectors = fetch_sector_realtime()
        print(f"  ✓ 板块行情获取成功（{len(sectors)} 个板块）")
    except Exception as e:
        print(f"  ⚠️ 板块行情获取失败（将跳过板块确认）: {e}")
        sectors = pd.DataFrame()

    sector_map = get_stock_sector_map(lake)
    print(f"  ✓ 板块映射加载（{len(sector_map)} 只）")
    print()

    # 检查条件
    result = check_conditions(orders, quotes, sectors, sector_map)

    # 输出结果
    buy_list = result[result["verdict"] == "✅ 可买入"]
    warn_list = result[result["verdict"] == "⚠️ 谨慎"]
    cancel_list = result[result["verdict"].str.contains("放弃|无行情")]

    print("━" * 70)
    print(f"  📊 确认结果: ✅可买{len(buy_list)} | ⚠️谨慎{len(warn_list)} | ❌放弃{len(cancel_list)}")
    print("━" * 70)
    print()

    for _, r in result.iterrows():
        code = r["code"]
        name = r.get("name", "---")
        verdict = r["verdict"]
        score = r.get("score", 0)
        current = r.get("current")
        cur_str = f"{current:.2f}" if pd.notna(current) else "---"

        # 从 orders 中获取操作参数
        order_row = orders[orders["code"] == code]
        shares = int(order_row["shares"].iloc[0]) if not order_row.empty else 0
        tp = order_row.get("take_profit_price", pd.Series([None])).iloc[0] if not order_row.empty else None
        sl = order_row.get("stop_loss_price", pd.Series([None])).iloc[0] if not order_row.empty else None

        print(f"  {verdict}  {code} {name}  现价:{cur_str}  板块:{r.get('sector','?')}{r.get('sector_chg','')}")

        if "conditions" in r and isinstance(r["conditions"], list):
            for cond in r["conditions"]:
                print(f"         {cond}")

        if verdict == "✅ 可买入":
            tp_str = f"{float(tp):.2f}" if pd.notna(tp) else "---"
            sl_str = f"{float(sl):.2f}" if pd.notna(sl) else "---"
            print(f"         → 执行买入: {shares}股 | 止盈:{tp_str} | 止损:{sl_str}")

        print()

    # 汇总可买入清单
    if not buy_list.empty:
        print("━" * 70)
        print("  🎯 最终买入清单（复制到券商APP下单）")
        print("━" * 70)
        for _, r in buy_list.iterrows():
            code = r["code"]
            name = r.get("name", "---")
            current = r.get("current", 0)
            order_row = orders[orders["code"] == code]
            shares = int(order_row["shares"].iloc[0]) if not order_row.empty else 0
            amount = shares * current if pd.notna(current) else 0
            print(f"    {code} {name}  买入 {shares}股  约 {amount/10000:.1f}万  （现价{current:.2f}）")
        print()
    else:
        print("  💤 今日无满足全部条件的标的，空仓等待")
        print()


if __name__ == "__main__":
    main()
