from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import html as _html

import pandas as pd


@dataclass
class HtmlReportGenerator:
    out_dir: Path | str = Path("reports")

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _get_name_sector_maps(self, orders: pd.DataFrame | None, watchlist: pd.DataFrame | None) -> tuple[dict, dict]:
        """Best-effort stock name and sector maps."""
        name_map: dict[str, str] = {}
        sector_map: dict[str, str] = {}
        try:
            from stock_alpha.storage.cache import DataLake
            # Try common data roots
            for root in ["data_real_2000_10pct", "data_pipeline_smoke", "data"]:
                lake = DataLake(root)
                basics = lake.read_parquet("meta", "stock_basic")
                if not basics.empty:
                    basics["code"] = basics["code"].astype(str).str.extract(r"(\d{6})", expand=False)
                    name_map = dict(zip(basics["code"], basics["name"]))
                    if "industry" in basics.columns:
                        sector_map = dict(zip(basics["code"], basics["industry"].fillna("未知")))
                    break
        except Exception:
            pass
        # Fallback: use name from orders if available
        if orders is not None and not orders.empty and "name" in orders.columns:
            for _, r in orders.iterrows():
                c = str(r["code"]).zfill(6)
                if c not in name_map and r.get("name"):
                    name_map[c] = str(r["name"])
        return name_map, sector_map

    def generate(
        self,
        predictions: pd.DataFrame,
        trade_date: str | None = None,
        top_n: int = 20,
        backtest_metrics: pd.DataFrame | None = None,
        trade_stats: pd.DataFrame | None = None,
        feature_importance: pd.DataFrame | None = None,
        quality_summary: pd.DataFrame | None = None,
        monthly: pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        equity: pd.DataFrame | None = None,
        holdings: pd.DataFrame | None = None,
        signal_stability: pd.DataFrame | None = None,
        turnover: pd.DataFrame | None = None,
        risk_tags: pd.DataFrame | None = None,
        explanations: pd.DataFrame | None = None,
        orders: pd.DataFrame | None = None,
        watchlist: pd.DataFrame | None = None,
    ) -> Path:
        df = predictions.copy()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        d = pd.to_datetime(trade_date) if trade_date else df["date"].max()
        date_str = d.strftime("%Y-%m-%d")

        m = backtest_metrics.iloc[0].to_dict() if backtest_metrics is not None and not backtest_metrics.empty else {}
        t = trade_stats.iloc[0].to_dict() if trade_stats is not None and not trade_stats.empty else {}
        name_map, sector_map = self._get_name_sector_maps(orders, watchlist)

        # --- Build order cards HTML ---
        orders_html = self._build_orders_html(orders, name_map, sector_map)
        watchlist_html = self._build_watchlist_html(watchlist, name_map, sector_map)
        metrics_html = self._build_metrics_html(m, t)
        guide_html = self._build_guide_html()

        content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>明日操作清单 {date_str}</title>
{self._css()}
</head><body>
<header>
  <h1>🎯 明日操作清单</h1>
  <p class="subtitle">信号日期: {date_str} · 模型自动生成 · 不构成投资建议</p>
</header>

{metrics_html}
{orders_html}
{watchlist_html}
{guide_html}

<footer>
  <p>⚠️ 风险提示：本报告为量化模型信号，不构成任何投资建议。实盘前必须结合涨跌停、流动性、仓位管理和止损规则。</p>
</footer>
{self._realtime_js(orders)}
</body></html>"""
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"daily_report_{timestamp}.html"
        path.write_text(content, encoding="utf-8")
        return path

    def _build_metrics_html(self, m: dict, t: dict) -> str:
        annual = m.get("annual_return", 0)
        dd = m.get("max_drawdown", 0)
        sharpe = m.get("sharpe", 0)
        wr = t.get("win_rate", 0)
        plr = t.get("profit_loss_ratio", 0)
        return f"""<section class="metrics">
  <div class="metric"><span class="m-label">年化收益</span><span class="m-value green">{annual:.1%}</span></div>
  <div class="metric"><span class="m-label">最大回撤</span><span class="m-value red">{dd:.1%}</span></div>
  <div class="metric"><span class="m-label">Sharpe</span><span class="m-value">{sharpe:.2f}</span></div>
  <div class="metric"><span class="m-label">胜率</span><span class="m-value">{wr:.1%}</span></div>
  <div class="metric"><span class="m-label">盈亏比</span><span class="m-value">{plr:.2f}</span></div>
</section>"""

    def _build_orders_html(self, orders: pd.DataFrame | None, name_map: dict, sector_map: dict) -> str:
        if orders is None or orders.empty:
            return "<section class='empty'><h2>明日无交易计划</h2><p>未通过风控或无信号，空仓等待。</p></section>"
        cards = []
        for idx, (_, r) in enumerate(orders.iterrows(), 1):
            code = str(r["code"]).zfill(6)
            name = name_map.get(code, r.get("name", "---"))
            sector = sector_map.get(code, "未知")
            ref_price = float(r.get("ref_price", 0))
            tp = r.get("take_profit_price", None)
            sl = r.get("stop_loss_price", None)
            tp_val = float(tp) if pd.notna(tp) and tp else ref_price * 1.15
            sl_val = float(sl) if pd.notna(sl) and sl else ref_price * 0.93
            shares = int(r.get("shares", 0))
            amount = shares * ref_price
            score = float(r.get("score", 0))
            up_prob = r.get("up_probability", None)
            down_prob = r.get("down_probability", None)
            up_str = f"{float(up_prob):.0%}" if pd.notna(up_prob) else "---"
            down_str = f"{float(down_prob):.0%}" if pd.notna(down_prob) else "---"
            min_open = ref_price * 0.98

            # Score color
            score_cls = "high" if score >= 0.8 else "mid" if score >= 0.6 else "low"

            cards.append(f"""<div class="stock-card">
  <div class="card-header">
    <span class="rank">#{idx}</span>
    <span class="code">{_html.escape(code)}</span>
    <span class="name">{_html.escape(str(name))}</span>
    <span class="header-scores">
      <span class="hs-item"><span class="hs-label">模型分</span><span class="hs-value {score_cls}">{score:.4f}</span></span>
      <span class="hs-item"><span class="hs-label">上涨</span><span class="hs-value green">{up_str}</span></span>
      <span class="hs-item"><span class="hs-label">下跌</span><span class="hs-value red">{down_str}</span></span>
    </span>
    <span class="sector-tag">{_html.escape(str(sector))}</span>
  </div>
  <div class="card-prices">
    <div class="price-row">
      <div class="p-item"><span class="p-label">买入参考价</span><span class="p-value">¥{ref_price:.2f}</span></div>
      <div class="p-item"><span class="p-label">股数</span><span class="p-value">{shares}股</span></div>
      <div class="p-item"><span class="p-label">金额</span><span class="p-value">{amount/10000:.1f}万</span></div>
      <div class="p-item tp"><span class="p-label">止盈卖出</span><span class="p-value">¥{tp_val:.2f} <small>(+15%)</small></span></div>
      <div class="p-item sl"><span class="p-label">止损卖出</span><span class="p-value">¥{sl_val:.2f} <small>(-7%)</small></span></div>
      <div class="p-item"><span class="p-label">持有上限</span><span class="p-value">10个交易日</span></div>
    </div>
  </div>
  <div class="card-bottom-row">
  <div class="card-confirm">
    <div class="confirm-title">开盘确认条件（全部满足才买）</div>
    <div class="confirm-item"><span class="check">✓</span> 开盘价 ≥ <strong>¥{min_open:.2f}</strong>（不低开超2%）</div>
    <div class="confirm-item"><span class="check">✓</span> 9:30-9:45 不出现放量下杀（量比&gt;2 且股价持续下跌 → 放弃）</div>
    <div class="confirm-item"><span class="check">✓</span> 9:45前股价站上 <strong>¥{ref_price:.2f}</strong>（昨收）</div>
    <div class="confirm-item"><span class="check">✓</span> 板块「{_html.escape(str(sector))}」不弱势（运行 confirm_open.py 自动判断）</div>
  </div>
  <div class="realtime-signal" id="signal-{code}">
    <div class="signal-header">
      <span class="signal-title">📡 实时信号</span>
      <span class="signal-time" id="time-{code}">加载中...</span>
    </div>
    <div class="signal-indicators" id="indicators-{code}">
      <div class="signal-loading">正在获取实时数据...</div>
    </div>
    <div class="signal-overall" id="overall-{code}">
      <span class="overall-badge waiting">⏳ 等待数据</span>
    </div>
  </div>
  </div>
</div>""")
        return f"""<section class="orders-section">
  <h2>🎯 明日操作清单（{len(orders)} 只，按模型分数排序）</h2>
  <div class="stock-cards">{''.join(cards)}</div>
</section>"""

    def _build_watchlist_html(self, watchlist: pd.DataFrame | None, name_map: dict, sector_map: dict) -> str:
        if watchlist is None or watchlist.empty:
            return ""
        rows = []
        for _, r in watchlist.head(20).iterrows():
            code = str(r["code"]).zfill(6)
            name = name_map.get(code, "---")
            score = float(r.get("score", 0))
            rows.append(f"<tr><td>{_html.escape(code)}</td><td>{_html.escape(str(name))}</td><td>{score:.4f}</td></tr>")
        return f"""<section class="watch-section">
  <h2>👀 观察池（{len(watchlist)} 只，盘中确认后可作为替补）</h2>
  <table class="watch-table">
    <thead><tr><th>代码</th><th>名称</th><th>分数</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""

    def _build_guide_html(self) -> str:
        return """<section class="guide-section">
  <h2>📋 操作指南</h2>
  <div class="guide-grid">
    <div class="guide-card">
      <h3>⏰ 时间节奏</h3>
      <table class="guide-table">
        <tr><td class="time">15:15后</td><td>运行 run_daily.sh，生成次日操作清单</td></tr>
        <tr><td class="time">晚上</td><td>阅读本报告，标记拟买入标的</td></tr>
        <tr><td class="time">次日 9:15</td><td>集合竞价观察：看开盘价是否满足最低开盘价</td></tr>
        <tr><td class="time">次日 9:45</td><td>运行 <code>scripts/confirm_open.py</code> 自动检查4条件</td></tr>
        <tr><td class="time">持有期间</td><td>每日收盘后检查是否触及止盈/止损价</td></tr>
      </table>
    </div>
    <div class="guide-card">
      <h3>🟢 买入规则</h3>
      <ul>
        <li>4个确认条件<strong>必须全部满足</strong>，缺一不买</li>
        <li>每日最多买入 <strong>5 只</strong>，优先买模型分数最高的</li>
        <li>单只仓位不超过总资金 <strong>10%</strong></li>
      </ul>
    </div>
    <div class="guide-card">
      <h3>🔴 卖出规则（三选一，先到先执行）</h3>
      <ul>
        <li class="tp">止盈：股价 ≥ 止盈价 → 当日收盘前卖出</li>
        <li class="sl">止损：股价 ≤ 止损价 → <strong>立即卖出，不犹豫</strong></li>
        <li>到期：持有满 10 个交易日 → 次日开盘卖出</li>
      </ul>
    </div>
    <div class="guide-card">
      <h3>⚠️ 特殊情况</h3>
      <ul>
        <li>大盘跳水（沪指跌&gt;2%）→ 全部暂停买入</li>
        <li>开盘即涨停 → 不追，放弃</li>
        <li>集合竞价一字跌停 → 持仓不卖，次日再处理</li>
      </ul>
    </div>
  </div>
</section>"""

    def _css(self) -> str:
        return """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
  background: #0f172a; color: #e2e8f0; padding: 16px; line-height: 1.4; }
header { text-align: center; margin-bottom: 16px; }
h1 { font-size: 22px; color: #fff; margin-bottom: 2px; }
.subtitle { color: #94a3b8; font-size: 13px; }
h2 { font-size: 17px; color: #fff; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid #334155; }
h3 { font-size: 14px; color: #f1f5f9; margin-bottom: 8px; }

/* Metrics bar */
.metrics { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-bottom: 16px; }
.metric { background: #1e293b; border-radius: 8px; padding: 8px 18px; text-align: center; min-width: 100px; }
.m-label { display: block; font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.m-value { display: block; font-size: 20px; font-weight: 700; margin-top: 2px; color: #f1f5f9; }
.m-value.green { color: #4ade80; }
.m-value.red { color: #f87171; }

/* Stock cards */
.orders-section, .watch-section, .guide-section { margin-bottom: 16px; }
section { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 14px; }
.stock-cards { display: flex; flex-direction: column; gap: 12px; }
.stock-card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 14px; transition: border-color 0.2s; }
.stock-card:hover { border-color: #3b82f6; }

/* Card header */
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.rank { background: #3b82f6; color: white; font-weight: 700; font-size: 13px; padding: 3px 8px; border-radius: 5px; }
.code { font-family: 'SF Mono', monospace; font-size: 16px; font-weight: 600; color: #fff; }
.name { font-size: 16px; color: #e2e8f0; }
.header-scores { display: flex; gap: 14px; margin-left: 8px; }
.hs-item { display: flex; flex-direction: column; align-items: center; }
.hs-label { font-size: 10px; color: #64748b; }
.hs-value { font-size: 16px; font-weight: 700; }
.hs-value.high { color: #4ade80; }
.hs-value.mid { color: #fbbf24; }
.hs-value.low { color: #94a3b8; }
.hs-value.green { color: #4ade80; }
.hs-value.red { color: #f87171; }
.sector-tag { background: #334155; color: #94a3b8; font-size: 12px; padding: 3px 10px; border-radius: 20px; margin-left: auto; }

/* Prices */
.card-prices { margin-bottom: 8px; }
.price-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 0; }
.p-item { flex: 1; min-width: 80px; }
.p-label { display: block; font-size: 10px; color: #94a3b8; }
.p-value { font-size: 14px; font-weight: 600; color: #f1f5f9; }
.p-item.tp .p-value { color: #4ade80; }
.p-item.sl .p-value { color: #f87171; }
.p-item small { font-size: 12px; opacity: 0.7; }

/* Confirmation + Realtime row */
.card-bottom-row { display: flex; gap: 10px; margin-top: 10px; }
.card-confirm { flex: 0 0 32%; background: #1a2332; border: 1px dashed #475569; border-radius: 6px; padding: 10px 12px; }
.realtime-signal { flex: 1; background: #0c1929; border: 1px solid #1e40af; border-radius: 6px; padding: 10px 12px; }
.confirm-title { font-size: 12px; font-weight: 600; color: #fbbf24; margin-bottom: 6px; }
.confirm-item { font-size: 12px; color: #cbd5e1; padding: 2px 0; }
.confirm-item .check { color: #4ade80; font-weight: 700; margin-right: 6px; }
.confirm-item strong { color: #fff; }

/* Realtime Signal */
.signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.signal-title { font-size: 14px; font-weight: 600; color: #60a5fa; }
.signal-time { font-size: 11px; color: #64748b; }
.signal-indicators { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-bottom: 8px; }
.signal-row { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: #1e293b; border-radius: 5px; border-left: 3px solid #475569; }
.signal-row.pass { border-left-color: #4ade80; }
.signal-row.fail { border-left-color: #f87171; }
.signal-row.unknown { border-left-color: #fbbf24; }
.sig-main { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.sig-name { font-size: 11px; color: #94a3b8; }
.sig-value { font-size: 13px; font-weight: 600; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sig-sub { font-size: 10px; color: #64748b; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sig-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600; white-space: nowrap; flex-shrink: 0; }
.sig-badge.pass { background: #064e3b; color: #4ade80; }
.sig-badge.fail { background: #450a0a; color: #f87171; }
.sig-badge.unknown { background: #422006; color: #fbbf24; }
.signal-overall { text-align: center; padding-top: 6px; border-top: 1px solid #1e293b; }
.overall-badge { display: inline-block; font-size: 14px; font-weight: 700; padding: 4px 16px; border-radius: 6px; }
.overall-badge.buy { background: #064e3b; color: #4ade80; border: 1px solid #4ade80; }
.overall-badge.no_buy { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
.overall-badge.waiting { background: #422006; color: #fbbf24; border: 1px solid #fbbf24; }
.signal-loading { text-align: center; color: #64748b; font-size: 13px; padding: 12px; }

/* Watchlist */
.watch-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.watch-table th { text-align: left; color: #94a3b8; font-size: 12px; padding: 8px 12px; border-bottom: 1px solid #334155; }
.watch-table td { padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }
.watch-table tr:hover { background: #334155; }

/* Guide */
.guide-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.guide-card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 18px; }
.guide-table { width: 100%; font-size: 13px; }
.guide-table td { padding: 6px 8px; color: #cbd5e1; border-bottom: 1px solid #1e293b; }
.guide-table .time { font-weight: 600; color: #60a5fa; white-space: nowrap; width: 80px; }
.guide-card ul { list-style: none; }
.guide-card li { padding: 4px 0; font-size: 13px; color: #cbd5e1; }
.guide-card li::before { content: '• '; color: #60a5fa; }
.guide-card li.tp::before { content: '• '; color: #4ade80; }
.guide-card li.sl::before { content: '• '; color: #f87171; }
.guide-card code { background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #93c5fd; }
.guide-card strong { color: #fff; }

/* Empty & Footer */
.empty { text-align: center; color: #94a3b8; padding: 40px; }
footer { text-align: center; margin-top: 32px; padding: 16px; color: #64748b; font-size: 12px; }
footer p { background: #1e293b; display: inline-block; padding: 8px 16px; border-radius: 8px; }

@media (max-width: 640px) {
  body { padding: 12px; }
  .card-scores { flex-direction: column; gap: 8px; }
  .price-row { flex-direction: column; }
  .guide-grid { grid-template-columns: 1fr; }
  .signal-row { flex-direction: column; align-items: flex-start; }
  .card-bottom-row { flex-direction: column; }
}
</style>"""

    def _realtime_js(self, orders: pd.DataFrame | None) -> str:
        """Generate JavaScript that fetches real-time data via JSONP from 东方财富 API.

        No server needed. Works with local file:// HTML.
        Uses JSONP to bypass CORS restrictions.
        """
        if orders is None or orders.empty:
            return ""
        codes = orders["code"].astype(str).str.zfill(6).tolist()
        # Build order info for client-side signal computation
        order_info = {}
        for _, r in orders.iterrows():
            code = str(r["code"]).zfill(6)
            order_info[code] = {
                "ref_price": float(r.get("ref_price", 0)),
                "prev_close": float(r.get("ref_price", 0)),  # ref_price = prev close
            }
        # Sector map from name_map context
        codes_json = json.dumps(codes)
        orders_json = json.dumps(order_info, ensure_ascii=False)
        return f'''<script>
const CODES = {codes_json};
const ORDERS = {orders_json};

// === JSONP 工具：绕过跨域限制 ===
function jsonp(url) {{
  return new Promise((resolve, reject) => {{
    const cbName = "_cb_" + Math.random().toString(36).slice(2);
    const timeout = setTimeout(() => {{ delete window[cbName]; reject("timeout"); }}, 10000);
    window[cbName] = (data) => {{
      clearTimeout(timeout);
      delete window[cbName];
      resolve(data);
    }};
    const s = document.createElement("script");
    s.src = url + (url.includes("?") ? "&" : "?") + "cb=" + cbName;
    s.onerror = () => {{ clearTimeout(timeout); reject("error"); }};
    document.head.appendChild(s);
    setTimeout(() => document.head.removeChild(s), 12000);
  }});
}}

// === 东方财富 API ===
function getSecId(code) {{
  // 沪市: 6xx, 688xxx; 深市: 0xx, 3xx
  if (code.startsWith("6")) return "1." + code;
  return "0." + code;
}}

async function fetchStockQuotes() {{
  const results = {{}};
  const promises = CODES.map(async (code) => {{
    try {{
      const secid = getSecId(code);
      const url = "https://push2.eastmoney.com/api/qt/stock/get?secid=" + secid +
        "&fields=f43,f44,f45,f46,f50,f57,f58,f60,f170,f47,f48";
      const data = await jsonp(url);
      if (data && data.data) {{
        const d = data.data;
        results[code] = {{
          name: d.f58 || "---",
          current: d.f43 / 100,   // 东财返回分为单位
          open: d.f46 / 100,
          prev_close: d.f60 / 100,
          high: d.f44 / 100,
          low: d.f45 / 100,
          volume_ratio: d.f50 / 100,
          pct_chg: d.f170 / 100,
        }};
      }}
    }} catch (e) {{ console.warn("fetch " + code + " failed", e); }}
  }});
  await Promise.all(promises);
  return results;
}}

async function fetchSectors() {{
  try {{
    const url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:90+t:2&fields=f3,f14,f104,f105";
    const data = await jsonp(url);
    if (data && data.data && data.data.diff) {{
      const sectors = {{}};
      data.data.diff.forEach(item => {{
        sectors[item.f14] = {{
          pct_chg: item.f3,
          up_count: item.f104,
          down_count: item.f105,
        }};
      }});
      return sectors;
    }}
  }} catch (e) {{ console.warn("sector fetch failed", e); }}
  return {{}};
}}

// === 板块模糊匹配 ===
function matchSector(sectorName, boardNames) {{
  if (!sectorName || sectorName === "未知") return null;
  if (boardNames.includes(sectorName)) return sectorName;
  const clean = sectorName.replace(/[行业]$/g, "");
  const found = boardNames.find(b => b === clean);
  if (found) return found;
  // 关键词匹配
  const kw = clean.length > 2 ? clean.slice(0, 2) : clean;
  if (kw.length < 2) return null;
  const candidates = boardNames.filter(b => b.includes(kw));
  return candidates.length ? candidates.sort((a,b) => a.length - b.length)[0] : null;
}}

// === 计算信号 ===
function computeSignals(quotes, sectors) {{
  const boardNames = Object.keys(sectors);
  const signals = {{}};
  const sectorTags = document.querySelectorAll(".sector-tag");
  // 从页面中提取板块名称
  const codeSectorMap = {{}};
  document.querySelectorAll(".stock-card").forEach(card => {{
    const codeEl = card.querySelector(".code");
    const sectorEl = card.querySelector(".sector-tag");
    if (codeEl && sectorEl) {{
      codeSectorMap[codeEl.textContent.trim()] = sectorEl.textContent.trim();
    }}
  }});

  for (const code of CODES) {{
    const q = quotes[code];
    const order = ORDERS[code] || {{}};
    const sectorName = codeSectorMap[code] || "未知";

    if (!q) {{
      signals[code] = {{ status: "no_data", indicators: [], overall: "unknown", overall_text: "⚠️ 无数据" }};
      continue;
    }}

    const prevClose = q.prev_close > 0 ? q.prev_close : order.ref_price;
    const minOpen = prevClose * 0.98;
    const indicators = [];

    // 指标1: 开盘价
    if (q.open > 0) {{
      const gap = ((q.open / prevClose) - 1) * 100;
      const pass = q.open >= minOpen;
      indicators.push({{ name: "开盘价", value: `¥${{q.open.toFixed(2)}} (${{gap >= 0 ? "+" : ""}}${{gap.toFixed(1)}}%)`, threshold: `≥ ¥${{minOpen.toFixed(2)}}`, pass, detail: pass ? "不低开超2%" : `低开${{gap.toFixed(1)}}%，超过阈值` }});
    }} else {{
      indicators.push({{ name: "开盘价", value: "未开盘", threshold: `≥ ¥${{minOpen.toFixed(2)}}`, pass: null, detail: "等待开盘" }});
    }}

    // 指标2: 量价关系
    const isKill = q.volume_ratio > 2.0 && q.current < q.open;
    const c2 = !isKill;
    indicators.push({{ name: "量价关系", value: `量比 ${{q.volume_ratio.toFixed(1)}} / 现价${{q.current >= q.open ? "↑" : "↓"}}`, threshold: "量比≤2 或 价格不跌破开盘", pass: c2, detail: c2 ? "正常" : "放量下杀（量比>2且破开盘价）" }});

    // 指标3: 站上昨收
    const c3 = q.current >= prevClose;
    const pct = ((q.current / prevClose) - 1) * 100;
    indicators.push({{ name: "站上昨收", value: `¥${{q.current.toFixed(2)}} (${{pct >= 0 ? "+" : ""}}${{pct.toFixed(1)}}%)`, threshold: `≥ ¥${{prevClose.toFixed(2)}}`, pass: c3, detail: c3 ? "已站上昨收" : `低于昨收${{Math.abs(pct).toFixed(1)}}%` }});

    // 指标4: 板块
    const matched = matchSector(sectorName, boardNames);
    if (matched && sectors[matched]) {{
      const s = sectors[matched];
      const weak = s.pct_chg < -1.0;
      const ratioBad = s.up_count > 0 && s.down_count / s.up_count > 2.0;
      const c4 = !(weak || ratioBad);
      let detail = "板块正常";
      if (!c4) {{
        const parts = [];
        if (weak) parts.push(`跌${{s.pct_chg.toFixed(1)}}%`);
        if (ratioBad) parts.push(`跌${{s.down_count}}/涨${{s.up_count}}`);
        detail = "板块弱势: " + parts.join(", ");
      }}
      indicators.push({{ name: "板块强弱", value: `[${{matched}}] ${{s.pct_chg >= 0 ? "+" : ""}}${{s.pct_chg.toFixed(1)}}% (涨${{s.up_count}}/跌${{s.down_count}})`, threshold: "涨跌幅≥-1% 且 跌/涨<2", pass: c4, detail }});
    }} else {{
      indicators.push({{ name: "板块强弱", value: `「${{sectorName}}」未匹配`, threshold: "需手动确认", pass: null, detail: "未匹配到东财板块" }});
    }}

    // 总信号
    const passed = indicators.filter(i => i.pass !== null).map(i => i.pass);
    const allPass = passed.length === 4 && passed.every(Boolean);
    const anyFail = passed.some(p => p === false);
    const overall = allPass ? "buy" : anyFail ? "no_buy" : "waiting";
    const overall_text = allPass ? "✅ 满足买入条件" : anyFail ? "❌ 不满足买入条件" : "⏳ 等待确认";

    signals[code] = {{ status: "ok", name: q.name, current: q.current, pct_chg: q.pct_chg, indicators, overall, overall_text }};
  }}
  return signals;
}}

// === 渲染 ===
function renderSignals(signals, timestamp) {{
  for (const code of CODES) {{
    const sig = signals[code];
    if (!sig || sig.status === "no_data") continue;

    const timeEl = document.getElementById("time-" + code);
    if (timeEl) timeEl.textContent = "更新: " + timestamp;

    const indEl = document.getElementById("indicators-" + code);
    if (indEl) {{
      let html = "";
      for (const ind of sig.indicators) {{
        const cls = ind.pass === true ? "pass" : ind.pass === false ? "fail" : "unknown";
        const badge = ind.pass === true ? "✅ 通过" : ind.pass === false ? "❌ 未达标" : "⏳ 待确认";
        html += `<div class="signal-row ${{cls}}"><div class="sig-main"><span class="sig-name">${{ind.name}}</span><span class="sig-value">${{ind.value}}</span></div><div class="sig-sub">${{ind.detail}}</div><span class="sig-badge ${{cls}}">${{badge}}</span></div>`;
      }}
      indEl.innerHTML = html;
    }}

    const overEl = document.getElementById("overall-" + code);
    if (overEl) {{
      overEl.innerHTML = `<span class="overall-badge ${{sig.overall}}">${{sig.overall_text}}</span>`;
    }}
  }}
}}

// === 启动 ===
function isMarketOpen() {{
  const now = new Date();
  const day = now.getDay(); // 0=周日, 6=周六
  if (day === 0 || day === 6) return false; // 周末
  const h = now.getHours(), m = now.getMinutes();
  const t = h * 100 + m;
  // 9:15 集合竞价开始 ~ 15:05 数据稳定
  return t >= 915 && t <= 1505;
}}

async function loadSignals() {{
  if (!isMarketOpen()) {{
    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-CN', {{hour:'2-digit',minute:'2-digit'}});
    const day = now.getDay();
    const dayNames = ['周日','周一','周二','周三','周四','周五','周六'];
    const msg = (day===0||day===6)
      ? `💤 周末不交易，实时信号开盘后可用（9:15起）`
      : `⏰ ${{dayNames[day]}} ${{timeStr}} 非交易时段，开盘后自动刷新（9:15~15:05）`;
    CODES.forEach(code => {{
      const el = document.getElementById('indicators-' + code);
      if (el) el.innerHTML = `<div class="signal-loading">${{msg}}</div>`;
      const ov = document.getElementById('overall-' + code);
      if (ov) ov.innerHTML = '<span class="overall-badge waiting">⏰ 等待开盘</span>';
      const tm = document.getElementById('time-' + code);
      if (tm) tm.textContent = '非交易时段';
    }});
    return;
  }}
  try {{
    document.querySelectorAll(".signal-loading").forEach(el => {{
      el.textContent = "⭐ 正在获取实时数据...";
    }});
    const [quotes, sectors] = await Promise.all([fetchStockQuotes(), fetchSectors()]);
    const signals = computeSignals(quotes, sectors);
    const now = new Date();
    const timestamp = now.getHours().toString().padStart(2,"0") + ":" + now.getMinutes().toString().padStart(2,"0") + ":" + now.getSeconds().toString().padStart(2,"0");
    renderSignals(signals, timestamp);
  }} catch (e) {{
    console.error(e);
    document.querySelectorAll(".signal-loading").forEach(el => {{
      el.textContent = "❌ 数据获取失败，请检查网络后刷新页面";
    }});
  }}
}}

// 页面加载即执行
loadSignals();
</script>'''

