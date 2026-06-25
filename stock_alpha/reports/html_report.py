from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html

import pandas as pd
from stock_alpha.reports.charts import add_drawdown, svg_line_chart


@dataclass
class HtmlReportGenerator:
    out_dir: Path | str = Path("reports")

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _df_html(self, df: pd.DataFrame | None, max_rows: int = 30) -> str:
        if df is None or df.empty:
            return "<p>无</p>"
        return df.head(max_rows).to_html(index=False, border=0, classes="table")

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
    ) -> Path:
        df = predictions.copy()
        df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        d = pd.to_datetime(trade_date) if trade_date else df["date"].max()
        day = df[df["date"] == d].copy()
        if "risk_blocked" in day.columns:
            day = day[~day["risk_blocked"].fillna(False)]
        score_col = "final_score_v2" if "final_score_v2" in day.columns else "final_score"
        day = day.sort_values(score_col, ascending=False).head(top_n)
        cols = [c for c in ["code", "up_probability", "down_probability", "risk_score", "intraday_score", score_col, "suggest_action", "suggest_action_v2"] if c in day.columns]
        m = backtest_metrics.iloc[0].to_dict() if backtest_metrics is not None and not backtest_metrics.empty else {}
        t = trade_stats.iloc[0].to_dict() if trade_stats is not None and not trade_stats.empty else {}
        cards = [
            ("总收益", f"{m.get('total_return', 0):.2%}"),
            ("年化收益", f"{m.get('annual_return', 0):.2%}"),
            ("最大回撤", f"{m.get('max_drawdown', 0):.2%}"),
            ("Sharpe", f"{m.get('sharpe', 0):.3f}"),
            ("胜率", f"{t.get('win_rate', 0):.2%}" if pd.notna(t.get('win_rate', None)) else "无"),
            ("盈亏比", f"{t.get('profit_loss_ratio', 0):.3f}" if pd.notna(t.get('profit_loss_ratio', None)) else "无"),
        ]
        cards_html = "".join(f"<div class='card'><div class='label'>{html.escape(k)}</div><div class='value'>{html.escape(v)}</div></div>" for k, v in cards)
        eq_chart = svg_line_chart(equity, "equity", "权益曲线", color="#2563eb") if equity is not None and not equity.empty else "<p>无权益数据</p>"
        dd_df = add_drawdown(equity) if equity is not None and not equity.empty else pd.DataFrame()
        dd_chart = svg_line_chart(dd_df, "drawdown", "回撤曲线", color="#dc2626") if not dd_df.empty else "<p>无回撤数据</p>"
        turnover_chart = svg_line_chart(turnover, "turnover", "TopN 换手率", color="#7c3aed") if turnover is not None and not turnover.empty else "<p>无换手率数据</p>"
        content = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>A股短线模型报告 {d.strftime('%Y-%m-%d')}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f7f8fa;color:#1f2937}}
h1{{margin-bottom:4px}} .sub{{color:#6b7280;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0}}
.card{{background:white;border-radius:12px;padding:16px;box-shadow:0 1px 4px #0001}}
.label{{font-size:12px;color:#6b7280}} .value{{font-size:22px;font-weight:700;margin-top:6px}}
section{{background:white;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 1px 4px #0001}}
.table{{border-collapse:collapse;width:100%;font-size:13px}} .table th,.table td{{border-bottom:1px solid #e5e7eb;padding:8px;text-align:right}} .table th:first-child,.table td:first-child{{text-align:left}}
</style></head><body>
<h1>A股短线模型报告</h1><div class='sub'>{d.strftime('%Y-%m-%d')} · 候选 {len(day)} 只</div>
<div class='grid'>{cards_html}</div>
<section><h2>图表</h2>{eq_chart}{dd_chart}{turnover_chart}</section>
<section><h2>Top 候选</h2>{self._df_html(day[cols] if cols else day, top_n)}</section>
<section><h2>Top 特征重要性</h2>{self._df_html(feature_importance, 15)}</section>
<section><h2>数据质量摘要</h2>{self._df_html(quality_summary, 10)}</section>
<section><h2>月度收益</h2>{self._df_html(monthly, 24)}</section>
<section><h2>次日交易计划</h2>{self._df_html(orders, 30)}</section>
<section><h2>候选风险标签</h2>{self._df_html(risk_tags, 30)}</section>
<section><h2>为什么入选</h2>{self._df_html(explanations, 30)}</section>
<section><h2>信号稳定性</h2>{self._df_html(signal_stability, 30)}</section>
<section><h2>TopN 换手率</h2>{self._df_html(turnover.tail(30) if turnover is not None and not turnover.empty else turnover, 30)}</section>
<section><h2>最近持仓</h2>{self._df_html(holdings.tail(50) if holdings is not None and not holdings.empty else holdings, 50)}</section>
<section><h2>最近交易</h2>{self._df_html(trades.tail(50) if trades is not None and not trades.empty else trades, 50)}</section>
<section><h2>权益曲线明细</h2>{self._df_html(equity.tail(30) if equity is not None and not equity.empty else equity, 30)}</section>
<section><h2>风险提示</h2><p>本报告为模型信号，不构成投资建议；实盘需结合流动性、涨跌停、仓位和止损。</p></section>
</body></html>"""
        path = self.out_dir / f"daily_report_{d.strftime('%Y%m%d')}.html"
        path.write_text(content, encoding="utf-8")
        return path
