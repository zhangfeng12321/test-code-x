from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DailyReportGenerator:
    out_dir: Path | str = Path("reports")

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _table(self, df: pd.DataFrame, max_rows: int = 10) -> list[str]:
        if df is None or df.empty:
            return ["无"]
        return df.head(max_rows).to_markdown(index=False).splitlines()

    def generate(
        self,
        predictions: pd.DataFrame,
        trade_date: str | None = None,
        top_n: int = 20,
        only_buy: bool = False,
        backtest_metrics: pd.DataFrame | None = None,
        trade_stats: pd.DataFrame | None = None,
        feature_importance: pd.DataFrame | None = None,
        quality_summary: pd.DataFrame | None = None,
        risk_tags: pd.DataFrame | None = None,
        explanations: pd.DataFrame | None = None,
        orders: pd.DataFrame | None = None,
        watchlist: pd.DataFrame | None = None,
    ) -> Path:
        df = predictions.copy()
        df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        d = pd.to_datetime(trade_date) if trade_date else df["date"].max()
        day_df = df[df["date"] == d].copy()
        if "risk_blocked" in day_df.columns:
            day_df = day_df[~day_df["risk_blocked"].fillna(False)]
        buy_df = day_df[day_df.get("suggest_action", "") == "BUY"].sort_values("final_score", ascending=False)
        watch_df = day_df[day_df.get("suggest_action", "") != "BUY"].sort_values("final_score", ascending=False)
        picks = buy_df if only_buy else pd.concat([buy_df, watch_df]).head(top_n)
        path = self.out_dir / f"daily_report_{d.strftime('%Y%m%d')}.md"
        score_col = "final_score_v2" if "final_score_v2" in picks.columns else "final_score"
        lines = [
            f"# A股短线模型每日选股报告 - {d.strftime('%Y-%m-%d')}",
            "",
            "## 概览",
            "",
            f"- 当日候选数：{len(day_df)}",
            f"- BUY 信号数：{len(buy_df)}",
            f"- 输出 TopN：{min(top_n, len(picks))}",
            "",
        ]
        if backtest_metrics is not None and not backtest_metrics.empty:
            m = backtest_metrics.iloc[0]
            lines += [
                "## 回测摘要",
                "",
                f"- 总收益：{m.get('total_return', 0):.2%}",
                f"- 年化收益：{m.get('annual_return', 0):.2%}",
                f"- 最大回撤：{m.get('max_drawdown', 0):.2%}",
                f"- Sharpe：{m.get('sharpe', 0):.3f}",
                f"- 交易次数：{int(m.get('trade_count', 0))}",
                "",
            ]
        if trade_stats is not None and not trade_stats.empty:
            t = trade_stats.iloc[0]
            lines += [
                "## 交易统计",
                "",
                f"- 完整买卖回合：{int(t.get('round_trips', 0))}",
                f"- 胜率：{t.get('win_rate', 0):.2%}" if pd.notna(t.get('win_rate', None)) else "- 胜率：无",
                f"- 盈亏比：{t.get('profit_loss_ratio', 0):.3f}" if pd.notna(t.get('profit_loss_ratio', None)) else "- 盈亏比：无",
                f"- 最大连续亏损：{int(t.get('max_consecutive_losses', 0))}",
                "",
            ]
        lines += [
            "## Top 候选",
            "",
            "| 排名 | 股票 | 上涨概率 | 下跌概率 | 风险分 | V2分时 | 综合分 | 建议 |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for i, r in enumerate(picks.head(top_n).itertuples(index=False), 1):
            lines.append(
                f"| {i} | {r.code} | {getattr(r, 'up_probability', 0):.2%} | "
                f"{getattr(r, 'down_probability', 0):.2%} | {getattr(r, 'risk_score', 0):.3f} | "
                f"{getattr(r, 'intraday_score', 0):.3f} | {getattr(r, score_col, 0):.4f} | {getattr(r, 'suggest_action_v2', getattr(r, 'suggest_action', ''))} |"
            )
        if picks.empty:
            lines.append("| - | 无 | - | - | - | - | - | - |")
        if orders is not None and not orders.empty:
            lines += ["", "## 次日交易计划", ""] + self._table(orders, 20)
        if watchlist is not None and not watchlist.empty:
            lines += ["", "## 观察池（需次日盘中确认，不是买入清单）", ""] + self._table(watchlist, 30)
        if risk_tags is not None and not risk_tags.empty:
            lines += ["", "## 候选风险标签", ""] + self._table(risk_tags, 20)
        if explanations is not None and not explanations.empty:
            lines += ["", "## 为什么入选", ""] + self._table(explanations, 20)
        if feature_importance is not None and not feature_importance.empty:
            lines += ["", "## Top 特征重要性", ""] + self._table(feature_importance, 10)
        if quality_summary is not None and not quality_summary.empty:
            lines += ["", "## 数据质量摘要", ""] + self._table(quality_summary, 10)
        lines += [
            "",
            "## 风险提示",
            "",
            "- 本报告为模型信号，不构成投资建议。",
            "- 实盘前必须结合涨跌停、流动性、仓位、止损规则。",
            "- 免费数据源可能存在延迟、缺失或接口变动。",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
