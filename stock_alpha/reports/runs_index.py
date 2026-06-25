from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def collect_runs(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root) / "runs"
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for summary in root.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = summary.parent
        events = run_dir / "events.jsonl"
        metrics = None
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                    if e.get("step") == "backtest" and e.get("status") == "done":
                        ms = e.get("metrics") or []
                        metrics = ms[0] if ms else None
                except Exception:
                    pass
        rows.append({**data, **(metrics or {})})
    return pd.DataFrame(rows)


def write_runs_index(data_root: str | Path, out_path: str | Path | None = None) -> Path:
    df = collect_runs(data_root)
    out = Path(out_path) if out_path else Path(data_root) / "runs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        html = "<html><body><h1>Runs</h1><p>No runs.</p></body></html>"
    else:
        cols = [c for c in ["run_id", "total_return", "annual_return", "max_drawdown", "sharpe", "trade_count", "model", "archived_html", "recommended_config"] if c in df.columns]
        table = df[cols].sort_values("run_id", ascending=False).to_html(index=False, border=0, classes="table")
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Runs Index</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f7f8fa}} section{{background:#fff;padding:16px;border-radius:12px}} .table{{border-collapse:collapse;width:100%;font-size:13px}} .table th,.table td{{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}}</style>
</head><body><h1>量化运行历史</h1><section>{table}</section></body></html>"""
    out.write_text(html, encoding="utf-8")
    return out
