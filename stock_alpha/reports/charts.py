from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _scale(vals, width: int, height: int, pad: int = 20):
    vals = list(vals)
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if math.isclose(mx, mn):
        mx = mn + 1e-9
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = pad + (width - 2 * pad) * (i / max(n - 1, 1))
        y = pad + (height - 2 * pad) * (1 - (v - mn) / (mx - mn))
        pts.append((x, y))
    return pts


def svg_line_chart(df: pd.DataFrame, y_col: str, title: str, width: int = 860, height: int = 260, color: str = "#2563eb") -> str:
    if df is None or df.empty or y_col not in df.columns:
        return "<p>无图表数据</p>"
    x = df.copy()
    vals = pd.to_numeric(x[y_col], errors="coerce").dropna().tolist()
    if not vals:
        return "<p>无图表数据</p>"
    pts = _scale(vals, width, height)
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    mn, mx = min(vals), max(vals)
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="20" y="18" font-size="14" font-weight="700" fill="#111827">{title}</text>
  <text x="20" y="40" font-size="11" fill="#6b7280">min={mn:.4f} max={mx:.4f}</text>
  <line x1="20" y1="{height-20}" x2="{width-20}" y2="{height-20}" stroke="#e5e7eb"/>
  <line x1="20" y1="20" x2="20" y2="{height-20}" stroke="#e5e7eb"/>
  <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>
</svg>"""


def add_drawdown(equity: pd.DataFrame) -> pd.DataFrame:
    if equity is None or equity.empty or "equity" not in equity.columns:
        return pd.DataFrame()
    df = equity.copy()
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df
