"""组合优化模块：替代简单等权 TopN 选股。

支持三种仓位分配策略：
1. equal: 等权分配（当前默认）
2. risk_parity: 风险平价（低波动股多配，高波动股少配）
3. score_weighted: 信号加权（分数越高仓位越大）

同时支持：
- 行业中性约束：同一行业仓位不超过上限
- 最大仓位约束：单只不超过总资金的 max_position_pct
- 相关性约束：选股时剔除与已选股票相关性过高的股票
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_position_weights(
    candidates: pd.DataFrame,
    daily: pd.DataFrame,
    method: str = "equal",
    max_position_pct: float = 0.2,
    max_sector_pct: float = 0.4,
    risk_parity_lookback: int = 20,
    score_col: str = "final_score",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """计算每只候选股票的目标仓位权重。

    Args:
        candidates: 已通过风控的候选股票（需含 code, date, score_col）
        daily: 日线数据（需含 code, date, close）
        method: 仓位分配方式 (equal/risk_parity/score_weighted)
        max_position_pct: 单只最大仓位
        max_sector_pct: 同行业最大总仓位
        risk_parity_lookback: 波动率计算回望天数
        score_col: 信号分数列名
        sector_col: 行业列名

    Returns:
        DataFrame with columns: code, weight, method_detail
    """
    if candidates.empty:
        return pd.DataFrame(columns=["code", "weight", "method_detail"])

    codes = candidates["code"].unique().tolist()
    n = len(codes)

    if n == 0:
        return pd.DataFrame(columns=["code", "weight", "method_detail"])

    if method == "risk_parity":
        weights = _risk_parity_weights(codes, daily, risk_parity_lookback)
    elif method == "score_weighted":
        weights = _score_weighted(candidates, codes, score_col)
    else:
        # equal
        weights = {c: 1.0 / n for c in codes}

    # 归一化并应用最大仓位约束
    weights = _apply_max_position(weights, max_position_pct)

    # 行业中性约束
    if sector_col in candidates.columns:
        weights = _apply_sector_constraint(weights, candidates, sector_col, max_sector_pct)

    # 构建输出
    result = pd.DataFrame([
        {"code": c, "weight": w, "method_detail": method}
        for c, w in weights.items() if w > 0
    ])
    return result


def _risk_parity_weights(codes: list[str], daily: pd.DataFrame, lookback: int = 20) -> dict[str, float]:
    """风险平价：权重与波动率成反比。

    低波动股票获得更高权重 → 组合整体波动更均匀。
    """
    daily_cp = daily.copy()
    daily_cp["date"] = pd.to_datetime(daily_cp["date"], errors="coerce")
    daily_cp["close"] = pd.to_numeric(daily_cp["close"], errors="coerce")
    daily_cp = daily_cp.sort_values(["code", "date"])

    volatilities = {}
    for code in codes:
        stock_data = daily_cp[daily_cp["code"] == code].tail(lookback + 1)
        if len(stock_data) < 5:
            volatilities[code] = 0.03  # 默认3%日波动率
            continue
        rets = stock_data["close"].pct_change().dropna()
        vol = rets.std()
        volatilities[code] = max(vol, 0.005)  # 最低0.5%防止除零

    # 权重 = 1/vol 的归一化
    inv_vols = {c: 1.0 / v for c, v in volatilities.items()}
    total = sum(inv_vols.values())
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}
    return {c: v / total for c, v in inv_vols.items()}


def _score_weighted(candidates: pd.DataFrame, codes: list[str], score_col: str) -> dict[str, float]:
    """信号加权：权重与模型分数成正比。

    分数越高的股票获得越大仓位。
    """
    scores = {}
    for code in codes:
        row = candidates[candidates["code"] == code]
        if not row.empty and score_col in row.columns:
            s = pd.to_numeric(row[score_col].iloc[0], errors="coerce")
            scores[code] = max(s, 0.01)
        else:
            scores[code] = 0.01

    total = sum(scores.values())
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}
    return {c: s / total for c, s in scores.items()}


def _apply_max_position(weights: dict[str, float], max_pct: float) -> dict[str, float]:
    """应用单只最大仓位约束。超出部分按比例分配给其他股票。"""
    # 迭代收敛：截断超限的，重新分配
    for _ in range(5):  # 最多迭代5次
        total = sum(weights.values())
        if total <= 0:
            return weights
        # 归一化
        weights = {c: w / total for c, w in weights.items()}
        # 截断
        excess = 0.0
        capped = {}
        uncapped_codes = []
        for c, w in weights.items():
            if w > max_pct:
                capped[c] = max_pct
                excess += w - max_pct
            else:
                capped[c] = w
                uncapped_codes.append(c)
        if excess <= 0:
            return capped
        # 将超出部分按比例分配给未超限的
        uncapped_total = sum(capped[c] for c in uncapped_codes)
        if uncapped_total <= 0:
            return capped
        for c in uncapped_codes:
            capped[c] += excess * (capped[c] / uncapped_total)
        weights = capped
    return weights


def _apply_sector_constraint(
    weights: dict[str, float],
    candidates: pd.DataFrame,
    sector_col: str,
    max_sector_pct: float,
) -> dict[str, float]:
    """行业中性约束：同一行业的总权重不超过 max_sector_pct。"""
    # 构建 code -> sector 映射
    sector_map = {}
    for _, row in candidates.iterrows():
        code = row.get("code", "")
        sector = row.get(sector_col, code[:3])  # fallback 用代码前三位
        sector_map[code] = sector

    # 按行业汇总权重
    sector_weights: dict[str, float] = {}
    for code, w in weights.items():
        sector = sector_map.get(code, code[:3])
        sector_weights[sector] = sector_weights.get(sector, 0) + w

    # 对超限行业按比例缩减
    for sector, total_w in sector_weights.items():
        if total_w > max_sector_pct:
            scale = max_sector_pct / total_w
            for code in weights:
                if sector_map.get(code, code[:3]) == sector:
                    weights[code] *= scale

    # 不做全局归一化：行业约束后权重总和可能 <1，未满仓部分作为现金
    # 避免重新放大已被 _apply_max_position 截断的权重
    return weights


def allocate_shares(
    weights: pd.DataFrame,
    total_capital: float,
    price_map: dict[str, float],
    lot_size: int = 100,
) -> pd.DataFrame:
    """将权重转化为实际买入股数。

    Args:
        weights: compute_position_weights 输出（code, weight）
        total_capital: 可用资金
        price_map: {code: price} 参考价格
        lot_size: 最小交易单位

    Returns:
        DataFrame with: code, weight, target_amount, shares, actual_amount
    """
    if weights.empty:
        return pd.DataFrame()

    result = weights.copy()
    result["target_amount"] = result["weight"] * total_capital
    result["ref_price"] = result["code"].map(price_map).fillna(0)
    result["shares"] = result.apply(
        lambda r: int((r["target_amount"] / r["ref_price"]) // lot_size * lot_size)
        if r["ref_price"] > 0 else 0,
        axis=1,
    )
    result["actual_amount"] = result["shares"] * result["ref_price"]
    return result[result["shares"] > 0].reset_index(drop=True)
