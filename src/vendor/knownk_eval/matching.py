# -*- coding: utf-8 -*-
"""一对一最优匹配（独立实现，不 import 历史 evaluate.py）。

规则：固定距离门限 0.30 m + 1e-12 数值容差；先最大化匹配数，再在等匹配数下最小化
总距离；平局按配对序列 (truth_index, pred_index) 字典序最小；禁止自匹配（同一索引
不可配对两次，真值/预测指数不同即视为不同个体）。
"""

from __future__ import annotations

import math

from .protocol import MatchingRule


class MatchingError(ValueError):
    """真值 / 预测坐标不合规，或枚举规模超界（须显式失败，不得静默丢弃）。"""


def _coerce_points(points, name: str) -> list[tuple[float, float]]:
    out = []
    for idx, item in enumerate(points):
        try:
            x, y = item
        except (TypeError, ValueError):
            raise MatchingError("%s[%d] 必须是 (x, y) 二元组，收到 %r" % (name, idx, item)) from None
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise MatchingError("%s[%d] 含 NaN/Inf：%r" % (name, idx, item))
        out.append((xf, yf))
    return out


def match_targets(truth_xy, pred_xy, rule: MatchingRule):
    truth = _coerce_points(truth_xy, "truth_xy")
    preds = _coerce_points(pred_xy, "pred_xy")
    n_t, n_p = len(truth), len(preds)
    if n_t > rule.max_n_truth or n_p > rule.max_n_pred:
        raise MatchingError("匹配枚举规模超界：n_truth=%d (max %d) n_pred=%d (max %d)"
                            % (n_t, rule.max_n_truth, n_p, rule.max_n_pred))
    gate = float(rule.gate_m)
    effective_gate = gate + float(rule.gate_tolerance_m)
    dist = [[math.hypot(truth[i][0] - preds[j][0], truth[i][1] - preds[j][1])
             for j in range(n_p)] for i in range(n_t)]
    allowed = [[j for j in range(n_p) if dist[i][j] <= effective_gate] for i in range(n_t)]
    best = {"key": (float("inf"), float("inf"), ()), "pairs": []}

    def rec(i: int, used: set, pairs, total: float):
        if i == n_t:
            key = (-len(pairs), total, tuple(pairs))
            if key < best["key"]:
                best["key"] = key
                best["pairs"] = list(pairs)
            return
        rec(i + 1, used, pairs, total)
        for j in allowed[i]:
            if j in used:
                continue
            pairs.append((i, j))
            rec(i + 1, used | {j}, pairs, total + dist[i][j])
            pairs.pop()

    if n_t and n_p:
        rec(0, set(), [], 0.0)
    pairs = best["pairs"]
    matched_t = {t for t, _ in pairs}
    matched_p = {p for _, p in pairs}
    return {
        "pairs": [(int(t), int(p), float(dist[t][p])) for t, p in pairs],
        "tp": len(pairs),
        "fn": [t for t in range(n_t) if t not in matched_t],
        "fp": [p for p in range(n_p) if p not in matched_p],
        "total_distance": float(sum(dist[t][p] for t, p in pairs)),
        "gate_m": gate,
        "effective_gate_m": float(effective_gate),
        "truth": truth,
        "predictions": preds,
    }
