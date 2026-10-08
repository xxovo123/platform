# -*- coding: utf-8 -*-
"""独立通用仿真指标：known-K 匹配 + TP/FN/FP + 条件误差 + 标准 OSPA。

口径冻结来源（**不重新定义**，逐条对齐）：

* ``大修执行_20260923/01_投稿口径/p0b/protocol/knownk_protocol.json``

  - ``matching``：先最大化匹配数，再在等匹配数下最小化 **总距离（非平方）**；
    ``gate_m = 0.30``、``gate_tolerance_m = 1e-12``（``dist <= gate + tol``）；
    平局按配对序列 ``(truth_index, pred_index)`` 字典序最小；禁止自匹配；
    枚举上限 ``n_truth <= 8``、``n_pred <= 16``。
  - ``metrics.conventions``：视轴 = ``-Y``，``theta > 0`` = ``+X`` 侧；
    ``r = hypot(x, y)``、``theta_deg = degrees(atan2(x, -y))``；角度差 wrap 到 ``(-180, 180]``。
  - 逐目标误差：``e_r = abs(r_hat - r)``、``e_theta = abs(wrap(theta_hat - theta))``、
    ``e_xy = hypot(dx, dy)``。误差**只条件于匹配 TP**。
  - 失败窗保留在分母：``tp=0, fn=n_truth, fp=0``，状态标 ``failed``。

* ``大修执行_20260923/04_Q6/hardware/位置综合误差_OSPA方案_20260924.md``

  - 标准 OSPA ``p=2``、``c=0.30`` m：**先截断平方、再求最优单射**；
    空点集按定义处理（都空 -> 0；一边空一边非空 -> ``c``）。
  - **不复用 TP gate 匹配**：OSPA 的指派独立于 ``match_by_gate`` 的结果。
  - 跨 trial：``RMS-OSPA = sqrt(mean_t OSPA_t^2)``，不是 ``mean_t OSPA_t``。

本模块是仿真侧独立实现，**不 import** ``knownk_eval`` / ``ospa_postprocess``。

公共接口
--------
``evaluate_predictions(truth_xy, pred_xy, resolution_distance_m=None, status='ok') -> dict``
``aggregate(rows) -> dict``

分母纪律（**五点正式集全覆盖**）
--------------------------------
* 失败 / 空输出 / ``not_applicable`` 一律**不删窗**：``status_used='empty'`` -> ``tp=0, fn=K, fp=0``、
  ``OSPA_t = c``、``success=False``，全部计入主 summary 分母。
* 缺 ``metrics`` 的行按行的 ``n_truth`` / ``k`` 补全为 empty/failed；**连真值数量都没有则显式 raise**，
  不产出看似有效却缩小 DR 分母的 summary。
* ``n_not_applicable`` 单列；主 summary 仍按 empty 计入。另给可选条件子表
  ``conditional_excluding_not_applicable``（不改变主口径）。

命名说明
--------
``fp_per_estimate`` 统一为 **FP per window**（沿用 knownk_protocol：``sum(FP)/有效估计窗数``）；
逐预测的虚警占比另名 ``false_discovery_fraction = FP / n_predictions``。
"""

from __future__ import annotations

import itertools
import math

import numpy as np

try:  # scipy 只用于 OSPA 的截断平方最优指派；缺失时有纯 Python 回退
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover - 环境缺 scipy 时走回退
    _lsa = None

SCHEMA_VERSION = 1

#: TP 匹配门限（knownk_protocol.json matching.gate_m）
DEFAULT_GATE_M = 0.30
#: 门限数值容差（knownk_protocol.json matching.gate_tolerance_m）
GATE_TOLERANCE_M = 1e-12
#: 匹配枚举上限（knownk_protocol.json matching.enumerate_limits）
MAX_N_TRUTH = 8
MAX_N_PRED = 16
#: OSPA 参数（位置综合误差_OSPA方案_20260924.md §2）
OSPA_P = 2
OSPA_C_M = 0.30
#: E2 分辨率成功判据（仿真实验完整方案 §E2）：gate = min(0.10, d/3)
RESOLUTION_ABS_GATE_M = 0.10
RESOLUTION_REL_DIVISOR = 3.0


class MetricError(ValueError):
    """真值/预测不合规或枚举规模超界（显式失败，不静默丢弃）。"""


# --------------------------------------------------------------------------- #
# 坐标口径（不重新定义）
# --------------------------------------------------------------------------- #
def wrap_deg(angle_deg: float) -> float:
    """把角度差 wrap 到 (-180, 180]；-180 归到 +180（与 knownk_protocol 一致）。"""
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if value <= -180.0 + 1e-12:
        value = 180.0
    return float(value)


def polar_of(x: float, y: float):
    """视轴 = -Y、theta>0 = +X 侧：返回 (r, theta_deg)。"""
    return (math.hypot(float(x), float(y)),
            math.degrees(math.atan2(float(x), -float(y))))


def _coerce_points(points, name: str):
    # 显式 None 检查：numpy 数组不能做 ``points or []``（会触发 ambiguous truth value）
    if points is None:
        return []
    out = []
    for idx, item in enumerate(points):
        try:
            x, y = item
        except (TypeError, ValueError):
            raise MetricError("%s[%d] 必须是 (x, y) 二元组，收到 %r" % (name, idx, item)) from None
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise MetricError("%s[%d] 含 NaN/Inf：%r" % (name, idx, item))
        out.append((xf, yf))
    return out


# --------------------------------------------------------------------------- #
# TP 匹配：先最大化匹配数，再最小化总距离（非平方），平局序列字典序最小
# --------------------------------------------------------------------------- #
def match_by_gate(truth_xy, pred_xy, gate_m: float = DEFAULT_GATE_M,
                  gate_tolerance_m: float = GATE_TOLERANCE_M,
                  max_n_truth: int = MAX_N_TRUTH, max_n_pred: int = MAX_N_PRED) -> dict:
    """known-K 一对一精确匹配；返回 pairs/tp/fn/fp/total_distance。"""
    truth = _coerce_points(truth_xy, "truth_xy")
    preds = _coerce_points(pred_xy, "pred_xy")
    n_t, n_p = len(truth), len(preds)
    if n_t > int(max_n_truth) or n_p > int(max_n_pred):
        raise MetricError("匹配枚举规模超界：n_truth=%d (max %d) n_pred=%d (max %d)"
                          % (n_t, max_n_truth, n_p, max_n_pred))
    effective_gate = float(gate_m) + float(gate_tolerance_m)
    dist = [[math.hypot(truth[i][0] - preds[j][0], truth[i][1] - preds[j][1])
             for j in range(n_p)] for i in range(n_t)]
    allowed = [[j for j in range(n_p) if dist[i][j] <= effective_gate] for i in range(n_t)]
    best = {"key": (float("inf"), float("inf"), ()), "pairs": []}

    def rec(i: int, used, pairs, total: float):
        if i == n_t:
            key = (-len(pairs), total, tuple(pairs))
            if key < best["key"]:
                best["key"] = key
                best["pairs"] = list(pairs)
            return
        rec(i + 1, used, pairs, total)  # 允许该真值不匹配
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
        "gate_m": float(gate_m),
        "effective_gate_m": float(effective_gate),
        "truth": truth,
        "predictions": preds,
    }


# --------------------------------------------------------------------------- #
# 标准 OSPA（p=2, c=0.30）：先截断平方，再求最优单射；独立于 TP gate 匹配
# --------------------------------------------------------------------------- #
def _min_assignment_cost(cost: np.ndarray) -> float:
    """矩形代价矩阵的最小指派（规模 = min(m,n)），返回被指派项代价之和。"""
    m, n = cost.shape
    if _lsa is not None:
        rows, cols = _lsa(cost)
        return float(cost[rows, cols].sum())
    if m <= n:
        return float(min(sum(cost[i, perm[i]] for i in range(m))
                         for perm in itertools.permutations(range(n), m)))
    return float(min(sum(cost[perm[j], j] for j in range(n))
                     for perm in itertools.permutations(range(m), n)))


def ospa_squared(truth_xy, pred_xy, c: float = OSPA_C_M, p: int = OSPA_P) -> float:
    """标准 OSPA^p（此处 p=2）：空集按定义处理，不抛错、不删窗。"""
    X = _coerce_points(truth_xy, "truth_xy")
    Y = _coerce_points(pred_xy, "pred_xy")
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c ** p)
    d = np.array([[math.hypot(X[i][0] - Y[j][0], X[i][1] - Y[j][1])
                   for j in range(n)] for i in range(m)], dtype=float)
    cost = np.minimum(d, float(c)) ** int(p)
    matched = _min_assignment_cost(cost)
    card = abs(m - n) * float(c) ** int(p)
    return (matched + card) / float(max(m, n))


def ospa(truth_xy, pred_xy, c: float = OSPA_C_M, p: int = OSPA_P) -> float:
    return math.sqrt(ospa_squared(truth_xy, pred_xy, c=c, p=p))


# --------------------------------------------------------------------------- #
# 逐窗评估
# --------------------------------------------------------------------------- #
def _error_row(ti, pi, truth, pred) -> dict:
    dx = float(pred[0] - truth[0])
    dy = float(pred[1] - truth[1])
    r_t, th_t = polar_of(truth[0], truth[1])
    r_p, th_p = polar_of(pred[0], pred[1])
    return {
        "truth_index": int(ti),
        "pred_index": int(pi),
        "e_xy": float(math.hypot(dx, dy)),
        "e_r": float(abs(r_p - r_t)),
        "e_theta": float(abs(wrap_deg(th_p - th_t))),
        "distance_m": float(math.hypot(dx, dy)),
    }


def evaluate_predictions(truth_xy, pred_xy, resolution_distance_m=None, status='ok') -> dict:
    """单窗指标（known-K）。返回 JSON 可序列化 dict（字段见模块 docstring / README）。

    * ``status != 'ok'`` -> 预测集按空处理，仍计入分母（failure 不漏窗）。
    * ``resolution_distance_m is None`` -> 通用口径（gate 0.30 m）。
    * ``resolution_distance_m = d``（必须有限且 > 0）-> E2 分辨率口径，``gate = min(0.10, d/3)``。
    """
    truth = _coerce_points(truth_xy, "truth_xy")
    status_in = str(status)
    ok = status_in == "ok"
    preds = _coerce_points(pred_xy, "pred_xy") if ok else []

    m = match_by_gate(truth, preds)
    n_t, n_p = len(truth), len(preds)
    tp, fn, fp = int(m["tp"]), int(len(m["fn"])), int(len(m["fp"]))
    ospa2 = ospa_squared(truth, preds)

    errors = [_error_row(ti, pi, truth[ti], preds[pi]) for ti, pi, _d in m["pairs"]]
    fn_items = [{"truth_index": int(t), "truth_xy": [float(truth[t][0]), float(truth[t][1])]}
                for t in m["fn"]]
    fp_items = [{"pred_index": int(p),
                 "prediction_xy": [float(preds[p][0]), float(preds[p][1])]} for p in m["fp"]]

    resolution = None
    if resolution_distance_m is not None:
        d = float(resolution_distance_m)
        if not math.isfinite(d) or d <= 0.0:
            raise MetricError("resolution_distance_m 必须是有限且 > 0 的数，收到 %r" % (resolution_distance_m,))
        res_gate = min(RESOLUTION_ABS_GATE_M, d / RESOLUTION_REL_DIVISOR)
        exactly_two = (n_t == 2 and n_p == 2)
        max_err = None
        res_success = False
        if exactly_two:
            dmat = [[math.hypot(truth[i][0] - preds[j][0], truth[i][1] - preds[j][1])
                     for j in range(2)] for i in range(2)]
            perms = ((0, 1), (1, 0))
            res_success = any(all(dmat[i][perm[i]] <= res_gate + GATE_TOLERANCE_M for i in range(2))
                              for perm in perms)
            max_err = float(min(max(dmat[i][perm[i]] for i in range(2)) for perm in perms))
        resolution = {
            "distance_m": float(d),
            "gate_m": float(res_gate),
            "exactly_two": bool(exactly_two),
            "success": bool(res_success),
            "max_matched_error_m": max_err,
        }
        success = bool(res_success)
        success_rule = "resolution_exactly2_and_all_le_min(0.10,d/3)"
    else:
        # K=0（无真值）不构成“整场景成功”
        success = bool(n_t > 0 and n_p == n_t and fn == 0 and fp == 0)
        success_rule = "whole_scene_count_equals_K_gt0_and_all_matched_within_gate"

    return {
        "schema_version": SCHEMA_VERSION,
        "status_in": status_in,
        "status_used": "ok" if ok else "empty",
        "n_truth": int(n_t),
        "n_predictions": int(n_p),
        "gate_m": float(DEFAULT_GATE_M),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "dr": (tp / n_t) if n_t else None,
        "miss_rate": (fn / n_t) if n_t else None,
        # fp_per_estimate 统一为 FP per window（单窗即 fp）；逐预测虚警占比另名
        "fp_per_window": int(fp),
        "fp_per_estimate": int(fp),
        "false_discovery_fraction": (fp / n_p) if n_p else None,
        "ospa2": float(ospa2),
        "ospa": float(math.sqrt(ospa2)),
        "success": success,
        "success_rule": success_rule,
        "resolution": resolution,
        "errors": errors,
        "fn_items": fn_items,
        "fp_items": fp_items,
    }


# --------------------------------------------------------------------------- #
# 跨 trial 聚合
# --------------------------------------------------------------------------- #
_TRUTH_COUNT_KEYS = ("n_truth", "k", "K", "n_targets")


def _synthesise_empty(status: str, k: int) -> dict:
    """缺 metrics 或失败行：按 k 补全为 empty（failed/NA 一致计分母）。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "status_in": status,
        "status_used": "empty",
        "n_truth": int(k),
        "n_predictions": 0,
        "gate_m": float(DEFAULT_GATE_M),
        "tp": 0,
        "fn": int(k),
        "fp": 0,
        "dr": (0.0 if k else None),
        "miss_rate": (1.0 if k else None),
        "fp_per_window": 0,
        "fp_per_estimate": 0,
        "false_discovery_fraction": None,
        "ospa2": (float(OSPA_C_M) ** OSPA_P if k else 0.0),
        "ospa": (float(OSPA_C_M) if k else 0.0),
        "success": False,
        "success_rule": "missing_metrics_synthesised_empty",
        "resolution": None,
        "errors": [],
        "fn_items": [],
        "fp_items": [],
        "synthesised_from_missing_metrics": True,
    }


def _row_to_metrics(row):
    """一行 -> (metrics dict, source)。缺 metrics 时按 n_truth/K 补全；无真值数量则 raise。"""
    if not isinstance(row, dict):
        raise MetricError("aggregate 收到非 dict 行：%r（既无 metrics 也无 n_truth）"
                          % (type(row).__name__,))
    inner = row.get("metrics")
    if isinstance(inner, dict) and ("ospa2" in inner or ("n_truth" in inner and "tp" in inner)):
        return inner, "metrics"
    if "ospa2" in row or ("n_truth" in row and "tp" in row):
        return row, "row"
    src = inner if isinstance(inner, dict) else row
    k = None
    for key in _TRUTH_COUNT_KEYS:
        if key in src and src[key] is not None:
            try:
                k = int(src[key])
                break
            except (TypeError, ValueError):
                continue
    if k is None or k < 0:
        raise MetricError(
            "aggregate 行既无可用 metrics，也无 n_truth/K，无法登记失败分母（拒绝产出缩小分母的 summary）："
            "%s" % {key: src.get(key) for key in list(src)[:8]})
    status = str(src.get("status", src.get("status_in", "failed")))
    return _synthesise_empty(status, k), "synthesised"


def _error_stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "median": None, "p95": None}
    return {
        "n": int(arr.size),
        "rmse": float(math.sqrt(float(np.mean(arr ** 2)))),
        "mae": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _summarize(metrics_list) -> dict:
    """对一组 metrics dict 求 summary（全部计入分母，不删窗）。"""
    ospa2_vals, ospa_vals = [], []
    tp = fn = fp = n_truth_total = n_pred_total = 0
    n_ok = n_failed = n_na = 0
    n_success = 0
    e_xy, e_r, e_theta = [], [], []
    res_rows = []
    for m in metrics_list:
        if m is None:
            raise MetricError("内部错误：_summarize 收到 None（应先经 _row_to_metrics 补全）")
        if str(m.get("status_in", "ok")) == "not_applicable":
            n_na += 1
        if str(m.get("status_used", "ok")) == "ok":
            n_ok += 1
        else:
            n_failed += 1
        ospa2_vals.append(float(m["ospa2"]))
        ospa_vals.append(float(m["ospa"]))
        tp += int(m.get("tp", 0))
        fn += int(m.get("fn", 0))
        fp += int(m.get("fp", 0))
        n_truth_total += int(m.get("n_truth", 0))
        n_pred_total += int(m.get("n_predictions", 0))
        n_success += 1 if m.get("success") else 0
        for e in m.get("errors", []) or []:
            if e.get("e_xy") is not None:
                e_xy.append(float(e["e_xy"]))
            if e.get("e_r") is not None:
                e_r.append(float(e["e_r"]))
            if e.get("e_theta") is not None:
                e_theta.append(float(e["e_theta"]))
        if m.get("resolution") is not None:
            res_rows.append(m["resolution"])
    n_rows = len(metrics_list)
    resolution = None
    if res_rows:
        gates = [float(r["gate_m"]) for r in res_rows]
        resolution = {
            "n": len(res_rows),
            "success_rate": sum(1 for r in res_rows if r.get("success")) / len(res_rows),
            "gate_min_m": min(gates),
            "gate_max_m": max(gates),
        }
    return {
        "n_rows": int(n_rows),
        "n_ok": int(n_ok),
        "n_failed": int(n_failed),
        "n_not_applicable": int(n_na),
        "n_truth_total": int(n_truth_total),
        "n_predictions_total": int(n_pred_total),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "dr": (tp / n_truth_total) if n_truth_total else None,
        "miss_rate": (fn / n_truth_total) if n_truth_total else None,
        "fp_per_window": (fp / n_rows) if n_rows else None,
        "fp_per_estimate": (fp / n_rows) if n_rows else None,
        "false_discovery_fraction": (fp / n_pred_total) if n_pred_total else None,
        "rms_ospa_m": (math.sqrt(sum(ospa2_vals) / len(ospa2_vals)) if ospa2_vals else None),
        "mean_ospa_m": (sum(ospa_vals) / len(ospa_vals)) if ospa_vals else None,
        "gate_m": float(DEFAULT_GATE_M),
        "ospa_c_m": float(OSPA_C_M),
        "success_rate": (n_success / n_rows) if n_rows else None,
        "tp_errors": {
            "e_xy": _error_stats(e_xy),
            "e_r": _error_stats(e_r),
            "e_theta": _error_stats(e_theta),
        },
        "resolution": resolution,
    }


def aggregate(rows) -> dict:
    """把逐窗/逐 trial 的 metrics 聚合成一个 summary dict。

    * ``rows`` 中每项或为 ``evaluate_predictions`` 返回的 dict，或为带 ``metrics`` 字段的包装 dict。
    * **所有行都计入主 summary 分母**（含 empty / failed / not_applicable）；
      ``not_applicable`` 同时单列 ``n_not_applicable``，并另给可选条件子表
      ``conditional_excluding_not_applicable``（不改变主口径）。
    * 缺 ``metrics`` 的行按行的 ``n_truth`` / ``k`` 补全为 empty；**无真值数量则 raise**。
    * ``rms_ospa_m = sqrt(mean(ospa2))``；``mean_ospa_m`` 仅作辅助值。
    """
    metrics_list = []
    n_missing = 0
    for row in rows or []:
        m, source = _row_to_metrics(row)
        if source == "synthesised":
            n_missing += 1
        metrics_list.append(m)

    main = _summarize(metrics_list)
    main["schema_version"] = SCHEMA_VERSION
    main["n_missing_rows"] = int(n_missing)
    non_na = [m for m in metrics_list if str(m.get("status_in", "ok")) != "not_applicable"]
    main["conditional_excluding_not_applicable"] = _summarize(non_na) if len(non_na) != len(metrics_list) else None
    return main
