#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标：公开 `evaluate_row` + **池化** `aggregate`（规范 §9）。

口径（严格按规范，不沿用旧实现的可疑处）：
  * RMSE 配对：**无门限**最大基数、最小平方位置代价的一对一指派（位置/距离/方位共用配对）。
  * RMSE 池化：跨 trial 汇 SSE 与配对数后开方（`sqrt(Σsse/Σpairs)`），**不平均各次 RMSE**。
  * 未配对目标不计入 RMSE，始终并列 pairs / missing。
  * DR 单独用 0.30 m 一对一门限（先最大 TP，再最小总距离）；FN=真值数-TP，FP=预测数-TP。
  * 全部失败时 RMSE=None（不是 0）。`pairs/n_truth` 只能称配对覆盖率，不叫 DR。
  * E2 分辨率成功判据：输出恰好 2 点，存在一对一匹配使两点误差均 <= min(0.10, d/3)。
  * 不确定度：成功率 Wilson 95%；RMSE 用固定 seed 的 **trial 整组** bootstrap 2000 次。

本模块不 import 冻结的 `knownk_eval`，是仿真侧独立实现；DR 匹配结果会与冻结规则做一致性核对
（tests 中逐例比对）。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np

SCHEMA_VERSION = 1
GATE_M = 0.30
GATE_TOLERANCE_M = 1e-12
OSPA_C_M = 0.30
OSPA_P = 2
RESOLUTION_ABS_GATE_M = 0.10
RESOLUTION_REL_DIVISOR = 3.0
MAX_N_TRUTH = 8
MAX_N_PRED = 16
BOOTSTRAP_SEED = 20260929  # 固定 seed，整组重采样
BOOTSTRAP_REPS = 2000


class MetricError(ValueError):
    """真值/预测不合规或枚举规模超界（显式失败）。"""


# --------------------------------------------------------------------------- #
# 坐标
# --------------------------------------------------------------------------- #
def wrap_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if value <= -180.0 + 1e-12:
        value = 180.0
    return float(value)


def polar_of(x: float, y: float):
    return (math.hypot(float(x), float(y)), math.degrees(math.atan2(float(x), -float(y))))


def _points(points, name: str):
    if points is None:
        return []
    out = []
    for idx, item in enumerate(points):
        try:
            x, y = item
        except (TypeError, ValueError):
            raise MetricError("%s[%d] 必须是 (x,y)，收到 %r" % (name, idx, item)) from None
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise MetricError("%s[%d] 含 NaN/Inf" % (name, idx))
        out.append((xf, yf))
    return out


def _dist_matrix(truth, preds):
    return [[math.hypot(truth[i][0] - preds[j][0], truth[i][1] - preds[j][1])
             for j in range(len(preds))] for i in range(len(truth))]


def _enum_assignments(n_t: int, n_p: int):
    """所有注入（含真值不匹配）- 预测指派，规模受 MAX_N_* 约束。"""
    for size in range(min(n_t, n_p), 0, -1):
        for tsub in itertools.combinations(range(n_t), size):
            for pperm in itertools.permutations(range(n_p), size):
                yield list(zip(tsub, pperm))


# --------------------------------------------------------------------------- #
# 两种配对
# --------------------------------------------------------------------------- #
def match_rmse(truth, preds) -> dict:
    """无门限最大基数 + 最小平方位置代价（等代价时按 (t,p) 序列字典序最小）。"""
    return _match(truth, preds, cost=lambda d: d * d, gate=None)


def match_dr(truth, preds, gate_m: float = GATE_M) -> dict:
    """0.30 m 门限：先最大 TP，再最小总距离（非平方）。"""
    return _match(truth, preds, cost=lambda d: d, gate=float(gate_m) + GATE_TOLERANCE_M)


def _match(truth, preds, *, cost, gate) -> dict:
    n_t, n_p = len(truth), len(preds)
    if n_t > MAX_N_TRUTH or n_p > MAX_N_PRED:
        raise MetricError("配对枚举超界：n_truth=%d(max %d) n_pred=%d(max %d)"
                          % (n_t, MAX_N_TRUTH, n_p, MAX_N_PRED))
    dist = _dist_matrix(truth, preds)
    allowed = [[j for j in range(n_p) if (gate is None or dist[i][j] <= gate)] for i in range(n_t)]
    best = {"key": None, "pairs": []}

    def rec(i, used, pairs, total):
        if i == n_t:
            key = (-len(pairs), total, tuple(pairs))
            if best["key"] is None or key < best["key"]:
                best["key"] = key
                best["pairs"] = list(pairs)
            return
        rec(i + 1, used, pairs, total)
        for j in allowed[i]:
            if j in used:
                continue
            pairs.append((i, j))
            rec(i + 1, used | {j}, pairs, total + cost(dist[i][j]))
            pairs.pop()

    if n_t and n_p:
        rec(0, set(), [], 0.0)
    pairs = best["pairs"]
    matched_t = {t for t, _ in pairs}
    matched_p = {p for _, p in pairs}
    return {
        "pairs": [[int(t), int(p), float(dist[t][p])] for t, p in pairs],
        "tp": len(pairs),
        "fn_index": [int(t) for t in range(n_t) if t not in matched_t],
        "fp_index": [int(p) for p in range(n_p) if p not in matched_p],
        "total_distance_m": float(sum(dist[t][p] for t, p in pairs)),
        "sse_m2": float(sum(dist[t][p] ** 2 for t, p in pairs)),
        "gate_m": None if gate is None else float(gate - GATE_TOLERANCE_M),
    }


def ospa(truth, preds, c: float = OSPA_C_M, p: int = OSPA_P) -> float:
    """标准 OSPA^p（仅作辅助值；主口径是池化 RMSE）。先截断再求最优单射。"""
    X, Y = _points(truth, "truth"), _points(preds, "pred")
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)
    d = np.array(_dist_matrix(X, Y), dtype=float)
    cost = np.minimum(d, float(c)) ** int(p)
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        rows, cols = lsa(cost)
        matched = float(cost[rows, cols].sum())
    except Exception:  # pragma: no cover
        size = min(m, n)
        matched = float(min(sum(cost[i, perm[i]] for i in range(size))
                            for perm in itertools.permutations(range(n), size)))
    card = abs(m - n) * float(c) ** int(p)
    return math.sqrt((matched + card) / float(max(m, n)))


# --------------------------------------------------------------------------- #
# 逐行评估
# --------------------------------------------------------------------------- #
def evaluate_row(truth_xy, pred_xy, *, resolution_distance_m=None, status="ok") -> dict:
    """单次（trial）指标。返回 JSON 可序列化 dict。

    * status != 'ok' -> 预测集按空处理，仍计入分母（失败不漏行）。
    * resolution_distance_m=None -> 通用 0.30 m 整场景成功口径。
    * resolution_distance_m=d -> E2 分辨率口径，gate = min(0.10, d/3)。
    """
    truth = _points(truth_xy, "truth_xy")
    status_in = str(status)
    ok = status_in == "ok"
    preds = _points(pred_xy, "pred_xy") if ok else []

    m_rmse = match_rmse(truth, preds)
    m_dr = match_dr(truth, preds, GATE_M)
    n_t, n_p = len(truth), len(preds)

    errors = []
    for ti, pi, d in m_rmse["pairs"]:
        r_t, th_t = polar_of(*truth[ti])
        r_p, th_p = polar_of(*preds[pi])
        errors.append({"truth_index": int(ti), "pred_index": int(pi),
                       "e_xy": float(d), "e_r": float(abs(r_p - r_t)),
                       "e_theta": float(abs(wrap_deg(th_p - th_t)))})

    resolution = None
    if resolution_distance_m is not None:
        d_res = float(resolution_distance_m)
        if not math.isfinite(d_res) or d_res <= 0.0:
            raise MetricError("resolution_distance_m 必须有限且 >0，收到 %r" % (resolution_distance_m,))
        gate = min(RESOLUTION_ABS_GATE_M, d_res / RESOLUTION_REL_DIVISOR)
        exactly_two = (n_t == 2 and n_p == 2)
        success = False
        max_err = None
        if exactly_two:
            dm = _dist_matrix(truth, preds)
            success = any(all(dm[i][perm[i]] <= gate + GATE_TOLERANCE_M for i in range(2))
                          for perm in ((0, 1), (1, 0)))
            max_err = float(min(max(dm[i][perm[i]] for i in range(2))
                                for perm in ((0, 1), (1, 0))))
        resolution = {"distance_m": d_res, "gate_m": float(gate),
                      "exactly_two": bool(exactly_two), "success": bool(success),
                      "max_matched_error_m": max_err}
        success_rule = "resolution_exactly2_and_all_le_min(0.10,d/3)"
    else:
        success = bool(n_t > 0 and n_p == n_t and int(m_dr["tp"]) == n_t)
        success_rule = "n_pred_eq_K_gt0_and_all_within_0.30m"

    return {
        "schema_version": SCHEMA_VERSION,
        "status_in": status_in,
        "status_used": "ok" if ok else "empty",
        "n_truth": int(n_t),
        "n_predictions": int(n_p),
        # RMSE 侧（无门限配对）
        "pairs": int(m_rmse["tp"]),
        "missing": int(len(m_rmse["fn_index"])),
        "spurious": int(len(m_rmse["fp_index"])),
        "sse_m2": float(m_rmse["sse_m2"]),
        "pair_coverage": (m_rmse["tp"] / n_t) if n_t else None,
        # DR 侧（0.30 m 门限，独立匹配）
        "gate_m": float(GATE_M),
        "tp": int(m_dr["tp"]),
        "fn": int(len(m_dr["fn_index"])),
        "fp": int(len(m_dr["fp_index"])),
        "dr": (int(m_dr["tp"]) / n_t) if n_t else None,
        "miss_rate": (len(m_dr["fn_index"]) / n_t) if n_t else None,
        # 兼容别名：明确不代表检出率
        "pairs_over_n_truth_NOT_DR": (m_rmse["tp"] / n_t) if n_t else None,
        "ospa_m": float(ospa(truth, preds)),
        "success": bool(success),
        "success_rule": success_rule,
        "resolution": resolution,
        "errors": errors,
        "fn_items": [{"truth_index": int(t), "truth_xy": [float(truth[t][0]), float(truth[t][1])]}
                     for t in m_dr["fn_index"]],
        "fp_items": [{"pred_index": int(p), "prediction_xy": [float(preds[p][0]), float(preds[p][1])]}
                     for p in m_dr["fp_index"]],
    }


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def _stat(values):
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "sd": (float(arr.std(ddof=1)) if arr.size > 1 else None),
            "min": float(arr.min()), "max": float(arr.max())}


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict:
    """Wilson 95% 区间。"""
    if n <= 0:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {"n": int(n), "successes": int(successes), "rate": float(p),
            "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "method": "wilson_95"}


def _rng_for(condition: dict):

    key = json.dumps(condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(("boot|%d|%s" % (BOOTSTRAP_SEED, key)).encode("utf-8")).digest()[:8],
                          "big") % (2 ** 63 - 1)
    return np.random.default_rng(seed)


def pooled_rmse(rows, *, bootstrap: bool = True) -> dict:
    """池化 RMSE 与不确定度。rows = 逐 trial 记录，需含 sse_m2 / pairs / resolution / truth/pred 数。

    trial 整组 bootstrap：重采样 **整次 trial**（连 sse 与 pairs 一起），不把目标当独立样本。
    """
    sse = float(sum(float(r.get("sse_m2") or 0.0) for r in rows))
    pairs = int(sum(int(r.get("pairs") or 0) for r in rows))
    rmse = math.sqrt(sse / pairs) if pairs else None
    trial_rmse = []
    for r in rows:
        p = int(r.get("pairs") or 0)
        if p:
            trial_rmse.append(math.sqrt(float(r.get("sse_m2") or 0.0) / p))
    out = {
        "n_trials": int(len(rows)),
        "n_trials_with_pairs": int(len(trial_rmse)),
        "pairs_total": int(pairs),
        "missing_total": int(sum(int(r.get("missing") or 0) for r in rows)),
        "spurious_total": int(sum(int(r.get("spurious") or 0) for r in rows)),
        "rmse_pooled_m": (float(rmse) if rmse is not None else None),
        "trial_rmse_sd_m": (float(np.std(trial_rmse, ddof=1)) if len(trial_rmse) > 1 else None),
        "trial_rmse_stats": _stat(trial_rmse),
        "pair_coverage": (pairs / float(sum(int(r.get("n_truth") or 0) for r in rows))
                          if sum(int(r.get("n_truth") or 0) for r in rows) else None),
        "note": ("RMSE=sqrt(Σsse/Σpairs)，跨 trial 汇 SSE 与配对数；未配对目标不计入 RMSE，"
                 "单列 missing；trial_rmse_sd 只解释已输出配对的误差波动。"),
    }
    if bootstrap and len(rows) > 1:
        cond = {k: rows[0].get(k) for k in
                ("task", "family", "layer", "scene", "snr_db", "L", "layout", "direction",
                 "d_nominal_m") if k in rows[0]}
        rng = _rng_for(cond)
        sse_arr = np.asarray([float(r.get("sse_m2") or 0.0) for r in rows], dtype=float)
        pair_arr = np.asarray([int(r.get("pairs") or 0) for r in rows], dtype=float)
        idx = np.arange(len(rows))
        vals = np.empty(BOOTSTRAP_REPS, dtype=float)
        for b in range(BOOTSTRAP_REPS):
            pick = rng.choice(idx, size=idx.size, replace=True)
            s = sse_arr[pick].sum()
            p = pair_arr[pick].sum()
            vals[b] = math.sqrt(s / p) if p > 0 else np.nan
        vals = vals[np.isfinite(vals)]
        out["rmse_bootstrap"] = {
            "reps": int(BOOTSTRAP_REPS), "seed": int(BOOTSTRAP_SEED),
            "n_valid": int(vals.size),
            "ci95_lo": (float(np.percentile(vals, 2.5)) if vals.size else None),
            "ci95_hi": (float(np.percentile(vals, 97.5)) if vals.size else None),
            "unit": "trial_整组重采样（不把目标当独立样本）",
        }
    return out


def dr_summary(rows) -> dict:
    tp = int(sum(int(r.get("tp") or 0) for r in rows))
    fn = int(sum(int(r.get("fn") or 0) for r in rows))
    fp = int(sum(int(r.get("fp") or 0) for r in rows))
    n_truth = int(sum(int(r.get("n_truth") or 0) for r in rows))
    n_pred = int(sum(int(r.get("n_predictions") or 0) for r in rows))
    return {"gate_m": float(GATE_M), "tp": tp, "fn": fn, "fp": fp,
            "n_truth_total": n_truth, "n_predictions_total": n_pred,
            "dr": (tp / n_truth) if n_truth else None,
            "miss_rate": (fn / n_truth) if n_truth else None,
            "fp_per_window": (fp / len(rows)) if rows else None,
            "note": "DR 单独用 0.30 m 一对一门限；FN=真值数-TP，FP=预测数-TP。"}


def aggregate(rows, *, bootstrap: bool = True) -> dict:
    """池化聚合：RMSE 与 DR 分开；失败行保留在分母。"""
    if not rows:
        return {"n_rows": 0, "rmse_pooled_m": None, "dr": None}
    ok_rows = [r for r in rows if str(r.get("status_used", "ok")) == "ok"]
    res_rows = [r for r in rows if r.get("resolution") is not None]
    n_success = sum(1 for r in rows if r.get("success"))
    out = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(len(rows)),
        "n_ok": int(len(ok_rows)),
        "n_empty_or_failed": int(len(rows) - len(ok_rows)),
        "rmse": pooled_rmse(rows, bootstrap=bootstrap),
        "dr_summary": dr_summary(rows),
        "success": wilson_interval(n_success, len(rows)),
        "conditional_ok_only": (pooled_rmse(ok_rows, bootstrap=bootstrap)
                               if len(ok_rows) != len(rows) else None),
        "errors_within_gate_only": _gate_conditional(rows),
    }
    if res_rows:
        out["resolution"] = {
            "n": len(res_rows),
            "success": wilson_interval(sum(1 for r in res_rows if r["resolution"].get("success")),
                                       len(res_rows)),
            "gate_min_m": min(float(r["resolution"]["gate_m"]) for r in res_rows),
            "gate_max_m": max(float(r["resolution"]["gate_m"]) for r in res_rows),
            "rule": "输出恰好 2 点，存在一对一匹配使两点误差均 <= min(0.10, d/3)",
        }
    out["ospa"] = {"rms_ospa_m": float(math.sqrt(np.mean([float(r.get("ospa_m") or 0.0) ** 2
                                                          for r in rows]))),
                   "mean_ospa_m": float(np.mean([float(r.get("ospa_m") or 0.0) for r in rows])),
                   "role": "辅助值，主口径是池化 RMSE"}
    return out


def _gate_conditional(rows) -> dict:
    """门限内条件误差（仅内部对照；不替代主 RMSE）。"""
    e_xy, e_r, e_th = [], [], []
    for r in rows:
        for e in r.get("errors") or []:
            d = float(e["e_xy"])
            if d <= GATE_M + GATE_TOLERANCE_M:
                e_xy.append(d)
                e_r.append(float(e["e_r"]))
                e_th.append(float(e["e_theta"]))
    return {"note": "只作内部对照：0.30 m 内的配对误差；不是主 RMSE（主 RMSE 不设门限）",
            "e_xy": _stat(e_xy), "e_r": _stat(e_r), "e_theta": _stat(e_th),
            "n_gate_pairs": len(e_xy)}


CONDITION_KEYS = ("task", "family", "layer", "scene", "snr_db", "L", "layout", "direction",
                  "d_nominal_m", "method")


def group_by_condition(rows, keys=CONDITION_KEYS) -> dict:
    """按条件列分组池化。真实条件列与分母都保留。"""
    groups = {}
    for r in rows:
        key = tuple(r.get(k) for k in keys if k in r)
        groups.setdefault(key, []).append(r)
    out = {}
    for key, sub in groups.items():
        label = dict(zip([k for k in keys if k in sub[0]], key))
        out[json.dumps(label, ensure_ascii=False, sort_keys=True)] = {
            "condition": label, "metrics": aggregate(sub)}
    return out


__all__ = ["MetricError", "evaluate_row", "aggregate", "pooled_rmse", "dr_summary",
           "wilson_interval", "group_by_condition", "match_rmse", "match_dr", "ospa",
           "wrap_deg", "polar_of", "GATE_M", "SCHEMA_VERSION", "BOOTSTRAP_REPS",
           "BOOTSTRAP_SEED", "CONDITION_KEYS"]
