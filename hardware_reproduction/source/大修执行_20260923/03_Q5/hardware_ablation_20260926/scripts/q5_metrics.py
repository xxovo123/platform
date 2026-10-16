#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q5 指标层：OSPA（p=2, c=0.30 m）与描述性统计。

OSPA 口径**与 04_Q6/hardware/analysis_ospa_20260924/ospa_postprocess.py 一致**：

    OSPA_t^2 = [ min_{单射 pi} sum_i min(||x_i - y_pi(i)||, c)^2 + |m-n| c^2 ] / max(m, n)

* **先截断、再求最优单射**（截断平方代价矩阵 + scipy 的 linear_sum_assignment；
  矩形矩阵给出规模 min(m,n) 的最优单射）；不得先按未截断距离配对再截断。
* 两边都空 -> 0；一边空、另一边非空 -> c（平方意义下 c^2）。
* 每窗所有最终输出点都纳入（含 FN/FP），失败窗按空输出处理、**不删窗**。
* 主值 **RMS-OSPA = sqrt(mean_t OSPA_t^2)**（等权窗）；辅值 mean OSPA 与目标槽加权值。

不报 Pfa（本批无空场景录制）；DR/FN/FP 仍是正式报告口径，OSPA 为补充综合量。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

P_ORDER = 2
C_M = 0.30


def ospa_squared(truth_xy, pred_xy, c: float = C_M) -> float:
    X = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    Y = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    m, n = int(X.shape[0]), int(Y.shape[0])
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c ** 2)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("OSPA 输入含 NaN/Inf")
    d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    cost = np.minimum(d, float(c)) ** P_ORDER
    rows, cols = linear_sum_assignment(cost)
    matched = float(cost[rows, cols].sum())
    card = abs(m - n) * float(c) ** P_ORDER
    return (matched + card) / float(max(m, n))


def ospa(truth_xy, pred_xy, c: float = C_M) -> float:
    return math.sqrt(ospa_squared(truth_xy, pred_xy, c))


def extract_sets(rec: dict, where: str = "") -> tuple[list, list]:
    """由 WindowResult.to_dict() 严格重建 (truths, preds)；不一致直接报错，不静默。"""
    need = ("n_truth", "n_predictions", "tp", "fn", "fp",
            "truth_xy", "errors", "fn_items", "fp_items")
    missing = [k for k in need if k not in rec]
    if missing:
        raise ValueError("%s：缺少字段 %s" % (where, missing))
    truths = [[float(v) for v in p] for p in (rec["truth_xy"] or [])]
    n_truth, n_pred = int(rec["n_truth"]), int(rec["n_predictions"])
    errors = rec["errors"] or []
    fn_items = rec["fn_items"] or []
    fp_items = rec["fp_items"] or []
    if len(truths) != n_truth:
        raise ValueError("%s：truth_xy=%d != n_truth=%d" % (where, len(truths), n_truth))
    if len(errors) + len(fp_items) != n_pred:
        raise ValueError("%s：errors+fp_items=%d != n_predictions=%d"
                         % (where, len(errors) + len(fp_items), n_pred))
    if len(errors) + len(fn_items) != n_truth:
        raise ValueError("%s：errors+fn_items=%d != n_truth=%d"
                         % (where, len(errors) + len(fn_items), n_truth))
    if int(rec["tp"]) != len(errors) or int(rec["fn"]) != len(fn_items) \
            or int(rec["fp"]) != len(fp_items):
        raise ValueError("%s：tp/fn/fp=%s/%s/%s 与 items 长度 %d/%d/%d 不一致"
                         % (where, rec["tp"], rec["fn"], rec["fp"],
                            len(errors), len(fn_items), len(fp_items)))
    slots: dict[int, list] = {}
    for e in errors:
        idx = int(e["pred_index"])
        if idx in slots:
            raise ValueError("%s：pred_index=%d 重复" % (where, idx))
        slots[idx] = [float(v) for v in e["estimated_xy"]]
    for f in fp_items:
        idx = int(f["pred_index"])
        if idx in slots:
            raise ValueError("%s：pred_index=%d 重复（error 与 fp 冲突）" % (where, idx))
        slots[idx] = [float(v) for v in f["prediction_xy"]]
    if sorted(slots) != list(range(n_pred)):
        raise ValueError("%s：pred_index 集合 %s != 0..%d" % (where, sorted(slots), n_pred - 1))
    return truths, [slots[i] for i in range(n_pred)]


def window_ospa(rec: dict, where: str = "") -> float:
    truths, preds = extract_sets(rec, where)
    return ospa(truths, preds)


def desc_stats(values) -> dict:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "rmse": None, "median": None, "p95": None, "std": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
    }


#: 非 ok 状态的语义划分（**不得**把 not_applicable 与 failed 混为一谈）
STATUS_KINDS = ("ok", "failed", "invalid", "not_applicable")

#: 失败处理口径（**freeze 前写定，不按结果选择**）
FAILURE_POLICY = {
    "formal_run": ("任何 request 窗出现非 ok（failed / invalid / not_applicable）**立即终止正式运行**"
                   "并返回非零退出码：先修执行错误再重跑，不产出该配置的正式指标。"
                   "正常“无峰”仍为 status=ok（只是 n_predictions=0），不是失败。"),
    "not_applicable_distinct": ("not_applicable 表示方法按物理/结构原因主动声明该窗不可评估，"
                               "与技术失败（failed）**语义不同**；本批若出现，单列结构原因，"
                               "不得当作空输出，也不计入 DR/误差。"),
    "invalid_distinct": ("invalid 表示维度/秩等**结构条件**不满足（本批唯一硬条件 M>K）；"
                        "该类窗单列原因，不得伪造成空输出或 0 预测。"),
    "main_denominator": "主口径 DR/FP/误差只条件于 status=ok 的 metric-eligible 窗，并在报告中明写 n_eligible。",
    "conservative_all_windows": ("另给**显式标注失败惩罚**的保守全窗上界：非 ok 窗按 FN=N_true、"
                                "FP=0、空输出预测计入；与真实“无峰”不同，只作上界，不作主值。"),
}


def aggregate_config(records: list, *, config_id: str) -> dict:
    """由逐窗记录聚合一个配置的指标。

    主口径：只条件于 status=ok（metric-eligible），分子分母分别累加。
    另给保守全窗上界（非 ok 按 FN=N_true 计入，显式标注失败惩罚）。
    OSPA 主值同样只对 ok 窗聚合；非 ok 窗的 OSPA 单列在 conservative 块，避免与主口径混淆。
    """
    n_win = len(records)
    kinds = {k: 0 for k in STATUS_KINDS}
    for r in records:
        kinds[str(r["status"])] = kinds.get(str(r["status"]), 0) + 1
    non_ok = [r for r in records if r["status"] != "ok"]
    na = [r for r in records if r["status"] == "not_applicable"]
    inv = [r for r in records if r["status"] == "invalid"]
    bad = [r for r in records if r["status"] == "failed"]
    ok = [r for r in records if r["status"] == "ok"]
    n_ok = len(ok)
    tp = int(sum(int(r["tp"]) for r in ok))
    fn = int(sum(int(r["fn"]) for r in ok))
    fp = int(sum(int(r["fp"]) for r in ok))
    n_pred_total = int(sum(int(r["n_predictions"]) for r in ok))
    e_xy = [e["e_xy"] for r in ok for e in (r["errors"] or [])]
    e_r = [e["e_r"] for r in ok for e in (r["errors"] or [])]
    e_th = [e["e_theta"] for r in ok for e in (r["errors"] or [])]
    ospa_sq_ok = [float(r["ospa_sq"]) for r in ok]
    # 保守上界：非 ok 视作空输出预测（含 cardinality 惩罚）
    ospa_sq_all = ospa_sq_ok + [float(C_M) ** P_ORDER for _ in non_ok]
    n_truth_total = int(sum(int(r["n_truth"]) for r in records))
    n_truth_ok = int(sum(int(r["n_truth"]) for r in ok))
    return {
        "config_id": config_id,
        "n_windows_requested": n_win,
        "n_windows_ok": n_ok,
        "n_windows_non_ok": len(non_ok),
        "status_breakdown": kinds,
        "n_eligible": n_ok,
        "eligible_scope": "status=ok 窗（主口径；分母不缩水靠“非 ok 即终止正式运行”保证）",
        "n_recordings": len({r["recording_id"] for r in records}),
        "n_truth_total_requested": n_truth_total,
        "n_truth_total_eligible": n_truth_ok,
        "n_predictions_total": n_pred_total,
        "tp": tp, "fn": fn, "fp": fp,
        "dr": (tp / (tp + fn)) if (tp + fn) else None,
        "miss_rate": (fn / (tp + fn)) if (tp + fn) else None,
        "fp_per_estimate": (fp / n_ok) if n_ok else None,
        "n_windows_with_fp": int(sum(1 for r in ok if int(r["fp"]) > 0)),
        "p_window_with_fp": ((sum(1 for r in ok if int(r["fp"]) > 0) / n_ok) if n_ok else None),
        "e_xy": desc_stats(e_xy),
        "e_r": desc_stats(e_r),
        "e_theta_deg": desc_stats(e_th),
        "rms_ospa_m": (math.sqrt(float(np.mean(ospa_sq_ok))) if ospa_sq_ok else None),
        "mean_ospa_m": (float(np.mean([math.sqrt(v) for v in ospa_sq_ok])) if ospa_sq_ok else None),
        "median_ospa_m": (float(np.median([math.sqrt(v) for v in ospa_sq_ok])) if ospa_sq_ok else None),
        "p95_ospa_m": (float(np.percentile([math.sqrt(v) for v in ospa_sq_ok], 95)) if ospa_sq_ok else None),
        "conservative_all_windows": {
            "n_windows": n_win,
            "tp_all": tp,
            "fn_all": fn + (n_truth_total - n_truth_ok),
            "fp_all": fp,
            "n_truth_total": n_truth_total,
            "tp_over_n_truth_total": (tp / n_truth_total) if n_truth_total else None,
            "dr_failure_penalized": (tp / (tp + fn + (n_truth_total - n_truth_ok)))
            if (tp + fn + (n_truth_total - n_truth_ok)) else None,
            "rms_ospa_m_all_windows": (math.sqrt(float(np.mean(ospa_sq_all))) if ospa_sq_all else None),
            "note": ("非 ok 窗按 FN=N_true、FP=0、空输出预测计入；**显式失败惩罚**上界，"
                     "与真实无峰不同，仅作保守参考，不作主值。"),
        },
        "non_ok_windows": [{"window_id": r["window_id"], "status": r["status"],
                            "reason": r.get("reason")} for r in non_ok],
        "failed_windows": [r["window_id"] for r in bad],
        "invalid_windows": [{"window_id": r["window_id"], "reason": r.get("reason")}
                            for r in inv],
        "not_applicable_windows": [{"window_id": r["window_id"], "reason": r.get("reason")}
                                   for r in na],
        "failure_policy": FAILURE_POLICY,
    }


def aggregate_config_by_k(records: list) -> list:
    """按给定 K 分组（分子分母各自累加；**不作独立样本推断**）。"""
    out = []
    for k in sorted({int(r["k"]) for r in records}):
        sel = [r for r in records if int(r["k"]) == k]
        agg = aggregate_config(sel, config_id="k=%d" % k)
        agg["k"] = k
        out.append(agg)
    return out
