#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q6 硬件侧公共合同：case 结构、方法估计接口、共享后处理与输出校验。

设计目标（对应主审裁定）
------------------------
1. **case 绝不含真值**：方法只能看到数据与配置；真值另存，仅在 evaluation 阶段使用。
2. **不做"逐点同网格"的虚假声明**：case 里给的是**候选域与分辨率预算**（Cartesian 网格、
   距离环、FoV、1-D 角度/时延预算），方法按各自原生结构在自己的网格上搜索，并在
   diagnostics 里**如实申报** grid_kind / 候选点数 / 字典大小；报告只按申报值比较预算与代价。
3. **共享后处理只作用于方法原生候选**：native/连续域方法返回自身候选 (x, y, score)。
   共享 NMS（0.10 m、分值降序、确定性、cap=K）只在候选列表上做去重与截断，
   **没有**谱路径的 threshold_rel 语义，也不得被描述成"全谱相对阈值等同"。
4. **适配一次冻结**：任何网格/后处理/参数选择在跑正式结果前写定并记哈希，不按真值结果调整。

估计接口（方法代理必须实现）::

    def estimate(case: dict) -> dict:
        return {
            "predictions": [{"x": float, "y": float, "score": float}, ...],
            "status": "ok" | "failed" | "not_applicable",
            "diagnostics": {...},
        }

可选键：
  * "spectrum": 2-D float 数组（完整谱）——若有，则走 knownk_eval.evaluate_spectrum 的统一取峰；
    此时必须同时给 diagnostics["spectrum_grid"] = {"grid_x":..., "grid_y":...}（与 case 网格一致）。
  * status="not_applicable" 时必须给 diagnostics["reason"]：说明**物理/结构原因**
    （例如两 Rx 孔径不足以支撑该方法的根结构），不得写"结果不好"。
  * status="ok" 时必须给 diagnostics["grid_kind"] 与 diagnostics["n_candidates_evaluated"]；
    字典型方法另给 diagnostics["dictionary_size"]（无字典则 None，并给 dictionary_size_note）。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

CASE_SCHEMA_VERSION = 1
ESTIMATE_STATUSES = ("ok", "failed", "not_applicable")

#: case 必需键（方法可见输入；**不含真值**）
REQUIRED_CASE_KEYS = (
    "case_schema_version",
    "id", "recording_id", "environment", "config_id", "t_end_s",
    "X", "frequencies_hz", "pairs", "tx_positions", "rx_positions",
    "k", "grid_x", "grid_y", "config", "search", "model",
    "feature_order", "n_snapshots", "n_features", "n_freq_bins", "n_channels",
    "snapshots_available",
)
#: 任何键名命中这些子串即视为真值泄漏（含大小写变体）
TRUTH_KEY_MARKERS = ("truth", "gt_", "ground_truth", "label")


class CaseError(ValueError):
    """case 结构不合规（**显式失败**，不得静默降级）。"""


class OutputError(ValueError):
    """方法估计输出不合规（调用方须记为 failed 并保留在分母）。"""


# --------------------------------------------------------------------------------------
# case 校验
# --------------------------------------------------------------------------------------
def find_truth_leaks(case: dict) -> list[str]:
    """返回 case 中疑似真值泄漏的键路径（递归）。"""
    leaks: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_path = "%s.%s" % (path, key) if path else str(key)
                low = str(key).lower()
                if any(marker in low for marker in TRUTH_KEY_MARKERS):
                    leaks.append(key_path)
                walk(value, key_path)
        elif isinstance(node, (list, tuple)) and len(node) <= 64:
            for idx, value in enumerate(node):
                walk(value, "%s[%d]" % (path, idx))
        elif isinstance(node, np.ndarray):
            if node.dtype.kind in "OUS":
                for idx, value in enumerate(node.ravel()[:64]):
                    walk(value, "%s[%d]" % (path, idx))

    walk(case, "")
    return sorted(set(leaks))


def validate_case(case: dict, *, allow_leaks: bool = False) -> None:
    """结构 + 真值隔离 + 基本形状校验；不合规直接抛 CaseError。"""
    if not isinstance(case, dict):
        raise CaseError("case 必须是 dict，收到 %r" % type(case).__name__)
    missing = [key for key in REQUIRED_CASE_KEYS if key not in case]
    if missing:
        raise CaseError("case 缺少必需键：%s" % ", ".join(missing))
    if int(case["case_schema_version"]) != CASE_SCHEMA_VERSION:
        raise CaseError("case_schema_version=%r，本版合同要求 %d"
                        % (case["case_schema_version"], CASE_SCHEMA_VERSION))
    leaks = find_truth_leaks(case)
    if leaks and not allow_leaks:
        raise CaseError("case 含疑似真值字段（禁止传入估计器）：%s" % ", ".join(leaks[:8]))

    X = np.asarray(case["X"])
    if X.ndim != 2:
        raise CaseError("X 必须是二维 (L, F*C)，收到 shape=%s" % (X.shape,))
    freqs = np.asarray(case["frequencies_hz"], dtype=float)
    pairs = np.asarray(case["pairs"])
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise CaseError("pairs 必须是 (C, 2)，收到 shape=%s" % (pairs.shape,))
    n_freq = int(freqs.size)
    n_ch = int(pairs.shape[0])
    if int(X.shape[1]) != n_freq * n_ch:
        raise CaseError("X 列数 %d 与 F*C=%d*%d 不一致"
                        % (X.shape[1], n_freq, n_ch))
    for key in ("grid_x", "grid_y"):
        grid = np.asarray(case[key], dtype=float)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise CaseError("%s 必须是一维严格递增网格，收到 shape=%s" % (key, grid.shape))
    k = case["k"]
    if not isinstance(k, (int, np.integer)) or int(k) < 0:
        raise CaseError("k 必须是非负整数，收到 %r" % (k,))
    if int(k) >= int(X.shape[1]):
        raise CaseError("k=%d 必须小于特征维数 M=%d" % (int(k), int(X.shape[1])))
    if not isinstance(case["config"], dict) or not isinstance(case["search"], dict):
        raise CaseError("config / search 必须是标准 dict")
    if not np.all(np.isfinite(X)):
        raise CaseError("X 含 NaN/Inf（本批真实数据不应出现；请显式记录为 failed 而非静默处理）")
    if "snapshots" in case and case["snapshots"] is not None:
        if not isinstance(case["snapshots"], list):
            raise CaseError("snapshots 必须是 list（与硬件函数 list 口径一致），"
                            "收到 %r" % type(case["snapshots"]).__name__)


def case_fingerprint(case: dict) -> str:
    """case 数据指纹：X 字节 + 关键参数（用于结果可追溯；不含真值）。"""
    X = np.ascontiguousarray(np.asarray(case["X"], dtype=np.complex128))
    parts = [
        "schema=%d" % int(case["case_schema_version"]),
        "id=%s" % case["id"],
        "X_shape=%s" % (X.shape,),
        "X_sha256=%s" % hashlib.sha256(X.tobytes()).hexdigest(),
        "freq=%s" % ",".join("%.6f" % v for v in np.asarray(case["frequencies_hz"], float)),
        "pairs=%s" % np.asarray(case["pairs"]).tolist(),
        "k=%d" % int(case["k"]),
        "grid=%d,%d" % (np.asarray(case["grid_x"]).size, np.asarray(case["grid_y"]).size),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# 共享后处理（native 候选专用）
# --------------------------------------------------------------------------------------
SHARED_NMS_MIN_DIST_M = 0.10

POSTPROCESS_SEMANTICS = (
    "共享后处理只作用于方法**原生候选列表**：按方法自身分值降序 + 输入序做确定性贪心物理"
    "最小间距抑制（>= %.2f m 保留），再 cap 到给定 K。它**没有**谱路径 threshold_rel 的"
    "全谱相对阈值语义（候选集不是全谱），也**不使用真值**；对所有连续域方法用同一规则。"
    % SHARED_NMS_MIN_DIST_M
)


def postprocess_native_candidates(candidates, k, *, min_dist_m: float = SHARED_NMS_MIN_DIST_M):
    """原生候选 → (保留列表, 统计)。确定性：分值降序、同分按输入序。"""
    items = []
    for idx, item in enumerate(candidates or []):
        x = float(item["x"])
        y = float(item["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise OutputError("候选 #%d 含 NaN/Inf：x=%r y=%r" % (idx, item["x"], item["y"]))
        score = float(item.get("score", 0.0))
        if not math.isfinite(score):
            raise OutputError("候选 #%d 的 score 非有限：%r" % (idx, item.get("score")))
        items.append({"x": x, "y": y, "score": score, "input_index": int(idx)})
    order = sorted(range(len(items)), key=lambda i: (-items[i]["score"], items[i]["input_index"]))
    accepted = []
    for i in order:
        cand = items[i]
        if all(math.hypot(cand["x"] - a["x"], cand["y"] - a["y"]) >= float(min_dist_m)
               for a in accepted):
            accepted.append(cand)
    limit = int(k)
    capped = [] if limit <= 0 else accepted[:limit]
    stats = {
        "min_dist_m": float(min_dist_m),
        "cap": limit,
        "n_input": len(items),
        "n_after_nms": len(accepted),
        "n_after_cap": len(capped),
        "semantics": POSTPROCESS_SEMANTICS,
        "uses_truth": False,
        "threshold_rel_equivalent_to_spectrum_path": False,
    }
    return capped, stats


# --------------------------------------------------------------------------------------
# 估计输出校验
# --------------------------------------------------------------------------------------
def validate_estimate_output(result, case: dict):
    """返回 (normalised_result, diagnostics)；不合规抛 OutputError。"""
    if not isinstance(result, dict):
        raise OutputError("estimate 必须返回 dict，收到 %r" % type(result).__name__)
    if "status" not in result:
        raise OutputError("estimate 输出缺少 status（ok/failed/not_applicable）")
    status = str(result["status"])
    if status not in ESTIMATE_STATUSES:
        raise OutputError("status=%r 不在 %s" % (status, list(ESTIMATE_STATUSES)))
    diag = result.get("diagnostics")
    if diag is None:
        diag = {}
    if not isinstance(diag, dict):
        raise OutputError("diagnostics 必须是 dict，收到 %r" % type(diag).__name__)

    predictions = result.get("predictions")
    if predictions is None:
        predictions = []
    if not isinstance(predictions, list):
        raise OutputError("predictions 必须是 list")
    norm_preds = []
    for idx, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise OutputError("predictions[%d] 必须是 dict，收到 %r" % (idx, type(item).__name__))
        for key in ("x", "y"):
            if key not in item:
                raise OutputError("predictions[%d] 缺少 %r" % (idx, key))
        x = float(item["x"])
        y = float(item["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise OutputError("predictions[%d] 坐标非有限：x=%r y=%r" % (idx, item["x"], item["y"]))
        score = item.get("score", 0.0)
        score = 0.0 if score is None else float(score)
        if not math.isfinite(score):
            raise OutputError("predictions[%d].score 非有限：%r" % (idx, item.get("score")))
        norm_preds.append({"x": x, "y": y, "score": score})

    spectrum = result.get("spectrum")
    if spectrum is not None:
        arr = np.asarray(spectrum, dtype=float)
        expected = (np.asarray(case["grid_y"]).size, np.asarray(case["grid_x"]).size)
        if arr.shape != expected:
            raise OutputError("spectrum shape=%s 与 (len(grid_y), len(grid_x))=%s 不一致"
                              % (arr.shape, expected))
        grid = diag.get("spectrum_grid") or {}
        gx = grid.get("grid_x", case["grid_x"])
        gy = grid.get("grid_y", case["grid_y"])
        if not (np.allclose(np.asarray(gx, dtype=float), np.asarray(case["grid_x"], float))
                and np.allclose(np.asarray(gy, dtype=float), np.asarray(case["grid_y"], float))):
            raise OutputError("spectrum_grid 与 case 网格不一致（谱法路径要求同一候选网格）")

    if status == "ok" and spectrum is None:
        if int(case["k"]) > 0 and len(norm_preds) > int(case["k"]):
            raise OutputError("原生候选数 %d 超过给定 K=%d：连续域方法须按原生结构给出 <=K 个"
                              "（或自行调用共享后处理并如实说明）" % (len(norm_preds), int(case["k"])))
        for key in ("grid_kind", "n_candidates_evaluated"):
            if key not in diag:
                raise OutputError("status=ok 必须在 diagnostics 申报 %r"
                                  "（公平适配要求如实记录搜索预算）" % key)
        if "dictionary_size" not in diag:
            raise OutputError("status=ok 必须在 diagnostics 给出 dictionary_size"
                              "（无字典写 None 并加 dictionary_size_note）")
    if status == "not_applicable" and not str(diag.get("reason", "")).strip():
        raise OutputError("status=not_applicable 必须给 diagnostics['reason']（物理/结构原因）")
    if status == "failed" and not str(diag.get("reason", result.get("error", ""))).strip():
        raise OutputError("status=failed 必须给 diagnostics['reason'] 或 error")

    normalised = {
        "predictions": norm_preds,
        "status": status,
        "spectrum": None if spectrum is None else np.asarray(spectrum, dtype=float),
        "diagnostics": diag,
    }
    return normalised, diag
