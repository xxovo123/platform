# -*- coding: utf-8 -*-
"""统一评估编排：谱图路径（统一取峰）与连续域预测路径（原生 (x, y)）。

两条入口共用同一匹配与指标口径（协议驱动）：
  * evaluate_spectrum(...)    : 谱方法路径 —— 先按统一取峰规则得 (x, y) 峰，再匹配。
  * evaluate_predictions(...) : 连续域方法路径 —— 直接吃方法原生 (x, y) 预测，
                                不强制先栅格化（路线 8.4：保留原生求解结构）。

失败处理：任何阶段失败都保留该窗记录并显式写出阶段与原因；失败窗仍计入分母
（tp = 0, fn = n_truth, fp = 0），状态标为 failed，聚合中单列 n_windows_failed。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import peaks as peaks_mod
from .matching import MatchingError, match_targets
from .metrics import DuplicateWindowError, aggregate, check_unique_windows, polar_of, wrap_deg
from .protocol import Protocol


class EvaluationError(ValueError):
    """输入不合规（K / 形状 / 坐标 / 谱图缺失）。调用方须记录为显式失败。"""


def _validate_k(k) -> int:
    if k is None:
        raise EvaluationError("缺少给定 K（known-K 评估必须逐窗给定 K）")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        if isinstance(k, float) and float(k).is_integer():
            k = int(k)
        else:
            raise EvaluationError("K 必须是整数，收到 %r" % (k,))
    k = int(k)
    if k < 0:
        raise EvaluationError("K 不能为负，收到 %d" % k)
    return k


def _validate_truth(truth_xy) -> list[tuple[float, float]]:
    if truth_xy is None:
        raise EvaluationError("缺少真值坐标列表")
    out = []
    for idx, item in enumerate(truth_xy):
        try:
            x, y = item
        except (TypeError, ValueError):
            raise EvaluationError("真值 #%d 不是 (x, y) 二元组：%r" % (idx, item)) from None
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise EvaluationError("真值 #%d 含 NaN/Inf：%r" % (idx, item))
        out.append((xf, yf))
    return out


def _validate_predictions(pred_xy) -> list[tuple[float, float]]:
    if pred_xy is None:
        raise EvaluationError("缺少预测坐标列表（native 路径必须显式给出，允许空列表）")
    out = []
    for idx, item in enumerate(pred_xy):
        try:
            x, y = item
        except (TypeError, ValueError):
            raise EvaluationError("预测 #%d 不是 (x, y) 二元组：%r" % (idx, item)) from None
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise EvaluationError("预测 #%d 含 NaN/Inf：%r" % (idx, item))
        out.append((xf, yf))
    return out


@dataclass
class WindowInput:
    window_id: str
    recording_id: str
    environment: str = ""
    config_id: str = ""
    n_truth: int = 0
    k: int | None = None
    truth_xy: list = field(default_factory=list)
    input_kind: str = "spectrum"          # spectrum | native_xy
    image: Any = None                     # spectrum 路径
    grid_x: Any = None
    grid_y: Any = None
    predictions_xy: Any = None            # native_xy 路径
    prediction_scores: Any = None
    k_cap_mode: str = "native"            # native | cap_by_score
    source: str = ""
    notes: list = field(default_factory=list)


@dataclass
class WindowResult:
    window_id: str
    recording_id: str
    environment: str
    config_id: str
    n_truth: int
    k: int | None
    input_kind: str
    status: str
    failures: list
    spectrum_state: dict | None
    prediction_source: str
    k_cap_applied: bool
    peaks: list
    n_predictions: int
    tp: int
    fn: int
    fp: int
    errors: list
    fn_items: list
    fp_items: list
    truth_xy: list
    wall_seconds: float
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "recording_id": self.recording_id,
            "environment": self.environment,
            "config_id": self.config_id,
            "n_truth": self.n_truth,
            "k": self.k,
            "input_kind": self.input_kind,
            "status": self.status,
            "failures": self.failures,
            "spectrum_state": self.spectrum_state,
            "prediction_source": self.prediction_source,
            "k_cap_applied": self.k_cap_applied,
            "peaks": self.peaks,
            "n_predictions": self.n_predictions,
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "errors": self.errors,
            "fn_items": self.fn_items,
            "fp_items": self.fp_items,
            "truth_xy": self.truth_xy,
            "wall_seconds": self.wall_seconds,
            "notes": self.notes,
        }


def _failed(win: WindowInput, k, truth, stage: str, error: str, *, truth_n: int,
            spectrum_state=None, prediction_source="", k_cap_applied=False,
            predictions=(), peaks=(), wall=0.0) -> WindowResult:
    return WindowResult(
        window_id=win.window_id,
        recording_id=win.recording_id,
        environment=win.environment,
        config_id=win.config_id,
        n_truth=truth_n,
        k=k,
        input_kind=win.input_kind,
        status="failed",
        failures=[{"stage": stage, "error": error}],
        spectrum_state=spectrum_state,
        prediction_source=prediction_source,
        k_cap_applied=k_cap_applied,
        peaks=list(peaks),
        n_predictions=len(predictions),
        tp=0,
        fn=int(truth_n),
        fp=0,
        errors=[],
        fn_items=[],
        fp_items=[],
        truth_xy=[list(p) for p in truth],
        wall_seconds=wall,
        notes=list(win.notes) + ["失败窗仍保留并计入分母（tp=0, fn=n_truth, fp=0）"],
    )


def _match_and_score(win: WindowInput, protocol: Protocol, k: int, truth, predictions,
                     *, peaks, spectrum_state, prediction_source, k_cap_applied,
                     wall: float) -> WindowResult:
    match = match_targets(truth, predictions, protocol.matching)
    errors = []
    for t_idx, p_idx, distance in match["pairs"]:
        tx, ty = truth[t_idx]
        ex, ey = predictions[p_idx]
        tr, tt = polar_of(tx, ty)
        er, et = polar_of(ex, ey)
        errors.append({
            "truth_index": int(t_idx),
            "pred_index": int(p_idx),
            "truth_xy": [float(tx), float(ty)],
            "estimated_xy": [float(ex), float(ey)],
            "truth_polar": [float(tr), float(tt)],
            "estimated_polar": [float(er), float(et)],
            "match_distance_m": float(distance),
            "e_r": float(abs(er - tr)),
            "e_theta": float(abs(wrap_deg(et - tt))),
            "e_xy": float(math.hypot(ex - tx, ey - ty)),
        })
    fn_items = [{"truth_index": int(t), "truth_xy": [float(truth[t][0]), float(truth[t][1])],
                 "reason": "no prediction within gate %.2f m" % match["gate_m"]}
                for t in match["fn"]]
    fp_items = [{"pred_index": int(p), "prediction_xy": [float(predictions[p][0]),
                                                         float(predictions[p][1])],
                 "reason": "prediction not within gate %.2f m of any unmatched truth"
                           % match["gate_m"]}
                for p in match["fp"]]
    return WindowResult(
        window_id=win.window_id,
        recording_id=win.recording_id,
        environment=win.environment,
        config_id=win.config_id,
        n_truth=win.n_truth,
        k=k,
        input_kind=win.input_kind,
        status="ok",
        failures=[],
        spectrum_state=spectrum_state,
        prediction_source=prediction_source,
        k_cap_applied=k_cap_applied,
        peaks=list(peaks),
        n_predictions=len(predictions),
        tp=int(match["tp"]),
        fn=len(fn_items),
        fp=len(fp_items),
        errors=errors,
        fn_items=fn_items,
        fp_items=fp_items,
        truth_xy=[list(p) for p in truth],
        wall_seconds=wall,
        notes=list(win.notes),
    )


def evaluate_window(win: WindowInput, protocol: Protocol) -> WindowResult:
    started = time.perf_counter()
    truth = []
    try:
        k = _validate_k(win.k)
    except EvaluationError as exc:
        try:
            truth = _validate_truth(win.truth_xy)
        except EvaluationError:
            truth = []
        return _failed(win, win.k, truth, "validate_k", str(exc), truth_n=win.n_truth,
                       wall=time.perf_counter() - started,
                       prediction_source=win.source)

    try:
        truth = _validate_truth(win.truth_xy)
    except EvaluationError as exc:
        return _failed(win, k, [], "validate_truth", str(exc), truth_n=win.n_truth,
                       wall=time.perf_counter() - started, prediction_source=win.source)
    if len(truth) != int(win.n_truth):
        return _failed(win, k, truth, "validate_truth",
                       "真值个数 %d 与 n_truth=%d 不一致" % (len(truth), int(win.n_truth)),
                       truth_n=win.n_truth, wall=time.perf_counter() - started,
                       prediction_source=win.source)

    # ---------------- 谱方法路径：统一取峰 ----------------
    if win.input_kind == "spectrum":
        if win.image is None or win.grid_x is None or win.grid_y is None:
            return _failed(win, k, truth, "spectrum_input",
                           "谱图/网格缺失（input_kind=spectrum 必须给出 image, grid_x, grid_y）",
                           truth_n=win.n_truth, wall=time.perf_counter() - started,
                           prediction_source="unified_nms_spectrum")
        try:
            peak_list, spectrum_state = peaks_mod.nms_peaks(
                win.image, win.grid_x, win.grid_y, protocol.peaks, k=k)
        except peaks_mod.PeakError as exc:
            return _failed(win, k, truth, "peak_rule", str(exc), truth_n=win.n_truth,
                           prediction_source="unified_nms_spectrum",
                           wall=time.perf_counter() - started)
        predictions = [(p["x"], p["y"]) for p in peak_list]
        try:
            return _match_and_score(win, protocol, k, truth, predictions, peaks=peak_list,
                                    spectrum_state=spectrum_state,
                                    prediction_source="unified_nms_spectrum",
                                    k_cap_applied=True,
                                    wall=time.perf_counter() - started)
        except MatchingError as exc:
            return _failed(win, k, truth, "matching", str(exc), truth_n=win.n_truth,
                           spectrum_state=spectrum_state,
                           prediction_source="unified_nms_spectrum",
                           peaks=peak_list, predictions=predictions,
                           wall=time.perf_counter() - started)

    # ---------------- 连续域方法路径：原生 (x, y) 预测 ----------------
    if win.input_kind != "native_xy":
        return _failed(win, k, truth, "input_kind",
                       "未知 input_kind=%r（只支持 spectrum / native_xy）" % (win.input_kind,),
                       truth_n=win.n_truth, wall=time.perf_counter() - started)
    try:
        predictions = _validate_predictions(win.predictions_xy)
    except EvaluationError as exc:
        return _failed(win, k, truth, "native_predictions", str(exc), truth_n=win.n_truth,
                       prediction_source=win.source, wall=time.perf_counter() - started)
    k_cap_applied = False
    if win.k_cap_mode == "cap_by_score":
        if len(predictions) > k:
            if win.prediction_scores is None:
                return _failed(win, k, truth, "native_predictions",
                               "cap_by_score 需要 prediction_scores（按方法自身分值排序）",
                               truth_n=win.n_truth, prediction_source=win.source,
                               wall=time.perf_counter() - started)
            scores = list(win.prediction_scores)
            if len(scores) != len(predictions):
                return _failed(win, k, truth, "native_predictions",
                               "prediction_scores 长度 %d 与预测数 %d 不一致"
                               % (len(scores), len(predictions)),
                               truth_n=win.n_truth, prediction_source=win.source,
                               wall=time.perf_counter() - started)
            order = sorted(range(len(predictions)), key=lambda i: (-float(scores[i]), i))
            keep = sorted(order[:k])
            predictions = [predictions[i] for i in keep]
            k_cap_applied = True
    elif len(predictions) > k:
        return _failed(win, k, truth, "native_predictions",
                       "原生预测数 %d 超过给定 K=%d，且 k_cap_mode=native："
                       "连续域方法须自行按原生结构给到 K 个（或以 cap_by_score 显式声明截断）"
                       % (len(predictions), k),
                       truth_n=win.n_truth, prediction_source=win.source,
                       wall=time.perf_counter() - started)
    try:
        return _match_and_score(win, protocol, k, truth, predictions, peaks=[],
                                spectrum_state=None, prediction_source=win.source or "native_xy",
                                k_cap_applied=k_cap_applied,
                                wall=time.perf_counter() - started)
    except MatchingError as exc:
        return _failed(win, k, truth, "matching", str(exc), truth_n=win.n_truth,
                       prediction_source=win.source or "native_xy", predictions=predictions,
                       wall=time.perf_counter() - started)


def evaluate_spectrum(image, grid_x, grid_y, k, truth_xy, protocol: Protocol, *,
                      window_id="window", recording_id="", environment="", config_id="",
                      n_truth=None, source="unified_nms_spectrum") -> WindowResult:
    """Python API：谱方法路径（统一取峰 + 匹配 + 指标）。"""
    win = WindowInput(
        window_id=window_id, recording_id=recording_id, environment=environment,
        config_id=config_id,
        n_truth=int(len(truth_xy)) if n_truth is None else int(n_truth),
        k=k, truth_xy=list(truth_xy), input_kind="spectrum",
        image=image, grid_x=grid_x, grid_y=grid_y, source=source)
    return evaluate_window(win, protocol)


def evaluate_predictions(truth_xy, predictions_xy, protocol: Protocol, *, k,
                         prediction_scores=None, k_cap_mode="native",
                         window_id="window", recording_id="", environment="", config_id="",
                         n_truth=None, source="native_xy") -> WindowResult:
    """Python API：连续域方法路径（原生 (x, y) 预测，不强制栅格化）。

    k_cap_mode:
      * "native"        方法自身原生输出必须 <= K（否则显式失败，不做隐式截断）；
      * "cap_by_score"  按方法自身分值降序确定性截断到 K，并在结果中记录 k_cap_applied。
    """
    win = WindowInput(
        window_id=window_id, recording_id=recording_id, environment=environment,
        config_id=config_id,
        n_truth=int(len(truth_xy)) if n_truth is None else int(n_truth),
        k=k, truth_xy=list(truth_xy), input_kind="native_xy",
        predictions_xy=list(predictions_xy) if predictions_xy is not None else None,
        prediction_scores=prediction_scores, k_cap_mode=k_cap_mode, source=source)
    return evaluate_window(win, protocol)


def evaluate_windows(windows, protocol: Protocol, *, progress=None):
    """批量评估；返回 (results, rows)。失败窗不丢弃；重复 window_id 直接报错。"""
    windows = list(windows)
    ids = [w.window_id for w in windows]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise DuplicateWindowError(
            "输入窗列表含重复 window_id（同一窗会被重复计数）：%s" % ", ".join(dup[:10]))
    results = []
    for idx, win in enumerate(windows):
        results.append(evaluate_window(win, protocol))
        if progress is not None and (idx + 1) % progress == 0:
            print("    [eval] %d/%d windows" % (idx + 1, len(windows)), flush=True)
    return results, aggregate(results)


def summarise(results, protocol: Protocol) -> dict:
    results = list(results)
    check_unique_windows(results)
    rows = aggregate(results, require_unique_windows=False)
    overall = next(r for r in rows if r["group_dim"] == "overall")
    failed = [r.window_id for r in results if r.status != "ok"]
    return {
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_revision": protocol.protocol_revision,
            "canonical_sha256": protocol.canonical_sha256_value,
            "file_sha256": protocol.file_sha256,
            "path": str(protocol.path),
            "peaks": {
                "threshold_rel": protocol.peaks.threshold_rel,
                "min_dist_m": protocol.peaks.min_dist_m,
                "cap": protocol.peaks.count_cap,
            },
            "matching": {
                "gate_m": protocol.matching.gate_m,
                "gate_tolerance_m": protocol.matching.gate_tolerance_m,
            },
        },
        "denominators": {
            "n_windows": overall["n_windows"],
            "n_recordings": overall["n_recordings"],
            "n_windows_failed": overall["n_windows_failed"],
            "n_truth_total": overall["n_truth_total"],
            "n_predictions_total": overall["n_predictions_total"],
            "unit": "每个窗为一次估计试验；分母为逐窗真值人数之和（分子分母分别累加）",
        },
        "overall": overall,
        "groups": {dim: [r for r in rows if r["group_dim"] == dim]
                   for dim in ("environment", "n_truth", "environment_n", "config")},
        "failed_windows": failed,
        "caveats": [
            "不报 Pfa（本批无空场景录制）；虚警只报有目标条件下的 per-estimate 值。",
            "不宣称计数性能：K 由 known-K 口径给定。",
            "描述性统计，不作独立样本 CI 或显著性推断。",
        ],
    }
