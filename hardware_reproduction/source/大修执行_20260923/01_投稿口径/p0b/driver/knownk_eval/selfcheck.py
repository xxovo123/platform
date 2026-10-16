# -*- coding: utf-8 -*-
"""协议语义自检：把协议里的“描述字段”与实际实现行为绑在一起。

背景（独立验收发现）：协议 JSON 里的 accept_rule / ordering / finite_handling /
allow_fewer_than_K / matching rule+tie_break / 指标分母等字段如果只是**文字描述**，
改掉文字再重签 SHA256 仍能通过之前的装载校验，而实现语义没变——这就是“描述与实现
漂移”。本模块用最小确定性算例把描述钉到行为上：任何一处描述或参数被改成与实现不符
（哪怕重签 hash），装载即失败。
"""

from __future__ import annotations

import numpy as np

from .matching import match_targets
from .metrics import group_stats, polar_of, wrap_deg
from .peaks import nms_peaks
from .protocol import Protocol, ProtocolError

# 描述字段必须逐字等于实现语义（改描述 = 改协议，必须同步改代码并重签）
SEMANTIC_STRINGS = (
    (("peaks", "nms", "accept_rule"), "dist >= min_dist_m（含等于）"),
    (("peaks", "nms", "ordering"), "value_desc_then_row_then_col"),
    (("peaks", "nms", "metric"), "euclidean_xy_m"),
    (("peaks", "threshold", "mode"), "linear_relative_to_global_finite_max"),
    (("peaks", "threshold", "finite_handling"),
     "非有限值按 -inf 参与比较；若全谱无有限值或有限最大值 <= 0，则本窗无候选峰"),
    (("peaks", "plateau", "representative"), "plateau_max"),
    (("peaks", "plateau", "tie_break"), "min_row_then_col"),
    (("peaks", "count", "cap"), "K"),
    (("matching", "rule"),
     "先最大化匹配数，再在等匹配数下最小化总距离（精确枚举全部单射部分匹配）"),
    (("matching", "tie_break"), "配对序列 (truth_index, pred_index) 字典序最小"),
    (("matching", "gate_rule"),
     "dist <= gate_m + gate_tolerance_m（1e-12 仅为浮点边界数值容差，不是门限调参）"),
    (("matching", "not"), "强度贪心匹配"),
    (("metrics", "aggregation", "denominators"),
     "分子分母分别累加；分组 DR = sum_i TP_i / sum_i (TP_i + FN_i)"),
    (("metrics", "aggregation", "error_population"),
     "e_r/e_theta/e_xy 只条件于匹配 TP；FN 与 FP 单列，不得剔除后宣称整体精度"),
    (("metrics", "count_metrics", "dr"), "sum(TP) / sum(TP + FN)（known-K 定位命中率）"),
    (("metrics", "count_metrics", "miss_rate"), "sum(FN) / sum(TP + FN)"),
    (("metrics", "count_metrics", "fp_per_estimate"),
     "sum(FP) / 有效估计窗数（仅“有目标”条件，须显式标注）"),
    (("metrics", "count_metrics", "p_window_with_fp"),
     "#窗(FP >= 1) / 有效估计窗数（仅“有目标”条件）"),
    (("metrics", "conventions", "polar"), "r = hypot(x, y); theta_deg = degrees(atan2(x, -y))"),
    (("metrics", "conventions", "angle_wrap"), "角度差 wrap 到 (-180, 180]"),
    (("failure_handling", "policy"),
     "任何窗口在任何阶段失败都必须保留该窗记录并显式写出失败阶段与原因，禁止静默丢弃；"
     "聚合中单列失败窗数"),
    (("failure_handling", "retained_on_failure"),
     "失败窗仍计入分母（tp=0, fn=n_truth, fp=0），与历史保留口径一致，但状态标为 failed"),
)
SEMANTIC_BOOLS = (
    (("peaks", "neighbourhood", "include_center"), True),
    (("peaks", "neighbourhood", "strict"), False),
    (("peaks", "count", "allow_fewer_than_K"), True),
    (("peaks", "count", "truth_used_for_peak_selection"), False),
    (("matching", "self_match_forbidden"), True),
    (("determinism", "rng_used"), False),
)
SEMANTIC_INTS = (
    (("peaks", "plateau", "connectivity"), 8),
)
SEMANTIC_LISTS = (
    (("peaks", "neighbourhood", "window"), [3, 3]),
    (("matching", "enumerate_limits", "max_n_truth"), "present"),
    (("matching", "enumerate_limits", "max_n_pred"), "present"),
)

# 冻结值：protocol_revision = 1 的数值口径必须逐项等于实现冻结值。
# 若将来要改口径，必须同时升 protocol_revision 并在这里登记新的一套值——
# 这样“改参数再重签 hash”不会静默生效。
FROZEN_REVISION = 1
FROZEN_VALUES = {
    "peaks.threshold.threshold_rel": 0.3,
    "peaks.threshold.mode": "linear_relative_to_global_finite_max",
    "peaks.nms.min_dist_m": 0.1,
    "peaks.nms.metric": "euclidean_xy_m",
    "peaks.plateau.connectivity": 8,
    "peaks.count.cap": "K",
    "matching.gate_m": 0.30,
    "matching.gate_tolerance_m": 1e-12,
}


def _dig(payload: dict, keys):
    node = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ProtocolError("协议缺少必需字段 %s" % ".".join(keys))
        node = node[key]
    return node


def validate_semantic_fields(payload: dict):
    """逐字核对描述字段（返回检查条数）。"""
    checked = 0
    revision = payload.get("protocol_revision")
    if int(revision) == FROZEN_REVISION:
        for dotted, expected in FROZEN_VALUES.items():
            actual = _dig(payload, dotted.split("."))
            same = (abs(float(actual) - float(expected)) < 1e-15
                    if isinstance(expected, float) else actual == expected)
            if not same:
                raise ProtocolError(
                    "协议冻结值被改动：%s = %r（revision %d 的实现冻结值为 %r）；"
                    "改口径必须升 protocol_revision 并在 selfcheck 中登记"
                    % (dotted, actual, FROZEN_REVISION, expected))
            checked += 1
    for keys, expected in SEMANTIC_STRINGS:
        actual = _dig(payload, keys)
        if actual != expected:
            raise ProtocolError(
                "协议描述字段与实现语义不一致：%s\n  协议: %r\n  实现: %r\n"
                "（改描述或改参数都必须同步改实现并重签 hash，否则拒绝装载）"
                % (".".join(keys), actual, expected))
        checked += 1
    for keys, expected in SEMANTIC_BOOLS:
        actual = _dig(payload, keys)
        if bool(actual) is not expected:
            raise ProtocolError("协议布尔字段与实现语义不一致：%s = %r（实现要求 %r）"
                                % (".".join(keys), actual, expected))
        checked += 1
    for keys, expected in SEMANTIC_INTS:
        actual = _dig(payload, keys)
        if int(actual) != int(expected):
            raise ProtocolError("协议整数字段与实现语义不一致：%s = %r（实现要求 %r）"
                                % (".".join(keys), actual, expected))
        checked += 1
    for keys, expected in SEMANTIC_LISTS:
        actual = _dig(payload, keys)
        if expected != "present":
            if list(actual) != list(expected):
                raise ProtocolError("协议列表字段与实现语义不一致：%s = %r（实现要求 %r）"
                                    % (".".join(keys), actual, expected))
        elif actual is None or int(actual) < 1:
            raise ProtocolError("协议缺少必需字段 %s" % ".".join(keys))
        checked += 1
    return checked


def behavioral_self_check(protocol: Protocol):
    """用最小确定性算例核对实现行为；任何一项不符即抛 ProtocolError。"""
    rule = protocol.peaks
    failures = []

    def note(name, ok, detail=""):
        if not ok:
            failures.append("%s %s" % (name, detail))

    grid = np.arange(5, dtype=float)
    one_row = np.array([-1.0])

    # 1) 平台去重：2x2 等值全为最大值 -> 单个平台、代表点 (0,0)
    g2 = np.array([0.0, 1.0])
    peaks, status = nms_peaks(np.full((2, 2), 0.5), g2, g2, rule, k=4)
    note("plateau_dedup", status["n_plateau_candidates"] == 1 and len(peaks) == 1
         and (peaks[0]["row"], peaks[0]["col"]) == (0, 0),
         "candidates=%d peaks=%d" % (status["n_plateau_candidates"], len(peaks)))

    # 2) 线性相对阈值：0.25 的局部极大相对最大值 1.0 是否被剔除，取决于 threshold_rel
    img = np.array([[0.0, 0.25, 0.0, 1.0, 0.0]])
    peaks, status = nms_peaks(img, grid, one_row, rule, k=5)
    expect_candidates = 2 if rule.threshold_rel <= 0.25 else 1
    note("threshold_rel", status["n_plateau_candidates"] == expect_candidates,
         "threshold_rel=%r candidates=%d expect=%d"
         % (rule.threshold_rel, status["n_plateau_candidates"], expect_candidates))

    # 3) NMS 门限语义：恰好 min_dist_m 保留、半个 min_dist_m 抑制
    d = rule.min_dist_m
    img2 = np.array([[1.0, 0.0, 0.9]])
    peaks, _ = nms_peaks(img2, np.array([0.0, d / 2.0, d]), one_row, rule, k=3)
    note("nms_accept_at_threshold", len(peaks) == 2, "peaks=%d" % len(peaks))
    peaks, _ = nms_peaks(img2, np.array([0.0, d / 4.0, d / 2.0]), one_row, rule, k=3)
    note("nms_reject_below_threshold", len(peaks) == 1, "peaks=%d" % len(peaks))

    # 4) 峰数上限 = K（含 K=0）与“允许少于 K”
    img3 = np.array([[1.0, 0.0, 0.9, 0.0, 0.8]])
    peaks, _ = nms_peaks(img3, np.arange(5, dtype=float) * max(rule.min_dist_m, 1.0),
                         one_row, rule, k=2)
    note("cap_K", len(peaks) == 2, "peaks=%d" % len(peaks))
    peaks, _ = nms_peaks(img3, np.arange(5, dtype=float) * max(rule.min_dist_m, 1.0),
                         one_row, rule, k=10)
    note("allow_fewer_than_K", len(peaks) == 3, "peaks=%d" % len(peaks))
    peaks, _ = nms_peaks(img3, np.arange(5, dtype=float) * max(rule.min_dist_m, 1.0),
                         one_row, rule, k=0)
    note("cap_K_zero", peaks == [], "peaks=%d" % len(peaks))

    # 5) 非有限 / 全零策略
    peaks, status = nms_peaks(np.zeros((2, 3)), np.arange(3, dtype=float),
                              np.array([-1.0, 0.0]), rule, k=2)
    note("zero_spectrum_empty", peaks == [] and status["spectrum_state"] == "nonpositive_max",
         "state=%s" % status["spectrum_state"])
    peaks, status = nms_peaks(np.full((2, 3), np.nan), np.arange(3, dtype=float),
                              np.array([-1.0, 0.0]), rule, k=2)
    note("all_nonfinite_empty", peaks == [] and status["spectrum_state"] == "all_nonfinite",
         "state=%s" % status["spectrum_state"])
    peaks, status = nms_peaks(np.array([[0.0, 1.0, np.nan, 0.0, 0.8]]), grid, one_row, rule, k=5)
    note("partial_nonfinite_ok", status["n_nonfinite_pixels"] == 1 and len(peaks) == 2,
         "nonfinite=%d peaks=%d" % (status["n_nonfinite_pixels"], len(peaks)))

    # 6) 匹配：基数优先于总距离；tie-break 字典序；门限边界含容差
    mrule = protocol.matching
    out = match_targets([(0.0, 0.0), (0.0, 0.29)], [(0.0, 0.001), (0.0, 0.291)], mrule)
    note("matching_cardinality_first", out["tp"] == 2, "tp=%d" % out["tp"])
    out = match_targets([(0.0, 0.0)], [(0.1, 0.0), (-0.1, 0.0)], mrule)
    note("matching_tie_break", out["pairs"][0][:2] == (0, 0), "pairs=%s" % (out["pairs"],))
    out = match_targets([(0.0, 0.0)], [(mrule.gate_m, 0.0)], mrule)
    note("matching_gate_boundary_accept", out["tp"] == 1, "tp=%d" % out["tp"])
    out = match_targets([(0.0, 0.0)], [(mrule.gate_m + 10.0 * mrule.gate_tolerance_m, 0.0)], mrule)
    note("matching_gate_boundary_reject", out["tp"] == 0, "tp=%d" % out["tp"])

    # 7) 指标口径：角度 wrap 与 polar 约定
    note("wrap_convention", abs(wrap_deg(190.0) - (-170.0)) < 1e-9, "wrap=%r" % wrap_deg(190.0))
    r, theta = polar_of(0.0, -1.0)
    note("polar_convention", abs(r - 1.0) < 1e-12 and abs(theta) < 1e-12, "r=%r th=%r" % (r, theta))

    # 8) 分母分别累加（两个不同窗：1 真值 1 命中、3 真值 1 命中 -> DR = 2/4）
    records = [
        {"window_id": "a", "recording_id": "r", "environment": "e", "config_id": "c",
         "n_truth": 1, "tp": 1, "fn": 0, "fp": 0, "n_predictions": 1, "status": "ok",
         "errors": [{"e_r": 0.1, "e_theta": 1.0, "e_xy": 0.1}]},
        {"window_id": "b", "recording_id": "r", "environment": "e", "config_id": "c",
         "n_truth": 3, "tp": 1, "fn": 2, "fp": 1, "n_predictions": 2, "status": "ok",
         "errors": [{"e_r": 0.2, "e_theta": 2.0, "e_xy": 0.2}]},
    ]
    stats = group_stats(records)
    note("denominator_accumulation",
         stats["tp"] == 2 and stats["fn"] == 2 and abs(stats["dr"] - 0.5) < 1e-12
         and abs(stats["fp_per_estimate"] - 0.5) < 1e-12,
         "tp=%d fn=%d dr=%r fp/est=%r"
         % (stats["tp"], stats["fn"], stats["dr"], stats["fp_per_estimate"]))

    if failures:
        raise ProtocolError("协议行为自检未通过（描述/参数与实现不一致）：\n  - "
                            + "\n  - ".join(failures))
    return True
