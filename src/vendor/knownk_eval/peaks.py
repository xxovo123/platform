# -*- coding: utf-8 -*-
"""统一取峰（独立实现，不 import 历史 evaluate.py / util.py）。

规则（与 protocols/knownk_protocol.json 的 peaks 段一一对应）：
  1) 3x3 非严格邻域局部极大（含中心点，边界按 edge 复制）；
  2) 等值平台按 8 连通域去重，代表点取域内最大值，并列取最小 (row, col)；
  3) 线性相对阈值 threshold_rel = 0.3（相对全谱最大有限值）；
  4) 按分值降序 + (row, col) 字典序的确定性贪心物理 NMS，门限 0.10 m（>= 保留）；
  5) 峰数上限 = 给定 K（cap=K；K <= 0 则为空）；不用真值；允许少于 K。

非有限值按 -inf 参与比较；全谱无有限值或最大有限值 <= 0 时本窗无候选峰（不是失败）。
"""

from __future__ import annotations

import math

import numpy as np

from .protocol import PeakRule


class PeakError(ValueError):
    """谱图 / 网格不合规（调用方须记录为显式失败，不得静默丢弃）。"""


def validate_grid(grid_x, grid_y) -> tuple[np.ndarray, np.ndarray]:
    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)
    for name, arr in (("grid_x", gx), ("grid_y", gy)):
        if arr.ndim != 1:
            raise PeakError("%s 必须是一维数组，收到 shape=%s" % (name, arr.shape))
        if arr.size < 1:
            raise PeakError("%s 为空" % name)
        if not np.all(np.isfinite(arr)):
            raise PeakError("%s 含 NaN/Inf" % name)
        if arr.size >= 2 and not np.all(np.diff(arr) > 0.0):
            raise PeakError("%s 必须严格递增（本 driver 不重排输入）" % name)
    return gx, gy


def validate_spectrum(image, grid_x, grid_y) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise PeakError("谱图必须是二维数组，收到 shape=%s" % (arr.shape,))
    expected = (int(np.asarray(grid_y).size), int(np.asarray(grid_x).size))
    if arr.shape != expected:
        raise PeakError("谱图 shape=%s 与 (len(grid_y), len(grid_x))=%s 不一致"
                        % (arr.shape, expected))
    return arr


def neighbourhood_max3x3(arr: np.ndarray) -> np.ndarray:
    """3x3 非严格邻域最大值，边界按 edge 复制。"""
    padded = np.pad(np.asarray(arr, dtype=float), 1, mode="edge")
    ny, nx = arr.shape
    out = np.full((ny, nx), -np.inf, dtype=float)
    for dr in range(3):
        for dc in range(3):
            np.maximum(out, padded[dr:dr + ny, dc:dc + nx], out=out)
    return out


def label_plateaus_8conn(mask: np.ndarray):
    """自实现 8 连通标号（行主序扫描 + 4 向/对角合并），返回 (labels, 坐标列表)。

    与 scipy.ndimage.label(structure=ones((3,3))) 的语义一致；标号顺序为行主序首遇顺序，
    但在取峰中候选随后按 (-value, row, col) 排序，故标号数值不影响结果。
    """
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.intp)
    groups: list[list[tuple[int, int]]] = []
    stack: list[tuple[int, int]] = []
    for row in range(ny):
        for col in range(nx):
            if not mask[row, col] or labels[row, col]:
                continue
            label = len(groups) + 1
            coords: list[tuple[int, int]] = []
            stack.append((row, col))
            labels[row, col] = label
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < ny and 0 <= cc < nx and mask[rr, cc] and not labels[rr, cc]:
                            labels[rr, cc] = label
                            stack.append((rr, cc))
            coords.sort()
            groups.append(coords)
    return labels, groups


def nms_peaks(image, grid_x, grid_y, rule: PeakRule, *, k=None):
    """返回 (peaks, status)。peaks 为 list[dict]，按 rank 升序。"""
    arr = validate_spectrum(image, grid_x, grid_y)
    gx, gy = validate_grid(grid_x, grid_y)
    arr_eff = np.where(np.isfinite(arr), arr, -np.inf)
    finite = np.isfinite(arr)
    status = {
        "spectrum_state": "unknown",
        "n_pixels": int(arr.size),
        "n_finite_pixels": int(finite.sum()),
        "n_nonfinite_pixels": int(arr.size - finite.sum()),
        "finite_max": None,
        "n_plateau_candidates": 0,
        "n_accepted": 0,
        "threshold_linear": None,
        "k_cap": None if k is None else int(k),
        "reasons": [],
    }
    if not finite.any():
        status["spectrum_state"] = "all_nonfinite"
        status["reasons"].append("谱图无任何有限像素（NaN/Inf），本窗无候选峰")
        return [], status
    finite_max = float(np.max(arr_eff))
    status["finite_max"] = finite_max if math.isfinite(finite_max) else None
    if not math.isfinite(finite_max) or finite_max <= 0.0:
        status["spectrum_state"] = "nonpositive_max"
        status["reasons"].append("全谱最大有限值 <= 0（全零谱或全负谱），本窗无候选峰")
        return [], status
    status["spectrum_state"] = "has_nonfinite" if status["n_nonfinite_pixels"] else "ok"

    threshold = float(rule.threshold_rel) * finite_max
    status["threshold_linear"] = threshold
    mask = (arr_eff >= neighbourhood_max3x3(arr_eff)) & (arr_eff >= threshold)
    if not mask.any():
        status["reasons"].append("无像元同时满足非严格局部极大与线性相对阈值 %.3g" % threshold)
        return [], status

    _labels, groups = label_plateaus_8conn(mask)
    candidates = []
    for coords in groups:
        best_value = -np.inf
        best_rc = None
        for r, c in coords:  # coords 已按 (row, col) 升序
            v = float(arr_eff[r, c])
            if v > best_value:
                best_value = v
                best_rc = (r, c)
        candidates.append((best_value, best_rc[0], best_rc[1]))
    status["n_plateau_candidates"] = len(candidates)
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    min_dist = float(rule.min_dist_m)
    accepted = []
    for value, row, col in candidates:
        x = float(gx[col])
        y = float(gy[row])
        if all(math.hypot(x - a["x"], y - a["y"]) >= min_dist for a in accepted):
            accepted.append({"x": x, "y": y, "value": value, "row": int(row), "col": int(col)})
    if k is not None:
        limit = int(k)
        if limit <= 0:
            status["reasons"].append("K <= 0：按协议返回空峰列表")
            accepted = []
        else:
            accepted = accepted[:limit]
    for rank, peak in enumerate(accepted):
        peak["rank"] = rank
    status["n_accepted"] = len(accepted)
    return accepted, status
