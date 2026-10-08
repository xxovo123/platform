# -*- coding: utf-8 -*-
"""B0 比对工具：逐格谱差、归一化谱相关、峰集一致性、掩膜外零检查、协方差共轭关系。

设计纪律（独立验收意见，2026-09-23）：
  * 不做“过滤掉非有限像元后再算统计”的隐式清洗：非有限像元数量单独计数，
    并把 finite_ok 作为**判定条件之一**交给调用方；非有限即判不通过。
  * 同时给出原始最大绝对差、相对参考最大值的差、以及**逐格自归一化最大绝对差**
    max|a/max(a) - b/max(b)|（最直观的可比量）。
  * 峰比较除像素 (row, col) 外，同时比较坐标 (x, y) 与分值。
  * 掩膜外必须为 0（伪谱只在掩膜内定义），单独检查并进入判定。
"""

from __future__ import annotations

import math

import numpy as np


def _finite_max(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0
    return float(np.max(np.where(finite, a, -np.inf)))


def normalise_image(arr: np.ndarray) -> np.ndarray:
    """按自身全谱最大有限值线性归一化（最大值 <= 0 时返回全零）。"""
    a = np.asarray(arr, dtype=float)
    m = _finite_max(a)
    if m <= 0.0:
        return np.zeros_like(a)
    return np.where(np.isfinite(a), a, 0.0) / m


def outside_mask_max_abs(image, mask) -> float:
    """掩膜外像元的最大绝对值（合同口径要求为 0）。"""
    a = np.asarray(image, dtype=float)
    sel = np.asarray(mask, dtype=bool)
    if a.shape != sel.shape:
        raise ValueError("谱与掩膜形状不一致：%s vs %s" % (a.shape, sel.shape))
    outside = a[~sel]
    if outside.size == 0:
        return 0.0
    return float(np.max(np.abs(outside)))


def spectrum_comparison(a, b, mask=None) -> dict:
    """逐格比较两个伪谱（b 为参考）；非有限像元单独计数并给出 finite_ok 判定。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("谱形状不一致：%s vs %s" % (a.shape, b.shape))
    sel = np.ones(a.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if sel.shape != a.shape:
        raise ValueError("掩膜形状与谱不一致：%s vs %s" % (sel.shape, a.shape))
    av_all = a[sel]
    bv_all = b[sel]
    fa = np.isfinite(av_all)
    fb = np.isfinite(bv_all)
    both = fa & fb
    out = {
        "n_pixels_total": int(a.size),
        "n_pixels_in_scope": int(av_all.size),
        "n_finite_a": int(fa.sum()),
        "n_nonfinite_a": int((~fa).sum()),
        "n_finite_b": int(fb.sum()),
        "n_nonfinite_b": int((~fb).sum()),
        "n_pixels_compared": int(both.sum()),
        "finite_ok": bool(fa.all() and fb.all()),
        "reference_max": None,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "max_abs_diff_normalised": None,
        "rms_abs_diff": None,
        "cosine_similarity_normalised": None,
        "pearson_normalised": None,
    }
    if not both.any():
        return out
    av = av_all[both]
    bv = bv_all[both]
    bmax = float(np.max(np.abs(bv)))
    amax = float(np.max(np.abs(av)))
    out["reference_max"] = bmax
    diff = np.abs(av - bv)
    out["max_abs_diff"] = float(np.max(diff))
    out["max_rel_diff"] = (float(np.max(diff)) / bmax) if bmax > 0 else None
    out["rms_abs_diff"] = float(math.sqrt(float(np.mean(diff ** 2))))
    na = (av / amax) if amax > 0 else np.zeros_like(av)
    nb = (bv / bmax) if bmax > 0 else np.zeros_like(bv)
    out["max_abs_diff_normalised"] = float(np.max(np.abs(na - nb)))
    if np.any(nb) and np.linalg.norm(na) > 0:
        denom = float(np.linalg.norm(na) * np.linalg.norm(nb))
        out["cosine_similarity_normalised"] = float(np.dot(na, nb) / denom) if denom > 0 else None
    if na.size > 1 and np.std(na) > 0 and np.std(nb) > 0:
        out["pearson_normalised"] = float(np.corrcoef(na, nb)[0, 1])
    return out


def peak_set_comparison(peaks_a, peaks_b, *, tol: float = 1e-9) -> dict:
    """比较两组峰（按 rank 顺序）：像素 (row, col)、坐标 (x, y) 与分值都必须在容差内。"""
    a = list(peaks_a)
    b = list(peaks_b)
    mismatches = []
    if len(a) != len(b):
        mismatches.append({"issue": "峰数不一致", "a": len(a), "b": len(b)})
    max_xy_delta = 0.0
    max_value_delta = 0.0
    for idx, (pa, pb) in enumerate(zip(a, b)):
        if (int(pa["row"]) != int(pb["row"])) or (int(pa["col"]) != int(pb["col"])):
            mismatches.append({"rank": idx, "issue": "像素不一致",
                               "a": [int(pa["row"]), int(pa["col"])],
                               "b": [int(pb["row"]), int(pb["col"])]})
            continue
        dx = abs(float(pa["x"]) - float(pb["x"]))
        dy = abs(float(pa["y"]) - float(pb["y"]))
        dv = abs(float(pa["value"]) - float(pb["value"]))
        max_xy_delta = max(max_xy_delta, dx, dy)
        max_value_delta = max(max_value_delta, dv)
        if dx > tol or dy > tol:
            mismatches.append({"rank": idx, "issue": "坐标超容差",
                               "a": [float(pa["x"]), float(pa["y"])],
                               "b": [float(pb["x"]), float(pb["y"])]})
        if dv > tol * max(1.0, abs(float(pb["value"]))):
            mismatches.append({"rank": idx, "issue": "分值超容差",
                               "a": float(pa["value"]), "b": float(pb["value"])})
    return {
        "n_a": len(a),
        "n_b": len(b),
        "identical": not mismatches,
        "mismatches": mismatches,
        "tol_rel": tol,
        "max_xy_abs_delta": max_xy_delta,
        "max_value_abs_delta": max_value_delta,
        "peaks_a": [[p["row"], p["col"], p["x"], p["y"], p["value"]] for p in a],
        "peaks_b": [[p["row"], p["col"], p["x"], p["y"], p["value"]] for p in b],
    }


def conjugation_check(R_col, R_hw, tol_rel: float = 1e-12) -> dict:
    """核对 R_col == conj(R_hw)（逐元素）与 Hermitian 性。"""
    R_col = np.asarray(R_col, dtype=np.complex128)
    R_hw = np.asarray(R_hw, dtype=np.complex128)
    if R_col.shape != R_hw.shape:
        raise ValueError("协方差形状不一致：%s vs %s" % (R_col.shape, R_hw.shape))
    delta = np.abs(R_col - np.conj(R_hw))
    scale = max(float(np.max(np.abs(R_hw))), 1e-300)
    herm = float(np.max(np.abs(R_col - R_col.conj().T)))
    return {
        "max_abs_conj_delta": float(np.max(delta)),
        "max_rel_conj_delta": float(np.max(delta) / scale),
        "max_abs_nonhermitian": herm,
        "hermitian_rel": herm / scale,
        "tol_rel": tol_rel,
        "pass": float(np.max(delta) / scale) <= tol_rel,
    }


def music_form_equivalence_check(A, projector_col, projector_hw, tol_rel: float = 1e-12) -> dict:
    """核对列约定 a^H P_col a 与行约定 a^T P_hw a^* 给出同一实数（要求 P_hw = conj(P_col)）。"""
    A = np.asarray(A, dtype=np.complex128)
    Pc = np.asarray(projector_col, dtype=np.complex128)
    Ph = np.asarray(projector_hw, dtype=np.complex128)
    col = np.real(np.sum((np.conj(A) @ Pc) * A, axis=1))
    row = np.sum((A @ Ph) * np.conj(A), axis=1)
    imag = float(np.max(np.abs(np.imag(row)))) if row.size else 0.0
    row = np.real(row)
    scale = max(float(np.max(np.abs(col))), 1e-300)
    delta = float(np.max(np.abs(col - row))) if col.size else 0.0
    return {
        "max_abs_form_delta": delta,
        "max_rel_form_delta": delta / scale,
        "max_abs_imag_row_form": imag,
        "tol_rel": tol_rel,
        "pass": (delta / scale) <= tol_rel,
    }

