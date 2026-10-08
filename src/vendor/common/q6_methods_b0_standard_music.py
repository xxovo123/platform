#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B0（标准全二维 TOA-AOA MUSIC）在公共 case 上的适配器 —— 同模型控制组，不是性能基线。

用途（与主审口径一致）
----------------------
* B0 只作**同模型数值正确性/一致性控制组**：在本批硬件数据上，它与本方法在同模型下等价
  （B0 包已用独立实现 214/214 窗逐格核对，谱差 0.0；本适配器从 case['X'] 出发再算一遍）。
* 它同时充当**集成自检**：本方法的适配器用 case['snapshots'] 走硬件原函数，本适配器只吃
  case['X']（已 FF 且已校准）；两者谱若不一致，说明缓存的 X 与 snapshots 口径不一致。
* 谱的计算核心复用 B0 包的 sample_covariance / steering_matrix（列约定 R = X^T X^*/L，
  denom = a^H E_n E_n^H a），本文件只做装配，不另写第二套 MUSIC。

边界：B0 **不是** Deng/Kakouche/Wang 的复现，**不是**性能基线，不用于宣称优越性。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

WS = Path(__file__).resolve().parents[3]
B0_DIR = WS / "大修执行_20260923" / "04_Q6" / "B0_同模型校验"
P0B_DRIVER = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "driver"
PROTOCOL = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "protocol" / "knownk_protocol.json"
for _p in (str(B0_DIR), str(P0B_DRIVER)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from b0 import standard_music  # noqa: E402
from knownk_eval import load_protocol  # noqa: E402
from knownk_eval import peaks as knownk_peaks  # noqa: E402

_PROTOCOL = None


def _protocol():
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = load_protocol(str(PROTOCOL))
    return _PROTOCOL


def cfg_from_case(case) -> standard_music.MusicConfig:
    cfg = case["config"]
    return standard_music.MusicConfig(
        fft_length=int(cfg["fft_length"]),
        tap_interval_si=float(cfg["tap_interval_si"]),
        center_freq_si=float(cfg["center_freq_si"]),
        bandwidth_si=float(cfg["bandwidth_si"]),
        epsilon_r_background=float(cfg["epsilon_r_background"]),
        system_delay_si=float(cfg["system_delay_si"]),
        diag_load_factor=float(cfg["diag_load_factor"]),
        is_forward_backward=bool(cfg["is_forward_backward"]),
        fov_degrees=float(cfg["fov_degrees"]),
        cir_window=tuple(int(v) for v in cfg["cir_window"]),
        k=int(case["k"]),
        pairs=tuple(tuple(int(v) for v in p) for p in np.asarray(case["pairs"]).tolist()),
        deg_corr=(None if cfg["deg_corr"] is None else tuple(cfg["deg_corr"])),
        ant_calib_phase=(None if cfg["ant_calib_phase"] is None
                         else tuple(cfg["ant_calib_phase"])),
        freq_range_si=tuple(float(v) for v in cfg["freq_range_si"]),
    )


def spectrum_from_X(case):
    """从 case['X'] 直接装配标准全二维 MUSIC 伪谱（核心复用 B0 包实现）。"""
    cfg = cfg_from_case(case)
    X = np.asarray(case["X"], dtype=np.complex128)
    grid_x = np.asarray(case["grid_x"], dtype=float)
    grid_y = np.asarray(case["grid_y"], dtype=float)
    image = np.zeros((grid_y.size, grid_x.size), dtype=float)
    M = int(X.shape[1])
    k = int(case["k"])
    if k >= M:
        raise ValueError("K=%d 必须小于 M=%d" % (k, M))
    R, cov_info = standard_music.sample_covariance(X, cfg)
    if cov_info.get("zero_energy"):
        return image, "zero_energy", cov_info
    eigvals, eigvecs = np.linalg.eigh(R)
    En = eigvecs[:, : M - k]
    projector = En @ En.conj().T
    A, mask, _vx, _vy = standard_music.steering_matrix(
        cfg, case["tx_positions"], case["rx_positions"], grid_x, grid_y)
    denom = np.real(np.sum((np.conj(A) @ projector) * A, axis=1))
    denom = np.where(np.isnan(denom), np.inf, denom)
    denom = np.maximum(denom, 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    spectrum = np.maximum(spectrum, 0.0)
    image[mask] = spectrum
    return image, "ok", {"M": M, "L": int(X.shape[0]), "k": k,
                         "eigenvalues_first_last": [float(eigvals[0]), float(eigvals[-1])]}


def estimate(case: dict) -> dict:
    t0 = time.perf_counter()
    image, status, info = spectrum_from_X(case)
    t_spectrum = time.perf_counter() - t0
    if status != "ok":
        return {"predictions": [], "status": "failed",
                "diagnostics": {"reason": "标准 MUSIC 返回 %s（零能量/退化）" % status}}
    peaks, _peak_status = knownk_peaks.nms_peaks(
        np.asarray(image, dtype=float), np.asarray(case["grid_x"], dtype=float),
        np.asarray(case["grid_y"], dtype=float), _protocol().peaks, k=int(case["k"]))
    return {
        "predictions": [{"x": float(p["x"]), "y": float(p["y"]),
                         "score": float(p["value"])} for p in peaks],
        "status": "ok",
        "spectrum": np.asarray(image, dtype=float),
        "diagnostics": {
            "eval_path": "spectrum",
            "role": "同模型控制组（不是性能基线）",
            "grid_kind": "cartesian_2d_full_grid_with_distance_ring_mask",
            "n_candidates_evaluated": int(np.count_nonzero(np.asarray(image) != 0.0)),
            "dictionary_size": None,
            "dictionary_size_note": "非字典型方法：全网格扫描",
            "n_peaks_from_own_picker": len(peaks),
            "fb_used": bool(case["config"]["is_forward_backward"]),
            "n_channels_used": int(np.asarray(case["pairs"]).shape[0]),
            "M_features": int(case["n_features"]),
            "L_snapshots": int(case["n_snapshots"]),
            "timing_scope": "covariance+FB+diag_load+EVD+search+own_peak_picking",
            "t_spectrum_s": t_spectrum,
            "convention": "column convention R = X^T X^*/L, denom = a^H EnEnH a（B0 包实现）",
        },
    }


def make_estimate(case: dict) -> dict:
    return estimate(case)
