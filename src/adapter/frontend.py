#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共同前端：raw32 -> crop_start（v2=0，全 32 tap）-> fft(n=32) -> 带内 17 频点 -> X。

实现复用 sim_common（协议唯一事实来源），并提供与 vendor b0 包 build_observation_matrix 的
逐格核对（verify_vs_b0），证明本前端与冻结后端逐字等价（含特征序 = 频率优先、通道其次）。
"""

from __future__ import annotations

import sys

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np


def cropped_from_raw(raw_cir):
    return sc.crop_raw(raw_cir)


def x_from_cropped(cropped, *, ant_calib_phase=None):
    return sc.x_from_cropped(cropped, ant_calib_phase=ant_calib_phase)


def x_from_raw(raw_cir, *, ant_calib_phase=None):
    return sc.build_x_from_raw(raw_cir, ant_calib_phase=ant_calib_phase)


def verify_vs_b0(cropped, *, ant_calib_phase=None) -> dict:
    """逐格核对：本前端 vs vendor b0 包 build_observation_matrix（需 vendor 存在）。"""
    if str(sc.VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(sc.VENDOR_DIR))
    try:
        from b0 import standard_music as sm
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": "vendor b0 不可导入：%s" % exc}
    cfg = sm.MusicConfig(
        fft_length=int(sc.FFT_LENGTH), tap_interval_si=float(sc.SAMPLE_INTERVAL_S),
        center_freq_si=float(sc.FC_HZ), bandwidth_si=float(sc.BANDWIDTH_HZ),
        epsilon_r_background=float(sc.EPSILON_R_BACKGROUND), system_delay_si=float(sc.SYSTEM_DELAY_SI),
        diag_load_factor=float(sc.DIAG_LOAD_FACTOR), is_forward_backward=bool(sc.IS_FORWARD_BACKWARD),
        fov_degrees=float(sc.FOV_DEGREES), cir_window=(int(sc.CROP_START), int(sc.FFT_LENGTH)),
        k=1, pairs=tuple(tuple(int(v) for v in p) for p in sc.PAIRS.tolist()),
        deg_corr=None, ant_calib_phase=ant_calib_phase,
        freq_range_si=tuple(float(v) for v in sc.freq_range_si()))
    # b0 快照形式 (n_rx, n_tx, n_taps)：按 pairs 把 4 通道放回 rx×tx 位置
    cropped = np.asarray(cropped)
    pairs = np.asarray(sc.PAIRS, dtype=int)
    n_rx = int(pairs[:, 1].max()) + 1
    n_tx = int(pairs[:, 0].max()) + 1
    snaps = []
    for i in range(cropped.shape[0]):
        arr = np.zeros((n_rx, n_tx, cropped.shape[-1]), dtype=np.complex128)
        for ch, (tx_i, rx_i) in enumerate(pairs):
            arr[int(rx_i), int(tx_i), :] = cropped[i, ch, :]
        snaps.append(arr)
    X_b0, info = sm.build_observation_matrix(snaps, cfg)
    X_here = x_from_cropped(cropped, ant_calib_phase=ant_calib_phase)
    scale = max(float(np.max(np.abs(X_b0))), 1e-300)
    max_abs = float(np.max(np.abs(X_here - X_b0)))
    freq_b0 = np.asarray(info.get("freq_axis_in_band_si"), dtype=float)
    freq_here = np.asarray(sc.frequencies_hz(), dtype=float)
    freq_match = bool(freq_b0.size == freq_here.size
                      and np.allclose(freq_b0, freq_here, rtol=0.0, atol=0.0))
    return {"available": True, "identical_bitwise": bool(np.array_equal(X_here, X_b0)),
            "max_abs_diff": max_abs, "max_rel_diff": max_abs / scale,
            "shape": list(X_here.shape), "n_freq_bins": int(info["n_freq_bins"]),
            "feature_order": info["feature_order"],
            "frequencies_match": freq_match,
            "frequencies_hz_b0_first3": [float(v) for v in freq_b0[:3]],
            "frequencies_hz_here_first3": [float(v) for v in freq_here[:3]],
            "frequencies_abs_diff_max": (float(np.max(np.abs(freq_b0 - freq_here)))
                                         if freq_b0.size == freq_here.size else None)}


__all__ = ["cropped_from_raw", "x_from_cropped", "x_from_raw", "verify_vs_b0"]
