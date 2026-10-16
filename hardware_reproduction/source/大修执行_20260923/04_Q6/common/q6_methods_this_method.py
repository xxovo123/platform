#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本方法（JSF-MUSIC / 论文方法）的硬件侧适配器（**公平计时版**）。

修复背景（2026-09-24，硬件侧 README §6.3）
------------------------------------------
* 旧版 estimate() 走 hardware_spectrum(case)：该函数把 case['snapshots']（时域 CIR）
  交给硬件原函数，后者**内部重新做** fft(n=fft_length) 构造观测矩阵；旧版却把整段耗时
  计入 t_spectrum，同时在 diagnostics 里声称"前处理/解码/FFT 属缓存构建、不计入本计时"。
  **计时范围与实测不符 → 与其它方法计时不公平。**

* 本版把**公平计时路径**改成从 case['X']（缓存里已构造好的观测矩阵）开始：
  协方差 → FB → 对角加载 → EVD/噪声子空间 → 导向/掩膜 → MUSIC 伪谱 → 自身取峰。

* 原 hardware_spectrum(case) **保留为核验参考路径**（内部仍从 snapshots 重跑 FFT），
  只用于和公平路径做逐格谱核对；**不参与** estimate() 计时，命名上明确区分。

算法身份（本方法身份不变）
--------------------------
公平路径是硬件 radar_algorithm._run_radar_algorithm_optimized **FFT 之后全部步骤**的
逐行等价实现——同一行约定 R = X^H X / L、同一 denom = Re Σ (A P) conj(A)、
同一被动双程精确距离流形、同一 FB/对角加载/掩膜/通道相位校准口径——只是观测矩阵
由缓存提供，而不是在计时区内现场重算 FFT。

"FFT 之后全部步骤只依赖观测矩阵"这一等价性不是假定，而是实测核对项：
verification_independent/test_input_wrapper_fairness.py 在单/双/三目标多录制代表窗上
逐格比对"从 X 起算"与"从 snapshots 起算（硬件原函数）"两条路径。

口径边界（不做的事）
--------------------
* **不**另行裁剪 tap、**不**做 2 ns 补偿、**不**重新校相位：case['X'] 已是冻结节
  链（帧解析→背景 EMA→128 帧环→裁掉前 cir_offset 个 tap→fft(n=fft_length) 补零→
  取带内频点→数据侧 exp(-1j*phi)）的产物，本模块原样使用。
* **不用**真值：case 由 q6_contract.validate_case 硬门控真值字段；本模块另做一次
  find_truth_leaks 显式失败检查。
* 非有限输入 / 非法维数 / 频点轴不一致 / K >= M 一律**显式失败**，不静默降级。
* **子集 + 非空 deg_corr 不在支持范围**：q6_case_cache.subset_case 只收窄 pairs/X 列，
  **不裁切** config.deg_corr（仍为全长）。本批 deg_corr 全为 None，故不受影响；
  若将来在非空 deg_corr 上取子集，本模块会因长度不匹配**显式失败**，而不是静默按错
  通道数套用相位校准。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

WS = Path(__file__).resolve().parents[3]
COMMON_DIR = Path(__file__).resolve().parent
SRC_DIR = WS / "Q2 数据采集" / "uwb-occupancy-detection-main"
P0B_DRIVER = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "driver"
PROTOCOL = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "protocol" / "knownk_protocol.json"
for _p in (str(SRC_DIR), str(P0B_DRIVER), str(COMMON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import radar_algorithm  # noqa: E402
import q6_contract as contract  # noqa: E402
from knownk_eval import load_protocol  # noqa: E402
from knownk_eval import peaks as knownk_peaks  # noqa: E402

_PROTOCOL = None
_PEAK_RULE_TEXT = ("P0b 统一取峰（3x3 非严格极大 + 8 连通平台 + 0.3 线性相对阈值 "
                   "+ 0.10 m 确定性 NMS + cap=K）")
TIMED_PATH = "covariance_FB_diag_load_EVD_steering_spectrum_own_peaks_from_case_X"
REFERENCE_PATH = "hardware_reference_from_snapshots_recomputes_fft_NOT_TIMED"


def _protocol():
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = load_protocol(str(PROTOCOL))
    return _PROTOCOL


# ======================================================================================
# 核验参考路径（**不参与计时**）：原封不动地调用硬件原函数，从 snapshots 重跑 FFT
# ======================================================================================
def hardware_spectrum(case):
    """核验参考路径：调用硬件原函数，返回 (image, k_hat)。只读 case，不使用真值。

    **注意**：该函数内部对 case['snapshots'] 重做 fft(n=fft_length)，因此其耗时**包含**
    FFT/观测矩阵构造，属于"前处理"范围；它仅用于和公平路径做逐格谱核对，**不得**用于
    estimate() 的计时。公平计时入口见 hardware_spectrum_from_X。
    """
    cfg = case["config"]
    grid_x = np.asarray(case["grid_x"], dtype=float)
    grid_y = np.asarray(case["grid_y"], dtype=float)
    return radar_algorithm._run_radar_algorithm_optimized(
        cir_snapshots=list(case["snapshots"]),
        tx_positions_si=np.asarray(case["tx_positions"], dtype=float),
        rx_positions_si=np.asarray(case["rx_positions"], dtype=float),
        selected_pairs=[tuple(int(v) for v in p) for p in np.asarray(case["pairs"]).tolist()],
        freq_axis_si_abs=(np.fft.fftfreq(int(cfg["fft_length"]),
                                         d=float(cfg["tap_interval_si"]))
                          + float(cfg["center_freq_si"])),
        freq_range_si=(float(cfg["freq_range_si"][0]), float(cfg["freq_range_si"][1])),
        epsilon_r_background=float(cfg["epsilon_r_background"]),
        imaging_grid=(grid_x, grid_y),
        max_targets=int(case["config"].get("mdl_max_targets", 3)),
        cir_window=tuple(int(v) for v in cfg["cir_window"]),
        tap_interval_si=float(cfg["tap_interval_si"]),
        diag_load_factor=float(cfg["diag_load_factor"]),
        system_delay_si=float(cfg["system_delay_si"]),
        is_forward_backward=bool(cfg["is_forward_backward"]),
        fov_degrees=float(cfg["fov_degrees"]),
        deg_corr=cfg["deg_corr"],
        ant_calib_phase=cfg["ant_calib_phase"],
        signal_subspace_dim=int(case["k"]),
        fft_length=int(cfg["fft_length"]),
        center_freq_si=float(cfg["center_freq_si"]),
    )


# ======================================================================================
# 公平计时路径：从 case['X'] 起算（协方差 → FB → 加载 → EVD → 导向 → 谱）
# ======================================================================================
def _selected_freqs(cfg):
    """复算硬件频率轴与带内频点选择（与 radar_algorithm 逐字一致）。"""
    axis = radar_algorithm.fft_frequency_axis(int(cfg["fft_length"]),
                                              float(cfg["tap_interval_si"]),
                                              float(cfg["center_freq_si"]))
    low, high = float(cfg["freq_range_si"][0]), float(cfg["freq_range_si"][1])
    idx = np.where((axis >= low) & (axis <= high))[0]
    return axis, idx, axis[idx]


def _check_deg_corr_supported(case, n_channels):
    """显式声明 deg_corr 的子集边界：subset_case 不裁切 config.deg_corr。"""
    deg_corr = case["config"].get("deg_corr")
    if deg_corr is None or len(deg_corr) == 0:
        return
    if len(deg_corr) != int(n_channels):
        raise ValueError(
            "config.deg_corr 长度 %d != 通道数 %d：本模块不支持"
            "「子集 + 非空 deg_corr」（q6_case_cache.subset_case 只收窄 pairs/X，"
            "不裁切 deg_corr）。本批 deg_corr 全为 None，故正常路径不受影响。"
            % (len(deg_corr), int(n_channels)))


def music_image_from_X(case, X, *, timing=None):
    """硬件函数 **FFT 之后全部步骤** 的逐行等价实现，观测矩阵由调用方给出。

    返回 (image, k_hat, info)；info 记录 FB/加载/掩膜等口径，便于诊断与核对。
    非法输入（维数/非有限/K>=M/频点不一致/deg_corr 长度）抛 ValueError。
    """
    cfg = case["config"]
    grid_x = np.asarray(case["grid_x"], dtype=float)
    grid_y = np.asarray(case["grid_y"], dtype=float)
    n_channels = int(np.asarray(case["pairs"]).shape[0])

    axis, freq_idx, selected_freqs = _selected_freqs(cfg)
    declared = np.asarray(case["frequencies_hz"], dtype=float)
    if declared.size != selected_freqs.size or not np.allclose(declared, selected_freqs,
                                                              rtol=0.0, atol=0.0):
        raise ValueError("case['frequencies_hz'] 与 fft 轴带内频点不一致："
                         "n_declared=%d n_axis=%d" % (declared.size, selected_freqs.size))

    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise ValueError("case['X'] 必须是二维 (L, F*C)，收到 shape=%s" % (X.shape,))
    L, M = X.shape
    if L < 1 or M < 1:
        raise ValueError("case['X'] 形状非法：%s" % (X.shape,))
    n_freqs = int(freq_idx.size)
    if M != n_freqs * n_channels:
        raise ValueError("case['X'] 列数 %d != F*C = %d*%d" % (M, n_freqs, n_channels))
    if not np.all(np.isfinite(X)):
        raise ValueError("case['X'] 含 NaN/Inf（应显式记为 failed，不静默处理）")
    k = int(case["k"])
    if k < 0 or k >= M:
        raise ValueError("K=%d 必须落在 [0, M=%d)" % (k, M))
    _check_deg_corr_supported(case, n_channels)
    deg_corr_arr = radar_algorithm._resolve_deg_corr(cfg.get("deg_corr"), n_channels)

    # ---- 协方差（行约定 R = X^H X / L，与硬件一致） ----
    t0 = time.perf_counter()
    R_averaged = (X.T.conj() @ X) / float(L)
    if bool(cfg["is_forward_backward"]):
        J = np.flip(np.eye(M), axis=0)
        R_averaged = 0.5 * (R_averaged + J @ R_averaged.conj() @ J)
    trace_before_load = complex(np.trace(R_averaged))
    if (not np.isfinite(trace_before_load.real)) or (not np.isfinite(trace_before_load.imag)) \
            or trace_before_load.real <= 0.0:
        raise ValueError("协方差无能量或非有限（全零/退化输入）：trace=%s" % (trace_before_load,))
    R_averaged = R_averaged + np.eye(M) * float(cfg["diag_load_factor"]) * trace_before_load
    if timing is not None:
        timing["t_covariance_fb_load_s"] = time.perf_counter() - t0

    # ---- EVD → 噪声子空间 ----
    t0 = time.perf_counter()
    _eigvals_asc, eigenvectors = np.linalg.eigh(R_averaged)
    if (M - k) <= 0:
        raise ValueError("M - K <= 0（K=%d, M=%d），噪声子空间为空" % (k, M))
    noise_subspace = eigenvectors[:, :M - k]
    EnEnH = noise_subspace @ noise_subspace.conj().T
    if timing is not None:
        timing["t_eigh_s"] = time.perf_counter() - t0

    # ---- 掩膜（与硬件一致：y<0、|θ|<=fov/2、距离环 r ∈ cir_window·dist_per_tap） ----
    c0 = radar_algorithm.DEFAULT_C0
    v_prop = c0 / np.sqrt(float(cfg["epsilon_r_background"]))
    Xg, Yg = np.meshgrid(grid_x, grid_y)
    R_dist = np.sqrt(Xg ** 2 + Yg ** 2)
    Angles = np.arctan2(Xg, -Yg)
    dist_per_tap = (float(cfg["tap_interval_si"]) * v_prop) / 2.0
    r_start = float(cfg["cir_window"][0]) * dist_per_tap
    r_end = float(cfg["cir_window"][1]) * dist_per_tap
    max_angle_rad = np.deg2rad(float(cfg["fov_degrees"]) / 2.0)
    mask = ((Yg < 0) & (np.abs(Angles) <= max_angle_rad)
            & (R_dist >= r_start) & (R_dist <= r_end))
    image = np.zeros((grid_y.size, grid_x.size), dtype=np.float64)
    info = {"r_start_m": float(r_start), "r_end_m": float(r_end),
            "n_masked_grid_points": int(np.count_nonzero(mask)), "M": int(M), "L": int(L)}
    if not mask.any():
        return image, k, info

    # ---- 导向（被动双程精确距离流形）→ 伪谱 ----
    t0 = time.perf_counter()
    valid_x = Xg[mask]
    valid_y = Yg[mask]
    pixel_positions = np.stack([valid_x, valid_y, np.zeros_like(valid_x)], axis=1)
    pairs = np.asarray(case["pairs"], dtype=int)
    tx_idxs = pairs[:, 0]
    rx_idxs = pairs[:, 1]
    tx_positions = np.asarray(case["tx_positions"], dtype=float)
    rx_positions = np.asarray(case["rx_positions"], dtype=float)
    u_tx_idx, tx_inv = np.unique(tx_idxs, return_inverse=True)
    u_rx_idx, rx_inv = np.unique(rx_idxs, return_inverse=True)
    d_tx_unique = np.linalg.norm(pixel_positions[:, None, :] - tx_positions[u_tx_idx], axis=2)
    d_rx_unique = np.linalg.norm(pixel_positions[:, None, :] - rx_positions[u_rx_idx], axis=2)
    tof = ((d_tx_unique[:, tx_inv] + d_rx_unique[:, rx_inv]) / v_prop
           + float(cfg["system_delay_si"]))
    phases = -2j * np.pi * selected_freqs[None, :, None] * tof[:, None, :]
    lambda_ = c0 / float(cfg["center_freq_si"])
    phases = radar_algorithm.phase_cal(phases, deg_corr_arr, lambda_)
    A = np.exp(phases).reshape(valid_x.size, -1)
    denom = np.real(np.sum((A @ EnEnH) * np.conj(A), axis=1))
    denom = np.where(np.isnan(denom), np.inf, denom)
    denom = np.maximum(denom, 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    spectrum = np.maximum(spectrum, 0.0)
    image[mask] = spectrum
    if timing is not None:
        timing["t_steering_spectrum_s"] = time.perf_counter() - t0
    return image, k, info


def hardware_spectrum_from_X(case, *, timing=None):
    """公平路径：从 case['X'] 起算，返回 (image, k_hat)。**estimate() 的计时入口。**"""
    if not isinstance(case, dict) or "X" not in case or case["X"] is None:
        raise ValueError("case 缺少 'X'（公平路径从缓存观测矩阵起算，需要 case['X']）")
    image, k_hat, _info = music_image_from_X(case, case["X"], timing=timing)
    return image, k_hat


# 兼容旧名：指向公平路径，避免再被误当成"含 FFT 的计时入口"。
spectrum_from_case = hardware_spectrum_from_X


def verify_fair_vs_reference(case):
    """核验：公平路径（从 X）与参考路径（硬件原函数，从 snapshots，重跑 FFT）逐格比对。

    返回 dict（最大绝对/相对差、归一化谱余弦、峰集是否一致）。不用真值。
    需要 case 同时带 snapshots（参考路径）与 X（公平路径）。
    """
    fair, k_fair = hardware_spectrum_from_X(case)
    if case.get("snapshots") is None:
        raise ValueError("参考路径需要 case['snapshots']（本函数只用于核验，不进计时）")
    ref, k_ref = hardware_spectrum(case)
    fair = np.asarray(fair, dtype=float)
    ref = np.asarray(ref, dtype=float)
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    max_abs = float(np.max(np.abs(fair - ref)))
    fa = fair.ravel()
    ra = ref.ravel()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(ra))
    cos = float(fa @ ra / denom) if denom > 0.0 else 1.0
    peaks_fair, _ = knownk_peaks.nms_peaks(fair, np.asarray(case["grid_x"], float),
                                           np.asarray(case["grid_y"], float),
                                           _protocol().peaks, k=int(case["k"]))
    peaks_ref, _ = knownk_peaks.nms_peaks(ref, np.asarray(case["grid_x"], float),
                                          np.asarray(case["grid_y"], float),
                                          _protocol().peaks, k=int(case["k"]))
    pk_fair = [(round(p["x"], 12), round(p["y"], 12), round(p["value"], 12)) for p in peaks_fair]
    pk_ref = [(round(p["x"], 12), round(p["y"], 12), round(p["value"], 12)) for p in peaks_ref]
    return {
        "window_id": case.get("id"), "k": int(case["k"]),
        "k_hat_fair": int(k_fair), "k_hat_reference": int(k_ref),
        "max_abs_diff": max_abs, "max_rel_diff": max_abs / scale,
        "spectrum_cosine": cos,
        "identical_bitwise": bool(np.array_equal(fair, ref)),
        "peaks_identical": pk_fair == pk_ref,
        "peaks_fair": pk_fair, "peaks_reference": pk_ref,
        "n_finite_fair": int(np.count_nonzero(np.isfinite(fair))),
    }


# ======================================================================================
# 公共合同入口
# ======================================================================================
def _fail(reason):
    return {"predictions": [], "status": "failed", "diagnostics": {"reason": str(reason)}}


def estimate(case: dict) -> dict:
    """公共合同入口：公平计时路径（从 case['X'] 起算），含自身取峰。

    计时范围 = 协方差 + FB + 对角加载 + EVD + 导向/掩膜 + 全网格伪谱 + **自身取峰**。
    解码/重放/裁剪/FFT 构造 X 属**前处理**，在缓存构建时另计，**不含**在本计时内。
    不需要 case['snapshots']。任何非有限/非法维度/真值字段一律显式 failed。
    """
    if not isinstance(case, dict):
        return _fail("case 必须是 dict，收到 %r" % type(case).__name__)
    leaks = contract.find_truth_leaks(case)
    if leaks:
        return _fail("case 含疑似真值字段（禁止传入估计器）：%s" % ", ".join(leaks[:8]))
    for key in ("X", "k", "grid_x", "grid_y", "pairs", "config", "frequencies_hz"):
        if key not in case:
            return _fail("case 缺少必需键 %r" % key)
    if case["X"] is None:
        return _fail("case['X'] 为 None：公平路径需要缓存观测矩阵（不需要 snapshots）")

    timing = {}
    t_total0 = time.perf_counter()
    try:
        image, _k_hat, info = music_image_from_X(case, case["X"], timing=timing)
    except ValueError as exc:
        return _fail("公平路径输入不合规（显式失败，未静默降级）：%s" % exc)
    t_spectrum = time.perf_counter() - t_total0

    if not np.all(np.isfinite(image)):
        return _fail("公平路径产出的谱含 NaN/Inf（显式失败）")

    t1 = time.perf_counter()
    peaks, _peak_status = knownk_peaks.nms_peaks(
        np.asarray(image, dtype=float), np.asarray(case["grid_x"], dtype=float),
        np.asarray(case["grid_y"], dtype=float), _protocol().peaks, k=int(case["k"]))
    t_peaks = time.perf_counter() - t1

    return {
        "predictions": [{"x": float(p["x"]), "y": float(p["y"]),
                         "score": float(p["value"])} for p in peaks],
        "status": "ok",
        "spectrum": np.asarray(image, dtype=float),
        "diagnostics": {
            "eval_path": "spectrum",
            "timed_path": TIMED_PATH,
            "reference_path": REFERENCE_PATH,
            "x_source": "case['X']（缓存观测矩阵）；解码/重放/裁剪/FFT 构造 X 属前处理，"
                        "在缓存构建时另计，不计入本计时",
            "grid_kind": "cartesian_2d_full_grid_with_distance_ring_mask",
            "n_candidates_evaluated": int(np.count_nonzero(image != 0.0)),
            "dictionary_size": None,
            "dictionary_size_note": "非字典型方法：全网格扫描，规模由掩膜内网格点数决定",
            "n_masked_grid_points": int(info["n_masked_grid_points"]),
            "distance_ring_m": [info["r_start_m"], info["r_end_m"]],
            "n_peaks_from_own_picker": len(peaks),
            "peak_rule": _PEAK_RULE_TEXT,
            "k_used": int(case["k"]),
            "fb_used": bool(case["config"]["is_forward_backward"]),
            "n_channels_used": int(np.asarray(case["pairs"]).shape[0]),
            "M_features": int(info["M"]),
            "L_snapshots": int(info["L"]),
            "timing_scope": "covariance+FB+diag_load+EVD+mask+steering+full_grid_spectrum"
                            "+own_peak_picking（从 case['X'] 起算；不含 FFT/X 构造）",
            "t_covariance_fb_load_s": timing.get("t_covariance_fb_load_s"),
            "t_eigh_s": timing.get("t_eigh_s"),
            "t_steering_spectrum_s": timing.get("t_steering_spectrum_s"),
            "t_spectrum_s": t_spectrum,
            "t_own_peaks_s": t_peaks,
            "t_estimate_total_s": time.perf_counter() - t_total0,
            "response_assumption": case["model"]["response_assumption"]["mode"],
        },
    }


def make_estimate(case: dict) -> dict:
    """兼容别名（部分运行器按 make_estimate 探测）。"""
    return estimate(case)
