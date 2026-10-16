#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q5 参数化**等价后端**：显式接收（选定频点、对应 X 列、pairs），其余公式逐项原样。

为什么需要它（主审 2026-09-26 口径）
-----------------------------------
04_Q6/common/q6_methods_this_method.py 的 music_image_from_X 把选定频点写死为
"FFT 轴与 freq_range 的**连续带内**选择"（_selected_freqs），因此
**全跨度稀疏抽点**（因素 A 的 F=5/9）根本无法通过改 case 的 frequencies_hz 传入。
本模块把"选定频点"提为**显式入参**，X 列与 pairs 同时显式给出；
**协方差 / FB / 对角加载 / EVD 噪声子空间 / 掩膜 / 被动双程导向 / 伪谱**
逐行保持与 music_image_from_X 相同（见下方逐项对照），**不改源后端、不猴子补丁、
不伪造 tap/FFT 配置凑频轴**。这是**实验参数接口适配**，不是算法改进。

逐项对照（源 -> 本模块）
-----------------------
1. R = X^H X / L                                    -> 同
2. FB: R = 0.5 (R + J R* J), J = flip(I_M)          -> 同（J 沿本特征序，M = F_used*C_used）
3. trace 检查 + 对角加载 R += I * diag_load * trace  -> 同（同一 diag_load_factor）
4. eigh -> 噪声子空间 eigenvectors[:, :M-K]          -> 同（同一 K = case["k"]）
5. 掩膜 y<0、|theta|<=fov/2、距离环 r 属于 cir_window·dist_per_tap -> 同
6. tof = (||p-tx||+||p-rx||)/v_prop + system_delay；phases = -2j pi f tof
   -> phase_cal(deg_corr, lambda_) -> exp                    -> 同（直接复用 phase_cal）
7. denom = Re sum((A EnEnH) * conj(A))；spectrum = 1/(denom+1e-12)，nan->inf、负->0 -> 同

不做的事：不改取峰（取峰仍走 knownk_eval 统一 NMS）、不用真值、不改源工程文件、
不改 04_Q6 的任何模块、不在 case 里塞真值字段。

额外诊断（不改变谱值）：由"R + cI 只是特征值整体平移"这一恒等式，直接给出
**加载前**协方差矩阵的特征值极值与数值秩，用于"维度/秩不满足条件则标无效"。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[4]
SRC_DIR = WS / "Q2 数据采集" / "uwb-occupancy-detection-main"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import radar_algorithm  # noqa: E402

#: 默认数值秩判据（相对最大特征值）；写定，不按结果调整
RANK_TOL_REL = 1e-9


class Q5BackendError(ValueError):
    """输入不合规（显式失败，不静默降级）。"""


def offset_index_map(case_freqs_hz, center_freq_si, df_hz, *, atol_rel=1e-9):
    """case 的 frequencies_hz（原始 FFT 顺序）-> {df 偏移: 列内频率下标}。

    要求每个频点都能被 center + o·df 精确表示，且偏移互不相同；否则显式失败。
    """
    freqs = np.asarray(case_freqs_hz, dtype=float)
    offsets = np.rint((freqs - float(center_freq_si)) / float(df_hz)).astype(int)
    recon = float(center_freq_si) + offsets.astype(float) * float(df_hz)
    if not np.allclose(recon, freqs, rtol=0.0, atol=float(df_hz) * atol_rel):
        raise Q5BackendError("frequencies_hz 不能表示为 center + o*df（df=%g）" % df_hz)
    if len(set(int(o) for o in offsets)) != offsets.size:
        raise Q5BackendError("frequencies_hz 存在重复 df 偏移")
    return {int(o): int(i) for i, o in enumerate(offsets)}


def select_columns(case, offsets_df, channel_indices, *, df_hz):
    """按 df 偏移集合 + 通道下标从 case['X'] 取出子观测矩阵（频率优先、通道其次）。

    返回 (X_sel, freqs_sel, pairs_sel, freq_indices, offset_to_index)；
    频率保持**原始 FFT 顺序**（freq_indices 升序），不按物理升序重排。
    """
    cfg = case["config"]
    center = float(cfg["center_freq_si"])
    omap = offset_index_map(case["frequencies_hz"], center, df_hz)
    want = list(int(o) for o in offsets_df)
    missing = [o for o in want if o not in omap]
    if missing:
        raise Q5BackendError("偏移 %s 不在该窗带内频点中（可用：%s）" % (missing, sorted(omap)))
    freq_indices = sorted(omap[o] for o in want)
    if len(set(freq_indices)) != len(want):
        raise Q5BackendError("偏移集合映射后出现重复频率下标：%s" % want)
    keep = [int(c) for c in channel_indices]
    pairs_all = np.asarray(case["pairs"], dtype=int)
    if keep and (min(keep) < 0 or max(keep) >= pairs_all.shape[0]):
        raise Q5BackendError("通道下标越界：%s（n_channels=%d）" % (keep, pairs_all.shape[0]))
    n_ch_all = int(pairs_all.shape[0])
    col_idx = [f * n_ch_all + c for f in freq_indices for c in keep]
    X = np.asarray(case["X"], dtype=np.complex128)
    X_sel = np.ascontiguousarray(X[:, col_idx])
    freqs_sel = np.asarray(case["frequencies_hz"], dtype=float)[freq_indices]
    pairs_sel = np.asarray(pairs_all[keep], dtype=np.int64)
    return X_sel, freqs_sel, pairs_sel, freq_indices, omap


def apply_config(case, config, *, df_hz):
    """把冻结配置作用到 case：返回 (X_used, freqs_used, pairs_used, info)。

    * 频率：按 config['freq_offsets_df'] 选列（保持原 FFT 顺序）；
    * 通道：按 config['channel_indices'] 选列；
    * 快照：取**每窗末** L 行（X[-L:]），窗口结束时刻不变；
    * FB：显式开关（config['is_forward_backward']）；**不修改 case 本身**。
    """
    L = int(config.get("snapshot_last") or int(np.asarray(case["X"]).shape[0]))
    X_sel, freqs_sel, pairs_sel, freq_indices, _omap = select_columns(
        case, config["freq_offsets_df"], config["channel_indices"], df_hz=df_hz)
    n_all = int(np.asarray(case["X"]).shape[0])
    if L <= 0 or L > n_all:
        raise Q5BackendError("snapshot_last=%d 越界（该窗 L=%d）" % (L, n_all))
    X_used = np.ascontiguousarray(X_sel[n_all - L:, :])
    fb = bool(config["is_forward_backward"])
    info = {
        "config_id": config["id"],
        "L_used": int(L),
        "L_available": int(n_all),
        "snapshot_take": "last_%d_of_%d（窗口结束时刻不变）" % (L, n_all),
        "freq_indices_in_case": [int(i) for i in freq_indices],
        "freq_offsets_df": [int(o) for o in config["freq_offsets_df"]],
        "frequencies_hz_used": [float(v) for v in freqs_sel],
        "channel_indices": [int(c) for c in config["channel_indices"]],
        "pairs_used": pairs_sel.tolist(),
        "is_forward_backward": bool(fb),
        "M_features": int(X_used.shape[1]),
        "n_freq_used": int(freqs_sel.size),
        "n_channels_used": int(pairs_sel.shape[0]),
    }
    return X_used, freqs_sel, pairs_sel, info


def music_image_param(*, cfg, X, selected_freqs, pairs, tx_positions, rx_positions,
                      grid_x, grid_y, k, is_forward_backward, timing=None,
                      rank_tol_rel=RANK_TOL_REL):
    """参数化 MUSIC 谱（逐行等价于 q6_methods_this_method.music_image_from_X）。

    返回 (image, info)。任何维度/有限性/秩/deg_corr 不满足条件 -> Q5BackendError（显式失败）。
    """
    selected_freqs = np.asarray(selected_freqs, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise Q5BackendError("pairs 必须是 (C,2)，收到 shape=%s" % (pairs.shape,))
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise Q5BackendError("X 必须是二维 (L, F*C)，收到 shape=%s" % (X.shape,))
    L, M = int(X.shape[0]), int(X.shape[1])
    n_channels = int(pairs.shape[0])
    n_freqs = int(selected_freqs.size)
    if L < 1 or M < 1:
        raise Q5BackendError("X 形状非法：%s" % (X.shape,))
    if M != n_freqs * n_channels:
        raise Q5BackendError("X 列数 %d != F*C = %d*%d" % (M, n_freqs, n_channels))
    if not np.all(np.isfinite(X)):
        raise Q5BackendError("X 含 NaN/Inf（应显式记 failed，不静默处理）")
    k = int(k)
    if k < 0 or k >= M:
        raise Q5BackendError("K=%d 必须落在 [0, M=%d)" % (k, M))
    deg_corr_cfg = cfg.get("deg_corr")
    if deg_corr_cfg is not None and len(deg_corr_cfg) not in (0, n_channels):
        raise Q5BackendError(
            "config.deg_corr 长度 %d 与本次使用的通道数 %d 不一致：本模块不支持"
            "「子集 + 非空 deg_corr」（不静默按错通道数套用相位校准）"
            % (len(deg_corr_cfg), n_channels))
    deg_corr_arr = radar_algorithm._resolve_deg_corr(deg_corr_cfg, n_channels)
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)

    # ---- 协方差（R = X^H X / L） ----
    t0 = time.perf_counter()
    R_averaged = (X.T.conj() @ X) / float(L)
    if bool(is_forward_backward):
        J = np.flip(np.eye(M), axis=0)
        R_averaged = 0.5 * (R_averaged + J @ R_averaged.conj() @ J)
    trace_before_load = complex(np.trace(R_averaged))
    if (not np.isfinite(trace_before_load.real)) or (not np.isfinite(trace_before_load.imag)) \
            or trace_before_load.real <= 0.0:
        raise Q5BackendError("协方差无能量或非有限（全零/退化输入）：trace=%s" % (trace_before_load,))
    load_shift = float(cfg["diag_load_factor"]) * trace_before_load
    R_averaged = R_averaged + np.eye(M) * load_shift
    t_cov = time.perf_counter() - t0

    # ---- EVD -> 噪声子空间 ----
    t0 = time.perf_counter()
    eigvals_asc, eigenvectors = np.linalg.eigh(R_averaged)
    if (M - k) <= 0:
        raise Q5BackendError("M - K <= 0（K=%d, M=%d），噪声子空间为空" % (k, M))
    noise_subspace = eigenvectors[:, :M - k]
    EnEnH = noise_subspace @ noise_subspace.conj().T
    t_eigh = time.perf_counter() - t0

    # ---- 加载前特征值（R + cI 的特征值 = R 的特征值 + c，这条恒等式给出 rank） ----
    eigvals_before = np.asarray(eigvals_asc, dtype=float) - float(np.real(load_shift))
    abs_max = float(np.max(np.abs(eigvals_before))) if eigvals_before.size else 0.0
    tol = float(rank_tol_rel) * abs_max
    rank_before = int(np.count_nonzero(eigvals_before > tol))
    rank_bound = int(min(M, L * (2 if bool(is_forward_backward) else 1)))

    # ---- 掩膜（与硬件一致） ----
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
    info = {
        "M_features": M, "L_snapshots": L, "n_freq_used": n_freqs,
        "n_channels_used": n_channels, "k": k,
        "is_forward_backward": bool(is_forward_backward),
        "r_start_m": float(r_start), "r_end_m": float(r_end),
        "n_masked_grid_points": int(np.count_nonzero(mask)),
        "trace_before_load": float(np.real(trace_before_load)),
        "diag_load_shift": float(np.real(load_shift)),
        "eig_min_before_load": float(np.min(eigvals_before)),
        "eig_max_before_load": float(abs_max),
        "rank_tol_rel": float(rank_tol_rel),
        "rank_before_load": rank_before,
        "rank_bound": rank_bound,
        "rank_ge_k": bool(rank_before >= k),
        "noise_subspace_ok": bool(M - k > 0),
        "t_covariance_fb_load_s": t_cov,
        "t_eigh_s": t_eigh,
    }
    if not mask.any():
        return image, info

    # ---- 导向（被动双程精确距离流形）-> 伪谱 ----
    t0 = time.perf_counter()
    valid_x = Xg[mask]
    valid_y = Yg[mask]
    pixel_positions = np.stack([valid_x, valid_y, np.zeros_like(valid_x)], axis=1)
    tx_idxs = pairs[:, 0]
    rx_idxs = pairs[:, 1]
    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)
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
    info["t_steering_spectrum_s"] = time.perf_counter() - t0
    return image, info
