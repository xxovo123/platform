# -*- coding: utf-8 -*-
"""B0：标准全二维联合 (TOA-AOA) MUSIC 噪声子空间校验（独立列向量实现）。

算法身份（不是任何单篇论文方法的复现）：
  * 这里实现的是**标准全二维联合 TOA-AOA MUSIC**：同一被动双程精确流形下，
    在 (x, y) 网格上直接求噪声子空间正交性伪谱 P(p) = 1 / (a(p)^H E_n E_n^H a(p) + eps)。
  * **不是** Deng 2023 的角度→条件 TOA 降维、**不是** Two-Stage / 分离一维 MUSIC、
    也不是 B1/B2/B3/B4 任何一条近年基线的复现；B0 只作同模型数值正确性校验。
  * B0 也不是性能基线、不是 FDTD/旧仿真结论的验证。

与硬件实现的约定关系（重要，2026-09-23 首次运行据此修正过一处真实错误）：
  * 硬件 radar_algorithm 用行约定 R_hw = X^H X / L 与 denom_hw = Re sum((A @ P_hw) * conj(A))，
    即 a^T P_hw a^*；本模块用标准列约定 R_col = X^T X^* / L 与 denom_col = a^H P_col a。
  * 两个**协方差矩阵**逐元素互为共轭：R_col = conj(R_hw)，因此噪声投影算子也互为共轭：
    P_hw = conj(P_col)。于是 a^T P_hw a^* = a^H P_col a（两边等于同一实数），
    两种约定才给出同一个伪谱。
  * **不是**“a^H P a 与 a^T P a^* 互为共轭”这么简单：对同一个 P，a^T P a^* 与 a^H P a
    一般不等（它们只在 P 为实矩阵时相同）。谱能对上靠的是 **P 本身也取了共轭**
    （P_hw = conj(P_col)），不是靠把两种二次型当成同一件事。混用（例如用 R_col 配
    a^T P a^*、或用 R_hw 配 a^H P a）会给出完全不同的伪谱——首次运行时就是这样错的，
    已由“分母逐格比对 + 单位测试”固定下来。

独立实现边界：
  * 谱投影路径（协方差、FB 平均、对角加载、EVD、噪声子空间投影、导向矢量与掩膜）全部
    由本模块自行实现；**不调用** radar_algorithm 的 MUSIC 函数来生成 B0 谱，
    硬件函数只作为被比对对象。
  * 数据侧（bin 解析、背景 EMA、加窗）由 cfr_replay 独立实现，并用历史缓存谱核对。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

C0 = 299792458.0


class MusicError(ValueError):
    """配置或输入不合规（K/M、形状、频率范围等）：显式失败，不静默退化。"""


@dataclass(frozen=True)
class MusicConfig:
    fft_length: int = 32
    tap_interval_si: float = 1e-9
    center_freq_si: float = 7.9872e9
    bandwidth_si: float = 500e6
    epsilon_r_background: float = 1.0
    system_delay_si: float = 0.0
    diag_load_factor: float = 1e-5
    is_forward_backward: bool = True
    fov_degrees: float = 180.0
    cir_window: tuple = (0, 32)
    k: int = 1
    pairs: tuple = ((0, 2), (1, 2), (0, 3), (1, 3))
    deg_corr: tuple | None = None
    ant_calib_phase: tuple | None = None
    freq_range_si: tuple | None = None
    extra: dict = field(default_factory=dict)


def frequency_axis(fft_length: int, tap_interval_si: float, center_freq_si: float) -> np.ndarray:
    """与 np.fft.fftfreq 一致、首 bin 为中心频率的物理频率轴（Hz）。"""
    return np.fft.fftfreq(int(fft_length), d=float(tap_interval_si)) + float(center_freq_si)


def in_band_indices(cfg: MusicConfig):
    axis = frequency_axis(cfg.fft_length, cfg.tap_interval_si, cfg.center_freq_si)
    if cfg.freq_range_si is None:
        low = cfg.center_freq_si - cfg.bandwidth_si / 2.0
        high = cfg.center_freq_si + cfg.bandwidth_si / 2.0
    else:
        low, high = float(cfg.freq_range_si[0]), float(cfg.freq_range_si[1])
    idx = np.where((axis >= low) & (axis <= high))[0]
    if idx.size == 0:
        raise MusicError("带宽范围 [%.6g, %.6g] 内没有频点" % (low, high))
    return axis, low, high, idx


def build_observation_matrix(snapshots, cfg: MusicConfig):
    """独立构造观测矩阵 X (L, M)：M = 频点数 x 通道数，特征序 = 频点优先、通道其次。

物理模型：x_l[f, ch] = FFT(cir_l[:, rx_ch, tx_ch])[f] 经通道相位校准后取值。
通道相位校准按“数据侧乘 exp(-1j*phi_ch)”施加。注意（与硬件实现一致的口径说明）：
数据侧乘 exp(-1j*phi) 与导向矢量侧乘 exp(+1j*phi) **只在不做前后向平均时才严格等价**；
一旦启用 FB，J 共轭反转必须作用于**已校准的数据**，只有数据侧施加才是严格正确的，
因此这里与硬件一致地选择数据侧。
    """
    if snapshots is None or len(snapshots) == 0:
        raise MusicError("没有可用快照")
    first = np.asarray(snapshots[0])
    if first.ndim != 3:
        raise MusicError("每个快照必须是 (n_rx, n_tx, n_taps)")
    n_rx, n_tx, n_taps = first.shape
    if n_taps < 1:
        raise MusicError("tap 维必须 >= 1")
    axis, low, high, freq_idx = in_band_indices(cfg)
    n_channels = len(cfg.pairs)
    if n_channels == 0:
        raise MusicError("虚拟通道数为 0")
    for idx, pair in enumerate(cfg.pairs):
        tx_i, rx_i = int(pair[0]), int(pair[1])
        if not (0 <= tx_i < n_tx) or not (0 <= rx_i < n_rx):
            raise MusicError("通道 %d 的 (tx, rx)=(%d, %d) 超界（n_tx=%d, n_rx=%d）"
                             % (idx, tx_i, rx_i, n_tx, n_rx))
    ant_calib = None
    if cfg.ant_calib_phase is not None and len(cfg.ant_calib_phase):
        ant_calib = np.asarray(cfg.ant_calib_phase, dtype=float)
        max_virt = max(int(t) * n_rx + int(r) for t, r in cfg.pairs)
        if max_virt >= ant_calib.size:
            raise MusicError("ant_calib_phase 长度 %d 不足（需要 >= %d）"
                             % (ant_calib.size, max_virt + 1))
    L = len(snapshots)
    cir = np.empty((L, n_rx, n_tx, n_taps), dtype=np.complex128)
    for i, snap in enumerate(snapshots):
        arr = np.asarray(snap, dtype=np.complex128)
        if arr.shape != (n_rx, n_tx, n_taps):
            raise MusicError("快照 %d 形状 %s 与首个快照 %s 不一致"
                             % (i, arr.shape, (n_rx, n_tx, n_taps)))
        if not np.all(np.isfinite(arr)):
            raise MusicError("快照 %d 含 NaN/Inf" % i)
        cir[i] = arr
    cfr = np.fft.fft(cir, n=int(cfg.fft_length), axis=-1)
    tx_idx = [int(p[0]) for p in cfg.pairs]
    rx_idx = [int(p[1]) for p in cfg.pairs]
    data = cfr[:, rx_idx, tx_idx]
    data = data[:, :, freq_idx]
    if ant_calib is not None:
        virt = np.array([t * n_rx + r for t, r in cfg.pairs], dtype=int)
        data = data * np.exp(-1j * ant_calib[virt])[None, :, None]
    X = data.transpose(0, 2, 1).reshape(L, -1)
    info = {
        "n_snapshots": int(L),
        "n_channels": int(n_channels),
        "n_freq_bins": int(freq_idx.size),
        "n_features": int(X.shape[1]),
        "freq_axis_in_band_si": axis[freq_idx].tolist(),
        "freq_range_si": [float(low), float(high)],
        "feature_order": "frequency-major then channel (index = f*n_channels + ch)",
    }
    return X, info


def steering_matrix(cfg: MusicConfig, tx_positions, rx_positions, grid_x, grid_y):
    """返回 (A, mask, valid_x, valid_y, start_index_map)。

    A 的第 p 行对应掩膜内第 p 个像素的导向矢量（列约定，长度 M，顺序同 X 的特征序）：
      a_ch(p) = exp(-2j*pi*f*tau_ch(p)) * exp(-1j*2*pi/lambda*deg_corr_ch)
      tau_ch(p) = (||p - x_Tx|| + ||p - x_Rx||) / v_prop + system_delay
    这是被动双程的**真实距离**流形（不是虚拟天线平面波求和）。
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    if grid_x.ndim != 1 or grid_y.ndim != 1:
        raise MusicError("网格必须是一维数组")
    if not (np.all(np.isfinite(grid_x)) and np.all(np.isfinite(grid_y))):
        raise MusicError("网格含 NaN/Inf")
    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)
    axis, low, high, freq_idx = in_band_indices(cfg)
    freqs = axis[freq_idx]
    c0 = C0
    v_prop = c0 / math.sqrt(float(cfg.epsilon_r_background))
    xg, yg = np.meshgrid(grid_x, grid_y)
    r_dist = np.sqrt(xg ** 2 + yg ** 2)
    angles = np.arctan2(xg, -yg)
    dist_per_tap = (float(cfg.tap_interval_si) * v_prop) / 2.0
    r_start = float(cfg.cir_window[0]) * dist_per_tap
    r_end = float(cfg.cir_window[1]) * dist_per_tap
    max_angle = math.radians(float(cfg.fov_degrees) / 2.0)
    mask = ((yg < 0.0) & (np.abs(angles) <= max_angle)
            & (r_dist >= r_start) & (r_dist <= r_end))
    if not mask.any():
        raise MusicError("掩膜为空：检查 fov_degrees / cir_window / 网格")
    valid_x = xg[mask]
    valid_y = yg[mask]
    pixel = np.stack([valid_x, valid_y, np.zeros_like(valid_x)], axis=1)
    n_channels = len(cfg.pairs)
    n_freqs = freqs.size
    tof = np.empty((pixel.shape[0], n_channels), dtype=float)
    for idx, (tx_i, rx_i) in enumerate(cfg.pairs):
        d_tx = np.linalg.norm(pixel - tx_positions[int(tx_i)], axis=1)
        d_rx = np.linalg.norm(pixel - rx_positions[int(rx_i)], axis=1)
        tof[:, idx] = (d_tx + d_rx) / v_prop + float(cfg.system_delay_si)
    lam = c0 / float(cfg.center_freq_si)
    path_offsets = np.zeros(n_channels, dtype=float)
    if cfg.deg_corr is not None and len(cfg.deg_corr):
        path_offsets = np.asarray(cfg.deg_corr, dtype=float)
        if path_offsets.size != n_channels:
            raise MusicError("deg_corr 长度 %d 必须等于通道数 %d"
                             % (path_offsets.size, n_channels))
    phase = (-2j * np.pi * freqs[None, :, None] * tof[:, None, :]
             - 1j * (2.0 * np.pi / lam) * path_offsets[None, None, :])
    A = np.exp(phase).reshape(pixel.shape[0], n_freqs * n_channels)
    return A, mask, valid_x, valid_y


def sample_covariance(X, cfg: MusicConfig):
    """标准列约定样本协方差 R = X^T X^* / L，含 FB 平均与对角加载；返回 (R, info)。"""
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise MusicError("X 必须是二维 (L, M)")
    L, M = X.shape
    if L < 1 or M < 1:
        raise MusicError("X 形状非法：%s" % (X.shape,))
    R = (X.T @ X.conj()) / float(L)
    info = {"L": int(L), "M": int(M), "fb_applied": bool(cfg.is_forward_backward)}
    if cfg.is_forward_backward:
        J = np.flip(np.eye(M), axis=0)
        R = 0.5 * (R + J @ R.conj() @ J)
    trace_before = complex(np.trace(R))
    info["trace_before_load"] = [trace_before.real, trace_before.imag]
    if (not np.isfinite(trace_before.real)) or (not np.isfinite(trace_before.imag)) \
            or trace_before.real <= 0.0:
        info["zero_energy"] = True
        return None, info
    info["zero_energy"] = False
    R = R + np.eye(M) * float(cfg.diag_load_factor) * trace_before
    return R, info


def row_convention_covariance(X, cfg: MusicConfig):
    """硬件行约定 R_hw = X^H X / L（含同样的 FB 平均与对角加载），仅供共轭关系核对。

    两种约定互为逐元素共轭：R_col = conj(R_hw)。本函数不参与 B0 谱的计算路径。
    """
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise MusicError("X 必须是二维 (L, M)")
    L, M = X.shape
    R = (X.T.conj() @ X) / float(L)
    if cfg.is_forward_backward:
        J = np.flip(np.eye(M), axis=0)
        R = 0.5 * (R + J @ R.conj() @ J)
    trace_before = complex(np.trace(R))
    if (not np.isfinite(trace_before.real)) or trace_before.real <= 0.0:
        return None
    return R + np.eye(M) * float(cfg.diag_load_factor) * trace_before


def standard_music_spectrum(snapshots, cfg: MusicConfig, tx_positions, rx_positions,
                            grid_x, grid_y):
    """独立标准全二维 MUSIC 伪谱（列约定）。返回 dict（含中间量供比对）。"""
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    image = np.zeros((grid_y.size, grid_x.size), dtype=float)
    X, data_info = build_observation_matrix(snapshots, cfg)
    L, M = X.shape
    k = int(cfg.k)
    if k < 0:
        raise MusicError("K 不能为负：%d" % k)
    if k >= M:
        raise MusicError("K=%d 必须小于特征维数 M=%d" % (k, M))
    R, cov_info = sample_covariance(X, cfg)
    out = {
        "image": image,
        "X": X,
        "data_info": data_info,
        "cov_info": cov_info,
        "k": k,
        "M": int(M),
        "L": int(L),
        "eigenvalues": None,
        "denominator_image": np.zeros_like(image),
        "mask": np.zeros_like(image, dtype=bool),
    }
    if cov_info.get("zero_energy"):
        out["status"] = "zero_energy"
        return out
    eigvals, eigvecs = np.linalg.eigh(R)
    if (M - k) <= 0:
        raise MusicError("M - K <= 0，噪声子空间为空")
    En = eigvecs[:, : M - k]
    projector = En @ En.conj().T
    A, mask, valid_x, valid_y = steering_matrix(cfg, tx_positions, rx_positions, grid_x, grid_y)
    # 列约定：denom = a^H P a = sum_m (conj(a) P)_m a_m
    # （写成 A @ P 再乘 conj(A) 得到的是 a^T P a^*，即行约定，二者互为共轭，不能混用）
    PA = np.conj(A) @ projector
    denom = np.real(np.sum(PA * A, axis=1))
    denom = np.where(np.isnan(denom), np.inf, denom)
    denom = np.maximum(denom, 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    spectrum = np.maximum(spectrum, 0.0)
    image[mask] = spectrum
    denom_image = np.zeros_like(image)
    denom_image[mask] = denom
    out.update({
        "image": image,
        "denominator_image": denom_image,
        "mask": mask,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "eigenvalues": eigvals,
        "projector": projector,
        "R": R,
        "status": "ok",
    })
    return out


def config_from_meta(meta, *, k: int, pairs=None) -> MusicConfig:
    """从 meta 的 algo_params['2D-MUSIC'] 与 sampling 字段构造 B0 配置（只读）。"""
    mp = ((meta.get("algo_params") or {}).get("2D-MUSIC") or {})
    ft_len = int(meta.get("ft_len") or 32)
    cir_offset = int(mp.get("cir_offset") or 0)
    if not (0 <= cir_offset < ft_len):
        raise MusicError("cir_offset=%r 超界（ft_len=%d）" % (mp.get("cir_offset"), ft_len))
    ant_calib = mp.get("ant_calib_phase") if mp.get("ant_calib_en") else None
    deg_corr = mp.get("deg_corr") or None
    return MusicConfig(
        fft_length=ft_len,
        tap_interval_si=float(meta.get("tap_interval_si") or 1e-9),
        center_freq_si=float(mp.get("center_freq")),
        bandwidth_si=float(mp.get("bandwidth")),
        epsilon_r_background=float(mp.get("epsilon_r", 1.0)),
        system_delay_si=float(mp.get("system_delay", 0.0)),
        diag_load_factor=float(mp.get("diag_load", 1e-5)),
        is_forward_backward=bool(mp.get("forward_backward")),
        fov_degrees=float(mp.get("fov_degrees", 180.0)),
        cir_window=(cir_offset, ft_len),
        k=int(k),
        pairs=tuple(pairs or []),
        deg_corr=deg_corr,
        ant_calib_phase=ant_calib,
    )
