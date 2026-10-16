"""独立 2D-MUSIC 成像算法模块（纯 numpy，不依赖硬件/服务模块）。

移植来源：
    2dmusic-host-software-Encapsulation--paper_test/uwb_radar_lib/radar_algorithm.py
    中的 _run_radar_algorithm_optimized，以及同文件配套的 phase_cal 与
    uwb_radar_lib/util.py 中的 calculate_mdl_asc。

本模块去掉了源实现对 config_base / RadarDataService / internal_config 的耦合，
只依赖 numpy，可被采集端程序（gui_main.py 等）直接导入使用。

相对源实现的接口扩展（源函数签名前缀与返回 (image, k_hat) 保持不变）：
    signal_subspace_dim=None  信号子空间维数 K。
                              None = 用 MDL 自动估计；
                              0    = 背景（无目标，等价于整段噪声子空间）；
                              正整数 = 直接固定维数，必须小于特征维数 M。
    fft_length=32             CIR 做 FFT 的长度（源实现写死 n=32）。
    center_freq_si=7.9872e9   中心频率，用于相位校准的波长，并参与校验频率轴。
    deg_corr=None             通道级等效路径校准，单位：米。None 等价于全 0。

与源实现保持一致、不要随意改写的关键约定：
    * 协方差用 snapshots.T.conj() @ snapshots；导向矩阵按行摆放，用
      (A @ EnEnH) 与 conj(A) 配套计算分母；
    * 快照向量按"频率优先、通道其次"拼接，与 A 的 reshape 顺序一致；
    * FoV / 距离环 mask、系统时延、前后向平滑(FBA)、对角加载，以及按唯一物理
      天线距离做加速的写法都与源实现一致；
    * 频率轴必须严格等于 fft_frequency_axis(fft_length, tap_interval_si,
      center_freq_si) 生成的完整轴（顺序同 np.fft.fftfreq），不再使用
      FT_LEN - offset 的近似频率映射，也不再接受顺序错乱或改过间距的轴。

数值安全处理（只影响退化输入，不改动算法数学）：
    * 空快照 / 空通道 / 空频段 -> 返回全 0 图像与 k_hat=0；
    * 全零输入（协方差无能量）-> 返回全 0 图像（没有信号就没有可成像目标）；
    * 非法参数（NaN/维度/通道/FFT 轴/频率轴不匹配/窗口或频率范围逆序）-> 抛可读
      的 ValueError；快照里的 NaN/Inf 在 FFT 之前就会被拦下；
    * 谱值保证有限且非负，避免 1/0 或 NaN 造成的假峰；
    * calculate_mdl_asc 不再原地修改调用方传入的特征值数组。

人数与信号子空间维数的对应关系（例如 1 人用 K=1、2 人用 K=2、无人用 K=0）
由调用方决定并通过 signal_subspace_dim 传入，本模块只负责校验其合法性。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_C0",
    "DEFAULT_FFT_LENGTH",
    "DEFAULT_CENTER_FREQ_SI",
    "fft_frequency_axis",
    "calculate_mdl_asc",
    "phase_cal",
    "run_radar_algorithm_optimized",
    "_run_radar_algorithm_optimized",
]

DEFAULT_C0 = 2.99792458e8
DEFAULT_FFT_LENGTH = 32
DEFAULT_CENTER_FREQ_SI = 7.9872e9


# ---------------------------------------------------------------------------
# 参数校验小工具（统一在这里抛可读的 ValueError）
# ---------------------------------------------------------------------------
def _require_int(value, name):
    """校验整数参数（bool 视为非法）。"""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{name} 必须是整数，收到 {value!r}")


def _require_positive_int(value, name):
    ivalue = _require_int(value, name)
    if ivalue <= 0:
        raise ValueError(f"{name} 必须为正整数，收到 {value!r}")
    return ivalue


def _require_nonnegative_int(value, name):
    ivalue = _require_int(value, name)
    if ivalue < 0:
        raise ValueError(f"{name} 必须为非负整数，收到 {value!r}")
    return ivalue


def _require_finite(value, name):
    """校验有限实数参数（None / NaN / Inf / 非数值都报错）。"""
    if value is None:
        raise ValueError(f"{name} 不能为 None")
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是实数，收到 {value!r}") from None
    if not np.isfinite(fvalue):
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}")
    return fvalue


def _require_positive_finite(value, name):
    fvalue = _require_finite(value, name)
    if fvalue <= 0.0:
        raise ValueError(f"{name} 必须为正数，收到 {value!r}")
    return fvalue


def _require_finite_array(value, name, ndim=1):
    """校验实数数组：维度、空数组、NaN/Inf。"""
    if value is None:
        raise ValueError(f"{name} 不能为 None")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维数组，收到 shape={arr.shape}")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/Inf，请检查输入")
    return arr


def _require_positions(positions, n_antennas, name):
    """校验天线位置数组，形状必须是 (天线数, 3) 且覆盖快照里的天线索引。"""
    arr = _require_finite_array(positions, name, ndim=2)
    if arr.shape[1] != 3:
        raise ValueError(f"{name} 必须是 (天线数, 3) 的位置数组，收到 shape={arr.shape}")
    if arr.shape[0] < n_antennas:
        raise ValueError(
            f"{name} 只有 {arr.shape[0]} 个天线位置，但快照需要 {n_antennas} 个"
        )
    return arr


def _normalize_selected_pairs(selected_pairs, n_tx_total, n_rx_total):
    """把 selected_pairs 规范化成 [(tx_idx, rx_idx), ...] 并校验索引范围。"""
    if selected_pairs is None:
        raise ValueError("selected_pairs 不能为 None")
    try:
        pairs_raw = list(selected_pairs)
    except TypeError:
        raise ValueError("selected_pairs 必须是 (tx_idx, rx_idx) 序列") from None
    pairs = []
    for i, pair in enumerate(pairs_raw):
        try:
            tx_idx, rx_idx = pair
        except (TypeError, ValueError):
            raise ValueError(
                f"selected_pairs[{i}] 必须是 (tx_idx, rx_idx) 二元组，收到 {pair!r}"
            ) from None
        tx_idx = _require_int(tx_idx, f"selected_pairs[{i}][0]")
        rx_idx = _require_int(rx_idx, f"selected_pairs[{i}][1]")
        if not 0 <= tx_idx < n_tx_total:
            raise ValueError(
                f"selected_pairs[{i}] 的 tx 索引 {tx_idx} 超出 [0, {n_tx_total}) 范围"
            )
        if not 0 <= rx_idx < n_rx_total:
            raise ValueError(
                f"selected_pairs[{i}] 的 rx 索引 {rx_idx} 超出 [0, {n_rx_total}) 范围"
            )
        pairs.append((tx_idx, rx_idx))
    return pairs


def _resolve_deg_corr(deg_corr, n_channels):
    """把 deg_corr 规范化为长度 n_channels 的实数数组；None -> 全 0。"""
    if deg_corr is None:
        return np.zeros(n_channels, dtype=np.float64)
    if isinstance(deg_corr, (int, float, np.integer, np.floating)):
        deg_arr = np.asarray([deg_corr], dtype=np.float64)
    else:
        deg_arr = np.asarray(deg_corr, dtype=np.float64)
    if deg_arr.ndim != 1:
        raise ValueError(
            f"deg_corr 必须是一维序列（长度=通道数 {n_channels}），收到 shape={deg_arr.shape}"
        )
    if deg_arr.size != n_channels:
        raise ValueError(f"deg_corr 长度({deg_arr.size})必须等于通道数({n_channels})")
    if not np.all(np.isfinite(deg_arr)):
        raise ValueError("deg_corr 含 NaN/Inf，请检查输入")
    return deg_arr


def _normalize_snapshots(cir_snapshots):
    """把输入规范化为三维快照列表，并返回 (snapshots, shape)。

    shape 为 (n_rx, n_tx, n_taps)；空输入返回空列表与 None。
    """
    if cir_snapshots is None:
        return [], None
    if isinstance(cir_snapshots, np.ndarray):
        if cir_snapshots.ndim == 4:
            return list(cir_snapshots), tuple(cir_snapshots.shape[1:])
        if cir_snapshots.ndim == 3:
            return [cir_snapshots], tuple(cir_snapshots.shape)
        raise ValueError(
            "cir_snapshots 必须是快照列表或 (L, n_rx, n_tx, n_taps) 四维数组，"
            f"收到 shape={cir_snapshots.shape}"
        )
    try:
        snapshots = list(cir_snapshots)
    except TypeError:
        raise ValueError("cir_snapshots 必须是快照序列（list/tuple/ndarray）") from None
    if not snapshots:
        return [], None
    first = np.asarray(snapshots[0])
    if first.ndim != 3:
        raise ValueError(
            "cir_snapshots 的每个快照必须是 (n_rx, n_tx, n_taps) 三维数组，"
            f"收到 shape={first.shape}"
        )
    shape = tuple(first.shape)
    for idx, snap in enumerate(snapshots):
        actual = np.asarray(snap).shape
        if actual != shape:
            raise ValueError(
                f"cir_snapshots[{idx}] 形状 {actual} 与第一个快照 {shape} 不一致"
            )
    return snapshots, shape


# ---------------------------------------------------------------------------
# 公开辅助函数
# ---------------------------------------------------------------------------
def fft_frequency_axis(
    fft_length=DEFAULT_FFT_LENGTH,
    tap_interval_si=1e-9,
    center_freq_si=DEFAULT_CENTER_FREQ_SI,
):
    """生成与 fft_length 严格一致的物理频率轴（Hz）。

    等价于源实现里的 np.fft.fftfreq(FT_LEN, d=tap_interval_si) + cf，但长度由
    fft_length 决定，避免再出现 FT_LEN - offset 这类近似映射。
    """
    fft_length = _require_positive_int(fft_length, "fft_length")
    tap_interval_si = _require_positive_finite(tap_interval_si, "tap_interval_si")
    center_freq_si = _require_positive_finite(center_freq_si, "center_freq_si")
    return np.fft.fftfreq(fft_length, d=tap_interval_si) + center_freq_si


def calculate_mdl_asc(k, eigvals_asc, M_sub, L_prime):
    """MDL 模型阶数判据（源 uwb_radar_lib/util.py 的等价实现）。

    与源实现的唯一差别：不再原地把非正特征值写成 1e-12（源实现会污染调用方
    传入的数组），改用副本，数值结果完全一致。
    """
    k = _require_int(k, "k")
    M_sub = _require_int(M_sub, "M_sub")
    L_prime = _require_int(L_prime, "L_prime")
    eigvals_asc = np.asarray(eigvals_asc, dtype=np.float64)
    p = M_sub - k
    if p <= 0 or L_prime <= 0:
        return np.inf
    noise_eigvals = np.array(eigvals_asc[:p], dtype=np.float64, copy=True)
    if noise_eigvals.size == 0:
        return np.inf
    noise_eigvals = np.where(noise_eigvals <= 0.0, 1e-12, noise_eigvals)
    geom_mean = np.exp(np.mean(np.log(noise_eigvals)))
    arith_mean = np.mean(noise_eigvals)
    if arith_mean < 1e-12:
        return np.inf
    log_likelihood = -L_prime * p * np.log(geom_mean / arith_mean)
    penalty = 0.5 * k * (2 * M_sub - k) * np.log(L_prime)
    return float(log_likelihood + penalty)


def phase_cal(phases, deg_corr, lambda_):
    """通道级相位校准（沿用源实现，单位：米）。

    phase_slope = 2 * pi / lambda_ * deg_corr，其中 deg_corr 是每个通道的等效
    路径偏差（米）；deg_corr=None 表示不校准（全 0）。
    """
    phases = np.asarray(phases)
    if phases.ndim == 0:
        raise ValueError("phases 至少需要一维（最后一维为通道）")
    lambda_ = _require_positive_finite(lambda_, "lambda_")
    path_offsets = _resolve_deg_corr(deg_corr, phases.shape[-1])
    phase_slope = 2.0 * np.pi / lambda_ * path_offsets
    broadcast_shape = (1,) * (phases.ndim - 1) + (path_offsets.size,)
    return phases - 1j * phase_slope.reshape(broadcast_shape)


# ---------------------------------------------------------------------------
# 主算法
# ---------------------------------------------------------------------------
def _run_radar_algorithm_optimized(
        cir_snapshots,
        tx_positions_si,
        rx_positions_si,
        selected_pairs,
        freq_axis_si_abs,
        freq_range_si,
        epsilon_r_background,
        imaging_grid,
        max_targets,
        cir_window,
        tap_interval_si,
        c0=DEFAULT_C0,
        diag_load_factor=1e-5,
        system_delay_si=0.0,
        is_forward_backward=False,
        fov_degrees=180.0,
        deg_corr=None,
        ant_calib_phase=None,
        signal_subspace_dim=None,
        fft_length=DEFAULT_FFT_LENGTH,
        center_freq_si=DEFAULT_CENTER_FREQ_SI,
):
    """2D (X, Y) MUSIC 成像，返回 (joint_spectrum_image, K_hat)。

    参数与源实现 _run_radar_algorithm_optimized 一一对应（含默认值），并新增
    signal_subspace_dim / fft_length / center_freq_si 三个可选参数。

    Args:
        cir_snapshots: 三维 CIR 快照列表 [ (n_rx, n_tx, n_taps), ... ]，慢时间维。
        tx_positions_si: (n_tx, 3) 发射天线位置（米）。
        rx_positions_si: (n_rx, 3) 接收天线位置（米）。
        selected_pairs: [(tx_idx, rx_idx), ...] 虚拟通道（频率优先拼接）。
        freq_axis_si_abs: 物理频率轴（Hz），必须严格等于
            fft_frequency_axis(fft_length, tap_interval_si, center_freq_si) 的结果
            （长度 = fft_length，顺序同 np.fft.fftfreq）；空数组按空频段处理。
        freq_range_si: (min_freq, max_freq) 参与计算的频率范围（Hz）。
        epsilon_r_background: 背景相对介电常数。
        imaging_grid: (grid_x, grid_y) 成像网格坐标（米）。
        max_targets: MDL 搜索的最大目标数（K 的搜索上界为 max_targets+1）。
        cir_window: (起始 tap, 结束 tap)，用于距离环 mask。
        tap_interval_si: 每个 tap 的时间间隔（秒）。
        c0: 光速。
        diag_load_factor: 对角加载因子。
        system_delay_si: 系统时延（秒）。
        is_forward_backward: 是否做前后向平滑（解相干）。
        fov_degrees: 搜索视场总开角（度），与源实现一致按单侧 fov/2 限制。
        deg_corr: 通道级等效路径校准（米），None 表示全 0。
        ant_calib_phase: 通道级相位矫正（弧度），None 表示关闭。长度必须按虚拟通道序
            （tx_idx * n_rx_total + rx_idx，TX 主序，与 generator.py 的 8 通道顺序一致）
            覆盖到本帧用到的最大通道索引。约定与仓库内 RA-OCCUPY 移植（generator.py
            的 init_capon_steering）一致，但施加在数据侧：每通道乘 exp(-1j*phi)，
            等价于导向矢量侧乘 exp(+1j*phi)。选择数据侧是因为它与 forward_backward
            平均严格兼容（FB 的 J 共轭反转必须作用于已校准数据）。
        signal_subspace_dim: 信号子空间维数 K。None 表示用 MDL 估计；0 表示背景；
            正整数表示固定维数（必须小于 M = 频率点数 x 通道数）。人数到 K 的映射
            由调用方给出（如 1 人 -> 1、2 人 -> 2）。
        fft_length: CIR 的 FFT 长度，默认 32。
        center_freq_si: 中心频率（Hz），用于相位校准的波长，同时用于校验频率轴。

    Returns:
        (image, k_hat)：image 形状为 (len(grid_y), len(grid_x)) 的 float64 图像，
        mask 之外为 0；k_hat 为使用的信号子空间维数。
    """
    # ---------------- 1. 标量参数校验 ----------------
    fft_length = _require_positive_int(fft_length, "fft_length")
    tap_interval_si = _require_positive_finite(tap_interval_si, "tap_interval_si")
    epsilon_r_background = _require_positive_finite(
        epsilon_r_background, "epsilon_r_background"
    )
    c0 = _require_positive_finite(c0, "c0")
    center_freq_si = _require_positive_finite(center_freq_si, "center_freq_si")
    system_delay_si = _require_finite(system_delay_si, "system_delay_si")
    diag_load_factor = _require_finite(diag_load_factor, "diag_load_factor")
    if diag_load_factor < 0.0:
        raise ValueError(f"diag_load_factor 不能为负，收到 {diag_load_factor!r}")
    fov_degrees = _require_positive_finite(fov_degrees, "fov_degrees")
    max_targets = _require_nonnegative_int(max_targets, "max_targets")

    if cir_window is None:
        raise ValueError("cir_window 不能为 None")
    try:
        window_values = list(cir_window)
    except TypeError:
        raise ValueError("cir_window 必须是 (起始 tap, 结束 tap) 二元组") from None
    if len(window_values) != 2:
        raise ValueError("cir_window 必须是 (起始 tap, 结束 tap) 二元组")
    r_start_tap = _require_finite(window_values[0], "cir_window[0]")
    r_end_tap = _require_finite(window_values[1], "cir_window[1]")
    if r_start_tap < 0.0 or r_end_tap < 0.0:
        raise ValueError(f"cir_window 的 tap 不能为负，收到 {window_values!r}")
    if r_start_tap > r_end_tap:
        raise ValueError(f"cir_window 起始 tap 不能大于结束 tap，收到 {window_values!r}")

    if freq_range_si is None:
        raise ValueError("freq_range_si 不能为 None")
    try:
        range_values = list(freq_range_si)
    except TypeError:
        raise ValueError("freq_range_si 必须是 (min_freq, max_freq) 二元组") from None
    if len(range_values) != 2:
        raise ValueError("freq_range_si 必须是 (min_freq, max_freq) 二元组")
    min_freq = _require_finite(range_values[0], "freq_range_si[0]")
    max_freq = _require_finite(range_values[1], "freq_range_si[1]")
    if min_freq > max_freq:
        raise ValueError(f"freq_range_si 最小频率不能大于最大频率，收到 {range_values!r}")

    if imaging_grid is None:
        raise ValueError("imaging_grid 不能为 None")
    try:
        grid_x_raw, grid_y_raw = imaging_grid
    except (TypeError, ValueError):
        raise ValueError("imaging_grid 必须是 (grid_x, grid_y) 二元组") from None
    grid_x = _require_finite_array(grid_x_raw, "imaging_grid[0]")
    grid_y = _require_finite_array(grid_y_raw, "imaging_grid[1]")
    image_shape = (grid_y.size, grid_x.size)

    def _zero_image():
        return np.zeros(image_shape, dtype=np.float64)

    if freq_axis_si_abs is None:
        raise ValueError("freq_axis_si_abs 不能为 None")
    freq_axis = np.asarray(freq_axis_si_abs, dtype=np.float64)
    if freq_axis.ndim != 1:
        raise ValueError(
            f"freq_axis_si_abs 必须是一维频率轴，收到 shape={freq_axis.shape}"
        )
    if freq_axis.size:
        if freq_axis.size != fft_length:
            raise ValueError(
                f"频率轴长度({freq_axis.size})必须与 fft_length({fft_length})一致"
            )
        if not np.all(np.isfinite(freq_axis)):
            raise ValueError("freq_axis_si_abs 含 NaN/Inf，请检查输入")
        expected_axis = fft_frequency_axis(fft_length, tap_interval_si, center_freq_si)
        if not np.allclose(freq_axis, expected_axis, rtol=1e-9, atol=1e-2):
            raise ValueError(
                "freq_axis_si_abs 必须等于 fft_frequency_axis(fft_length, "
                "tap_interval_si, center_freq_si) 生成的完整频率轴（顺序同 np.fft.fftfreq，"
                "首个 bin 为 center_freq_si）；顺序错乱、改了间距或换了中心频率都会让"
                "频率索引与 FFT bin 对不上"
            )

    if signal_subspace_dim is not None:
        signal_subspace_dim = _require_int(signal_subspace_dim, "signal_subspace_dim")
        if signal_subspace_dim < 0:
            raise ValueError(f"signal_subspace_dim 不能为负，收到 {signal_subspace_dim}")

    # ---------------- 2. 快照与通道 ----------------
    snapshots, snapshot_shape = _normalize_snapshots(cir_snapshots)
    if not snapshots:
        return _zero_image(), 0
    n_rx_total, n_tx_total, n_taps = snapshot_shape
    if n_taps < 1:
        raise ValueError(f"cir_snapshots 的 tap 维必须大于等于 1，收到 n_taps={n_taps}")
    if n_rx_total == 0 or n_tx_total == 0:
        return _zero_image(), 0

    pairs = _normalize_selected_pairs(selected_pairs, n_tx_total, n_rx_total)
    n_channels = len(pairs)
    if n_channels == 0:
        return _zero_image(), 0

    deg_corr_arr = _resolve_deg_corr(deg_corr, n_channels)
    ant_calib_arr = None
    if ant_calib_phase is not None:
        ant_calib_arr = _require_finite_array(ant_calib_phase, "ant_calib_phase", ndim=1)
        if ant_calib_arr.size == 0:
            ant_calib_arr = None
    tx_positions = _require_positions(tx_positions_si, n_tx_total, "tx_positions_si")
    rx_positions = _require_positions(rx_positions_si, n_rx_total, "rx_positions_si")

    # ---------------- 3. 频率筛选 ----------------
    freq_indices = np.where((freq_axis >= min_freq) & (freq_axis <= max_freq))[0]
    selected_freqs = freq_axis[freq_indices]
    n_freqs = int(selected_freqs.size)
    if n_freqs == 0:
        return _zero_image(), 0

    n_features = n_freqs * n_channels
    if signal_subspace_dim is not None and signal_subspace_dim >= n_features:
        raise ValueError(
            f"signal_subspace_dim({signal_subspace_dim}) 必须小于特征维数 "
            f"M={n_features}（= 频率点数 n_freqs({n_freqs}) x 通道数 n_channels({n_channels})）"
        )

    # ---------------- 4. 协方差矩阵（频率优先、通道其次） ----------------
    L = len(snapshots)
    all_cir = np.empty((L, n_rx_total, n_tx_total, n_taps), dtype=np.complex128)
    for i, snap in enumerate(snapshots):
        snap_arr = np.asarray(snap, dtype=np.complex128)
        if not np.all(np.isfinite(snap_arr)):
            raise ValueError(f"cir_snapshots[{i}] 含 NaN/Inf，请检查输入")
        all_cir[i] = snap_arr
    all_mdm_freq = np.fft.fft(all_cir, n=fft_length, axis=-1)

    tx_idxs = np.array([p[0] for p in pairs], dtype=np.intp)
    rx_idxs = np.array([p[1] for p in pairs], dtype=np.intp)
    # 虚拟通道序 = tx_idx * n_rx_total + rx_idx（TX 主序）
    virt_ids = tx_idxs * n_rx_total + rx_idxs
    if ant_calib_arr is not None and int(virt_ids.max()) >= ant_calib_arr.size:
        raise ValueError(
            f"ant_calib_phase 长度({ant_calib_arr.size})不足：虚拟通道序 = tx_idx*n_rx + rx_idx"
            f"（TX 主序），与 generator.py 的 8 通道一致；本帧最大通道索引为 "
            f"{int(virt_ids.max())}，至少需要 {int(virt_ids.max()) + 1} 个通道相位"
        )

    # 索引后形状: [L, n_channels, fft_length] -> [L, n_channels, n_freqs]
    data_channels = all_mdm_freq[:, rx_idxs, tx_idxs]
    data_extracted = data_channels[:, :, freq_indices]
    if ant_calib_arr is not None:
        # RA-OCCUPY 式通道相位矫正: 数据侧乘 exp(-1j*phi), 精确还原无相偏数据。
        # 与导向矢量侧乘 exp(+1j*phi) 在不开启前后向平均时严格恒等; 开启 FB 时只有
        # 数据侧严格正确(FB 的 J 共轭反转须作用于已校准数据), 故选数据侧。
        pair_phase = ant_calib_arr[virt_ids]
        data_extracted = data_extracted * np.exp(-1j * pair_phase)[None, :, None]
    # 转置后 [L, n_freqs, n_channels] -> [L, n_freqs * n_channels]，与 A 的行顺序一致
    snapshots_mat = data_extracted.transpose(0, 2, 1).reshape(L, -1)

    R_averaged = (snapshots_mat.T.conj() @ snapshots_mat) / float(L)
    M_sub = int(R_averaged.shape[0])
    if M_sub == 0:
        return _zero_image(), 0

    if is_forward_backward:
        J = np.flip(np.eye(M_sub), axis=0)
        R_averaged = 0.5 * (R_averaged + J @ R_averaged.conj() @ J)

    trace_before_load = complex(np.trace(R_averaged))
    if (
        not np.isfinite(trace_before_load.real)
        or not np.isfinite(trace_before_load.imag)
        or trace_before_load.real <= 0.0
    ):
        # 全零输入或非有限输入：没有信号能量，直接给零图
        return _zero_image(), 0

    R_averaged = R_averaged + np.eye(M_sub) * diag_load_factor * trace_before_load

    # ---------------- 5. 特征分解与子空间维数 ----------------
    try:
        eigenvalues_asc, eigenvectors = np.linalg.eigh(R_averaged)
    except np.linalg.LinAlgError:
        return _zero_image(), 0
    eigenvalues_asc = np.abs(eigenvalues_asc)

    if signal_subspace_dim is None:
        search_range_max = min(max_targets + 1, M_sub)
        mdl_values = [
            calculate_mdl_asc(k, eigenvalues_asc, M_sub, L)
            for k in range(search_range_max)
        ]
        mdl_arr = np.asarray(mdl_values, dtype=np.float64)
        finite_mask = np.isfinite(mdl_arr)
        if finite_mask.any():
            K_hat = int(np.argmin(np.where(finite_mask, mdl_arr, np.inf)))
        else:
            K_hat = 0
    else:
        if signal_subspace_dim >= M_sub:
            raise ValueError(
                f"signal_subspace_dim({signal_subspace_dim}) 必须小于特征维数 M={M_sub}"
            )
        K_hat = int(signal_subspace_dim)

    if (M_sub - K_hat) <= 0:
        return _zero_image(), K_hat

    noise_subspace = eigenvectors[:, :M_sub - K_hat]
    EnEnH = noise_subspace @ noise_subspace.conj().T

    # ---------------- 6. FoV 与距离环 mask（与源实现一致） ----------------
    X, Y = np.meshgrid(grid_x, grid_y)
    v_prop = c0 / np.sqrt(epsilon_r_background)
    R_dist = np.sqrt(X ** 2 + Y ** 2)
    Angles = np.arctan2(X, -Y)
    dist_per_tap = (tap_interval_si * v_prop) / 2.0
    r_start = r_start_tap * dist_per_tap
    r_end = r_end_tap * dist_per_tap
    max_angle_rad = np.deg2rad(fov_degrees / 2.0)
    mask = (
        (Y < 0)
        & (np.abs(Angles) <= max_angle_rad)
        & (R_dist >= r_start)
        & (R_dist <= r_end)
    )

    joint_spectrum_image = _zero_image()
    valid_x = X[mask]
    valid_y = Y[mask]
    if valid_x.size == 0:
        return joint_spectrum_image, K_hat

    # ---------------- 7. 导向向量与 MUSIC 谱 ----------------
    pixel_positions = np.stack([valid_x, valid_y, np.zeros_like(valid_x)], axis=1)

    # 只用唯一物理天线的距离，再由映射关系组合出通道距离
    u_tx_idx, tx_inv = np.unique(tx_idxs, return_inverse=True)
    u_rx_idx, rx_inv = np.unique(rx_idxs, return_inverse=True)
    d_tx_unique = np.linalg.norm(
        pixel_positions[:, None, :] - tx_positions[u_tx_idx], axis=2
    )
    d_rx_unique = np.linalg.norm(
        pixel_positions[:, None, :] - rx_positions[u_rx_idx], axis=2
    )
    tof = (d_tx_unique[:, tx_inv] + d_rx_unique[:, rx_inv]) / v_prop + system_delay_si

    phases = -2j * np.pi * selected_freqs[None, :, None] * tof[:, None, :]
    lambda_ = c0 / center_freq_si
    phases = phase_cal(phases, deg_corr_arr, lambda_)
    A = np.exp(phases).reshape(valid_x.size, -1)

    denom = np.real(np.sum((A @ EnEnH) * np.conj(A), axis=1))
    denom = np.where(np.isnan(denom), np.inf, denom)
    denom = np.maximum(denom, 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    spectrum = np.maximum(spectrum, 0.0)
    joint_spectrum_image[mask] = spectrum

    return joint_spectrum_image, K_hat


# 便于采集端按公开名字调用
run_radar_algorithm_optimized = _run_radar_algorithm_optimized
