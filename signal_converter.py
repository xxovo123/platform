"""
FDTD 通带信号 → 基带 CIR 信号转换模块

功能:
  1. 生成基带参考脉冲 (Blackman-Harris 导数脉冲的 IQ 解调版本)
  2. 将 FDTD 输出的通带 MDM 矩阵转换为下采样后的基带 CIR 矩阵

无算法内容，纯信号处理流水线。
"""

import numpy as np
from scipy.signal import hilbert, resample
from scipy.signal import fftconvolve


def generate_reference_pulse_baseband(time_axis_si, fc_si, duration_si, target_fs_si):
    """
    根据 Blackman-Harris 导数脉冲公式，生成对应的基带参考脉冲。

    参数:
        time_axis_si (np.ndarray): FDTD 时间轴 [秒]
        fc_si (float): 中心载波频率 [Hz]
        duration_si (float): 脉冲持续时间 [秒]
        target_fs_si (float): 目标采样率 [Hz] (e.g. 1e9 for 1 GSa/s)

    返回:
        np.ndarray: 下采样后的基带参考脉冲 (复数)
    """
    t = time_axis_si

    # Blackman-Harris 窗函数系数
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    t0 = duration_si / 2.0
    t_rel = t - t0 + duration_si / 2.0

    in_pulse = np.where((t >= 0) & (t <= duration_si), 1, 0)
    arg = 2 * np.pi * t_rel / duration_si

    w_t = (a0 - a1 * np.cos(arg) + a2 * np.cos(2 * arg) - a3 * np.cos(3 * arg)) * in_pulse
    dw_dt = (2 * np.pi / duration_si) * (a1 * np.sin(arg) - 2 * a2 * np.sin(2 * arg) + 3 * a3 * np.sin(3 * arg)) * in_pulse

    carrier = np.cos(2 * np.pi * fc_si * (t - t0))
    carrier_deriv = -2 * np.pi * fc_si * np.sin(2 * np.pi * fc_si * (t - t0))

    passband_pulse = dw_dt * carrier + w_t * carrier_deriv

    # IQ 解调
    analytic_pulse = hilbert(passband_pulse)
    baseband_pulse_fullrate = analytic_pulse * np.exp(-1j * 2 * np.pi * fc_si * t)

    # 下采样
    original_duration = time_axis_si[-1]
    new_num_points = int(original_duration * target_fs_si)
    baseband_pulse_downsampled = resample(baseband_pulse_fullrate, new_num_points)

    return baseband_pulse_downsampled


def prepare_baseband_data(
    mdm_passband,
    time_axis_si,
    fc_si,
    target_fs_si,
    reference_pulse_baseband=None,
    target_num_points=None
):
    """
    将 FDTD 通带 MDM 信号矩阵转换为基带 CIR 矩阵。

    处理流程:
      1. IQ 解调 (Hilbert 变换 + 数字下变频)
      2. 下采样到目标采样率
      3. (可选) 匹配滤波
      4. (可选) 截断到指定点数

    参数:
        mdm_passband (np.ndarray): FDTD 通带信号 [n_rx, n_tx, n_timesteps]
        time_axis_si (np.ndarray): 时间轴 [秒]
        fc_si (float): 中心载波频率 [Hz]
        target_fs_si (float): 目标采样率 [Hz]
        reference_pulse_baseband (np.ndarray): 基带参考脉冲，用于匹配滤波。None 则跳过。
        target_num_points (int): 截断后保留的点数。None 则不截断。

    返回:
        tuple: (cir_final, time_axis_final, freq_axis_final)
            - cir_final (np.ndarray): 基带 CIR 矩阵 [n_rx, n_tx, n_points]
            - time_axis_final (np.ndarray): 新时间轴 [秒]
            - freq_axis_final (np.ndarray): 基带频率轴 [Hz] (0Hz 为中心)
    """
    print("\n--- [Baseband Workflow] IQ Demodulation + Downsampling ---")

    # 1. IQ 解调
    analytic_signal = hilbert(mdm_passband, axis=2)
    complex_lo = np.exp(-1j * 2 * np.pi * fc_si * time_axis_si)
    cir_baseband_fullrate = analytic_signal * complex_lo

    # 2. 下采样
    original_duration = time_axis_si[-1]
    new_num_points = int(original_duration * target_fs_si)
    downsampled_signal = resample(cir_baseband_fullrate, new_num_points, axis=2)
    new_time_axis = np.linspace(0, original_duration, new_num_points, endpoint=False)

    # 2.5 匹配滤波 (可选)
    if reference_pulse_baseband is not None:
        print("--- Applying Matched Filtering ---")
        matched_filter = np.conj(reference_pulse_baseband[::-1])
        matched_cir_full = np.apply_along_axis(
            lambda x: fftconvolve(x, matched_filter, mode='full'),
            axis=2, arr=downsampled_signal
        )
        delay_samples = len(reference_pulse_baseband) - 1
        matched_cir = matched_cir_full[:, :, delay_samples:delay_samples + new_num_points]
    else:
        matched_cir = downsampled_signal

    # 3. 截断 (可选)
    if target_num_points and new_num_points > target_num_points:
        print(f"--- Truncating CIR to {target_num_points} points ---")
        cir_final = matched_cir[:, :, :target_num_points]
        time_axis_final = new_time_axis[:target_num_points]
    else:
        cir_final = matched_cir
        time_axis_final = new_time_axis

    # 4. 频率轴
    final_dt = time_axis_final[1] - time_axis_final[0] if len(time_axis_final) > 1 else 0
    freq_axis_final = np.fft.fftfreq(len(time_axis_final), d=final_dt)

    return cir_final, time_axis_final, freq_axis_final
