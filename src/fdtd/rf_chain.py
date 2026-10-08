#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RF → Hilbert-IQ → 抗混叠重采样(精确 1 ns 轴) → 匹配滤波 → raw32。

口径与旧链的关系（2026-09-26 核对与修正）
-----------------------------------------
旧 ``signal_processing.prepare_baseband_data`` 有三处必须修的问题，本模块逐条改 1
并保留可复算的对照：

1. **时间轴不精确**：旧式 ``new_num_points = int(T*fs)`` + ``linspace(0, T, N, endpoint=False)``
   → 实际步长 ``T/N``（本工作区旧档 T≈66.7 ns、N=66 时 ≈1.0101 ns），相对声称的 1 ns
   **偏大约 1%（相对偏差）**；换算到第 k 个 tap 上约 ``0.01·k``，k=32 时约 **0.32 tap**。
   本模块输出**精确** ``t[k] = k·Ts``，不引入该偏差。
2. **抗混叠未声明**：旧 ``scipy.signal.resample`` 是 FFT 重采样，等效于在**新 Nyquist
   (=500 MHz) 处做砖墙截断**，没有保护带、也没有写下来。本模块用显式截止
   （默认 ``aa_cutoff_hz = 0.4·fs = 400 MHz``，含 100 MHz 保护带）并把参数写进 provenance。
3. **周期性回绕**：FFT 重采样把整段记录当一个周期；记录末尾未衰减时会把尾部折回。
   本模块先**零填充**到 ``pad_min_factor×N``（默认 2×），压低回绕污染，并把该设置写盘。

匹配滤波对齐（保留，属物理 filter lag，不是按真值补偿）
----------------------------------------------------
``y = x ⊛ conj(ref[::-1])`` 用 ``mode='full'``，再从下标 ``len(ref)-1`` 起截取。
推导：若 ``x[m] = ref[m-d]``，则 ``y`` 的峰在 ``n = (N_ref-1) + d``，故截 ``N_ref-1``
后第 ``k`` 个样本对应真实时延 ``k·Ts``。该偏移是滤波器群延迟，**与真值无关**。
真值不参与任何对齐/补偿；``crop_start``（硬件 2）留给 adapter。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.signal import fftconvolve, hilbert

from . import pulse as pulse_mod


def hilbert_iq(x: np.ndarray, t_s: np.ndarray, fc_hz: float, axis: int = -1) -> np.ndarray:
    """解析信号 → 复基带：``hilbert(x)·exp(-j2π fc t)``（与旧链同式）。"""
    x = np.asarray(x)
    t_s = np.asarray(t_s, dtype=float)
    if t_s.shape[0] != x.shape[axis]:
        raise ValueError("时间轴长度与信号轴不一致：%d vs %d" % (t_s.shape[0], x.shape[axis]))
    analytic = hilbert(x, axis=axis)
    phase = np.exp(-1j * 2.0 * np.pi * float(fc_hz) * t_s)
    shape = [1] * x.ndim
    shape[axis] = t_s.shape[0]
    return analytic * phase.reshape(shape)


def band_limited_resample(
    x: np.ndarray,
    fs_in_hz: float,
    t_out_s: np.ndarray,
    cutoff_hz: float,
    pad_min_factor: float = 2.0,
    axis: int = -1,
    *,
    t_in0_s: float,
) -> np.ndarray:
    """带限（抗混叠）重采样到任意精确输出时刻。

    做法：对（零填充后的）输入做 FFT，只保留 ``|f| <= cutoff_hz`` 的 bin，再按
    ``y(t) = (1/N)·Σ_k X[k]·exp(j2π f_k t)`` 在给定时刻直接求和。
    因此（a）抗混叠截止**显式可控**，（b）输出时刻**精确**（不需要等间隔）。
    周期性回绕由零填充压低（见模块 docstring）。

    **时间原点**（2026-09-26 修正）：FFT 的样本序号从 0 计，对应输入数组的**首样本**；
    若输入首样本的绝对时间是 ``t_in0_s``（Meep ``at_every`` 首回调在 t=dt，故 t_in0_s=dt），
    则相位必须用**相对**输入起点的 ``(t_out - t_in0_s)``。否则会给整条波形额外乘上
    ``exp(j2πf·t_in0)``，等效整体多移一个 ``t_in0``。
    """
    x = np.asarray(x, dtype=complex)
    t_out_s = np.asarray(t_out_s, dtype=float)
    if x.shape[axis] == 0:
        raise ValueError("输入为空")
    if fs_in_hz <= 0:
        raise ValueError("fs_in_hz 必须 > 0")
    if cutoff_hz <= 0 or cutoff_hz >= fs_in_hz / 2.0:
        raise ValueError(
            "cutoff_hz=%.6g 必须落在 (0, fs_in/2=%.6g) 内" % (cutoff_hz, fs_in_hz / 2.0)
        )
    n_in = x.shape[axis]
    n_pad = n_in
    if pad_min_factor and pad_min_factor > 1.0:
        n_pad = int(np.ceil(n_in * float(pad_min_factor)))
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, n_pad - n_in)
    x_pad = np.pad(x, pad_width)
    spec = np.fft.fft(x_pad, axis=axis)
    freqs = np.fft.fftfreq(n_pad, d=1.0 / float(fs_in_hz))
    keep = np.flatnonzero(np.abs(freqs) <= float(cutoff_hz))
    spec_k = np.take(spec, keep, axis=axis)
    t_rel = np.asarray(t_out_s, dtype=float) - float(t_in0_s)
    phase = np.exp(2j * np.pi * np.outer(t_rel, freqs[keep]))  # [n_out, n_keep]
    moved = np.moveaxis(spec_k, axis, -1)  # [..., n_keep]
    out = np.matmul(moved, phase.T) / float(n_pad)  # [..., n_out]
    return np.moveaxis(out, -1, axis)


def validate_chain_params(cfg: Mapping[str, Any], fs_in_hz: float | None = None) -> Dict[str, float]:
    """链路参数合法性（显式报错，不静默用错值）。

    * ``sample_interval_s > 0``；
    * 抗混叠截止 ``0 < aa_cutoff_hz < fs_out/2``（复数采样 Nyquist = ±fs_out/2，取严格小于）；
    * 给了输入采样率时还要求 ``aa_cutoff_hz < fs_in/2``（输入自身不能已混叠）。
    """
    ts = float(cfg["sample_interval_s"])
    if not (ts > 0.0):
        raise ValueError("sample_interval_s 必须 > 0，得到 %r" % ts)
    fs_out = 1.0 / ts
    cutoff = float(cfg.get("aa_cutoff_hz", 0.4 / ts))
    if not (cutoff > 0.0):
        raise ValueError("aa_cutoff_hz 必须 > 0，得到 %r" % cutoff)
    if cutoff >= fs_out / 2.0:
        raise ValueError(
            "aa_cutoff_hz=%.6g Hz 不小于输出 Nyquist（fs_out/2=%.6g Hz）：会混叠"
            % (cutoff, fs_out / 2.0)
        )
    if fs_in_hz is not None:
        fs_in = float(fs_in_hz)
        if not (fs_in > 0.0):
            raise ValueError("输入采样率必须 > 0，得到 %r" % fs_in)
        if cutoff >= fs_in / 2.0:
            raise ValueError(
                "aa_cutoff_hz=%.6g Hz 不小于输入 Nyquist（fs_in/2=%.6g Hz）" % (cutoff, fs_in / 2.0)
            )
    return {"fs_out_hz": fs_out, "aa_cutoff_hz": cutoff}


def matched_filter_align(x: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, int]:
    """``full`` 卷积后按物理 filter lag ``len(ref)-1`` 对齐；返回 (aligned, lag)。

    ``scipy.signal.fftconvolve`` 要求两个入参 **ndim 相同**；因此参考脉冲必须 reshape 成
    ``(1,)*(x.ndim-1) + (N,)`` 再卷积，否则多维输入会直接报错（2026-09-26 修正）。
    """
    x = np.asarray(x)
    ref = np.asarray(ref)
    if ref.ndim != 1:
        ref = ref.reshape(-1)
    h = np.conj(ref[::-1])
    h = h.reshape((1,) * (x.ndim - 1) + (h.shape[-1],))
    y = fftconvolve(x, h, mode="full", axes=-1)
    lag = int(ref.shape[-1] - 1)
    n_out = x.shape[-1]
    return y[..., lag : lag + n_out], lag


def reference_baseband_on_grid(t_s: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """在仿真时间轴上重建通带参考并走同一条 IQ 链（源与 MF 参考同公式）。"""
    duration_s = pulse_mod.duration_from_bandwidth(cfg["bandwidth_hz"])
    pb = pulse_mod.bh_derivative_passband(t_s, cfg["fc_hz"], duration_s)
    return hilbert_iq(pb, t_s, cfg["fc_hz"])


def output_time_axis(t_s: np.ndarray, sample_interval_s: float, t_start_s: float | None = None) -> np.ndarray:
    """精确 Ts 轴：``t[k] = t_start + k·Ts``，覆盖到记录末尾。

    ``t_start`` 默认取记录首样本时间：Meep 的 ``at_every(dt)`` 首个回调在 **t=dt**（不是 0），
    参考与数据必须共用同一条绝对时间轴，因此输出网格从记录起点开始（**不外推**到记录之前）。
    tap k 的绝对时间 = ``t_start + k·Ts``；相对源起始（t=0）的系统性偏移 ``t_start``
    会写进 info/provenance（约 1 个 FDTD 步，量级 ps，不补偿）。
    """
    ts = float(sample_interval_s)
    t0 = float(t_s[0]) if t_start_s is None else float(t_start_s)
    n_out = int(np.floor((float(t_s[-1]) - t0) / ts)) + 1
    if n_out < 1:
        raise ValueError("记录长度不足以产生 1 个 Ts 采样：t_s[-1]=%g, t0=%g, Ts=%g" % (t_s[-1], t0, ts))
    return t0 + np.arange(n_out, dtype=float) * ts


def prepare_reference_ds(t_s: np.ndarray, cfg: Mapping[str, Any], t_out: np.ndarray | None = None) -> np.ndarray:
    """源参考走**与数据完全相同**的 IQ + 抗混叠链，输出到精确 Ts 网格。

    分块处理时必须复用同一条参考（``records_to_aligned_cir(ref_ds=...)``），否则每块重算
    会浪费算力、且一旦参数不同会悄悄错配。
    """
    t_s = np.asarray(t_s, dtype=float)
    if t_s.size < 2:
        raise ValueError("时间轴至少需 2 个样本")
    fs_in = 1.0 / float(t_s[1] - t_s[0])
    checked = validate_chain_params(cfg, fs_in_hz=fs_in)
    if t_out is None:
        t_out = output_time_axis(t_s, cfg["sample_interval_s"])
    ref_bb = reference_baseband_on_grid(t_s, cfg)
    return band_limited_resample(
        ref_bb, fs_in, t_out, float(checked["aa_cutoff_hz"]), float(cfg.get("aa_pad_min_factor", 2.0)),
        axis=-1, t_in0_s=float(t_s[0]),
    )


def records_to_aligned_cir(
    records: np.ndarray,
    bg: np.ndarray | None,
    t_s: np.ndarray,
    cfg: Mapping[str, Any],
    keep_taps: int,
    background_mode: str = "empty_field",
    ema_alpha: float | None = None,
    ref_ds: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """把全速率场记录转成对齐后的基带 CIR（保留 ``keep_taps`` tap）。

    ``records``: ``[L, n_tx, n_rx, n_time]``（实数通带 Ez）。
    背景口径：
      * ``empty_field``：``total - bg``（无目标空场差分；**与硬件 EMA 不等同**）；
      * ``ema_windowed``：状态域一阶 EMA（冻结 ``alpha = 1/bg_m_factor``，首个输出为原值），
        见 ``apply_state_ema``。
    """
    records = np.asarray(records, dtype=float)
    if records.ndim != 4:
        raise ValueError("records 维度应为 [L, n_tx, n_rx, n_time]，得到 %s" % (records.shape,))
    if background_mode == "empty_field":
        if bg is None:
            raise ValueError("empty_field 背景模式需要 bg")
        work = records - np.asarray(bg, dtype=float)[None, ...]
    elif background_mode == "ema_windowed":
        work = records  # EMA 在 CIR 域施加（线性算子可交换，见 apply_state_ema）
    else:
        raise ValueError("未知 background_mode: %s" % background_mode)

    bb = hilbert_iq(work, t_s, cfg["fc_hz"], axis=-1)
    t_out = output_time_axis(t_s, cfg["sample_interval_s"])
    fs_in = 1.0 / float(t_s[1] - t_s[0])
    checked = validate_chain_params(cfg, fs_in_hz=fs_in)
    cutoff = float(checked["aa_cutoff_hz"])
    pad_factor = float(cfg.get("aa_pad_min_factor", 2.0))
    bb_ds = band_limited_resample(bb, fs_in, t_out, cutoff, pad_factor, axis=-1, t_in0_s=float(t_s[0]))

    ref_bb = reference_baseband_on_grid(t_s, cfg)
    if ref_ds is None:
        ref_ds = band_limited_resample(ref_bb, fs_in, t_out, cutoff, pad_factor, axis=-1, t_in0_s=float(t_s[0]))
    else:
        ref_ds = np.asarray(ref_ds)
        if ref_ds.shape[-1] != t_out.shape[0]:
            raise ValueError("传入 ref_ds 长度 %d 与输出轴 %d 不一致" % (ref_ds.shape[-1], t_out.shape[0]))
    aligned, lag = matched_filter_align(bb_ds, ref_ds)
    keep = int(keep_taps)
    if keep < 1:
        raise ValueError("keep_taps 必须 >= 1，得到 %d" % keep)
    if aligned.shape[-1] < keep:
        raise ValueError(
            "对齐后只有 %d 个 tap，少于请求的 keep_taps=%d：请增大 run_time_meep 或减小 raw_taps，"
            "不静默截短" % (aligned.shape[-1], keep)
        )
    info: Dict[str, Any] = {
        "background_mode": background_mode,
        "aa_cutoff_hz": cutoff,
        "aa_pad_min_factor": pad_factor,
        "fs_in_hz": fs_in,
        "n_full_rate": int(records.shape[-1]),
        "n_out": int(t_out.shape[0]),
        "rf_absolute_start_s": float(t_s[0]),
        "cir_lag_time0": "tap k 相对源起始的时延 = k*Ts（zero lag 在 tap 0）：数据与参考共用同一 RF 起点，公共偏移在相关中抵消",
        "output_grid_start_absolute_s": float(t_out[0]),
        "mf_lag_samples": int(lag),
        "mf_ref_len_samples": int(ref_ds.shape[-1]),
        "keep_taps": keep,
        "ema_alpha": (None if ema_alpha is None else float(ema_alpha)),
        "reference": ref_ds,
        "reference_closed_form_max_abs_err": float(
            np.max(np.abs(pulse_mod.bh_derivative_analytic_baseband(t_s, cfg["fc_hz"],
                     pulse_mod.duration_from_bandwidth(cfg["bandwidth_hz"])) - ref_bb))
        ),
    }
    return aligned[..., :keep], info


def apply_state_ema(cir: np.ndarray, alpha: float, first_raw: bool = True, burnin: int = 0) -> np.ndarray:
    """状态域一阶 EMA（与硬件 ``b0.cfr_replay`` 的逐帧递归同式）。

    硬件实测语义（只读核对，2026-09-26）：
      * 每通道独立；首帧 ``out = cur`` 且 ``ref = cur``；
      * 其后 ``out = cur - ref``，再 ``ref = (1-alpha)·ref + alpha·cur``；
      * ``alpha = 1/bg_m_factor``（默认 ``bg_m_factor=4.0`` → ``alpha=0.25``）；
      * 跨窗连续、不重置；窗取最后 ``ring`` 条 ``out``。

    线性算子（Hilbert-IQ / 重采样 / 匹配滤波）与 EMA 可交换，因此在 **CIR 域**做 EMA
    与在**全速率域**做 EMA 数值等价（量化差异除外，FDTD 侧无 int16 量化）。

    ``burnin`` = 丢弃前多少个状态的 ``out``（这些状态必须**真实求解**，不得复制）。
    """
    cir = np.asarray(cir)
    if cir.ndim < 1 or cir.shape[0] < 1:
        raise ValueError("cir 至少需 1 个状态")
    n = cir.shape[0]
    out = np.empty_like(cir)
    ref = np.array(cir[0], copy=True)
    out[0] = cir[0] if first_raw else (cir[0] - ref)
    for l in range(1, n):
        out[l] = cir[l] - ref
        ref = (1.0 - float(alpha)) * ref + float(alpha) * cir[l]
    burnin = int(burnin)
    if burnin < 0 or burnin >= n:
        raise ValueError("burnin=%d 必须落在 [0, %d)" % (burnin, n))
    return out[burnin:]


def mean_spectrum(cir: np.ndarray, sample_interval_s: float) -> Dict[str, np.ndarray]:
    """测得的平均频谱（用于 provenance：记录**实测**频响，不声称严格矩形）。"""
    cir = np.asarray(cir)
    n_taps = cir.shape[-1]
    spec = np.fft.fft(cir, n=n_taps, axis=-1)
    mag = np.abs(spec).mean(axis=tuple(range(spec.ndim - 1)))
    freq = np.fft.fftfreq(n_taps, d=float(sample_interval_s))
    order = np.argsort(freq)
    return {"freq_hz": freq[order], "mean_magnitude": mag[order]}
