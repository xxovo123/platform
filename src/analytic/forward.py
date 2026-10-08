#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立解析正向生成（层 A）。

模型（与统一协议一致）
----------------------
* 目标真值位置 p_k = (r*sin(theta), -r*cos(theta))（视轴 -Y）。
* 通道 ch（pairs[ch] = (tx_i, rx_i)）双程绝对到达时延：
      tau[k, ch] = (||p_k - tx[tx_i]|| + ||p_k - rx[rx_i]||) / v_prop + system_delay
* 频率轴 = np.fft.fftfreq(fft_length, Ts) + fc（32 个 bin 全带）。
* 单快照 CFR（全 32 bin）：H[l, ch, m] = sum_k a_k(l) * exp(-2j*pi*f_m*tau[k,ch])
  —— 时间原点 = 物理零延迟（raw tap 0 对应 tau = 0），目标回波写在 raw tap = tau/Ts。
     当前后端 crop_start 会截掉前 2 个 tap，故数据等效延迟为 tau - crop_start*Ts，与绝对
     tau 流形存在固定偏移（双程距离约 -0.29979 m）。该偏移**如实披露、不补偿**。
* raw CIR：raw[l, ch, :] = ifft(H[l, ch, :], n=fft_length)（32 tap，复数）。

诊断 profile（compensated，非正式、默认关闭）
--------------------------------------------
raw_cir_from_coeffs(..., compensate_crop=True) 会在生成端以 tau + crop_start*Ts 造回波，
使 crop2 前端对该延迟参考透明，仅用于对照/诊断（adapter/e0_checks.py 的 zero-offset CFR
诊断）；不得作为主协议，默认 False。

带宽（物理可核）
----------------
主档 band_mode = "bandlimited_500mhz"：发射谱 S(f) = 1 当 |f - fc| <= B/2（=500 MHz），
否则 S(f) = 0。于是 raw CIR 的物理带宽显式限到 500 MHz（可核），32-bin fftfreq 上恰好保留
17 个带内 bin（|k|<=8）。crop 后 30-tap 有限窗会有带外泄漏，如实披露。
诊断档 band_mode = "full_band_ideal"（S=1 全 Nyquist，非正式、默认关闭）仅作理想对照。

系数序列（相干性基准）
----------------------
* 主档 = K 个**独立**复系数序列（rho = 0；非相干基准）：a_k(l) = amp_k * exp(j*phi_k(l))，
  phi_k(l) 独立均匀（每目标、每快照独立）。固定 seed 只按 case_id（clean 序列），
  与方法无关；噪声 trial 另按 (case_id, trial) 独立（见 sim_common）。
* 相干挑战（可选，单独标注）：a_k(l) = amp_k * exp(j*(psi(l) + delta_k))，共同 psi(l) 按
  rho 混合（rho=1 完全相干；rho=0.9 部分相干）。

本模块只用 numpy；不读真值文件、不写历史产物、不 import 任何被测算法模块。
"""

from __future__ import annotations

import math

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np


class ForwardError(RuntimeError):
    """正向生成输入不合规（显式失败）。"""


BAND_MODES = ("bandlimited_500mhz", "full_band_ideal")


def band_mask(axis, band_mode: str = "bandlimited_500mhz") -> np.ndarray:
    """发射谱掩膜 S(f)：主档只保留 |f - fc| <= B/2；诊断档全 Nyquist 为 1。"""
    axis = np.asarray(axis, dtype=float)
    if band_mode == "full_band_ideal":
        return np.ones_like(axis, dtype=float)
    if band_mode != "bandlimited_500mhz":
        raise ForwardError("未知 band_mode=%r（只支持 %s）" % (band_mode, list(BAND_MODES)))
    return (np.abs(axis - float(sc.FC_HZ)) <= 0.5 * float(sc.BANDWIDTH_HZ) + 1e-6).astype(float)


def delays_for_targets(targets_xy, *, tx_positions=None, rx_positions=None, pairs=None,
                       v_prop: float | None = None, system_delay_si: float | None = None) -> np.ndarray:
    """返回 tau[k, ch]（K x C），绝对双程到达时延（s）。"""
    tx = sc.TX_POSITIONS_M if tx_positions is None else np.asarray(tx_positions, dtype=float)
    rx = sc.RX_POSITIONS_M if rx_positions is None else np.asarray(rx_positions, dtype=float)
    pr = sc.PAIRS if pairs is None else np.asarray(pairs, dtype=np.int64)
    v = float(sc.C0) if v_prop is None else float(v_prop)
    sysd = float(sc.SYSTEM_DELAY_SI) if system_delay_si is None else float(system_delay_si)
    pts = np.asarray(targets_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ForwardError("targets_xy 必须是 (K,2)，收到 %s" % (pts.shape,))
    K = pts.shape[0]
    C = pr.shape[0]
    p3 = np.concatenate([pts, np.zeros((K, 1))], axis=1)
    tau = np.empty((K, C), dtype=float)
    for ch in range(C):
        t_i, r_i = int(pr[ch, 0]), int(pr[ch, 1])
        d_tx = np.linalg.norm(p3 - tx[t_i][None, :], axis=1)
        d_rx = np.linalg.norm(p3 - rx[r_i][None, :], axis=1)
        tau[:, ch] = (d_tx + d_rx) / v + sysd
    return tau


def coefficients_from_spec(k: int, L: int, *, rng: np.random.Generator,
                           amps=None, coherence: float = 0.0,
                           phases_deg=None) -> np.ndarray:
    """生成复系数序列 a[l, k]。

    coherence = 0 -> K 个独立复系数序列（rho=0）；
    0 < coherence < 1 -> 部分相干（共同分量按 sqrt(rho) 混合）；
    coherence = 1 -> 完全相干（共同 psi(l) + 固定相位差）。
    """
    if k < 1:
        raise ForwardError("k 必须 >= 1，收到 %d" % k)
    amp = np.ones(k, dtype=float) if amps is None else np.asarray(amps, dtype=float).reshape(-1)
    if amp.size != k:
        raise ForwardError("amps 长度 %d != k=%d" % (amp.size, k))
    if amp.min() < 0:
        raise ForwardError("amps 必须非负")
    rho = float(min(1.0, max(0.0, coherence)))
    phi_ind = rng.uniform(0.0, 2.0 * math.pi, size=(L, k))            # 独立分量
    if phi_ind.shape[0] and rng is not None and False:                  # pragma: no cover - 明确无静默
        pass
    common = rng.uniform(0.0, 2.0 * math.pi, size=(L, 1))            # 共同分量
    if phases_deg is not None:
        delta = np.radians(np.asarray(phases_deg, dtype=float).reshape(-1))
        if delta.size != k:
            raise ForwardError("phases_deg 长度 %d != k=%d" % (delta.size, k))
    else:
        delta = np.zeros(k, dtype=float)
    coeff = np.empty((L, k), dtype=np.complex128)
    for i in range(k):
        if rho >= 1.0:
            coeff[:, i] = amp[i] * np.exp(1j * (common[:, 0] + delta[i]))
        elif rho <= 0.0:
            coeff[:, i] = amp[i] * np.exp(1j * phi_ind[:, i])
        else:
            # 以 rho 的比例把共同分量与其独立实现之和重归一，使相关系数近似 rho
            mix = math.sqrt(rho) * np.exp(1j * (common[:, 0] + delta[i])) + \
                math.sqrt(1.0 - rho) * np.exp(1j * phi_ind[:, i])
            coeff[:, i] = amp[i] * mix
    return coeff


def raw_cir_from_coeffs(tau: np.ndarray, coeff: np.ndarray, *, freqs_axis=None,
                        fft_length: int | None = None, crop_start: int | None = None,
                        compensate_crop: bool = False,
                        band_mode: str = "bandlimited_500mhz") -> np.ndarray:
    """由时延 tau(K,C) 与复系数 coeff(L,K) 生成 raw CIR raw[L,C,32]。

    主协议（compensate_crop=False，默认）：H[l,ch,m] = sum_k a[l,k]*exp(-2j*pi*f_m*tau[k,ch])，
    raw = ifft(H, n=fft_length)；raw tap 0 = 物理零延迟，不预补偿 crop 偏移。
    诊断（compensate_crop=True）：改用 tau + crop_start*Ts 造回波，仅作对照，非正式主协议。
    """
    tau = np.asarray(tau, dtype=float)                 # (K, C)
    coeff = np.asarray(coeff, dtype=np.complex128)     # (L, K)
    if tau.ndim != 2:
        raise ForwardError("tau 必须是 (K,C)，收到 %s" % (tau.shape,))
    K, C = tau.shape
    if coeff.ndim != 2 or coeff.shape[1] != K:
        raise ForwardError("coeff 必须是 (L,K=%d)，收到 %s" % (K, coeff.shape))
    L = coeff.shape[0]
    n = int(sc.FFT_LENGTH if fft_length is None else fft_length)
    cs = int(sc.CROP_START if crop_start is None else crop_start)
    ts = float(sc.SAMPLE_INTERVAL_S)
    axis = sc.raw_freq_axis() if freqs_axis is None else np.asarray(freqs_axis, dtype=float)
    if axis.size != n:
        raise ForwardError("freqs_axis 长度 %d != fft_length=%d" % (axis.size, n))
    # 主协议：raw tap0 = 物理零延迟；诊断 profile 才加 crop_start*Ts 预补偿（默认 False）
    tau_gen = tau + (float(cs) * ts if bool(compensate_crop) else 0.0)
    max_tap = float(np.max(tau_gen)) / ts
    if max_tap >= float(n):
        raise ForwardError("最大延迟 %.4f tap >= fft_length=%d：会圆周回卷，请扩大 fft_length "
                           "或缩小目标距离" % (max_tap, n))
    if max_tap > float(n - 2):
        raise ForwardError("最大延迟 %.4f tap 超过 %d（后 2 tap 会被 crop 丢弃）：拒绝潜在能量丢失"
                           % (max_tap, n - 2))
    phase = np.exp(-2j * math.pi * tau_gen[:, :, None] * axis[None, None, :])   # (K,C,M)
    H = np.einsum("lk,kcm->lcm", coeff, phase)                                   # (L,C,M)
    # BANDLIMIT-FIX-20260929：原冻结源定义 band_mask 却从未调用，发射谱未按 500 MHz 带限。
    # 显式乘掩膜：32 个 fftfreq bin 中 17 个带内（17/32），其余 15 个置零；
    # full_band_ideal 保留为**诊断档**（S=1 全 Nyquist，非正式），默认不启用。
    s_mask = band_mask(axis, band_mode)
    H = H * s_mask[None, None, :]
    raw = np.fft.ifft(H, n=n, axis=-1)
    return np.ascontiguousarray(raw.astype(np.complex128))


class MethodFreeForward:
    """场景 -> clean raw CIR 的独立正向器（不依赖任何被测算法模块）。"""

    def __init__(self, targets_xy, *, k: int, L: int = sc.SNAPSHOTS, rng: np.random.Generator,
                 amps=None, coherence: float = 0.0, phases_deg=None,
                 tx_positions=None, rx_positions=None, pairs=None, v_prop=None,
                 system_delay_si=None, compensate_crop: bool = False,
                 band_mode: str = "bandlimited_500mhz"):
        self.targets_xy = np.asarray(targets_xy, dtype=float).reshape(-1, 2)
        if self.targets_xy.shape[0] != int(k):
            raise ForwardError("targets_xy 数量 %d != k=%d" % (self.targets_xy.shape[0], k))
        self.k = int(k)
        self.L = int(L)
        self.tau = delays_for_targets(self.targets_xy, tx_positions=tx_positions,
                                      rx_positions=rx_positions, pairs=pairs,
                                      v_prop=v_prop, system_delay_si=system_delay_si)
        self.coeff = coefficients_from_spec(self.k, self.L, rng=rng, amps=amps,
                                            coherence=coherence, phases_deg=phases_deg)
        self.compensate_crop = bool(compensate_crop)
        self.band_mode = str(band_mode)
        self.raw = raw_cir_from_coeffs(self.tau, self.coeff, compensate_crop=self.compensate_crop,
                                       band_mode=self.band_mode)

    def raw_cir(self) -> np.ndarray:
        return self.raw

    def meta(self) -> dict:
        return {
            "k": int(self.k), "L": int(self.L),
            "targets_xy_m": [[float(x), float(y)] for x, y in self.targets_xy],
            "tau_s": [[float(v) for v in row] for row in self.tau],
            "max_delay_tap": float(np.max(self.tau) / sc.SAMPLE_INTERVAL_S + sc.CROP_START),
            "model": "independent_analytic_two_way_raw_cir_physical_zero",
            "time_origin": "raw tap 0 = physical zero delay; crop_start offset retained & disclosed",
            "compensate_crop_diagnostic_only": bool(self.compensate_crop),
            "band_mode": self.band_mode,
            "bandlimited_primary": bool(self.band_mode == "bandlimited_500mhz"),
            "crop_offset_range_m": float(sc.crop_offset_range_m()),
        }


def invariant_check(case_id: str, targets_xy, *, k: int, L: int = sc.SNAPSHOTS,
                    coherence: float = 0.0, compensate_crop: bool = False,
                    band_mode: str = "bandlimited_500mhz") -> dict:
    """自检记录：seed 确定性 + 秩（不按好坏筛数据，只如实记录）。"""
    rng = sc.rng_from(sc.clean_seed(case_id))
    fwd = MethodFreeForward(targets_xy, k=k, L=L, rng=rng, coherence=coherence,
                            compensate_crop=compensate_crop)
    again = MethodFreeForward(targets_xy, k=k, L=L, rng=sc.rng_from(sc.clean_seed(case_id)),
                              coherence=coherence, compensate_crop=compensate_crop)
    cropped = sc.crop_raw(fwd.raw)
    X = sc.x_from_cropped(cropped)
    sv = np.linalg.svd(X, compute_uv=False)
    s_sorted = np.sort(sv)[::-1]
    denom = float(s_sorted[0]) if s_sorted.size else 0.0
    rank = int(np.count_nonzero(s_sorted > 1e-8 * max(denom, 1e-300)))
    return {
        "case_id": case_id,
        "deterministic_regeneration": bool(np.array_equal(fwd.raw, again.raw)),
        "raw_shape": list(fwd.raw.shape),
        "X_shape": list(X.shape),
        "time_origin": "physical_zero(raw tap0 = tau 0); crop offset retained",
        "crop_offset_range_m": float(sc.crop_offset_range_m()),
        "n_dropped_tap_energy_fraction": float(
            np.sum(np.abs(fwd.raw[:, :, :sc.CROP_START]) ** 2)
            / max(np.sum(np.abs(fwd.raw) ** 2), 1e-300)),
        "singular_values_top8": [float(v) for v in s_sorted[:8]],
        "numerical_rank_rel_tol_1e-8": int(rank),
        "k": int(k), "note": "rank/奇异值只记录，不按好坏筛数据；known-K 不修复秩。",
    }
