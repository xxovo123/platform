#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""源脉冲与匹配滤波参考的**唯一**公式实现（源与 MF 参考必须同公式）。

与旧 ``robust_version`` 的关系（2026-09-26 已核对）
-------------------------------------------------
旧源脉冲 ``simulation_module_bpvs2dmusic.py`` L94-102 ``blackmann_harris_derivative_pulse``
与旧 ``signal_processing.py`` ``generate_reference_pulse_baseband`` 第 1 步**逐行相同**，
两者都用 ``duration = 2/B``。本模块把该公式抽成唯一实现：

* Meep 源：``bh_derivative_passband(t_meep, fc_meep, duration_meep)``；
* MF 参考：先在同一时间轴上重建**同一条通带波形**，再走与数据完全相同的
  ``rf_chain``（Hilbert-IQ → 抗混叠重采样 → 匹配滤波）。

因此“源与 MF 参考同公式”是结构保证，而不是靠人工保持两处同步。

另外提供闭式解析基带（``bh_derivative_analytic_baseband``）作**交叉校验**：
对 ``p(t)=Re{w(t)e^{jθ(t)}}``（θ = 2π fc (t - T/2)），其解析信号为
``(w' + j2πfc·w)·e^{-jπ fc T}``（w 外为 0）。用它比对 Hilbert 路径，量化
Hilbert 的近似误差，而不是假定两者一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Blackman-Harris 四项系数（与旧实现逐字相同）。
BH_COEF = (0.35875, 0.48829, 0.14128, 0.01168)

#: 与 protocol.PULSE_FORMULA_ID 对应的公式标识。
FORMULA_ID = "bh_derivative_v1"


def _envelope_and_derivative(t_s: np.ndarray, duration_s: float):
    """BH 窗与其时间导数（内部函数；t 与旧实现一致地取 t_rel = t）。"""
    a0, a1, a2, a3 = BH_COEF
    arg = 2.0 * np.pi * t_s / duration_s
    w = a0 - a1 * np.cos(arg) + a2 * np.cos(2.0 * arg) - a3 * np.cos(3.0 * arg)
    dw = (2.0 * np.pi / duration_s) * (a1 * np.sin(arg) - 2.0 * a2 * np.sin(2.0 * arg) + 3.0 * a3 * np.sin(3.0 * arg))
    return w, dw


def bh_derivative_passband(t: Any, fc_hz: float, duration_s: float) -> Any:
    """通带源脉冲 ``d/dt[ w(t)·cos(2π fc (t - T/2)) ]``，窗内有效、窗外 0。

    标量输入 → 标量输出（Meep ``CustomSource`` 的 ``src_func`` 需要标量）。
    """
    t_arr = np.asarray(t, dtype=float)
    duration_s = float(duration_s)
    t0 = duration_s / 2.0
    inside = (t_arr >= 0.0) & (t_arr <= duration_s)
    w, dw = _envelope_and_derivative(t_arr, duration_s)
    theta = 2.0 * np.pi * float(fc_hz) * (t_arr - t0)
    p = dw * np.cos(theta) + w * (-2.0 * np.pi * float(fc_hz) * np.sin(theta))
    p = np.where(inside, p, 0.0)
    if np.ndim(t) == 0:
        return float(p)
    return p


def bh_derivative_analytic_baseband(t: Any, fc_hz: float, duration_s: float) -> Any:
    """同一通带脉冲的闭式解析基带（交叉校验用，不用于数据路径）。"""
    t_arr = np.asarray(t, dtype=float)
    duration_s = float(duration_s)
    inside = (t_arr >= 0.0) & (t_arr <= duration_s)
    w, dw = _envelope_and_derivative(t_arr, duration_s)
    b = (dw + 1j * 2.0 * np.pi * float(fc_hz) * w) * np.exp(-1j * np.pi * float(fc_hz) * duration_s)
    b = np.where(inside, b, 0j)
    if np.ndim(t) == 0:
        return complex(b)
    return b


def duration_from_bandwidth(bandwidth_hz: float) -> float:
    """冻结口径：``duration = 2/B``（与旧实现一致）。"""
    return 2.0 / float(bandwidth_hz)
