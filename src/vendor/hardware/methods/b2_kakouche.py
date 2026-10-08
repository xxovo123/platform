# -*- coding: utf-8 -*-
"""B2 硬件侧适配实现：Kakouche, Maali, El Korso, Mesloub & Azzaz 2021
（Electronics 10(22):2805）式(13)—(22) 及其后正文的分级估计流程。

原文结构（**保身份**，不做结构替换；公式号按原文准确标注）
------------------------------------------------------------------------------
* 式(12)(13) 逐频点阵列模型 X(t,w) = A(theta,w)S(t,w) + N(t,w)，
      a(theta_l0,w) = [1, exp(j beta(w) sin(theta_l0)), ..., exp(j beta(w)(M-1) sin(theta_l0))]^T，
      beta(w) = -w d/c（原文印刷符号，见下 A2）。
* 式(14)  R_hat(w_k) = (1/N) sum_t X(t,w_k) X^H(t,w_k)（**逐频点各自一个 C x C 协方差**）。
* 式(15)  R_hat_n(w_k) = Q_n(w_k) Q_n^H(w_k)（噪声子空间，维 C-K）。
* 式(16)  P_IMUSIC(theta) = sum_{k=1}^{K} 1/(a^H(theta,w_k) R_hat_n(w_k) a(theta,w_k))
      —— **直接求和**：不逐频自归一、不取平均、不取乘积。
* 式(17)(18)(19) LCMV：min w_k^H R_hat(w_k) w_k s.t. w_k^H C = f，
      C = [a(theta_hat_1,w), ..., a(theta_hat_K,w)]（含**全部** K 个估计角），f = e_l（第 l 个人），
      w_k = R_hat^{-1}(w_k) C (C^H R_hat^{-1}(w_k) C)^{-1} f。
* 式(20)(21) 波束输出 S_hat_l(t,w_k) = w_k^H X(t,w_k) = e^{-j2 w_k d_l0/c} s_l(t) + e_l(t,w_k)。
* 式(22)  跨频模型 S_hat_l(t) = a(d_l0) s_l(t) + e_l(t)，
      a(d_l0) = [e^{-j2 w_1 d_l0/c}, ..., e^{-j2 w_K d_l0/c}]^T（K = 频点数；本数据的频点数记作 F）。
* **式(22) 之后的正文（无独立公式号）**：对式(22) 的跨频模型用 MUSIC，且**源数固定为 1**
      （"we use the MUSIC method in which the number of sources is fixed to one"），对每个 l 重复。
      本模块的测距（距离估计）步骤即此正文所述步骤，**不是**式(23)。
* 式(23)  ẑ_l(t) = a^H(d_hat_l0) S_hat_l(t)：**波束聚焦提取呼吸波形**（不是测距谱）。
* 式(24)(25) 呼吸波形 FFT 求呼吸率。
      本硬件基线的评测量是 (x,y) **位置**，不需要呼吸波形，故**不实现式(23)—(25)** 的呼吸环节；
      本节只取式(22) 之后正文所述"源数固定为 1 的距离 MUSIC"峰值 d_hat_l0。这一点在
      methods/docs/B1_B2_适配记录.md 与本模块 diagnostics 中同时写明。

本数据下的适配（逐条；公式对应与偏离见 methods/docs/B1_B2_适配记录.md，测试见 methods/tests/）
------------------------------------------------------------------------------
* A1 观测：原文是单站 P440 + 射频开关切换接收天线的 SIMO 雷达（式(5)：tau_lm = 2d_l0/c + (m-1)d sin(theta)/c）。
     本数据是 Tx -> 目标 -> Rx 的被动双程 CIR，且阵元为**绝对虚拟坐标** v_ch = tx + rx。
     **只实现原生平面波适配**：角导向 g_c(theta,f) = exp(+j2pi f v_x,ch sin(theta)/v_prop)，
     距离导向 a(d) = [exp(-j2pi f_k * 2d / v_prop)]_k（tau = 2r/v_prop）；r_hat = v_prop * tau_hat / 2。
     不引入 h 空第二参考偏移（不做"先减参考通道 x_ref sin(theta)/v 再测距"的二次平移）；
     不实现也不冒充 two_way_exact / r_ref 的"假 exact"变体。
* A2 符号：原文式(12) 印刷为 e^{j beta(w) sin(theta)}（beta = -w d/c，即 e^{-j 2pi f d sin(theta)/c}）；
     本坐标（theta = atan2(x,-y)，视轴 -Y）由解析例给出 e^{+j2pi f (v_x,ch - v_x,ref) sin(theta)/v_prop}
     （反号）。按本坐标实现，不照抄原文符号。绝对 v_x 与"相对参考通道"两种写法只差一个模为 1 的
     公共标量，而式(16) 的 a^H R_n a 与该标量无关，故二者谱完全相同（测试覆盖）。
* A3 ULA 结构：由 case 的 pairs/tx/rx 计算 v_ch，要求 v_x 等间距且 v_y/v_z 全同，才认为原文
     a(theta,w) 的等间距阵结构成立；否则 not_applicable（写明物理原因）。
* A4 S（发射脉冲谱）：原文式(12) 把 P(w) 放在源项 S_l(t,w) = alpha_l J1(beta_lr(w))P(w)... 内，
     式(22) 的距离导向 a(d) 是**纯相位**。本模块默认 S = 1（**共同纯相位假设**，不是已知真实发射谱），
     此时与原文式(22) 完全一致；若调用方显式给出非平坦 S（params['s_response'] 或
     case.model.response_assumption.vector），则按"源项含 P(w)"的原生分解把 S 并入距离导向
     （a(d) 正比于 S o exp(-j2pi f 2d/v)），并在 diagnostics 标为**声明式适配（非原文纯相位形式）**。
     角阶段本身与 S 无关：式(14) 是逐频点协方差，任何"逐频点复标量"都被源项吸收。
* A5 频点顺序：X 为 (L, F*C) **频率优先**，case 的原频率序**非单调**；本模块**不排序**，
     一律按 frequencies_hz 原始序逐点计算（不需要 bare J，不做额外置换）。
* A6 协方差/快照：式(14) 的 N = 本数据慢时间帧数 L；原文无前后向平均、无对角加载，本模块默认同原文
     （case.config.is_forward_backward 属主方法选择，不继承，只如实记录并标 not_used）。
* A7 LCMV 的 R^{-1}：原文假定可逆。本模块**不做隐式正则化**：逐 (波束, 频点) 记录条件数，
     cond > lcmv_cond_tol（冻结 1e12）或求解失败 -> 该波束**如实记为失败**并计数；
     全部波束失败 -> status='failed'。可选 params['lcmv_loading'] = lambda * trace(R)/C 为**额外**固定加载，
     启用时在 diagnostics 显式标注为"额外，非原法"。不用 pinv 冒充逆，也不按结果调参。

不适用（not_applicable + diagnostics['reason']）
------------------------------------------------------------------------------
* k = 0；C < 2；F < 2（式(22) 后正文的源数 1 距离 MUSIC 需 F-1 >= 1 维噪声子空间）；
* k >= C（式(15) 逐频点噪声子空间为空、式(19) 约束矩阵不可用）；
* 平面波适配下 v_ch 不构成沿 x 的等间距直线阵；
* params['manifold'] 不是 plane_wave。

失败（failed）：缺键/形状不符/含 NaN 或 Inf/S 长度不符等输入错误；或**全部**波束 LCMV 失败。

本模块只依赖 numpy；不读真值、不写任何历史产物；不修改 common/ 与其它方法。
* A8 时延/系统延迟口径：case.model 的延迟是**绝对到达时延** tau = (|p-tx|+|p-rx|)/v_prop + system_delay_si；
     search.one_dimensional_budget.round_trip_tau_span_s 与 config.distance_ring_m 只给**传播时间** 2r/v_prop，
     故距离网格仍覆盖距离环本身，而 system_delay 进入距离导向的相位（tau_abs = 2d/v_prop + system_delay），
     输出 r_hat = d_hat 直接是原点双程等价距离（不做第二次参考偏移、也不整体平移搜索域）。
* A9 共享后处理：候选去重/cap 直接调用 common/q6_contract.py::postprocess_native_candidates
     （0.10 m、分值降序、确定性、不用真值），本模块**不再**自实现一套规则；公共合同不可用即显式失败。
* A10 申报：所有实际采用的参数与阈值写入 diagnostics['params_used']；复数（如非平坦 S）
     在 diagnostics 内编码为实/虚数组，保证 JSON 可序列化。本模块**没有** e1 归一约束，
     故不要求 S 的某个参考分量非零；但 S 全为零会使距离导向退化，显式拒绝。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

C0 = 299792458.0
TINY = 1e-300
SHARED_NMS_MIN_DIST_M = 0.10

#: 冻结默认参数（写定，不按结果调整；实际取值一律在 diagnostics['params_used'] 申报）
FROZEN_PARAMS: dict[str, Any] = {
    "manifold": "plane_wave",
    "angle_step_deg": 0.05,
    "angle_grid_points_cap": 200001,
    "range_grid_points_cap": 400000,
    "nms_min_dist_m": SHARED_NMS_MIN_DIST_M,
    "lcmv_cond_tol": 1e12,
    "lcmv_loading": None,
    "range_source_count": 1,
    "s_response": None,
    "s_response_policy": "params_then_case_model_vector_else_unit",
    "range_steering_includes_s": True,
    "peak_edge_policy": "include_edges",
    "domain_tolerance": "half_cell",
    "default_tau_step_distance_m": 0.05,
    #: 显式调用方覆盖（search profile 用）：**距离步（米）**；None = 走 case 一维预算 -> 冻结默认。
    #: 显式覆盖优先于 case 的 `suggested_one_d_tau_step_s`（后者是**测角 TDOA**预算，见
    #: analysis_grid5cm/01_对齐方案：把它用在测距距离网格上属口径混用）。
    "tau_step_distance_m": None,
}

__all__ = [
    "C0",
    "FROZEN_PARAMS",
    "MethodError",
    "NotApplicable",
    "ParsedCase",
    "parse_case",
    "angle_grid_deg",
    "range_grid_m",
    "per_frequency_covariances",
    "noise_subspaces",
    "angle_steering",
    "imusic_spectrum",
    "lcmv_weights",
    "beam_output",
    "range_music_spectrum",
    "estimate",
]


class MethodError(ValueError):
    """输入或内部状态不合规：显式失败（status='failed'），不静默退化。"""


class NotApplicable(MethodError):
    """原文结构在本数据/配置下不成立：显式报告（status='not_applicable'）。"""


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _finite_array(value, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise MethodError("%s 必须是 %d 维数组，收到 shape=%s" % (name, ndim, arr.shape))
    if arr.size == 0:
        raise MethodError("%s 为空" % name)
    out = arr.astype(np.complex128 if arr.dtype.kind == "c" else np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise MethodError("%s 含 NaN/Inf：真实输入不应出现，显式失败而非静默处理" % name)
    return out


def _require(case: dict, key: str):
    if key not in case or case[key] is None:
        raise MethodError("case 缺少必需键 %r" % key)
    return case[key]


def _py(obj):
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "c":
            return {"kind": "complex_array", "n": int(obj.size),
                    "real": [float(v) for v in np.real(obj).ravel()],
                    "imag": [float(v) for v in np.imag(obj).ravel()]}
        if obj.size > 512:
            return {"kind": "array_summary", "n": int(obj.size),
                    "shape": [int(v) for v in obj.shape],
                    "min": float(np.min(obj)), "max": float(np.max(obj)),
                    "first8": [float(v) for v in obj.ravel()[:8]]}
        return _py(obj.tolist())
    if isinstance(obj, (complex, np.complexfloating)):
        return {"kind": "complex", "real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _peaks_incl_edges(values: np.ndarray) -> list:
    """**含端点**的确定性局部极大（平台取中点）：边界峰是合法候选，不得被丢弃。"""
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    if n == 0:
        return []
    peaks: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[j + 1] == v[i]:
            j += 1
        left_ok = (i == 0) or (v[i] > v[i - 1])
        right_ok = (j == n - 1) or (v[j] > v[j + 1])
        if left_ok and right_ok:
            peaks.append((i + j) // 2)
        i = j + 1
    return peaks


def _best_peak(values: np.ndarray) -> tuple[int, str]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise MethodError("一维搜索网格为空")
    v = np.where(np.isfinite(v), v, -np.inf)
    peaks = _peaks_incl_edges(v)
    if not peaks:
        return int(np.argmax(v)), "flat_no_strict_peak_fallback_argmax"
    if len(peaks) == 1 and float(np.max(v)) == float(np.min(v)):
        return int(np.argmax(v)), "flat_no_strict_peak_fallback_argmax"
    idx = sorted(peaks, key=lambda i: (-v[i], i))[0]
    return int(idx), "strict_local_max_including_edges"


def _top_peaks(values: np.ndarray, n_keep: int) -> tuple[list, str]:
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, -np.inf)
    peaks = _peaks_incl_edges(v)
    note = "strict_local_max_including_edges"
    if not peaks:
        peaks = [int(np.argmax(v))]
        note = "flat_no_strict_peak_fallback_argmax"
    order = sorted(peaks, key=lambda i: (-v[i], i))
    return order[: max(int(n_keep), 0)], note


# --------------------------------------------------------------------------- #
# case 解析（只读 case；不读真值）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedCase:
    case_id: str
    X: np.ndarray
    freqs: np.ndarray
    pairs: np.ndarray
    vcoord: np.ndarray
    v_order: np.ndarray
    spacing_m: float
    v_prop: float
    sys_delay_s: float
    k: int
    L: int
    F: int
    C: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    fov_deg: float
    ring_m: tuple | None
    tau_span_s: tuple
    ring_span_m: tuple
    range_step_m: float
    tau_step_s: float
    S: np.ndarray
    S_source: str
    config_fb_flag: bool


def _resolve_params(params: dict | None) -> tuple[dict, dict]:
    used = {key: value for key, value in FROZEN_PARAMS.items()}
    src = {key: "frozen_default" for key in FROZEN_PARAMS}
    if params:
        for key, value in params.items():
            if key not in FROZEN_PARAMS:
                raise MethodError("未知参数 %r（本模块参数冻结，见 FROZEN_PARAMS）" % key)
            used[key] = value
            src[key] = "caller_params"
    for key in ("angle_step_deg", "nms_min_dist_m", "lcmv_cond_tol"):
        if not (float(used[key]) > 0.0):
            raise MethodError("参数 %s 必须为正，收到 %r" % (key, used[key]))
    if int(used["range_source_count"]) != 1:
        raise MethodError("式(22) 之后正文规定测距 MUSIC 源数固定为 1；"
                          "range_source_count=%r 与原文不符" % used["range_source_count"])
    if used["lcmv_loading"] is not None and not (float(used["lcmv_loading"]) >= 0.0):
        raise MethodError("lcmv_loading 必须 >= 0 或 None")
    for key in ("angle_grid_points_cap", "range_grid_points_cap"):
        used[key] = int(used[key])
    for key in ("angle_step_deg", "nms_min_dist_m", "lcmv_cond_tol", "default_tau_step_distance_m"):
        used[key] = float(used[key])
    if used["lcmv_loading"] is not None:
        used["lcmv_loading"] = float(used["lcmv_loading"])
    used["range_source_count"] = int(used["range_source_count"])
    used["range_steering_includes_s"] = bool(used["range_steering_includes_s"])
    used["manifold"] = str(used["manifold"])
    if used["s_response"] is not None:
        used["s_response"] = np.asarray(used["s_response"], dtype=np.complex128).reshape(-1)
    return used, src


def _resolve_s_response(case: dict, params: dict, F: int) -> tuple[np.ndarray, str]:
    if params.get("s_response") is not None:
        S = np.asarray(params["s_response"], dtype=np.complex128).reshape(-1)
        if S.size != F:
            raise MethodError("params['s_response'] 长度 %d != F=%d" % (S.size, F))
        if not np.all(np.isfinite(S)):
            raise MethodError("params['s_response'] 含 NaN/Inf")
        return S, "caller_params.s_response"
    model = case.get("model") or {}
    resp = (model.get("response_assumption") or {}) if isinstance(model, dict) else {}
    vec = resp.get("vector") if isinstance(resp, dict) else None
    if vec is not None:
        S = np.asarray(vec, dtype=np.complex128).reshape(-1)
        if S.size != F:
            raise MethodError("case.model.response_assumption.vector 长度 %d != F=%d" % (S.size, F))
        if not np.all(np.isfinite(S)):
            raise MethodError("case.model.response_assumption.vector 含 NaN/Inf")
        return S, "case.model.response_assumption.vector"
    return np.ones(F, dtype=np.complex128), "assumption_S_equals_1(unknown_transmit_spectrum)"


def _ula_layout(vcoord: np.ndarray) -> tuple[np.ndarray, float]:
    order = np.argsort(vcoord[:, 0], kind="stable")
    v = vcoord[order]
    if v.shape[0] < 2:
        raise NotApplicable("SIMO 阵列模型需至少 2 个阵列通道（虚通道），收到 %d" % v.shape[0])
    for axis, name in ((1, "y"), (2, "z")):
        span = float(np.ptp(v[:, axis]))
        if span > 1e-9:
            raise NotApplicable(
                "原生平面波适配要求虚通道 %s 坐标一致（原文等间距线阵结构）：实测极差 %.3e m"
                % (name, span))
    dx = np.diff(v[:, 0])
    if not np.all(dx > 0.0):
        raise NotApplicable("虚通道 x 坐标存在重复（阵元间距为 0）：原文线阵结构不成立")
    spread = float(np.max(dx) - np.min(dx)) / float(np.max(dx))
    if spread > 1e-6:
        raise NotApplicable(
            "原生平面波适配要求虚通道沿 x **等间距**：相对间距散布 %.3e 超出 1e-6，结构不成立" % spread)
    return order, float(dx.mean())


def parse_case(case: dict, params: dict | None = None) -> tuple[ParsedCase, dict, dict]:
    """case -> (ParsedCase, params_used, parse_diag)。只读 case，不读真值。"""
    if not isinstance(case, dict):
        raise MethodError("case 必须是 dict，收到 %r" % type(case).__name__)
    used, _src = _resolve_params(params)
    if used["manifold"] != "plane_wave":
        raise NotApplicable(
            "本模块只实现原生平面波适配（manifold='plane_wave'）；收到 manifold=%r。"
            "旧稿的 two_way_exact/r_ref“假 exact”变体已移除，不提供等价替代" % used["manifold"])

    X = _finite_array(_require(case, "X"), "X", ndim=2)
    freqs = _finite_array(_require(case, "frequencies_hz"), "frequencies_hz").reshape(-1)
    pairs = _finite_array(_require(case, "pairs"), "pairs", ndim=2).astype(np.int64)
    tx = _finite_array(_require(case, "tx_positions"), "tx_positions", ndim=2).astype(float)
    rx = _finite_array(_require(case, "rx_positions"), "rx_positions", ndim=2).astype(float)
    grid_x = _finite_array(_require(case, "grid_x"), "grid_x").reshape(-1).astype(float)
    grid_y = _finite_array(_require(case, "grid_y"), "grid_y").reshape(-1).astype(float)
    k = int(_require(case, "k"))
    if k < 0:
        raise MethodError("k 必须非负，收到 %d" % k)

    F = int(freqs.size)
    C = int(pairs.shape[0])
    if pairs.shape[1] != 2:
        raise MethodError("pairs 必须是 (C,2)，收到 shape=%s" % (pairs.shape,))
    if tx.shape[1] != 3 or rx.shape[1] != 3:
        raise MethodError("tx_positions/rx_positions 必须是 (*,3)")
    if int(X.shape[1]) != F * C:
        raise MethodError("X 列数 %d != F*C = %d*%d（频率优先 f*C+c）" % (X.shape[1], F, C))
    if F < 2:
        raise NotApplicable("式(22) 之后正文的源数 1 距离 MUSIC 需 F-1 >= 1 维噪声子空间；本 case F=%d" % F)
    if C < 2:
        raise NotApplicable("式(13)(14) 的阵列模型需要 >=2 个阵列通道；本 case C=%d" % C)
    if k == 0:
        raise NotApplicable("k=0：式(16) 的角谱与式(17) 的约束都需至少 1 个源，无目标窗不给候选")
    if k >= C:
        raise NotApplicable("k=%d >= C=%d：式(15) 逐频点噪声子空间维 C-K <= 0，方法不成立" % (k, C))
    bad_tx = bool(np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= tx.shape[0]))
    bad_rx = bool(np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= rx.shape[0]))
    if bad_tx or bad_rx:
        raise MethodError("pairs 索引超出 tx_positions/rx_positions 范围")

    cfg = case.get("config") or {}
    if not isinstance(cfg, dict):
        raise MethodError("case['config'] 必须是 dict")
    eps_r = float(cfg.get("epsilon_r_background", 1.0) or 1.0)
    v_prop = cfg.get("propagation_speed_m_s")
    if v_prop is None:
        if eps_r <= 0.0:
            raise MethodError("config.epsilon_r_background 必须为正")
        v_prop = C0 / math.sqrt(eps_r)
        v_src = "derived_from_epsilon_r_background"
    else:
        v_src = "config.propagation_speed_m_s"
    v_prop = float(v_prop)
    if not (v_prop > 0.0):
        raise MethodError("传播速度必须为正，收到 %r" % v_prop)
    sys_delay = float(cfg.get("system_delay_si", 0.0) or 0.0)

    search = case.get("search") or {}
    if not isinstance(search, dict):
        raise MethodError("case['search'] 必须是 dict")
    budget = search.get("one_dimensional_budget") or {}
    if not isinstance(budget, dict):
        raise MethodError("search.one_dimensional_budget 必须是 dict")
    ring = cfg.get("distance_ring_m")
    ring_m = None if ring is None else (float(ring[0]), float(ring[1]))
    span = budget.get("round_trip_tau_span_s")
    if span is not None:
        tau_span = (float(span[0]), float(span[1]))
        span_src = "search.one_dimensional_budget.round_trip_tau_span_s"
    elif ring_m is not None:
        tau_span = (2.0 * ring_m[0] / v_prop, 2.0 * ring_m[1] / v_prop)
        span_src = "derived_from_config.distance_ring_m"
    else:
        raise NotApplicable(
            "既无 search.one_dimensional_budget.round_trip_tau_span_s，也无 config.distance_ring_m："
            "没有可冻结（且不依赖真值）的距离搜索域，拒绝默猜全域")
    if not (tau_span[1] > tau_span[0]):
        raise MethodError("距离搜索域必须严格递增，收到 %r" % (tau_span,))
    # A8：tau_span 只是传播时间 2r/v_prop；距离网格仍覆盖距离环本身，
    # system_delay 只进入距离导向的相位（见 range_music_spectrum）。
    ring_span_m = (0.5 * v_prop * tau_span[0], 0.5 * v_prop * tau_span[1])
    # 优先级（写定，不按结果调整）：**显式调用方覆盖 > case 一维预算 > 冻结默认**
    d_step = used.get("tau_step_distance_m")
    step = budget.get("suggested_one_d_tau_step_s")
    if d_step is not None:
        tau_step = 2.0 * float(d_step) / v_prop
        step_src = "caller_params.tau_step_distance_m(%.6g m 距离步 -> 双程时延步)" % float(d_step)
    elif step is not None:
        tau_step = float(step)
        step_src = "search.one_dimensional_budget.suggested_one_d_tau_step_s"
    else:
        tau_step = 2.0 * float(used["default_tau_step_distance_m"]) / v_prop
        step_src = ("frozen_default(%.3f m 距离步 -> 双程时延步)"
                    % float(used["default_tau_step_distance_m"]))
    if not (tau_step > 0.0):
        raise MethodError("距离搜索步长必须为正，收到 %r" % tau_step)
    range_step_m = 0.5 * v_prop * tau_step

    fov = float(cfg.get("fov_degrees", 180.0) or 180.0)
    vcoord = (tx[pairs[:, 0]] + rx[pairs[:, 1]]).astype(float)
    v_order, spacing = _ula_layout(vcoord)
    S, s_src = _resolve_s_response(case, params if params else {}, F)
    # 本模块没有 e1 归一约束（角阶段是逐频点协方差），故不要求某个参考分量非零；
    # 但 S 全为零会让距离导向 a(d) 整体退化为 0，显式拒绝。
    if int(np.count_nonzero(np.abs(S) > 0.0)) == 0:
        raise MethodError("S 全为零：距离导向 a(d) 全零，式(22) 之后正文的距离 MUSIC 不可用")

    parse_diag = {
        "manifold": "plane_wave (absolute virtual coordinate v = tx + rx)",
        "n_features": F * C,
        "feature_order": case.get("feature_order"),
        "frequency_order": "raw case order (NOT sorted); index = f*C + ch",
        "virtual_x_m": [float(v) for v in vcoord[:, 0]],
        "ula_spacing_m": spacing,
        "ula_channel_order": [int(i) for i in v_order],
        "propagation_speed_m_s": v_prop,
        "propagation_speed_source": v_src,
        "epsilon_r_background": eps_r,
        "system_delay_si": sys_delay,
        "tau_span_s": [tau_span[0], tau_span[1]],
        "ring_span_m": [ring_span_m[0], ring_span_m[1]],
        "range_step_m": range_step_m,
        "absolute_tau_note": ("tau_span_s 是传播时间 2r/v_prop；距离导向用绝对到达时延 "
                              "2d/v_prop + system_delay_si，因此 r_hat = d_hat 无需再减偏移"),
        "domain_policy": ("共同域用**严格** Cartesian 边界（仅 DOMAIN_EPS 数值容差，不做半格外延），"
                          "另检 y<0、r>0 与距离环；网格边界上的合法峰由含端点取峰器保留"),
        "tau_span_source": span_src,
        "tau_step_s": tau_step,
        "tau_step_source": step_src,
        "distance_ring_m": None if ring_m is None else [ring_m[0], ring_m[1]],
        "fov_degrees": fov,
        "s_response_source": s_src,
        "s_response_is_flat": bool(np.allclose(S, 1.0)),
        "s_response_n_zero_entries": int(np.count_nonzero(np.abs(S) == 0.0)),
        "s_reference_nonzero_required": False,
        "s_response_note": ("S = 1 是**共同纯相位假设**，不是已知真实发射谱；"
                            "角阶段（式14—19）与 S 无关，非平坦 S 只按声明式适配进入距离导向"),
        "formula_numbering": "测距 MUSIC（源数固定 1）出自式(22) 之后正文；式(23) 是波束聚焦呼吸波形，本模块不用",
        "case_config_is_forward_backward": bool(cfg.get("is_forward_backward", False)),
        "forward_backward_used": False,
        "forward_backward_note": "原文式(14) 无前后向平均；case.config.is_forward_backward 属主方法选择，本模块不继承",
        "channel_phase_calibration": "X 内已含数据侧 exp(-1j*phi) 校准；本模块不重复施加",
    }
    pc = ParsedCase(
        case_id=str(case.get("id", "?")), X=X, freqs=freqs, pairs=pairs, vcoord=vcoord,
        v_order=v_order, spacing_m=spacing, v_prop=v_prop, sys_delay_s=sys_delay, k=k,
        L=int(X.shape[0]), F=F, C=C, grid_x=grid_x, grid_y=grid_y, fov_deg=fov,
        ring_m=ring_m, tau_span_s=tau_span, ring_span_m=ring_span_m,
        range_step_m=range_step_m, tau_step_s=tau_step, S=S, S_source=s_src,
        config_fb_flag=bool(cfg.get("is_forward_backward", False)))
    return pc, used, parse_diag


# --------------------------------------------------------------------------- #
# 式(14)—(22) + 式(22) 后正文
# --------------------------------------------------------------------------- #
def per_frequency_covariances(pc: ParsedCase) -> np.ndarray:
    """式(14)：逐频点 C x C 样本协方差 R(w_k) = (1/N) sum_t X(t,w_k) X^H(t,w_k)。"""
    out = np.empty((pc.F, pc.C, pc.C), dtype=np.complex128)
    for f in range(pc.F):
        Xf = pc.X[:, f * pc.C:(f + 1) * pc.C]
        out[f] = (Xf.T @ Xf.conj()) / float(pc.L)
    return out


def noise_subspaces(pc: ParsedCase, R_all: np.ndarray | None = None) -> tuple[list, dict]:
    """式(15)：逐频点 eigh -> 噪声子空间基 Q_n(w_k)（维 C-K，升序特征值侧）。"""
    if R_all is None:
        R_all = per_frequency_covariances(pc)
    bases, info = [], {"eig_min": [], "eig_max": [], "noise_dim": int(pc.C - pc.k), "shape": []}
    for f in range(pc.F):
        w, V = np.linalg.eigh(0.5 * (R_all[f] + R_all[f].conj().T))
        bases.append(V[:, : pc.C - pc.k])
        info["eig_min"].append(float(w[0]))
        info["eig_max"].append(float(w[-1]))
        info["shape"].append([int(v) for v in R_all[f].shape])
    return bases, info


def angle_steering(pc: ParsedCase, theta_rad: float) -> np.ndarray:
    """式(13) 的 a(theta,w)（原生平面波、绝对虚拟坐标）：形状 (C, F)。

    g_c(theta,f) = exp(+j2pi f v_x,ch sin(theta)/v_prop)。与原文印刷符号反号（见 A2）；
    绝对 v_x 写法与“相对参考通道”写法只差模为 1 的公共标量，式(16) 的谱值不变。
    """
    s = math.sin(theta_rad)
    arg = (2.0 * math.pi / pc.v_prop) * np.outer(pc.vcoord[:, 0] * s, pc.freqs)
    return np.exp(1j * arg)


def imusic_spectrum(pc: ParsedCase, noise_bases: list, theta_deg_grid: np.ndarray) -> np.ndarray:
    """式(16)：P_IMUSIC(theta) = sum_k 1/(a^H R_hat_n(w_k) a) —— **直接求和**，不逐频归一。"""
    theta = np.asarray(theta_deg_grid, dtype=float).reshape(-1)
    values = np.zeros(theta.size, dtype=float)
    for i, th_deg in enumerate(theta):
        g = angle_steering(pc, math.radians(float(th_deg)))     # (C,F)
        total = 0.0
        for f in range(pc.F):
            B = noise_bases[f]                                   # (C, C-K)
            # 必须是 B^H g（即 conj(B)^T g），不能写 conj(g)^T B：
            # 复基下 B^T conj(g) != B^H g，会给出错误的象限（见 tests 的两种约定对照）。
            proj = B.conj().T @ g[:, f]
            denom = float(np.sum(np.abs(proj) ** 2))
            total += 1.0 / max(denom, TINY)
        values[i] = total
    return values


def lcmv_weights(R_f: np.ndarray, A_f: np.ndarray, *, loading: float | None = None,
                 cond_tol: float = 1e12) -> tuple[np.ndarray | None, dict]:
    """式(17)(18)(19)：W = R^{-1}C (C^H R^{-1}C)^{-1}，列 w_l 满足 w_l^H C = e_l（即 W^H C = I）。"""
    R = np.asarray(R_f, dtype=np.complex128)
    A = np.asarray(A_f, dtype=np.complex128)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise MethodError("R_f 必须是方阵，收到 shape=%s" % (R.shape,))
    if A.ndim != 2 or A.shape[0] != R.shape[0]:
        raise MethodError("A_f 必须是 (C, n_constraints)，收到 shape=%s（R_f %s）" % (A.shape, R.shape))
    C_dim = int(R.shape[0])
    n_c = int(A.shape[1])
    if n_c == 0:
        raise MethodError("A_f 约束数为 0")
    info: dict = {"C_dim": C_dim, "n_constraints": n_c, "loading": None, "ok": False}
    R_use = 0.5 * (R + R.conj().T)
    if loading is not None:
        trace = float(np.real(np.trace(R_use)))
        lam = float(loading) * trace / max(C_dim, 1)
        R_use = R_use + lam * np.eye(C_dim)
        info["loading"] = {"lambda": float(loading), "added": lam,
                           "note": "**额外**固定对角加载（非原法式19）"}
    cond = float(np.linalg.cond(R_use))
    info["cond_R"] = cond
    if not math.isfinite(cond) or cond > float(cond_tol):
        info["reason"] = "R_f 奇异/病态（cond=%.3e > %.3e）：原法式(19) 假定可逆，如实记为失败" % (cond, cond_tol)
        return None, info
    try:
        RinvA = np.linalg.solve(R_use, A)                 # (C, n_c) = R^{-1} C
        M = A.conj().T @ RinvA                            # (n_c, n_c) = C^H R^{-1} C
        cond_M = float(np.linalg.cond(M))
        info["cond_constraint"] = cond_M
        if not math.isfinite(cond_M) or cond_M > float(cond_tol):
            info["reason"] = ("约束矩阵 C^H R^{-1} C 奇异/病态（cond=%.3e）：如实失败" % cond_M)
            return None, info
        W = np.linalg.solve(M.conj().T, RinvA.conj().T).conj().T    # = R^{-1}C (C^H R^{-1}C)^{-1}
    except np.linalg.LinAlgError as exc:
        info["reason"] = "线性求解失败：%s" % exc
        return None, info
    resid = float(np.max(np.abs(W.conj().T @ A - np.eye(n_c))))
    info["max_constraint_residual_WHA_minus_I"] = resid
    info["ok"] = True
    return W, info


def beam_output(Xf: np.ndarray, w: np.ndarray) -> np.ndarray:
    """式(20)：S_hat(t,w) = w^H X(t,w) -> 对 (L,C) 的 Xf 即 Xf @ conj(w)。"""
    Xf = np.asarray(Xf, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128).reshape(-1)
    if Xf.ndim != 2 or Xf.shape[1] != w.size:
        raise MethodError("beam_output 形状不符：Xf %s, w %s" % (Xf.shape, w.shape))
    return Xf @ w.conj()


def range_music_spectrum(pc: ParsedCase, noise_basis: np.ndarray,
                         r_grid_m: np.ndarray) -> np.ndarray:
    """式(22) 之后正文：源数固定为 1 的距离 MUSIC（S=1 即原文式(22) 的纯相位导向）。"""
    B = np.asarray(noise_basis, dtype=np.complex128)
    r = np.asarray(r_grid_m, dtype=float).reshape(-1)
    if B.ndim != 2 or B.shape[0] != pc.F:
        raise MethodError("noise_basis 必须是 (F, F-1)，收到 shape=%s" % (B.shape,))
    # A8：绝对到达时延 tau_abs(d) = 2d/v_prop + system_delay_si（与 case.model 口径一致）；
    # S = 1 时即原文纯相位导向，非平坦 S 时按 A4 的声明式适配并入。
    tau_abs = 2.0 * r / pc.v_prop + pc.sys_delay_s
    A = np.exp(-2j * math.pi * np.outer(tau_abs, pc.freqs)) * pc.S[None, :]
    # 必须是 B^H a（按行 a 记即 A @ conj(B)），不能写 A @ B：
    # 复基下 B^T a != B^H a，会把时延轴镜像（tests 用 a^H P_n a 全矩阵路线独立断言）。
    proj = A @ B.conj()
    denom = np.sum(np.abs(proj) ** 2, axis=1)
    return 1.0 / np.maximum(denom, TINY)


def angle_grid_deg(pc: ParsedCase, *, angle_step_deg: float | None = None) -> np.ndarray:
    """角度搜索域：候选 Cartesian 网格包络 交 FoV/2 交 90 度（冻结，不依赖真值）。

    **端点生成规则（2026-09-24 起固定为 ceil）**：`n = ceil(2·half/Δθ) + 1`，端点对称 ±half 且含端点
    ⇒ **实际步长 <= Δθ**。原 round 规则在 Δθ = 0.6° 时给出 n = 254 ⇒ 0.600504°（超请求），故改 ceil；
    对本批默认 Δθ = 0.05° 该改动给出同一 n = 3040（网格逐位不变，旧结果仍可复现）。
    """
    step = float(FROZEN_PARAMS["angle_step_deg"] if angle_step_deg is None else angle_step_deg)
    geom = math.degrees(math.atan2(float(np.max(np.abs(pc.grid_x))),
                                   max(float(np.min(np.abs(pc.grid_y))), 1e-12)))
    half = min(90.0, 0.5 * abs(pc.fov_deg) if pc.fov_deg else 90.0, geom)
    n = int(math.ceil(2.0 * half / step - 1e-12)) + 1
    cap = int(FROZEN_PARAMS["angle_grid_points_cap"])
    if n > cap:
        raise MethodError("角度网格点数 %d 超过冻结上限 %d；请显式放宽 angle_step_deg" % (n, cap))
    return np.linspace(-half, half, max(n, 3))


def range_grid_m(pc: ParsedCase, *, range_step_m: float | None = None) -> np.ndarray:
    """距离搜索域：传播时间 span -> r = v*tau_prop/2（含端点，原点双程等价距离）。"""
    step = float(pc.range_step_m if range_step_m is None else range_step_m)
    lo, hi = float(pc.ring_span_m[0]), float(pc.ring_span_m[1])
    n = int(math.ceil((hi - lo) / step))
    cap = int(FROZEN_PARAMS["range_grid_points_cap"])
    if n + 1 > cap:
        raise MethodError("距离网格点数 %d 超过冻结上限 %d；请显式放宽步长" % (n + 1, cap))
    return np.linspace(lo, hi, n + 1)


#: 共同域边界只允许**数值**容差（远小于任何分辨率）：不做半格外延，
#: 因为外延会悄悄放大共同域定义、改变与其它方法的比较口径。
DOMAIN_EPS = 1e-9


def _in_domain(pc: ParsedCase, x: float, y: float, r_m: float) -> tuple[bool, str]:
    """共同域过滤：**严格** Cartesian 边界（仅 DOMAIN_EPS 数值容差）+ 半平面 + 距离环 + 正距离。"""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r_m)):
        return False, "non_finite_position_or_range"
    if x < float(pc.grid_x[0]) - DOMAIN_EPS or x > float(pc.grid_x[-1]) + DOMAIN_EPS:
        return False, "x_outside_cartesian_domain"
    if y < float(pc.grid_y[0]) - DOMAIN_EPS or y > float(pc.grid_y[-1]) + DOMAIN_EPS:
        return False, "y_outside_cartesian_domain"
    if y > -DOMAIN_EPS:
        return False, "not_in_half_plane_y_lt_0"
    if r_m <= DOMAIN_EPS:
        return False, "non_positive_range"
    if r_m < pc.ring_span_m[0] - DOMAIN_EPS or r_m > pc.ring_span_m[1] + DOMAIN_EPS:
        return False, "range_outside_ring"
    if pc.fov_deg and abs(math.degrees(math.atan2(x, -y))) > 0.5 * abs(pc.fov_deg) + DOMAIN_EPS:
        return False, "outside_fov"
    return True, ""


_SHARED_NMS_CACHE: dict = {}


def _shared_nms_impl():
    """加载公共合同里的共享 NMS（仅加载一次）。**不复制规则**：公共合同缺失即显式失败。"""
    if "fn" not in _SHARED_NMS_CACHE:
        import importlib
        import sys
        from pathlib import Path

        common_dir = Path(__file__).resolve().parents[2] / "common"
        if not (common_dir / "q6_contract.py").is_file():
            raise MethodError("找不到公共合同 %s/q6_contract.py：本模块拒绝使用私自实现的 NMS"
                              % common_dir)
        if str(common_dir) not in sys.path:
            sys.path.insert(0, str(common_dir))
        mod = importlib.import_module("q6_contract")
        _SHARED_NMS_CACHE["fn"] = mod.postprocess_native_candidates
        _SHARED_NMS_CACHE["min_dist"] = float(getattr(mod, "SHARED_NMS_MIN_DIST_M", 0.10))
    return _SHARED_NMS_CACHE["fn"], _SHARED_NMS_CACHE["min_dist"]


def _shared_nms(predictions: list, k: int, min_dist_m: float) -> dict:
    """直接调用 common/q6_contract.py::postprocess_native_candidates；规则不在本模块内重写。"""
    fn, shared_min = _shared_nms_impl()
    cands = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"])}
             for p in predictions]
    kept, stats = fn(cands, int(k), min_dist_m=float(min_dist_m))
    stats = dict(stats)
    stats["source"] = "common/q6_contract.py::postprocess_native_candidates"
    stats["shared_min_dist_m_constant"] = shared_min
    stats["constant_matches_shared"] = bool(abs(float(min_dist_m) - shared_min) <= 1e-12)
    return {"kept": kept, "stats": stats}


def _pack(predictions, status, diag, reason="") -> dict:
    diag = dict(diag)
    if reason:
        diag["reason"] = reason
    diag.setdefault("grid_kind", "angle_1d_imusic_then_lcmv_beam_then_range_music_1d")
    diag.setdefault("n_candidates_evaluated", 0)
    diag.setdefault("dictionary_size", None)
    diag.setdefault("dictionary_size_note",
                    "B2 无稀疏字典：式(16) 角谱 + LCMV 波束分离 + 式(22) 后正文的源数 1 距离 MUSIC")
    return {"predictions": list(predictions), "status": str(status), "diagnostics": _py(diag)}


def estimate(case: dict, params: dict | None = None) -> dict:
    """B2 公共接口：estimate(case[, params]) -> {predictions, status, diagnostics}。

    predictions = [{'x','y','score'}]（<= K，原生级联候选）；score = 距离 MUSIC 谱峰原值。
    本函数不读真值；失败/不适用一律显式返回并给 reason。
    """
    t_start = time.perf_counter()
    diag: dict = {
        "method": "B2",
        "source": "Kakouche, Maali, El Korso, Mesloub & Azzaz 2021, Electronics 10(22):2805",
        "doi": "10.3390/electronics10222805",
        "implemented_equations": "(13)(14)(15)(16)(17)(18)(19)(20)(21)(22) + 式(22) 后正文（距离 MUSIC 源数=1）",
        "structure": "IMUSIC 角谱（式16） -> LCMV 波束分离（式17-20） -> 跨频模型（式22）的源数 1 距离 MUSIC",
        "not_implemented": "式(23)—(25)（波束聚焦呼吸波形与呼吸率）与位置评测无关，本模块不做",
    }
    try:
        pc, used, parse_diag = parse_case(case, params)
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
    diag["params_used"] = used
    diag["parse"] = parse_diag

    try:
        R_all = per_frequency_covariances(pc)
        bases, cov_info = noise_subspaces(pc, R_all)
        diag["per_frequency_covariance"] = {
            "equation": "式(14)",
            "shape": [int(pc.F), int(pc.C), int(pc.C)],
            "noise_subspace_dim": int(pc.C - pc.k),
            "eig_min": cov_info["eig_min"],
            "eig_max": cov_info["eig_max"],
            "snapshots_N": int(pc.L),
            "forward_backward_used": False,
            "diag_load_used": 0.0,
            "per_frequency_normalization": "none（式(16) 为直接求和）",
        }
        theta_grid = angle_grid_deg(pc, angle_step_deg=float(used["angle_step_deg"]))
        im = imusic_spectrum(pc, bases, theta_grid)
        sel, peak_note = _top_peaks(im, pc.k)
        diag["angle_stage"] = {
            "equation": "式(16)（直接求和，不逐频自归一）",
            "grid_kind": "1d_angle_grid_deg",
            "n_points": int(theta_grid.size),
            "start_deg": float(theta_grid[0]),
            "stop_deg": float(theta_grid[-1]),
            "step_deg": float(used["angle_step_deg"]),
            "endpoint_rule": ("n = ceil(2*half/step)+1，端点对称 ±half 且含端点 ⇒ 实际步长 <= step"),
            "actual_step_deg": float(theta_grid[1] - theta_grid[0]),
            "peak_policy": peak_note,
            "n_angle_peaks_found": int(len(_peaks_incl_edges(im))),
            "selected_angle_deg": [float(theta_grid[i]) for i in sel],
            "selected_spectrum": [float(im[i]) for i in sel],
            "included_nonlocal_peaks": False,
        }
        theta_hat = [float(theta_grid[i]) for i in sel]

        rg = range_grid_m(pc)
        beams, beam_failures = [], []
        n_candidates = int(theta_grid.size)
        for l, th in enumerate(theta_hat):
            Sout = np.empty((pc.L, pc.F), dtype=np.complex128)
            failed_freqs, infos = [], []
            for f in range(pc.F):
                A_f = np.column_stack([angle_steering(pc, math.radians(t))[:, f]
                                       for t in theta_hat])
                W, info = lcmv_weights(R_all[f], A_f, loading=used["lcmv_loading"],
                                       cond_tol=float(used["lcmv_cond_tol"]))
                infos.append(info)
                if W is None:
                    failed_freqs.append(f)
                    continue
                Xf = pc.X[:, f * pc.C:(f + 1) * pc.C]
                Sout[:, f] = beam_output(Xf, W[:, l])
            if failed_freqs:
                beam_failures.append({"beam": int(l), "theta_hat_deg": float(th),
                                      "failed_frequencies": [int(v) for v in failed_freqs],
                                      "reason": infos[failed_freqs[0]].get("reason", "lcmv_failed"),
                                      "cond_R": [infos[f]["cond_R"] for f in failed_freqs]})
                continue
            max_resid = max(info["max_constraint_residual_WHA_minus_I"] for info in infos)
            conds = [info["cond_R"] for info in infos]
            R_range = (Sout.T @ Sout.conj()) / float(pc.L)
            w_r, V_r = np.linalg.eigh(0.5 * (R_range + R_range.conj().T))
            noise_basis = V_r[:, : pc.F - 1]
            spectrum = range_music_spectrum(pc, noise_basis, rg)
            j, r_note = _best_peak(spectrum)
            d_hat = float(rg[j])
            n_candidates += int(rg.size)
            x = d_hat * math.sin(math.radians(th))
            y = -d_hat * math.cos(math.radians(th))
            beams.append({
                "beam_index": int(l),
                "theta_hat_deg": float(th),
                "d_hat_m": d_hat,
                "range_peak_value": float(spectrum[j]),
                "range_peak_policy": r_note,
                "range_covariance_eig_min": float(w_r[0]),
                "range_covariance_eig_max": float(w_r[-1]),
                "range_source_count": 1,
                "lcmv_cond_R_max": float(np.max(conds)),
                "lcmv_max_constraint_residual": float(max_resid),
                "x": float(x), "y": float(y),
            })
        diag["range_stage"] = {
            "equation": "式(22) 后正文（MUSIC，源数固定为 1）",
            "grid_kind": "1d_range_grid_m",
            "endpoint_rule": ("linspace(lo, hi, ceil((hi-lo)/step)+1)，含端点（= 距离环端点），"
                              "实际步长 <= 请求步长"),
            "n_points_per_beam": int(rg.size),
            "span_m": [float(rg[0]), float(rg[-1])],
            "step_m": float(0.5 * pc.v_prop * pc.tau_step_s),
            "steering": "a(d) = S o exp(-j2pi f * 2d / v_prop)（绝对原点坐标；S=1 时即原文纯相位形式）",
            "range_steering_includes_s": bool(used["range_steering_includes_s"]),
            "s_response_is_flat": bool(np.allclose(pc.S, 1.0)),
            "reference_offset_second_time": False,
        }
        diag["lcmv"] = {
            "equation": "式(17)(18)(19)",
            "n_beams_requested": int(len(theta_hat)),
            "n_beams_usable": int(len(beams)),
            "n_beams_failed": int(len(beam_failures)),
            "loading": used["lcmv_loading"],
            "cond_tol": float(used["lcmv_cond_tol"]),
            "constraint_verified": "W^H A = I（残差在 beams 的 lcmv_max_constraint_residual 中逐束给出）",
        }
        diag["beam_failures"] = beam_failures
        kept, dropped = [], []
        for idx, p in enumerate(beams):
            ok, why = _in_domain(pc, p["x"], p["y"], p["d_hat_m"])
            if ok:
                kept.append(dict(p, beam_path_index=idx))
            else:
                dropped.append({"beam_index": idx, "reason": why, "x": p["x"], "y": p["y"],
                                "d_hat_m": p["d_hat_m"], "theta_hat_deg": p["theta_hat_deg"]})
        nms = _shared_nms([{"x": p["x"], "y": p["y"], "score": p["range_peak_value"]}
                           for p in kept], pc.k, float(used["nms_min_dist_m"]))
        predictions = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"])}
                       for p in nms["kept"]]
        diag["beams"] = kept
        diag["dropped_beams"] = dropped
        diag["postprocess"] = nms["stats"]
        diag["score_definition"] = "距离 MUSIC 谱在峰处的原值（未归一；排序只用于共享 NMS/cap）"
        diag["n_predictions"] = len(predictions)
        diag["n_candidates_evaluated"] = int(n_candidates)
        diag["evaluation_boundary"] = "只给 <=K 个原生候选；评测（含真值）由集成层做，本模块不读真值"
        diag["wall_seconds"] = time.perf_counter() - t_start
        if not beams:
            return _pack([], "failed", diag,
                         "全部波束的 LCMV 均失败（见 beam_failures），无可用候选")
        if not predictions:
            return _pack(predictions, "ok", diag,
                         "无候选：波束距离峰全部落在共同域外或被共享 NMS 抑制（见 dropped_beams/postprocess）")
        return _pack(predictions, "ok", diag, "")
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
