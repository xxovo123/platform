# -*- coding: utf-8 -*-
"""C3 硬件侧适配实现：Kim, Oh & Lee 2015（IEEE AWPL 14:1710-1713, DOI 10.1109/LAWP.2015.2420579）
——“DFT-ESPRIT 联合 TOA/DOA” 的 **CIR/CFR 适配版**（原文为 FMCW 去斜拍频观测）。

原文结构（**保身份**：顺序与算子逐条保留；PDF p2 逐式核对，渲染件见 logs/C3_pdf_render/）
--------------------------------------------------------------------------------
* 式(5)  去斜后每阵元 p_k[n] = sum_m a_m e^{j(mu tau_m T_s n + omega_s tau_m - mu/2 tau_m^2 + pi k sin(θ_m))} + w_k[n]。
* 式(6)  **只在第一个阵元**上做 1D-DFT：P_1[l] = sum_n p_1[n] e^{-j2pi n l/N}（l 为离散时延索引）。
* 式(7)  由峰值索引得 TOA：tau_hat_m = I_m / (N * T_s * mu)（**去斜拍频专有**；本适配**替换**）。
* 式(8)  在**每个 TOA 索引**上跨阵元取同一 l 的 DFT 值组成向量
         P_m = [P_1[I_m], P_2[I_m], ..., P_L[I_m]]^T，自相关矩阵 **R_m = P_m P_m^H**（单向量外积）。
* 式(9)  R_m 的 EVD：U_1 为信号子空间（列数写作 M）、U_2 为噪声子空间（L-M 列）。
* 式(10) U_{1,0} = J_0 U_1（删最后一行）、U_{1,1} = J_1 U_1（删第一行）。
* 式(11) U_{1,0} = J_0 A T、U_{1,1} = J_0 A Phi T（A 见式(12)；Phi = diag[e^{j pi sin(θ_m)}]）。
* 式(12) A 第 k 行为 e^{j pi k sin(θ_m)}（阵元相位与频率无关：窄带阵列模型，间距 = lambda_c/2）。
* 式(13) 广义 EVD：(U_{1,0}^dagger U_{1,1} - xi_m I) x_m = 0。
* 式(14) theta_hat_m = asin( (1/pi) * angle(xi_m) )。
* Step 1-4：第一阵元 DFT -> 峰值索引 -> 跨阵元自相关 -> 广义 EVD 得 DOA（顺序逐字保留）。

**原文 notation 矛盾（按 PDF p2 逐式核对后的结论；如实报告，不擅自增强）**
--------------------------------------------------------------------------------
* 第 II 节正文写 “ULA consisting of **L** elements” => 阵元数 = L；式(8) 的 P_m 长度即阵元数。
* 但式(9) 写 U_1 = [u_0 ... u_{M-1}]（**M** 个信号特征向量）、U_2 为 L-M 维；式(10)(12) 又用 **K**
  表示阵元数（J_0 = [I_{K-1} 0]、A 第 K 行为 e^{j pi (K-1) sin θ}）。**L / K / M 三者混用**。
* 更关键：式(8) 是**单向量外积**，rank(R_m) = 1，恒只有 1 个非零特征值 => “M 个信号特征向量”
  **只可能对应 M = 1（每个 TOA 索引上 1 个源）**；原文正文却把 M 当作目标总数，两者**不能同时成立**
  —— 同一 TOA（距离）索引上多目标时，式(8) 的秩结构**无法**分辨。
* 本实现的处置（与主审 2026-09-24 决策一致）：**每个距离峰只给 1 个角度**，空间维只取**首个**
  （最大）特征向量；**不做**每 bin 多源 / 子阵平滑 / 前后向 / 对角加载等任何增强；**同距离多目标保留
  原限制**。**不声称逐式原样复现**：这是**明确声明的 CIR 适配版**。

本机适配（逐条与理由见 methods/docs/C3_Kim_适配与公式.md）
--------------------------------------------------------------------------------
* A1 CFR -> 时延谱（替换式(7)）：case 的 X(L, F*C) **频率优先**，且 17 个带内频点**原始数组序非单调**
  （[fc, fc+df, ..., fc+8df, fc-8df, ..., fc-df]）。式(6) 的 DFT 要求等间隔有序样本，故先按频率
  **排序**（并对 X 列做**一致置换**，纯索引重标号），再在排序后等间隔频点上做逆 DFT：
      P_c[l] = sum_j X[f_j, c] * e^{+j 2pi (f_j - f_c) tau_l},   tau_l = l / (N_zp * df)
  **原生 range bin** = 1/(F*df)（本批 F=17、df=31.25 MHz => 1.882 ns，双程 0.282 m）；
  无模糊窗 = 1/df = 32 ns。默认 zeropad_factor = 1 = **原生网格**；zeropad > 1 只是把同一条
  Dirichlet 核插值到更密网格（**不新增信息、不提高分辨率**），取值一律在 diagnostics 申报。
  **不使用** FMCW 斜率式 tau = I/(N T_s mu)（原文属去斜拍频专有，本机无 mu）。
* A2 首阵元 -> 原点补偿（**只补一次**）：式(6) 的峰取自**第一个阵元**，其到达时延
      tau_ref = (2r - v_x,ref * sin θ) / v_prop + system_delay
  故 P_c[l_m] 的峰位是**该阵元**的时延。由 theta_hat 反解原点距离必须
      r = v_prop * (tau_ref - system_delay) / 2 + v_x,ref * sin(theta_hat) / 2      （**只加一次**）
  漏加/重复加都会产生 |v_x,ref| * sinθ / 2 的系统偏差（本批 v_x,ref = -0.038 m，最大 19 mm）。
  first_element 与 phase_center 两种参考阵元必须给出**同一个原点距离**（解析自测覆盖）。
* A3 fc 窄带空间相位与符号：本机坐标 theta = atan2(x, -y)（视轴 -Y），绝对虚拟坐标 v = tx + rx；
  解析例给出相邻阵元比值 xi 的相角 = **+** 2pi f_c dv_x sin θ / v_prop（与原文 pi sin θ **同号**，
  dv_x = lambda_c/2 时二者一致）：sin(theta_hat) = angle(xi) * v_prop / (2pi f_c dv_x)。
  **不照抄**可能相反的印刷符号。**相位绕回唯一性**：比值相位本身定义在 (-pi, pi]，
  无绕回条件为 |dv_x sin θ| < lambda_c/2 <=> |θ| < theta_wrap = asin(lambda_c/(2 dv_x))。
  本批 dv_x = 0.019 m 略大于 lambda_c/2 = 0.018767 m => theta_wrap = 81.02 度；
  **共同域**（bbox ∩ ring ∩ FoV ∩ y<0）的几何半角约 76 度 < 81.02 度 => **域内相位仍唯一**，
  故**不因 d > lambda_c/2 判 NA**，而是**按限定共同域声明辨识**并把该判据写进 diagnostics。
  若域内半角超过 theta_wrap，则本模块申报 principal-branch 限制（仍按式(14) 取主值解，不静默）。
* A4 S（发射谱）：原文式(5) 是**去斜后**的复指数模型（发射波形已移除）。故声明**非平坦** S
  （params['s_response'] 或 case.model.response_assumption.vector）时需**反卷积**（X 列除以 S_f）
  再做时延谱；S 含零分量 -> 显式 failed。**DOA 比值与 S 无关**（S 为逐频点公共因子，在跨阵元比值中
  约掉）——解析自测覆盖。S = 1 是**共同纯相位假设**，不是已知真实发射谱。
* A5 多快照 = 式(8) 的统计扩展（**明确声明**）：原文式(8) 是**单快照**单向量外积（rank 1）。本机有
  L 个慢时间快照，本模块用 R_m = (1/L) sum_t P_m^(t) (P_m^(t))^H —— **逐快照**保留原文外积、
  再**快照平均**（无 FB、无对角加载、无平滑）。L = 1 时**逐字**回到式(8)。单源时该矩阵仍 rank 1
  （随机慢时间相位被平均消去）；**空间维只取首个特征向量**（见上“原文矛盾”）。
  时延谱峰值检测用**逐快照功率的平均**（对慢时间随机相位稳健）。
* A6 时延 32 ns 周期性（主审 2026-09-24 修订）：时延谱以 1/df 为周期（本批 32 ns）。搜索窗不是
  固定 [0, 1/df)，而是 **物理 ref-delay 窗** = 原点环（含 system_delay）+- 首阵元双程投影余量
  |v_x|max/v_prop，并要求**窗宽 < 周期**；窗内 bin 用**整周期 unwrap** 的绝对时延标签
  （tau = l * dtau，l 可 <0 或 >= 周期 bin 数）。因此**周期端点不作为不可测域**、非零 system_delay
  造成的跨周期部分**不静默丢失**；窗宽 >= 周期时显式 not_applicable（不额外采样、不外延域）。
  候选的最后判据是**原点域**（r_origin 落在距离环内，strict 1e-9 m 容差）。
* A7 平面 x-y 近似：本批虚通道 v = tx + rx 的 v_y 全为 0、v_z 恒为 -0.02 m（**公共 z 偏移**）。
  原文式(12) 为 1-D ULA；本适配**在 x-y 平面内**建模，**公共 z 偏移不进入角度相位**（只作为公共
  标量/公共距离偏移），该近似在 diagnostics 明确声明；但若 v_y 或 v_z 在阵元间**不一致**，则
  原文 ULA 结构不成立 -> not_applicable（不做变体替换）。
* A8 共同域：严格 Cartesian 边界（仅 1e-9 数值容差）+ 半平面 y<0 + tau 落在距离环内；域外候选
  **不静默丢弃**，逐条记入 dropped_candidates（含坐标与原因）；**原生解域外可 ok + 零个**。
* A9 共享后处理：候选去重/cap **直接调用** common/q6_contract.py::postprocess_native_candidates
  （0.10 m、分值降序、确定性、cap=K、不用真值）；本模块**不**自实现规则，也**不**套用二维全谱
  3x3 + 0.3 阈值的取峰口径（C3 不是二维全谱方法）。
* A10 时延锚点模型（主审 2026-09-24 要求显式声明）：本适配实现**逐阵元时延锚点**
      tau_c = 2r/v_prop - v_x,c*sin(theta)/v_prop + system_delay，频域相位 exp(-j2pi f tau_c)
  （与 case.model.geometry 及 B1/B2 的流形一致）=> 式(6) 的峰落在**参考阵元**时延 tau_ref 上，
  故输出原点距离必须加 +v_x_ref*sin(theta)/2（只加一次）。**另一锚点**
      exp(-j2pi f tau_origin) * exp(+j2pi f_c v_x sin(theta)/v)
  的峰位即 tau_origin，**不应**再加该补偿；两种锚点不可混用（诊断里明确写出，测试分开，不混）。
* A11 共同域容差（主审 2026-09-24 **更正后**口径）：新 C 组用 **strict** 共同域 —— bbox、距离环、
  FoV 一律只给 DOMAIN_EPS = 1e-9 m 的**数值容差**，**不做** half-cell / 半格外延（与 b1_deng.py
  L655ff 的 _in_domain 实际实现逐项一致）。相位绕回唯一性也**按 strict bbox** 核。
  说明（**不改** B1/B2 历史代码）：其 FROZEN_PARAMS 里的 domain_tolerance="half_cell" 只是**陈旧标签**，
  代码与文档均为严格边界；本模块**不沿用该标签**，以免 docs 写错。

不适用（not_applicable + diagnostics['reason']）
--------------------------------------------------------------------------------
* k = 0；F < 2；C < 2（ESPRIT 相邻阵元结构需 >= 2 个通道）；k >= F*C；
* 物理 ref-delay 窗宽 >= 无模糊周期（整周期唯一 unwrap 不成立）；
* 频点在**排序后**不严格递增或不严格等间隔（原生 range bin / DFT 结构不成立）；
* 阵元 y/z 坐标在阵元间不一致，或 x 坐标沿阵元不等间距（原文 ULA / Phi 结构不成立）；
* 距离环与原生时延网格无交集。

失败（failed + diagnostics['reason']）：缺键 / 形状不符 / 含 NaN 或 Inf / S 长度不符 / S 含零分量。

本模块只依赖 numpy；**不读真值**、不写任何历史产物；不修改 common/ 与其它方法模块。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

C0 = 299792458.0
TINY = 1e-300
SHARED_NMS_MIN_DIST_M = 0.10
#: 共同域边界只允许数值容差（不做半格外延：外延会悄悄放大共同域定义）
DOMAIN_EPS = 1e-9
#: 时延维数值容差（秒）：坐标容差 1e-9 m 若误用在秒域等于 1 ns（~半个 range bin），故单列
TIME_EPS = 1e-15

#: 冻结默认参数（写定，不按结果调整；实际取值一律在 diagnostics['params_used'] 申报）
FROZEN_PARAMS: dict[str, Any] = {
    "delay_transform": "idft_over_sorted_uniform_inband_frequencies",
    "zeropad_factor": 1,
    #: 显式调用方覆盖（search profile 用）：目标**距离**步（米）。给定时按
    #: `zeropad_factor = max(1, ceil(native_range_bin / target))` 复算（本批 native = 0.2821576 m
    #: ⇒ factor = 6 ⇒ 0.0470263 m）。zeropad 仅把同一 Dirichlet 核插值到更密格点，
    #: 不新增信息、不改变带宽真实分辨力、不改原算法顺序。None = 用上面显式 zeropad_factor。
    "target_range_step_m": None,
    "n_bins_cap": 200000,
    "reference_channel": "first_element",
    "spatial_phase_model": "narrowband_at_center_frequency",
    "spatial_subspace_dim": 1,
    "covariance_extension": "snapshot_mean_outer_product",
    "delay_power_aggregation": "snapshot_mean_power",
    "adjacent_ratio": "first_order_adjacent_elements",
    "peak_edge_policy": "include_edges",
    "bin_mask": "ring_span_strict",
    "domain_tolerance": "strict_1e-9",
    "sin_out_of_range_policy": "drop_and_report",
    "s_response_policy": "params_then_case_model_vector_else_unit",
    "s_response": None,
    "nms_min_dist_m": SHARED_NMS_MIN_DIST_M,
    "max_delay_peaks": None,
}

#: 允许的枚举取值（改成其它值一律显式失败，不静默降级）
_ALLOWED_VALUES = {
    "delay_transform": {"idft_over_sorted_uniform_inband_frequencies"},
    "spatial_phase_model": {"narrowband_at_center_frequency"},
    "covariance_extension": {"snapshot_mean_outer_product"},
    "delay_power_aggregation": {"snapshot_mean_power"},
    "adjacent_ratio": {"first_order_adjacent_elements"},
    "peak_edge_policy": {"include_edges"},
    "bin_mask": {"ring_span_strict"},
    "domain_tolerance": {"strict_1e-9"},
    "sin_out_of_range_policy": {"drop_and_report"},
    "s_response_policy": {"params_then_case_model_vector_else_unit"},
    "reference_channel": {"first_element", "phase_center"},
}

__all__ = [
    "C0",
    "FROZEN_PARAMS",
    "MethodError",
    "NotApplicable",
    "ParsedCase",
    "parse_case",
    "delay_spectra",
    "delay_bin_grid",
    "peak_bins",
    "esprit_from_bin",
    "estimate",
]


class MethodError(ValueError):
    """输入或内部状态不合规：显式失败（status='failed'），不静默退化。"""


class NotApplicable(MethodError):
    """原文结构在本数据/配置下不成立：显式报告（status='not_applicable'）。"""


# --------------------------------------------------------------------------- #
# 通用小工具（本模块自带，便于独立审计；不依赖其它方法模块）
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
    """把 numpy / 复数递归转成 JSON 可序列化形式（diagnostics 落盘用）。"""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "c":
            return {"kind": "complex_array", "n": int(obj.size),
                    "real": [_py(float(v)) for v in np.real(obj).ravel()],
                    "imag": [_py(float(v)) for v in np.imag(obj).ravel()]}
        if obj.size > 512:
            return {"kind": "array_summary", "n": int(obj.size),
                    "shape": [int(v) for v in obj.shape],
                    "min": _py(float(np.min(obj))), "max": _py(float(np.max(obj))),
                    "first8": _py(obj.ravel()[:8].tolist())}
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
    """含端点的确定性局部极大（平台取中点）；与 B1 同一“含端点”政策的独立实现。"""
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


# --------------------------------------------------------------------------- #
# case 解析（只读 case；不读真值）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedCase:
    case_id: str
    X: np.ndarray                 # (L, F*C) **已排序 + 通道按 v_x 升序**（纯索引重排，已申报）
    col_order: np.ndarray         # 相对原始 X 的列置换（审计用）
    freqs: np.ndarray             # (F,) 排序后（严格递增、严格等间隔）
    freq_order_raw: np.ndarray
    freq_was_sorted: bool
    pairs: np.ndarray             # (C,2) 通道按 v_x 升序
    vcoord: np.ndarray            # (C,3) 绝对虚拟坐标 v = tx + rx（按 v_x 升序）
    v_order: np.ndarray
    delta_vx_m: float
    vx_ref_m: float
    ref_channel_index: int
    z_common_m: float
    z_ignored: bool
    fc_hz: float
    fc_source: str
    v_prop: float
    sys_delay_s: float
    k: int
    L: int
    F: int
    C: int
    df_hz: float
    unamb_s: float
    native_bin_s: float
    native_range_bin_m: float
    resolution_scale_s: float
    delta_tau_s: float
    tau_grid_s: np.ndarray
    n_bins: int
    ring_m: tuple | None
    ring_span_m: tuple
    tau_span_s: tuple
    tau_span_prop_s: tuple
    ring_abs_span_s: tuple
    ring_within_unambiguous_window: bool
    ring_fraction_beyond_window: float
    ref_delay_window_s: tuple
    ref_delay_margin_s: float
    window_fits_in_one_period: bool
    period_shift_applied: bool
    period_bin_index: np.ndarray  # (n_bins,) 未折叠的整周期索引
    bin_in_ring: np.ndarray       # bool (n_bins,)
    grid_x: np.ndarray
    grid_y: np.ndarray
    fov_deg: float
    geometric_half_angle_deg: float
    domain_half_angle_deg: float
    phase_wrap_free_sin: float
    phase_wrap_free_angle_deg: float
    phase_unique_in_domain: bool
    S: np.ndarray
    S_source: str
    s_deconv_applied: bool
    config_fb_flag: bool
    cfg_keys_used: dict


def _resolve_params(params: dict | None) -> tuple[dict, dict]:
    used = {key: value for key, value in FROZEN_PARAMS.items()}
    src = {key: "frozen_default" for key in FROZEN_PARAMS}
    if params:
        for key, value in params.items():
            if key not in FROZEN_PARAMS:
                raise MethodError("未知参数 %r（本模块参数冻结，见 FROZEN_PARAMS）" % key)
            used[key] = value
            src[key] = "caller_params"
    for key, allowed in _ALLOWED_VALUES.items():
        if str(used[key]) not in allowed:
            raise MethodError("参数 %s=%r 不在允许取值 %s（本模块不静默降级）"
                              % (key, used[key], sorted(allowed)))
    used["zeropad_factor"] = int(used["zeropad_factor"])
    if used["zeropad_factor"] < 1:
        raise MethodError("zeropad_factor 必须 >= 1，收到 %r" % used["zeropad_factor"])
    used["n_bins_cap"] = int(used["n_bins_cap"])
    if used["n_bins_cap"] < 2:
        raise MethodError("n_bins_cap 必须 >= 2")
    if used.get("target_range_step_m") is not None:
        used["target_range_step_m"] = float(used["target_range_step_m"])
        if not (used["target_range_step_m"] > 0.0):
            raise MethodError("target_range_step_m 必须为正，收到 %r" % used["target_range_step_m"])
    used["spatial_subspace_dim"] = int(used["spatial_subspace_dim"])
    if used["spatial_subspace_dim"] != 1:
        raise MethodError(
            "spatial_subspace_dim=%r：原文式(8) 为单向量外积（rank 1），只支持 1 个信号特征向量；"
            "每 bin 多源 / 子阵平滑属**增强**，本适配版不做（与主审 2026-09-24 决策一致）"
            % used["spatial_subspace_dim"])
    used["nms_min_dist_m"] = float(used["nms_min_dist_m"])
    if not (used["nms_min_dist_m"] > 0.0):
        raise MethodError("nms_min_dist_m 必须为正")
    if used["max_delay_peaks"] is not None:
        used["max_delay_peaks"] = int(used["max_delay_peaks"])
        if used["max_delay_peaks"] < 0:
            raise MethodError("max_delay_peaks 必须非负或 None")
    if used["s_response"] is not None:
        used["s_response"] = np.asarray(used["s_response"], dtype=np.complex128).reshape(-1)
    used["reference_channel"] = str(used["reference_channel"])
    return used, src


def _resolve_s_response(case: dict, params: dict, F: int) -> tuple[np.ndarray, str]:
    """S（发射谱）解析：params 优先，其次 case.model.response_assumption.vector，否则 S = 1 假设。"""
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


def delay_bin_grid(F: int, df_hz: float, zeropad_factor: int) -> dict:
    """原生时延网格：tau_l = l * dtau，dtau = 1/(N_zp * df)，N_zp = zeropad_factor * F，覆盖 [0, 1/df)。"""
    n_bins = int(zeropad_factor) * int(F)
    unamb = 1.0 / float(df_hz)
    delta_tau = 1.0 / (float(n_bins) * float(df_hz))
    tau = delta_tau * np.arange(n_bins, dtype=float)
    return {"n_bins": int(n_bins), "delta_tau_s": float(delta_tau), "tau_s": tau,
            "unamb_s": float(unamb), "native_bin_s": 1.0 / (float(F) * float(df_hz))}


def _resolve_ring_tau_span(cfg: dict, budget: dict, v_prop: float, sys_delay: float):
    ring = cfg.get("distance_ring_m")
    ring_m = None if ring is None else (float(ring[0]), float(ring[1]))
    span = budget.get("round_trip_tau_span_s")
    if span is not None:
        span_prop = (float(span[0]), float(span[1]))
        span_src = "search.one_dimensional_budget.round_trip_tau_span_s"
    elif ring_m is not None:
        span_prop = (2.0 * ring_m[0] / v_prop, 2.0 * ring_m[1] / v_prop)
        span_src = "derived_from_config.distance_ring_m"
    else:
        raise NotApplicable(
            "既无 search.one_dimensional_budget.round_trip_tau_span_s，也无 config.distance_ring_m："
            "没有可冻结（且不依赖真值）时延搜索域，拒绝默猜全域")
    if not (span_prop[1] > span_prop[0]):
        raise MethodError("时延搜索域必须严格递增，收到 %r" % (span_prop,))
    # 搜索域是**绝对到达时延** tau = 2r/v_prop + system_delay（与其它方法统一口径）
    return span_prop, span_src, ring_m


def parse_case(case: dict, params: dict | None = None) -> tuple[ParsedCase, dict, dict]:
    """case -> (ParsedCase, params_used, parse_diag)。只读 case，不读真值。"""
    if not isinstance(case, dict):
        raise MethodError("case 必须是 dict，收到 %r" % type(case).__name__)
    used, _src = _resolve_params(params)

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
    if int(X.shape[0]) < 1:
        raise MethodError("X 至少需要 1 个快照")
    if F < 2:
        raise NotApplicable("时延谱（式(6) 的 DFT）需 >= 2 个带内频点；本 case F=%d" % F)
    if C < 2:
        raise NotApplicable("式(8)-(14) 的 ESPRIT 需 >= 2 个阵元通道；本 case C=%d" % C)
    if k == 0:
        raise NotApplicable("k=0：式(6)(7) 的距离峰检测需至少 1 个源，无目标窗不给候选")
    if k >= F * C:
        raise NotApplicable("k=%d >= F*C=%d：时延谱/空间结构不可用" % (k, F * C))
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
    fov = float(cfg.get("fov_degrees", 180.0) or 180.0)

    search = case.get("search") or {}
    if not isinstance(search, dict):
        raise MethodError("case['search'] 必须是 dict")
    budget = search.get("one_dimensional_budget") or {}
    if not isinstance(budget, dict):
        raise MethodError("search.one_dimensional_budget 必须是 dict")
    span_prop, span_src, ring_m = _resolve_ring_tau_span(cfg, budget, v_prop, sys_delay)
    tau_span_prop = (span_prop[0], span_prop[1])
    # ---- 共同域容差：strict DOMAIN_EPS=1e-9 m（bbox + ring + FoV），对齐 b1_deng.py L655ff 实际实现 ----
    res_m = search.get("resolution_m")
    res_m = float(res_m) if (res_m is not None and float(res_m) > 0.0) else None
    step_x = float(np.min(np.diff(grid_x))) if grid_x.size > 1 else 0.0
    step_y = float(np.min(np.diff(grid_y))) if grid_y.size > 1 else 0.0
    cand_cells = [s for s in (step_x, step_y) if s > 0.0]
    case_cell_m = res_m if res_m is not None else (float(min(cand_cells)) if cand_cells else 0.0)
    case_cell_src = "search.resolution_m" if res_m is not None else "grid step"
    tau_span = (span_prop[0] + sys_delay, span_prop[1] + sys_delay)
    ring_span_m = (0.5 * v_prop * span_prop[0], 0.5 * v_prop * span_prop[1])

    # ---- 频率轴：原文式(6) 的 DFT 要求等间隔有序样本，故必须先按频率排序 ----
    freq_order = np.argsort(freqs, kind="stable")
    freqs_sorted = freqs[freq_order]
    freq_was_sorted = bool(np.array_equal(freq_order, np.arange(F)))
    dfs = np.diff(freqs_sorted)
    if np.any(dfs <= 0.0):
        dup = [int(i) for i in np.nonzero(dfs <= 0.0)[0]]
        raise NotApplicable(
            "带内频点排序后仍非严格递增（重复频点索引 %s）：CIR 逆 DFT 的原生 range bin/无模糊窗不成立"
            % dup[:6])
    df_hz = float(dfs[0])
    spread = float((np.max(dfs) - np.min(dfs)) / np.max(dfs))
    if spread > 1e-6:
        raise NotApplicable(
            "带内频点排序后**非等间隔**（相对散布 %.3e > 1e-6）：式(6) 的 DFT/原生 range bin 结构不成立；"
            "本适配版不做非均匀频点的额外重采样（那会改变方法本身）" % spread)

    # ---- 阵元：原文 ULA（式(12) 的 e^{j pi k sin θ}），要求沿 x 等间距、y/z 在阵元间一致 ----
    vcoord = (tx[pairs[:, 0]] + rx[pairs[:, 1]]).astype(float)
    v_order = np.argsort(vcoord[:, 0], kind="stable")
    v_sorted = vcoord[v_order]
    for axis, name in ((1, "y"), (2, "z")):
        span_axis = float(np.ptp(v_sorted[:, axis]))
        if span_axis > 1e-9:
            raise NotApplicable(
                "原文式(12) 的 ULA 结构要求各阵元 %s 坐标一致：本机极差 %.3e m，结构不成立"
                % (name, span_axis))
    dvx = np.diff(v_sorted[:, 0])
    if not np.all(dvx > 0.0):
        raise NotApplicable("虚拟阵元 x 坐标存在重复（间距为 0）：原文 ULA 结构不成立")
    dvx_spread = float((np.max(dvx) - np.min(dvx)) / np.max(dvx))
    if dvx_spread > 1e-6:
        raise NotApplicable(
            "原文式(12) 要求阵元沿 x **等间距**：相对间距散布 %.3e 超出 1e-6，结构不成立" % dvx_spread)
    delta_vx = float(dvx.mean())
    max_abs_vx = float(np.max(np.abs(v_sorted[:, 0])))
    z_common = float(np.mean(v_sorted[:, 2]))
    z_ignored = bool(abs(z_common) > 1e-12)

    # ---- X 列置换：先按频率排序（f*C 块），再在块内把通道按 v_x 升序排列 ----
    col_order = np.concatenate([f * C + v_order for f in freq_order]).astype(np.int64)
    Xs = X[:, col_order]

    # ---- S（发射谱）：非平坦 S 需反卷积（原文式(5) 为去斜后的复指数模型）----
    S, s_src = _resolve_s_response(case, params if params else {}, F)
    S_sorted = S[freq_order]
    s_deconv = False
    if not np.allclose(S_sorted, 1.0):
        if np.any(np.abs(S_sorted) == 0.0):
            zeros = [int(i) for i in np.nonzero(np.abs(S_sorted) == 0.0)[0]]
            raise MethodError(
                "S 含零分量（排序后索引 %s）：非平坦 S 的反卷积在零点处不可用；"
                "请声明 S 非零，或令 S=1（共同纯相位假设）" % zeros[:8])
        Xs = Xs / np.repeat(S_sorted, C)[None, :]
        s_deconv = True

    fc = cfg.get("center_freq_si")
    if fc is None or float(fc) <= 0.0:
        fc = float(0.5 * (freqs_sorted[0] + freqs_sorted[-1]))
        fc_src = "derived_band_center(frequencies_hz)"
    else:
        fc = float(fc)
        fc_src = "config.center_freq_si"

    # 目标距离步（search profile）-> 最小整数 zeropad（写定，不按结果/真值调整）
    zeropad_src = "params/frozen zeropad_factor（无 target_range_step_m）"
    tgt_range = used.get("target_range_step_m")
    if tgt_range is not None:
        native_bin_s_ = 1.0 / (float(F) * float(df_hz))
        native_range_m_ = v_prop * native_bin_s_ / 2.0
        factor_ = max(1, int(math.ceil(native_range_m_ / float(tgt_range) - 1e-12)))
        used["zeropad_factor"] = int(factor_)
        zeropad_src = ("caller_params.target_range_step_m(%.6g m)：原生 range bin = %.6g m "
                       "→ zeropad = max(1, ceil(native/target)) = %d"
                       % (float(tgt_range), native_range_m_, factor_))
    grid = delay_bin_grid(F, df_hz, used["zeropad_factor"])
    n_period = int(grid["n_bins"])
    delta_tau = float(grid["delta_tau_s"])
    unamb = float(grid["unamb_s"])
    # ---- 物理 ref-delay 搜索窗：宽度 < 周期 -> 整周期 unwrap（不把周期端点当不可测域）----
    lo, hi = tau_span
    ref_margin_s = max_abs_vx / v_prop     # 首（及各）阵元双程投影相对原点的最大偏移
    win_lo = lo - ref_margin_s
    win_hi = hi + ref_margin_s
    win_width = win_hi - win_lo
    window_fits = bool(win_width < unamb - TIME_EPS)
    if not window_fits:
        raise NotApplicable(
            "物理 ref-delay 搜索窗宽度 %.6g s >= 无模糊周期 %.6g s（环跨度 +- 首阵元余量）："
            "整周期唯一 unwrap 不成立，本适配版不额外采样也不外延域" % (win_width, unamb))
    l_min = int(math.ceil((win_lo - TIME_EPS) / delta_tau))
    l_max = int(math.floor((win_hi + TIME_EPS) / delta_tau))
    if l_max < l_min:
        raise NotApplicable(
            "物理 ref-delay 搜索窗 [%.6g, %.6g] s 在原生网格步长 %.6g s 下无可用 bin"
            % (win_lo, win_hi, delta_tau))
    period_bin_index = np.arange(l_min, l_max + 1, dtype=np.int64)   # **未折叠**整周期索引
    tau_grid = period_bin_index.astype(float) * delta_tau
    n_bins = int(tau_grid.size)
    if n_bins > int(used["n_bins_cap"]):
        raise MethodError("时延网格点数 %d 超过冻结上限 %d；请降低 zeropad_factor"
                          % (n_bins, int(used["n_bins_cap"])))
    bin_in_ring = np.ones(n_bins, dtype=bool)      # 掩膜 = 物理搜索窗本身
    n_valid = n_bins
    period_shift_applied = bool(np.any((period_bin_index < 0) | (period_bin_index >= n_period)))
    # 信息性申报：原点环相对**单周期**的位置（不再据此裁剪域）
    ring_within = bool(hi <= unamb + TIME_EPS)
    ring_beyond = float(max(0.0, hi - unamb) / max(unamb, TINY))

    if used["reference_channel"] == "phase_center":
        ref_ch = int(np.argmin(np.abs(v_sorted[:, 0])))
        ref_src = "phase_center(argmin|v_x|)"
    else:
        ref_ch = 0
        ref_src = "first_element(v_x ascending order, 即原文“first antenna array”)"

    # ---- 相位绕回唯一性：|dv_x sin θ| < lambda_c/2 <=> |θ| < asin(lambda_c/(2 dv_x)) ----
    half_lambda = 0.5 * v_prop / fc
    wrap_sin_bound = half_lambda / delta_vx         # 无量纲：sin 域内的无绕回上界
    wrap_sin_eff = min(1.0, float(wrap_sin_bound))
    wrap_angle = math.degrees(math.asin(wrap_sin_eff))
    # 相位唯一性按 **strict bbox** 核（主审 2026-09-24 更正口径）
    geom_half = math.degrees(math.atan2(float(np.max(np.abs(grid_x))),
                                        max(float(np.min(np.abs(grid_y))), 1e-12)))
    fov_half = 0.5 * abs(fov) if fov else 90.0
    domain_half = min(geom_half, fov_half)
    phase_unique = bool(math.sin(math.radians(domain_half)) <= wrap_sin_eff + 1e-12)

    parse_diag = {
        "manifold": "CFR -> delay spectrum (IDFT over sorted in-band frequencies) + closed-form ESPRIT ratio",
        "n_features": F * C,
        "feature_order": case.get("feature_order"),
        "frequency_order": ("raw case order is NOT sorted -> sorted ascending for the DFT; "
                            "X columns permuted consistently (pure index re-labelling)"),
        "frequency_was_already_sorted": freq_was_sorted,
        "freq_raw_ghz": [float(v / 1e9) for v in freqs],
        "freq_sorted_ghz": [float(v / 1e9) for v in freqs_sorted],
        "df_hz": df_hz,
        "unambiguous_window_s": unamb,
        "native_bin_s": float(grid["native_bin_s"]),
        "native_range_bin_m": float(v_prop * grid["native_bin_s"] / 2.0),
        "resolution_scale_s": float(1.0 / ((F - 1) * df_hz)),
        "resolution_scale_note": ("原生 range bin 由 N_zp=zeropad*F 的时延网格给出；"
                                  "物理分辨率尺度 ~ 1/(占用带宽)=1/((F-1)*df)"),
        "delta_tau_s": float(grid["delta_tau_s"]),
        "n_bins_total": int(n_bins),
        "n_bins_in_ring": int(n_valid),
        "zeropad_factor": int(used["zeropad_factor"]),
        "zeropad_source": zeropad_src,
        "target_range_step_m": (None if tgt_range is None else float(tgt_range)),
        "zeropad_is_interpolation_only": True,
        "delay_periodicity": {
            "period_s": unamb,
            "origin_ring_absolute_tau_span_s": [lo, hi],
            "reference_margin_s": float(ref_margin_s),
            "reference_delay_window_s": [float(win_lo), float(win_hi)],
            "window_width_s": float(win_width),
            "window_fits_in_one_period": bool(window_fits),
            "bin_index_range_unwrapped": [int(l_min), int(l_max)],
            "period_bin_count": int(n_period),
            "period_shift_applied": bool(period_shift_applied),
            "origin_ring_within_one_period_informational": ring_within,
            "origin_ring_fraction_beyond_period_informational": ring_beyond,
            "origin_ring_upper_edge_at_period_endpoint": bool(hi >= unamb - TIME_EPS),
            "note": ("时延谱以 1/df 为周期。搜索窗 = 原点环 +- 首阵元双程投影余量（含 system_delay），"
                     "按**整周期 unwrap** 到物理绝对时延：**周期端点不作为不可测域**，"
                     "环上限超过单周期不再被裁剪（覆盖不静默丢失）。"
                     "固有周期模糊：tau 与 tau + k/df 的 bin **同值**（端点与低时延/DC 混叠），"
                     "属该方法固有性质，按物理窗标签如实申报，不额外去混叠"),
        },
        "domain_tolerance": {
            "mode": "strict",
            "domain_eps_m": float(DOMAIN_EPS),
            "applied_to": ["cartesian_bbox_x", "cartesian_bbox_y", "distance_ring_r", "half_plane_y_lt_0", "fov"],
            "case_cell_m": float(case_cell_m),
            "case_cell_source": case_cell_src,
            "half_cell_used": False,
            "note": ("主审 2026-09-24 **更正后**口径：bbox / 距离环 / FoV 一律只用 DOMAIN_EPS=1e-9 m 的"
                     "数值容差，不做 half-cell 外延；与 b1_deng.py L655ff 的 _in_domain 实际实现一致"
                     "（其 FROZEN_PARAMS 的 domain_tolerance=half_cell 是陈旧标签，代码为严格边界）。"
                     "case_cell_m 仅作信息记录，**不**用于扩张共同域"),
        },
        "delay_anchor_model": {
            "anchor": "per_channel_delay(wideband_first_order)",
            "definition": ("tau_c = 2r/v_prop - v_x,c*sin(theta)/v_prop + system_delay；"
                           "频域相位 exp(-j2pi f tau_c)（与 case.model.geometry 及各基线流形一致）"),
            "consequence": ("式(6) 的峰落在**参考阵元**时延 tau_ref 上，输出原点距离需 "
                            "+v_x_ref*sin(theta)/2 补偿（只加一次）"),
            "alternative_not_used": ("另一锚点 exp(-j2pi f tau_origin)*exp(+j2pi f_c v_x sin θ/v)"
                                    "（窄带锚原点）的峰位即 tau_origin，**不应**再加该补偿；"
                                    "两种锚点不可混用，本模块只实现前者（与 case.model 一致）"),
        },
        "reference_channel": str(used["reference_channel"]),
        "reference_channel_source": ref_src,
        "reference_channel_index": int(ref_ch),
        "virtual_x_m": [float(v) for v in v_sorted[:, 0]],
        "virtual_y_m": [float(v) for v in v_sorted[:, 1]],
        "virtual_z_m": [float(v) for v in v_sorted[:, 2]],
        "ula_channel_order": [int(i) for i in v_order],
        "ula_spacing_m": delta_vx,
        "lambda_c_m": float(v_prop / fc),
        "lambda_c_over_2_m": half_lambda,
        "d_over_half_lambda": float(delta_vx / half_lambda),
        "planar_xy_model": ("原文式(12) 为 1-D ULA；本适配在 x-y 平面建模，**公共 z 偏移不进入角度相位**"
                            "（只作公共标量/公共距离偏移）——声明近似"),
        "common_z_offset_m": z_common,
        "common_z_ignored": z_ignored,
        "phase_wrap": {
            "criterion": "|dv_x * sin(theta)| < lambda_c/2",
            "wrap_free_sin_bound": float(wrap_sin_bound),
            "wrap_free_sin_bound_clipped_1": float(wrap_sin_eff),
            "wrap_free_angle_deg": float(wrap_angle),
            "geometric_half_angle_deg": float(geom_half),
            "fov_half_angle_deg": float(fov_half),
            "domain_half_angle_deg": float(domain_half),
            "phase_unique_in_common_domain": phase_unique,
            "note": ("d > lambda_c/2 **不**单独判 NA：只要共同域几何半角 <= wrap_free_angle_deg，"
                     "则式(14) 的主值解在域内唯一，按限定共同域声明辨识；否则申报 principal-branch 限制"),
        },
        "center_frequency_hz": float(fc),
        "center_frequency_source": fc_src,
        "spatial_phase_model": ("narrowband at fc: adjacent-element ratio phase = +2pi fc dv_x sin θ / v_prop"
                                "（dv_x = lambda_c/2 时退化为原文的 pi sin θ）"),
        "propagation_speed_m_s": v_prop,
        "propagation_speed_source": v_src,
        "epsilon_r_background": eps_r,
        "system_delay_si": sys_delay,
        "tau_span_s": [tau_span[0], tau_span[1]],
        "tau_span_propagation_s": [tau_span_prop[0], tau_span_prop[1]],
        "tau_span_absolute_note": ("搜索域是**绝对到达时延** tau = 2r/v_prop + system_delay_si；"
                                   "round_trip_tau_span_s 只给传播时间，已统一加 system_delay"),
        "tau_span_source": span_src,
        "ring_span_m": [ring_span_m[0], ring_span_m[1]],
        "distance_ring_m": None if ring_m is None else [ring_m[0], ring_m[1]],
        "fov_degrees": fov,
        "domain_policy": ("共同域用 **strict** Cartesian 边界 + 距离环 + FoV（仅 DOMAIN_EPS=1e-9 m 数值"
                          "容差，**不做**半格外延；与 b1_deng.py L655ff 实际实现一致）；半平面 y<0 按"
                          "物理约束；峰搜索用整周期 unwrap 的物理 ref-delay 窗，含端点取峰"),
        "s_response_source": s_src,
        "s_response_is_flat": bool(np.allclose(S_sorted, 1.0)),
        "s_deconvolution_applied": bool(s_deconv),
        "s_response_note": ("原文式(5) 是**去斜后**的复指数模型；声明非平坦 S 时按反卷积（X/S）处理。"
                            "DOA 比值与 S 无关（逐频公共因子在跨阵元比值中约掉），时延峰位置含 S"),
        "case_config_is_forward_backward": bool(cfg.get("is_forward_backward", False)),
        "forward_backward_used": False,
        "forward_backward_note": "原文式(8) 无前后向平均；case.config.is_forward_backward 属主方法选择，本模块不继承",
        "diag_load_used": 0.0,
        "channel_phase_calibration": "X 内已含数据侧 exp(-1j*phi) 校准；本模块不重复施加",
        "column_permutation_applied": [int(v) for v in col_order],
    }
    pc = ParsedCase(
        case_id=str(case.get("id", "?")), X=Xs, col_order=col_order, freqs=freqs_sorted,
        freq_order_raw=freq_order, freq_was_sorted=freq_was_sorted, pairs=pairs[v_order],
        vcoord=v_sorted, v_order=v_order, delta_vx_m=delta_vx,
        vx_ref_m=float(v_sorted[ref_ch, 0]), ref_channel_index=int(ref_ch),
        z_common_m=z_common, z_ignored=z_ignored, fc_hz=float(fc), fc_source=fc_src,
        v_prop=v_prop, sys_delay_s=sys_delay, k=k, L=int(X.shape[0]), F=F, C=C, df_hz=df_hz,
        unamb_s=unamb, native_bin_s=float(grid["native_bin_s"]),
        native_range_bin_m=float(v_prop * grid["native_bin_s"] / 2.0),
        resolution_scale_s=float(1.0 / ((F - 1) * df_hz)), delta_tau_s=float(grid["delta_tau_s"]),
        tau_grid_s=tau_grid, n_bins=n_bins, ring_m=ring_m, ring_span_m=ring_span_m,
        tau_span_s=tau_span, tau_span_prop_s=tau_span_prop,
        ring_abs_span_s=(lo, hi), ring_within_unambiguous_window=ring_within,
        ring_fraction_beyond_window=ring_beyond,
        ref_delay_window_s=(float(win_lo), float(win_hi)),
        ref_delay_margin_s=float(ref_margin_s), window_fits_in_one_period=bool(window_fits),
        period_shift_applied=bool(period_shift_applied), period_bin_index=period_bin_index,
        bin_in_ring=bin_in_ring, grid_x=grid_x,
        grid_y=grid_y, fov_deg=fov, geometric_half_angle_deg=float(geom_half),
        domain_half_angle_deg=float(domain_half), phase_wrap_free_sin=float(wrap_sin_eff),
        phase_wrap_free_angle_deg=float(wrap_angle), phase_unique_in_domain=phase_unique,
        S=S_sorted, S_source=s_src, s_deconv_applied=s_deconv,
        config_fb_flag=bool(cfg.get("is_forward_backward", False)),
        cfg_keys_used={"propagation_speed_m_s": v_src, "system_delay_si": "config",
                       "distance_ring_m": ring_m is not None,
                       "fov_degrees": "config" if "fov_degrees" in cfg else "default_180"})
    return pc, used, parse_diag


# --------------------------------------------------------------------------- #
# 式(6)：CFR -> 时延谱（逆 DFT）；式(8)：跨阵元外积（快照平均，L=1 时逐字回到原文）
# --------------------------------------------------------------------------- #
def delay_spectra(pc: ParsedCase) -> dict:
    """式(6) 的 CIR 适配：P_c[l] = sum_j X[f_j, c] * e^{+j 2pi (f_j - f_c) tau_l}，并按快照平均功率取峰。"""
    F, C, nb = pc.F, pc.C, pc.n_bins
    Xr = pc.X.reshape(pc.L, F, C)                      # 列序 f*C+c -> [t, f, c]
    kern = np.exp(2j * np.pi * np.outer(pc.freqs - pc.fc_hz, pc.tau_grid_s))   # (F, nb)
    P = np.einsum("tfc,fl->ctl", Xr, kern)             # (C, L, nb)
    power = np.mean(np.abs(P[pc.ref_channel_index, :, :]) ** 2, axis=0)        # (nb,)
    return {
        "P": P,
        "power": power,
        "reference_channel_index": int(pc.ref_channel_index),
        "aggregation": "snapshot-mean power of the first-element delay spectrum",
        "power_source": "mean_t |P_ref[t, l]|^2（逐快照功率平均，对慢时间随机相位稳健）",
        "kern_rel_fc_note": ("核用 (f_j - f_c) 的基带形式；与用绝对值 f_j 只差逐 bin 公共相位 "
                             "e^{+j 2pi f_c tau_l}，对 |P| 与跨阵元比值均无影响"),
    }


def peak_bins(power: np.ndarray, mask: np.ndarray) -> list:
    """环内 bin 上的含端点局部极大（返回**全局 bin 索引**，按输入序）。"""
    p = np.asarray(power, dtype=float)
    idx = np.nonzero(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    return [int(idx[i]) for i in _peaks_incl_edges(p[idx])]


def _select_peaks(power: np.ndarray, mask: np.ndarray, n_keep: int) -> tuple[list, str]:
    p = np.asarray(power, dtype=float)
    peaks = peak_bins(p, mask)
    note = "strict_local_max_including_edges_within_ring"
    if not peaks:
        idx = np.nonzero(np.asarray(mask, dtype=bool))[0]
        if idx.size == 0:
            return [], "no_bin_in_ring"
        peaks = [int(idx[int(np.argmax(p[idx]))])]
        note = "flat_no_strict_peak_fallback_argmax"
    order = sorted(peaks, key=lambda i: (-p[i], i))
    return order[: max(int(n_keep), 0)], note


def esprit_from_bin(pc: ParsedCase, P: np.ndarray, l: int) -> dict:
    """式(8)-(14)：给定 TOA 索引 l，跨阵元自相关（快照平均外积）-> 首个特征向量 -> 相邻比值 -> theta_hat。

    * R_m = (1/L) sum_t P_m^(t) (P_m^(t))^H（L=1 时**逐字**为原文式(8) 的 P_m P_m^H）；
    * 取**首个**（最大）特征向量 u（原文式(8) 的秩 1 结构 => 每距离峰最多 1 个源，不增强）；
    * xi = (J_0 u)^dagger (J_1 u)（式(10)(13) 在单向量下的闭式；广义 EVD 退化为标量）；
    * sin(theta) = angle(xi) * v_prop / (2 pi f_c dv_x)（式(12)(14) 的窄带空间相位；
      dv_x = lambda_c/2 时即 asin(angle(xi)/pi)）。
    """
    C = pc.C
    Pm = P[:, :, int(l)]                              # (C, L)
    R = (Pm @ Pm.conj().T) / float(pc.L)               # 式(8)（快照平均扩展）
    R = 0.5 * (R + R.conj().T)
    w, V = np.linalg.eigh(R)
    trace = float(np.real(np.trace(R)))
    if not math.isfinite(trace) or trace <= 0.0:
        raise MethodError("bin l=%d 的跨阵元外积能量为零/非有限，无法做 ESPRIT" % int(l))
    u = V[:, -1]                                       # 式(9) 的 U_1：只取首个信号特征向量
    u0 = u[: C - 1]                                    # 式(10) J_0 U_1
    u1 = u[1:]                                         # 式(10) J_1 U_1
    denom = complex(u0.conj() @ u0)
    if abs(denom) <= TINY:
        raise MethodError("bin l=%d：U_{1,0} 范数为零，式(13) 的 U_{1,0}^dagger U_{1,1} 不可用" % int(l))
    xi = complex(u0.conj() @ u1) / denom               # 式(13) 的 xi_m（单向量时退化为标量）
    phase = math.atan2(xi.imag, xi.real)               # angle(·)，主值 (-pi, pi]
    scale = 2.0 * math.pi * pc.fc_hz * pc.delta_vx_m / pc.v_prop
    sin_theta = phase / scale
    abs_sin = abs(sin_theta)
    return {
        "bin_index": int(l),
        "period_bin_index": int(pc.period_bin_index[int(l)]),
        "tau_ref_s": float(pc.tau_grid_s[int(l)]),
        "xi": xi,
        "xi_phase_rad": float(phase),
        "principal_branch_used": True,
        "sin_theta": float(sin_theta),
        "sin_theta_abs": float(abs_sin),
        "sin_theta_in_range": bool(abs_sin <= 1.0 + 1e-9),
        "sin_theta_within_wrap_free": bool(abs_sin <= pc.phase_wrap_free_sin + 1e-12),
        "theta_deg": (float(math.degrees(math.asin(max(-1.0, min(1.0, sin_theta)))))
                      if abs_sin <= 1.0 else None),
        "eig_values": [float(v) for v in w],
        "eig_smallest": float(w[0]),
        "eig_largest": float(w[-1]),
        "rank_one_ratio": float(abs(w[-1]) / max(abs(w[0]), TINY)),
        "cross_element_trace": trace,
        "snapshots_L": int(pc.L),
        "spatial_subspace_dim_used": 1,
        "spatial_scale_rad_per_sin": float(scale),
        "formula": ("xi = (J0 u)^H (J1 u) / ||J0 u||^2 ; "
                    "sin(theta) = angle(xi) * v_prop / (2*pi*fc*delta_vx)"),
    }


# --------------------------------------------------------------------------- #
# 已解析候选 -> 共同域过滤（严格边界 + 半平面 + 距离环 + FoV）
# --------------------------------------------------------------------------- #
def _in_domain(pc: ParsedCase, x: float, y: float, r_m: float) -> tuple[bool, str]:
    """共同域过滤（**strict**）：bbox / 距离环 / FoV 只用 DOMAIN_EPS=1e-9 m 数值容差，不做半格外延。

    与 b1_deng.py L655ff 的 _in_domain **逐项一致**（B1/B2 不改）；距离环检查用**原点**距离
    （首阵元补偿后的 r）。网格边界上的合法峰由含端点取峰器保留。
    """
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
    """加载公共合同里的共享 NMS（仅加载一次）。不复制规则：公共合同缺失即显式失败。"""
    if "fn" not in _SHARED_NMS_CACHE:
        import importlib
        import sys

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
    diag.setdefault("grid_kind", "cir_delay_bins_then_esperit_adjacent_ratio")
    diag.setdefault("n_candidates_evaluated", 0)
    diag.setdefault("dictionary_size", None)
    diag.setdefault("dictionary_size_note",
                    "C3 无稀疏字典：式(6) 为时延维 IDFT（原生 range bin），式(8)-(14) 为闭式比值")
    return {"predictions": list(predictions), "status": str(status), "diagnostics": _py(diag)}


def estimate(case: dict, params: dict | None = None) -> dict:
    """C3 公共接口：estimate(case[, params]) -> {predictions, status, diagnostics}。

    predictions = [{'x','y','score'}]（<= K 的原生候选；score = 参考阵元时延谱在该 bin 的功率）。
    本函数**不读真值**；失败/不适用一律显式返回并给 reason。
    """
    t_start = time.perf_counter()
    diag: dict = {
        "method": "C3",
        "source": "Kim, Oh & Lee 2015, IEEE AWPL 14:1710-1713",
        "doi": "10.1109/LAWP.2015.2420579",
        "implemented_equations": "(5)(6)(8)(9)(10)(11)(12)(13)(14)",
        "replaced_equation": "(7) FMCW 去斜斜率式 tau=I/(N*Ts*mu) 属去斜拍频专有；本适配改为 CIR/CFR 时延谱",
        "structure": ("第一阵元时延谱取峰 -> 同索引跨阵元自相关（快照平均外积）-> 首个特征向量 -> "
                      "相邻比值 xi -> sin(theta) = angle(xi)*v/(2 pi fc dv_x)"),
        "reproduction_claim": ("**不声称逐式原样复现**：这是明确声明的 CIR 适配版；原文 L/K/M 记法混用、"
                               "式(8) 单向量外积 rank 1 与“M 个信号特征向量”不一致，已在 docs 记录"),
        "not_enhanced": ("每个距离峰只给 1 个角度（式(8) 秩 1）；不做每 bin 多源/子阵平滑/前后向/对角加载；"
                         "同距离多目标保留原限制"),
    }
    try:
        pc, used, parse_diag = parse_case(case, params)
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001 - 显式失败，不静默
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
    diag["params_used"] = used
    diag["parse"] = parse_diag
    diag["phase_uniqueness"] = dict(parse_diag["phase_wrap"])
    diag["delay_periodicity"] = dict(parse_diag["delay_periodicity"])
    diag["domain_tolerance"] = dict(parse_diag["domain_tolerance"])
    diag["delay_anchor_model"] = dict(parse_diag["delay_anchor_model"])
    diag["planar_xy_model_declared"] = True

    try:
        energy = float(np.sum(np.abs(pc.X) ** 2))
        diag["input_energy"] = energy
        if not math.isfinite(energy) or energy <= 0.0:
            raise MethodError("输入能量为零/非有限（sum|X|^2 = %r）：拒绝捏造“正常结果”" % energy)
        spec = delay_spectra(pc)
        P, power = spec["P"], spec["power"]
        n_keep = int(used["max_delay_peaks"]) if used["max_delay_peaks"] is not None else int(pc.k)
        sel, peak_note = _select_peaks(power, pc.bin_in_ring, n_keep)
        n_local_peaks = len(peak_bins(power, pc.bin_in_ring))
        diag["toa_stage"] = {
            "grid_kind": "uniform_delay_bins_over_unambiguous_window",
            "n_bins_total": int(pc.n_bins),
            "n_bins_in_ring": int(np.count_nonzero(pc.bin_in_ring)),
            "delta_tau_s": float(pc.delta_tau_s),
            "delta_range_m": float(pc.v_prop * pc.delta_tau_s / 2.0),
            "native_bin_s": float(pc.native_bin_s),
            "native_range_bin_m": float(pc.native_range_bin_m),
            "resolution_scale_s": float(pc.resolution_scale_s),
            "unambiguous_window_s": float(pc.unamb_s),
            "zeropad_factor": int(used["zeropad_factor"]),
            "zeropad_note": "zeropad>1 仅把同一 Dirichlet 核插值到更密网格（不新增信息、不提高分辨率）",
            "peak_policy": peak_note,
            "n_local_peaks_in_ring": int(n_local_peaks),
            "n_peaks_selected": int(len(sel)),
            "selected_bin_indices": [int(i) for i in sel],
            "selected_period_bin_indices": [int(pc.period_bin_index[int(i)]) for i in sel],
            "selected_bins_period_alias_of_dc": [
                bool(int(pc.period_bin_index[int(i)])
                     % int(parse_diag["delay_periodicity"]["period_bin_count"]) == 0)
                for i in sel],
            "selected_tau_s": [float(pc.tau_grid_s[int(i)]) for i in sel],
            "search_window_s": [float(pc.ref_delay_window_s[0]), float(pc.ref_delay_window_s[1])],
            "window_fits_in_one_period": bool(pc.window_fits_in_one_period),
            "period_bin_count": int(parse_diag["delay_periodicity"]["period_bin_count"]),
            "period_shift_applied": bool(pc.period_shift_applied),
            "selected_power": [float(power[int(i)]) for i in sel],
            "aggregation": spec["aggregation"],
            "power_source": spec["power_source"],
            "reference_channel_index": int(pc.ref_channel_index),
            "reference_channel_source": parse_diag["reference_channel_source"],
        }
        candidates, dropped, algorithm_failures = [], [], []
        n_eval = int(np.count_nonzero(pc.bin_in_ring))
        compensation_note = {
            "formula": "r = v_prop*(tau_ref - system_delay)/2 + v_x_ref*sin(theta)/2",
            "v_x_ref_m": float(pc.vx_ref_m),
            "reference_channel_index": int(pc.ref_channel_index),
            "applied_once": True,
            "note": ("tau_ref 属参考阵元（原文 first antenna array）；转原点距离只加一次补偿，"
                     "重复加/漏加都会产生 |v_x_ref|*sin(theta)/2 的系统偏差"),
        }
        diag["compensation"] = compensation_note
        for l in sel:
            n_eval += int(pc.C) + 1
            try:
                est = esprit_from_bin(pc, P, int(l))
            except MethodError as exc:
                algorithm_failures.append({
                    "bin_index": int(l),
                    "period_bin_index": int(pc.period_bin_index[int(l)]),
                    "reason": "esprit_error: %s" % exc,
                })
                continue
            if not est["sin_theta_in_range"]:
                dropped.append({"bin_index": int(l), "tau_ref_s": est["tau_ref_s"],
                                "xi_phase_rad": est["xi_phase_rad"],
                                "sin_theta": est["sin_theta"],
                                "reason": "sin_theta_out_of_range"})
                continue
            sin_th = est["sin_theta"]
            theta = math.asin(max(-1.0, min(1.0, sin_th)))
            # A2：首阵元时延 -> 原点双程时延；r = v(tau_ref - sys)/2 + v_x,ref*sinθ/2（**只加一次**）
            r_origin = (pc.v_prop * (est["tau_ref_s"] - pc.sys_delay_s) / 2.0
                        + pc.vx_ref_m * sin_th / 2.0)
            x = r_origin * sin_th
            y = -r_origin * math.cos(theta)
            cand = {
                "bin_index": int(l),
                "period_bin_index": int(pc.period_bin_index[int(l)]),
                "tau_ref_s": float(est["tau_ref_s"]),
                "sin_theta": float(sin_th),
                "theta_deg": float(math.degrees(theta)),
                "xi": complex(est["xi"]),
                "xi_phase_rad": float(est["xi_phase_rad"]),
                "sin_theta_within_wrap_free": bool(est["sin_theta_within_wrap_free"]),
                "r_origin_m": float(r_origin),
                "x": float(x), "y": float(y),
                "score": float(power[int(l)]),
                "eig_values": est["eig_values"],
                "rank_one_ratio": est["rank_one_ratio"],
            }
            ok, why = _in_domain(pc, x, y, r_origin)
            if ok:
                candidates.append(cand)
            else:
                dropped.append({"bin_index": int(l), "tau_ref_s": cand["tau_ref_s"],
                                "theta_deg": cand["theta_deg"], "x": cand["x"], "y": cand["y"],
                                "r_origin_m": cand["r_origin_m"], "reason": why})
        diag["algorithm_failures"] = algorithm_failures
        n_attempted = int(len(sel))
        if n_attempted > 0 and len(algorithm_failures) == n_attempted and not candidates:
            diag["n_candidates_evaluated"] = int(n_eval)
            diag["wall_seconds"] = time.perf_counter() - t_start
            return _pack([], "failed", diag,
                         "全部被选距离峰均发生 ESPRIT 数值失败（algorithm_failure）；"
                         "沿 runner failed 口径保留分母，允许退化、不强 K。首个原因：%s"
                         % algorithm_failures[0]["reason"])
        nms = _shared_nms([{"x": c["x"], "y": c["y"], "score": c["score"]} for c in candidates],
                          int(pc.k), float(used["nms_min_dist_m"]))
        predictions = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"])}
                       for p in nms["kept"]]
        diag["candidates"] = candidates
        diag["dropped_candidates"] = dropped
        diag["postprocess"] = nms["stats"]
        diag["score_definition"] = ("参考阵元时延谱在该 bin 的**快照平均功率**（原生检测统计量；"
                                    "仅用于共享 NMS/cap 的排序）")
        diag["n_predictions"] = len(predictions)
        diag["n_candidates_evaluated"] = int(n_eval)
        diag["same_range_multitarget_limit"] = (
            "原文式(8) 为单向量外积（rank 1）=> 每个距离峰只承载 1 个角度；同距离多目标**不分离**"
            "（本适配不增强）。若两目标落在同一 range bin，只回 1 个候选")
        diag["snapshot_extension"] = {
            "covariance_extension": str(used["covariance_extension"]),
            "n_snapshots": int(pc.L),
            "note": ("R_m = (1/L) sum_t P_m^(t)(P_m^(t))^H 是式(8) 单快照外积的**统计扩展**（多快照平均）；"
                     "L=1 时逐字回到 P_m P_m^H；无 FB、无对角加载、无平滑；空间维只取首个特征向量"),
        }
        diag["evaluation_boundary"] = "只给 <=K 个原生候选；评测（含真值）由集成层做，本模块不读真值"
        diag["wall_seconds"] = time.perf_counter() - t_start
        if not predictions:
            return _pack(predictions, "ok", diag,
                         "无候选：环内无原生距离峰，或所选峰的候选全部落在共同域外/被共享 NMS 抑制"
                         "（见 dropped_candidates/postprocess）")
        return _pack(predictions, "ok", diag, "")
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
