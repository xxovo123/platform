# -*- coding: utf-8 -*-
"""B1 硬件侧适配实现：Deng, Li, Tang & Zhang 2023（Sensors 23(14):6363）§4.1 式(19)—(28)。

原文结构（**保身份**，不做结构替换）
------------------------------------------------------------------------------
* 式(8)   Z = A(tau,theta)B + V，列 a(tau,theta) = C(theta)[S e_tau]；ULA 阵元间距 d = lambda/2，
          C(theta) = [I; Phi(theta); ...; Phi^{M-1}(theta)]，Phi(theta) = diag(e^{-j w_p d sin(theta)/c})。
* 式(9)   R = Z Z^H / K（K = 频域快照数；本数据 K = 慢时间帧数 L），无 FB、无对角加载。
* 式(19)(20) f(tau,theta) = [S e_tau]^H Q(theta)[S e_tau]，Q(theta) = C(theta)^H U_N U_N^H C(theta)，
          是 P x P（P = 带内频点数）矩阵。
* 式(23)  e(tau) = S e_tau exp(+j w_c0 tau)/S(w_c0)，首元素恒为 1。
* 式(24)—(27) 用约束 e1^H e(tau) = 1（e1 为频点参考位置的单位向量）消掉平凡解 e(tau)=0，
          得 **先 DOA**：theta_hat = arg max_theta e1^H Q(theta)^{-1} e1。
* 式(28)  带回 theta_hat 做**条件一维 TOA**搜索：P1D(tau) = 1/(a(tau,theta_hat)^H U_N U_N^H a(tau,theta_hat))。
          顺序逐字保留：先角度、后时延；不做二维联合搜索、不引入额外配对阶段。

Q(theta) 的奇异性与数值正则（**关键修正，取代旧稿“删除奇异角候选”**）
------------------------------------------------------------------------------
* Q 属 C^{P x P}，P = 17；rank Q = rank(U_N^H C(theta)) <= min(P, F*C - K)。
  本机 F*C = 17*C >= 34、K <= 3，故 **Q 通常可逆**；只有真方向（无噪理想情形）才退化：
  无噪时 C(theta_l)(S o e_{tau_l}) 落在信号子空间 => U_N^H C(theta_l) 至少有一个零奇异方向
  => Q(theta_l) 奇异。因此**接近奇异的 theta 恰恰是真峰所在**，任何候选都不得因“奇异”被删除。
* 实现用固定**相对**正则：Q = sum_i w_i v_i v_i^H（eigh），
      obj(theta) = sum_i |v_i[0]|^2 / max(w_i, q_reg_rel * w_max)，q_reg_rel = 1e-12（冻结默认）。
  极限说明：eps -> 0 时 sum_i |v_i[0]|^2/(w_i+eps) 在且仅在 e1 于 Q 的零空间有分量处发散，
  即真方向处发散；固定 q_reg_rel 只把该发散截成有限值。
  **不声称固定 floor 下峰位置一概不变**：floor 是按 Q 自身的尺度（q_reg_rel * w_max）取的，
  若某窗内 Q 在所有候选上都整体偏小（例如噪声极低的近退化情形），floor 会同时抬高整段谱，
  峰位可能展宽或微移。该性质只按**实测覆盖范围**申报（见 methods/docs/B1_B2_适配记录.md 的
  “已验证范围”一节与 methods/tests/ 的用例），不当作通用保证；理想极限 eps -> 0（无噪）下
  真方向是精确峰。
  这与“把 pinv 当逆”**不同**：pinv 会把零奇异方向整段丢掉（该方向贡献置 0），本实现把该方向
  保留为 1/(q_reg_rel * w_max)（最大项），因此真峰不会被抹掉。
  只有 Q 恰为零矩阵（w_max = 0，无参考尺度）时报告固定饱和值 OBJ_SAT，并在 diagnostics 标注。

本数据下的适配（逐条，公式对应与偏离见 methods/docs/B1_B2_适配记录.md，测试见 methods/tests/）
------------------------------------------------------------------------------
* A1 观测：原文是“发射 -> 多径 -> 单阵列接收”**单向**信道；本数据是 Tx -> 目标 -> Rx 的**被动双程** CIR。
     **只实现原生平面波适配**：绝对虚拟坐标 v_ch = tx + rx，通道相位
          a[f,ch] = S_f * exp(-j2pi f tau) * exp(+j2pi f v_x,ch sin(theta) / v_prop)，tau = 2r / v_prop。
     即 tau_ch ~= (2r - v_x,ch sin(theta)) / v_prop，r 为**原点**双程等价距离（r_hat = v_prop*tau_hat/2）。
     不实现、也不冒充 two_way_exact / r_ref 的“假 exact”变体（请求该 manifold 时返回 not_applicable）。
* A2 符号：原文式(7) 印刷为 e^{-j w d sin(theta)/c}；本坐标（theta = atan2(x,-y)，视轴 -Y）由解析例
     给出**相对**相位 +2pi f (v_x,ch - v_x,ref) sin(theta)/v_prop（反号）。按本坐标实现，不照抄符号。
* A3 ULA 结构：由 case 的 pairs/tx/rx 计算 v_ch，要求 v_x 等间距且 v_y/v_z 全同，才认为原文阵列结构成立；
     否则 not_applicable（并写明物理原因）。
* A4 S 未知：S = 1 是**共同假设**（不是已知真实发射谱），除非 case.model.response_assumption.vector
     或 params['s_response'] 显式给出非平坦 S。角阶段式(19)—(27) 与 S 无关；式(28) 的 TOA 阶段含 S。
* A5 频点顺序：X 为 (L, F*C) **频率优先**，case 的原频率序**非单调**。本模块**不排序**，一律按
     frequencies_hz 原始序逐点构流形（不需要 bare J，也不做额外置换）。
* A6 协方差：原文式(9) 无 FB、无对角加载；case.config.is_forward_backward 属主方法选择，本模块不继承，
     只在 diagnostics 如实记录并标为 not_used。
* A7 旧稿缺陷：**已修** —— 不再删除“奇异角候选”；Q 的奇异/近奇异候选保留并计数。
* A8 时延口径：case.model 的延迟是**绝对到达时延** tau = (|p-tx|+|p-rx|)/v_prop + system_delay_si；
     search.one_dimensional_budget.round_trip_tau_span_s 与 config.distance_ring_m 都只给**传播时间**
     2r/v_prop，故一维搜索域整体加 system_delay 后再与式(23)(28) 的绝对 tau 对齐（输出 r_hat 再减回）。
     否则非零 system_delay 会让可搜域相对距离环整体平移。
* A9 共享后处理：候选去重/cap 直接调用 common/q6_contract.py::postprocess_native_candidates
     （0.10 m、分值降序、确定性、不用真值），本模块**不再**自实现一套规则；公共合同不可用即显式失败。
* A10 S 参考分量：式(23) 用 e1^H e(tau) = 1 归一（e1 在参考频点），故要求 S 的参考分量非零；
     S(reference_frequency_index) = 0 时该约束失效（e(tau) 的参考分量恒为 0），本模块显式拒绝，
     并提示改选 S 非零的频点为参考（params['reference_frequency_index'] —— 只改参考位置，
     不改 X 列序，因此不引入数据置换/bare J）。
* A11 申报：所有实际采用的参数与阈值写入 diagnostics['params_used']；复数（如非平坦 S）
     在 diagnostics 内编码为实/虚数组，保证 JSON 可序列化。

不适用（not_applicable + diagnostics['reason'])
------------------------------------------------------------------------------
* k = 0；F < 2；C < 2；K >= F*C（噪声子空间为空）；
* 平面波适配下 v_ch 不构成沿 x 的等间距直线阵（原文 ULA 结构不成立）；
* params['manifold'] 不是 plane_wave（本模块只实现原生平面波适配）。

失败（failed + diagnostics['reason']）：缺键/形状不符/含 NaN 或 Inf/S 长度不符等显式输入错误。

本模块只依赖 numpy；不读真值、不写任何历史产物；不修改 common/ 与其它方法。
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
#: Q 恰为零矩阵（无参考尺度）时的固定饱和值（有限，便于打分与排序）
OBJ_SAT = 1e30
SHARED_NMS_MIN_DIST_M = 0.10

#: 冻结默认参数（写定，不按结果调整；实际取值一律在 diagnostics['params_used'] 申报）
FROZEN_PARAMS: dict[str, Any] = {
    "manifold": "plane_wave",
    "angle_step_deg": 0.05,
    "angle_grid_points_cap": 200001,
    "tau_grid_points_cap": 400000,
    "q_reg_rel": 1e-12,
    "q_singular_tol_rel": 1e-8,
    "reference_frequency_index": 0,
    "nms_min_dist_m": SHARED_NMS_MIN_DIST_M,
    "s_response_policy": "params_then_case_model_vector_else_unit",
    "s_response": None,
    "peak_edge_policy": "include_edges",
    "domain_tolerance": "half_cell",
    "default_tau_step_distance_m": 0.05,
    #: 显式调用方覆盖（search profile 用）：**距离步（米）**；None = 走 case 一维预算 -> 冻结默认。
    #: 语义说明：显式覆盖优先于 case 的 `suggested_one_d_tau_step_s`（后者是**测角 TDOA**预算，
    #: 用于**测距**的条件一维 TOA 搜索属口径混用，见 analysis_grid5cm/01_对齐方案）。
    "tau_step_distance_m": None,
}

__all__ = [
    "C0",
    "OBJ_SAT",
    "FROZEN_PARAMS",
    "MethodError",
    "NotApplicable",
    "ParsedCase",
    "parse_case",
    "angle_grid_deg",
    "tau_grid",
    "angle_stage_spectrum",
    "conditional_toa_spectrum",
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
    """把 numpy / 复数递归转成 **JSON 可序列化** 形式（diagnostics 落盘用）。"""
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
            return {"kind": "array_summary", "n": int(obj.size), "shape": [int(v) for v in obj.shape],
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
    """**含端点**的确定性局部极大（平台取中点）。

    端点即网格边界：单调升/降到边界的峰是合法边界峰，不得因“严格局部极大不含边界”被丢弃。
    """
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
    """取唯一最好峰；全平坦时退回 argmax 并如实标注（不静默编造峰）。"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise MethodError("一维搜索网格为空")
    v = np.where(np.isfinite(v), v, -np.inf)
    peaks = _peaks_incl_edges(v)
    if not peaks:
        return int(np.argmax(v)), "flat_no_strict_peak_fallback_argmax"
    flat = (float(np.max(v)) == float(np.min(v)))
    if len(peaks) == 1 and flat:
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
    ring_span_m: tuple
    tau_span_s: tuple
    tau_span_propagation_s: tuple
    tau_step_s: float
    S: np.ndarray
    S_source: str
    ref_freq_index: int
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
    for key in ("angle_step_deg", "q_reg_rel", "nms_min_dist_m"):
        if not (float(used[key]) > 0.0):
            raise MethodError("参数 %s 必须为正，收到 %r" % (key, used[key]))
    if not (0.0 < float(used["q_singular_tol_rel"]) < 1.0):
        raise MethodError("q_singular_tol_rel 必须落在 (0,1)")
    used["angle_step_deg"] = float(used["angle_step_deg"])
    used["q_reg_rel"] = float(used["q_reg_rel"])
    used["q_singular_tol_rel"] = float(used["q_singular_tol_rel"])
    used["nms_min_dist_m"] = float(used["nms_min_dist_m"])
    used["reference_frequency_index"] = int(used["reference_frequency_index"])
    used["angle_grid_points_cap"] = int(used["angle_grid_points_cap"])
    used["tau_grid_points_cap"] = int(used["tau_grid_points_cap"])
    used["default_tau_step_distance_m"] = float(used["default_tau_step_distance_m"])
    used["manifold"] = str(used["manifold"])
    if used["s_response"] is not None:
        used["s_response"] = np.asarray(used["s_response"], dtype=np.complex128).reshape(-1)
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


def _ula_layout(vcoord: np.ndarray) -> tuple[np.ndarray, float]:
    """校验原生平面波适配下 v_ch 构成沿 x 的等间距直线阵（原文 ULA），返回 (顺序, 间距)。"""
    order = np.argsort(vcoord[:, 0], kind="stable")
    v = vcoord[order]
    if v.shape[0] < 2:
        raise NotApplicable("Deng 阵列模型需至少 2 个阵列通道（虚通道），收到 %d" % v.shape[0])
    for axis, name in ((1, "y"), (2, "z")):
        span = float(np.ptp(v[:, axis]))
        if span > 1e-9:
            raise NotApplicable(
                "原生平面波适配要求虚通道 %s 坐标一致（原文 ULA 的 Phi(theta) 结构）：实测极差 %.3e m，"
                "不引入额外波前模型就无法套用式(7)(20)" % (name, span))
    dx = np.diff(v[:, 0])
    if not np.all(dx > 0.0):
        raise NotApplicable("虚通道 x 坐标存在重复（阵元间距为 0）：原文 ULA 结构不成立")
    spread = float(np.max(dx) - np.min(dx)) / float(np.max(dx))
    if spread > 1e-6:
        raise NotApplicable(
            "原生平面波适配要求虚通道沿 x **等间距**（原文 Phi^{m-1} 结构）：相对间距散布 %.3e "
            "超出 1e-6，结构不成立" % spread)
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
        raise NotApplicable("式(23)(27) 需要 >=2 个带内频点（P>=2）；本 case F=%d" % F)
    if C < 2:
        raise NotApplicable("Deng 阵列模型需要 >=2 个阵列通道；本 case C=%d" % C)
    if k == 0:
        raise NotApplicable("k=0：式(27) 的角谱需要至少 1 个源，无目标窗不给候选")
    if k >= F * C:
        raise NotApplicable("k=%d >= F*C=%d：噪声子空间为空，式(19) 不可用" % (k, F * C))
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
        span_prop = (float(span[0]), float(span[1]))
        span_src = "search.one_dimensional_budget.round_trip_tau_span_s"
    elif ring_m is not None:
        span_prop = (2.0 * ring_m[0] / v_prop, 2.0 * ring_m[1] / v_prop)
        span_src = "derived_from_config.distance_ring_m"
    else:
        raise NotApplicable(
            "既无 search.one_dimensional_budget.round_trip_tau_span_s，也无 config.distance_ring_m："
            "没有可冻结（且不依赖真值）的一维时延搜索域，拒绝默猜全域")
    if not (span_prop[1] > span_prop[0]):
        raise MethodError("时延搜索域必须严格递增，收到 %r" % (span_prop,))
    # A8 口径统一：搜索域是**绝对到达时延**；span_prop 只是传播时间 2r/v_prop，必须整体加 system_delay。
    tau_span_prop = (span_prop[0], span_prop[1])
    tau_span = (span_prop[0] + sys_delay, span_prop[1] + sys_delay)
    ring_span_m = (0.5 * v_prop * span_prop[0], 0.5 * v_prop * span_prop[1])
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
        raise MethodError("一维时延步长必须为正，收到 %r" % tau_step)

    fov = float(cfg.get("fov_degrees", 180.0) or 180.0)
    vcoord = (tx[pairs[:, 0]] + rx[pairs[:, 1]]).astype(float)
    v_order, spacing = _ula_layout(vcoord)
    S, s_src = _resolve_s_response(case, params if params else {}, F)
    ref_idx = int(used["reference_frequency_index"])
    if not (0 <= ref_idx < F):
        raise MethodError("reference_frequency_index=%d 超界（F=%d）" % (ref_idx, F))
    # A10：式(23) 的 e(tau) 以参考频点归一（e1^H e(tau) = 1）；S(ref) = 0 时该约束失效
    # （e(tau) 的参考分量恒为 0），故显式拒绝，并提示改用 S 非零的参考频点。
    if abs(complex(S[ref_idx])) == 0.0:
        nonzero = [int(i) for i in np.nonzero(np.abs(S) > 0.0)[0]]
        if not nonzero:
            raise MethodError("S 全为零：没有可用的非零参考频点，式(23) 的 e1^H e(tau)=1 归一约束不可用")
        raise MethodError(
            "S 的参考分量 S[frequencies_hz[%d]] = 0：式(23) 以 e1^H e(tau) = 1 归一，参考分量为零时"
            "约束失效。请改用 S 非零的频点作参考：params['reference_frequency_index'] ∈ %s"
            "（只改参考位置，不改 X 列序，不引入数据置换/bare J）" % (ref_idx, nonzero[:8]))

    parse_diag = {
        "manifold": "plane_wave (absolute virtual coordinate v = tx + rx)",
        "n_features": F * C,
        "feature_order": case.get("feature_order"),
        "frequency_order": "raw case order (NOT sorted); index = f*C + ch",
        "reference_frequency_hz": float(freqs[ref_idx]),
        "reference_frequency_index": ref_idx,
        "virtual_x_m": [float(v) for v in vcoord[:, 0]],
        "ula_spacing_m": spacing,
        "ula_channel_order": [int(i) for i in v_order],
        "propagation_speed_m_s": v_prop,
        "propagation_speed_source": v_src,
        "epsilon_r_background": eps_r,
        "system_delay_si": sys_delay,
        "tau_span_s": [tau_span[0], tau_span[1]],
        "tau_span_propagation_s": [tau_span_prop[0], tau_span_prop[1]],
        "tau_span_absolute_note": ("搜索域是**绝对到达时延** tau = 2r/v_prop + system_delay_si；"
                                   "round_trip_tau_span_s 与 distance_ring_m 只给传播时间，"
                                   "已统一加上 system_delay 后再搜索"),
        "tau_span_source": span_src,
        "ring_span_m": [ring_span_m[0], ring_span_m[1]],
        "domain_policy": ("共同域用**严格** Cartesian 边界（仅 DOMAIN_EPS 数值容差，不做半格外延），"
                          "另检 y<0、r>0 与距离环；网格边界上的合法峰由含端点取峰器保留"),
        "tau_step_s": tau_step,
        "tau_step_source": step_src,
        "distance_ring_m": None if ring_m is None else [ring_m[0], ring_m[1]],
        "fov_degrees": fov,
        "s_response_source": s_src,
        "s_response_is_flat": bool(np.allclose(S, 1.0)),
        "s_response_n_zero_entries": int(np.count_nonzero(np.abs(S) == 0.0)),
        "s_reference_abs": float(abs(complex(S[ref_idx]))),
        "s_reference_nonzero_required": True,
        "s_response_note": ("S = 1 是**共同纯相位假设**，不是已知真实发射谱；"
                            "角阶段式(19)—(27) 与 S 无关，式(28) 的 TOA 阶段含 S"),
        "case_config_is_forward_backward": bool(cfg.get("is_forward_backward", False)),
        "forward_backward_used": False,
        "forward_backward_note": "原文式(9) 无前后向平均；case.config.is_forward_backward 属主方法选择，本模块不继承",
        "diag_load_used": 0.0,
        "channel_phase_calibration": "X 内已含数据侧 exp(-1j*phi) 校准；本模块不重复施加",
    }
    pc = ParsedCase(
        case_id=str(case.get("id", "?")), X=X, freqs=freqs, pairs=pairs, vcoord=vcoord,
        v_order=v_order, spacing_m=spacing, v_prop=v_prop, sys_delay_s=sys_delay, k=k,
        L=int(X.shape[0]), F=F, C=C, grid_x=grid_x, grid_y=grid_y, fov_deg=fov,
        ring_m=ring_m, ring_span_m=ring_span_m, tau_span_s=tau_span,
        tau_span_propagation_s=tau_span_prop, tau_step_s=tau_step, S=S, S_source=s_src,
        ref_freq_index=ref_idx, config_fb_flag=bool(cfg.get("is_forward_backward", False)),
        cfg_keys_used={"propagation_speed_m_s": v_src, "system_delay_si": "config",
                       "distance_ring_m": ring_m is not None,
                       "fov_degrees": "config" if "fov_degrees" in cfg else "default_180"})
    return pc, used, parse_diag


# --------------------------------------------------------------------------- #
# 式(9)(19)—(28)
# --------------------------------------------------------------------------- #
def _covariance_noise_subspace(pc: ParsedCase, diag: dict | None = None) -> np.ndarray:
    """式(9)：R = Z Z^H / L（Z = X^T），eigh -> 噪声子空间 U_N（(F*C) x (F*C-K)）。"""
    R = (pc.X.T @ pc.X.conj()) / float(pc.L)
    trace = float(np.real(np.trace(R)))
    if not math.isfinite(trace) or trace <= 0.0:
        raise MethodError("输入能量为零/非有限：R 的迹 = %r" % trace)
    w, V = np.linalg.eigh(0.5 * (R + R.conj().T))
    m = pc.F * pc.C
    U_N = V[:, : m - pc.k]
    if diag is not None:
        diag["covariance"] = {
            "shape": [int(m), int(m)],
            "trace": trace,
            "snapshots_L": int(pc.L),
            "source_count_K": int(pc.k),
            "noise_subspace_dim": int(m - pc.k),
            "diag_load_factor_used": 0.0,
            "eig_smallest": float(w[0]),
            "eig_largest": float(w[-1]),
            "rank_bound_note": "R 秩 <= min(F*C, L)；Q(theta) 秩 <= min(F, F*C-K)",
        }
    return U_N


def _channel_phase(pc: ParsedCase, theta_rad: float) -> np.ndarray:
    """原生平面波通道相位 exp(+j2pi f v_x,ch sin(theta)/v_prop)：形状 (C, F)，绝对虚拟坐标。"""
    s = math.sin(theta_rad)
    arg = (2.0 * math.pi / pc.v_prop) * np.outer(pc.vcoord[:, 0] * s, pc.freqs)
    return np.exp(1j * arg)


def _angle_manifold(pc: ParsedCase, chan_phase: np.ndarray) -> np.ndarray:
    """式(20) 的 C(theta)：((F*C) x F)，行 (f,c)、列 f；只含角相位（S 与 tau 由式(23) 的 e(tau) 承担）。"""
    C = pc.C
    F = pc.F
    mat = np.zeros((F * C, F), dtype=np.complex128)
    rows = (np.arange(F)[None, :] * C) + np.arange(C)[:, None]
    cols = np.tile(np.arange(F)[None, :], (C, 1))
    mat[rows, cols] = chan_phase
    return mat


def angle_stage_spectrum(pc: ParsedCase, U_N: np.ndarray | None = None,
                         theta_deg_grid: np.ndarray | None = None,
                         *, q_reg_rel: float | None = None) -> dict:
    """式(19)—(27)：角阶段谱 obj(theta) = e1^H Q(theta)^{-1} e1（固定相对正则，见模块 docstring）。

    返回 theta_deg / values / q_min_eig / q_max_eig / is_near_singular / n_near_singular。
    近奇异（含真峰）候选只计数、**不删除**。
    """
    if U_N is None:
        U_N = _covariance_noise_subspace(pc)
    if theta_deg_grid is None:
        theta_deg_grid = angle_grid_deg(pc)
    theta = np.asarray(theta_deg_grid, dtype=float).reshape(-1)
    reg = float(FROZEN_PARAMS["q_reg_rel"] if q_reg_rel is None else q_reg_rel)
    tol = float(FROZEN_PARAMS["q_singular_tol_rel"])
    ref = int(pc.ref_freq_index)
    UNH = U_N.conj().T
    values = np.empty(theta.size, dtype=float)
    qmin = np.empty(theta.size, dtype=float)
    qmax = np.empty(theta.size, dtype=float)
    n_sat = 0
    for i, th_deg in enumerate(theta):
        mat = _angle_manifold(pc, _channel_phase(pc, math.radians(float(th_deg))))
        B = UNH @ mat
        Q = B.conj().T @ B
        Q = 0.5 * (Q + Q.conj().T)
        w, V = np.linalg.eigh(Q)
        w = np.real(w)
        w_max = float(w[-1]) if w.size else 0.0
        qmin[i] = float(w[0]) if w.size else 0.0
        qmax[i] = w_max
        weights = np.abs(V[ref, :]) ** 2
        if w_max > 0.0:
            weff = np.maximum(w, reg * w_max)
            obj = float(np.sum(weights / weff))
        else:
            obj = OBJ_SAT
            n_sat += 1
        values[i] = min(obj, OBJ_SAT)
    near = qmin <= tol * np.maximum(qmax, TINY)
    return {
        "theta_deg": theta,
        "values": values,
        "q_min_eig": qmin,
        "q_max_eig": qmax,
        "is_near_singular": near,
        "n_near_singular": int(np.count_nonzero(near)),
        "n_saturated_exact_zero_Q": int(n_sat),
        "q_reg_rel": reg,
        "q_singular_tol_rel": tol,
        "objective": "e1^H Q(theta)^-1 e1 with fixed relative eigen floor (no candidate deleted)",
    }


def conditional_toa_spectrum(pc: ParsedCase, theta_deg: float, U_N: np.ndarray | None = None,
                             tau_grid_s: np.ndarray | None = None) -> dict:
    """式(28)：固定 theta_hat 后的条件一维 TOA 谱 P1D(tau) = 1/||U_N^H a(tau,theta_hat)||^2。"""
    if U_N is None:
        U_N = _covariance_noise_subspace(pc)
    if tau_grid_s is None:
        tau_grid_s = tau_grid(pc)
    tau = np.asarray(tau_grid_s, dtype=float).reshape(-1)
    chan_phase = _channel_phase(pc, math.radians(float(theta_deg)))
    eta = np.exp(-2j * math.pi * np.outer(tau, pc.freqs))
    # (T,F,1) * (1,F,C) -> (T,F,C)，再直接 reshape 成 (T, F*C)，行序 f*C+c（不做 transpose）
    blocks = (pc.S[None, :] * eta)[:, :, None] * chan_phase.T[None, :, :]
    A = blocks.reshape(tau.size, pc.F * pc.C)
    proj = A @ U_N.conj()
    denom = np.sum(np.abs(proj) ** 2, axis=1)
    values = 1.0 / np.maximum(denom, TINY)
    return {"tau_s": tau, "values": values, "theta_deg": float(theta_deg),
            "manifold": "a(tau,theta) = C(theta)[S o exp(-j2pi f tau)]"}


def angle_grid_deg(pc: ParsedCase, *, angle_step_deg: float | None = None) -> np.ndarray:
    """角度搜索域：候选 Cartesian 网格包络 交 FoV/2 交 90 度（冻结，不依赖真值）。

    **端点生成规则（2026-09-24 起固定为 ceil）**：`n = ceil(2·half/Δθ) + 1`，端点对称 ±half
    且含端点 ⇒ **实际步长 2·half/(n−1) <= Δθ**（保证“不超过请求步长”）。
    原 `round` 规则在 half = 75.9638°、Δθ = 0.6° 时给出 n = 254 ⇒ 实际 0.600504° > 0.6°（超请求），
    故改为 ceil；对本批默认 Δθ = 0.05° 该改动给出同一 n = 3040（网格逐位不变，旧结果仍可复现）。
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


def tau_grid(pc: ParsedCase, *, tau_step_s: float | None = None) -> np.ndarray:
    """一维时延搜索域：round_trip_tau_span_s（含端点），步长冻结。"""
    step = float(pc.tau_step_s if tau_step_s is None else tau_step_s)
    lo, hi = float(pc.tau_span_s[0]), float(pc.tau_span_s[1])
    n = int(math.ceil((hi - lo) / step))
    cap = int(FROZEN_PARAMS["tau_grid_points_cap"])
    if n + 1 > cap:
        raise MethodError("时延网格点数 %d 超过冻结上限 %d；请显式放宽步长" % (n + 1, cap))
    return np.linspace(lo, hi, n + 1)


#: 共同域边界只允许**数值**容差（远小于任何分辨率）：不做半格外延，
#: 因为外延会悄悄放大共同域定义、改变与其它方法的比较口径。
DOMAIN_EPS = 1e-9


def _in_domain(pc: ParsedCase, x: float, y: float, r_m: float) -> tuple[bool, str]:
    """共同域过滤：**严格** Cartesian 边界（仅 DOMAIN_EPS 数值容差）+ 半平面 + 距离环 + 正距离。

    网格**边界上**的合法峰由含端点的取峰器保留（见 _peaks_incl_edges）；本函数只判候选是否落在
    共同域内，不用半格外延改动共同域的定义。
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
    """加载公共合同里的共享 NMS（仅加载一次）。**不复制规则**：公共合同缺失即显式失败。"""
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
    diag.setdefault("grid_kind", "angle_1d_grid_then_conditional_tau_1d_grid")
    diag.setdefault("n_candidates_evaluated", 0)
    diag.setdefault("dictionary_size", None)
    diag.setdefault("dictionary_size_note",
                    "B1 无稀疏字典：式(27)(28) 为两段一维网格搜索（先角度、后条件时延）")
    return {"predictions": list(predictions), "status": str(status), "diagnostics": _py(diag)}


def estimate(case: dict, params: dict | None = None) -> dict:
    """B1 公共接口：estimate(case[, params]) -> {predictions, status, diagnostics}。

    predictions = [{'x','y','score'}]（<= K，原生两段式候选）；score = 末级一维 TOA 谱峰值（原值）。
    本函数不读真值；失败/不适用一律显式返回并给 reason。
    """
    t_start = time.perf_counter()
    diag: dict = {
        "method": "B1",
        "source": "Deng, Li, Tang & Zhang 2023, Sensors 23(14):6363",
        "doi": "10.3390/s23146363",
        "implemented_equations": "(8)(9)(19)(20)(23)(27)(28)",
        "structure": "先 DOA（式27）-> 固定 theta_hat 的条件一维 TOA（式28）；无二维联合搜索",
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

    try:
        U_N = _covariance_noise_subspace(pc, diag)
        theta_grid = angle_grid_deg(pc, angle_step_deg=float(used["angle_step_deg"]))
        spec = angle_stage_spectrum(pc, U_N, theta_grid, q_reg_rel=float(used["q_reg_rel"]))
        diag["angle_stage"] = {
            "grid_kind": "1d_angle_grid_deg",
            "n_points": int(theta_grid.size),
            "start_deg": float(theta_grid[0]),
            "stop_deg": float(theta_grid[-1]),
            "step_deg": float(used["angle_step_deg"]),
            "endpoint_rule": ("n = ceil(2*half/step)+1，端点对称 ±half 且含端点 ⇒ 实际步长 "
                              "2*half/(n-1) <= step（grid5cm 对齐要求“不超过请求步长”）"),
            "actual_step_deg": float(theta_grid[1] - theta_grid[0]),
            "n_near_singular_candidates": int(spec["n_near_singular"]),
            "n_saturated_exact_zero_Q": int(spec["n_saturated_exact_zero_Q"]),
            "near_singular_policy": "保留并计数（真峰即在近奇异处）；仅当 Q 恰为零矩阵时用固定饱和值",
            "q_reg_rel": float(spec["q_reg_rel"]),
            "q_singular_tol_rel": float(spec["q_singular_tol_rel"]),
            "objective": spec["objective"],
            "s_independent": True,
        }
        sel, peak_note = _top_peaks(spec["values"], pc.k)
        diag["angle_stage"]["peak_policy"] = peak_note
        diag["angle_stage"]["n_angle_peaks_found"] = int(len(_peaks_incl_edges(spec["values"])))
        diag["angle_stage"]["selected_angle_deg"] = [float(theta_grid[i]) for i in sel]
        diag["angle_stage"]["selected_objective"] = [float(spec["values"][i]) for i in sel]
        diag["angle_stage"]["selected_q_min_eig"] = [float(spec["q_min_eig"][i]) for i in sel]
        diag["angle_stage"]["selected_q_max_eig"] = [float(spec["q_max_eig"][i]) for i in sel]
        diag["angle_stage"]["selected_is_near_singular"] = [bool(spec["is_near_singular"][i]) for i in sel]

        tg = tau_grid(pc)
        diag["toa_stage"] = {
            "grid_kind": "1d_tau_grid_s",
            "n_points_per_angle": int(tg.size),
            "span_s": [float(tg[0]), float(tg[-1])],
            "step_s": float(pc.tau_step_s),
            "step_distance_m": float(pc.v_prop * pc.tau_step_s / 2.0),
            "endpoint_rule": ("linspace(lo, hi, ceil((hi-lo)/step)+1)，含端点（lo/hi = 距离环端点 "
                              "+ system_delay 平移到绝对时延）⇒ 实际步长 <= 请求步长"),
            "tau_to_range": "r_hat = v_prop * (tau_hat - system_delay) / 2（原点双程等价距离）",
            "uses_s_response": True,
        }
        paths, n_candidates = [], int(theta_grid.size)
        for i in sel:
            theta_hat = float(theta_grid[i])
            toa = conditional_toa_spectrum(pc, theta_hat, U_N, tg)
            n_candidates += int(tg.size)
            j, toa_note = _best_peak(toa["values"])
            tau_hat = float(tg[j])
            r_hat = float(pc.v_prop * (tau_hat - pc.sys_delay_s) / 2.0)
            x = r_hat * math.sin(math.radians(theta_hat))
            y = -r_hat * math.cos(math.radians(theta_hat))
            paths.append({
                "angle_objective": float(spec["values"][i]),
                "theta_hat_deg": theta_hat,
                "q_min_eig": float(spec["q_min_eig"][i]),
                "q_max_eig": float(spec["q_max_eig"][i]),
                "near_singular": bool(spec["is_near_singular"][i]),
                "tau_hat_s": tau_hat,
                "r_hat_m": r_hat,
                "toa_peak_value": float(toa["values"][j]),
                "toa_peak_policy": toa_note,
                "x": float(x), "y": float(y),
            })
        kept, dropped = [], []
        for idx, p in enumerate(paths):
            ok, why = _in_domain(pc, p["x"], p["y"], p["r_hat_m"])
            if ok:
                kept.append(dict(p, path_index=idx))
            else:
                dropped.append({"path_index": idx, "reason": why, "x": p["x"], "y": p["y"],
                                "r_hat_m": p["r_hat_m"], "theta_hat_deg": p["theta_hat_deg"]})
        nms = _shared_nms([{"x": p["x"], "y": p["y"], "score": p["toa_peak_value"]}
                           for p in kept], pc.k, float(used["nms_min_dist_m"]))
        predictions = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"])}
                       for p in nms["kept"]]
        diag["paths"] = kept
        diag["dropped_paths"] = dropped
        diag["postprocess"] = nms["stats"]
        diag["score_definition"] = "式(28) 条件一维 TOA 谱在峰处的原值（未归一；排序只用于共享 NMS/cap）"
        diag["n_predictions"] = len(predictions)
        diag["n_candidates_evaluated"] = int(n_candidates)
        diag["evaluation_boundary"] = "只给 <=K 个原生候选；评测（含真值）由集成层做，本模块不读真值"
        diag["wall_seconds"] = time.perf_counter() - t_start
        if not predictions:
            return _pack(predictions, "ok", diag,
                         "无候选：角度峰全部落在共同域外或被共享 NMS 抑制（见 dropped_paths/postprocess）")
        return _pack(predictions, "ok", diag, "")
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
