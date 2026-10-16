#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B5 硬件侧适配：Wang 2014 §3.1 传统 full-2D TOA–DOA MUSIC 的**四通道 CIR 扩展**
（多阵元结构参考 Deng 2023 §3.1 式(6)—(10)）。

身份与边界（诚实声明，不得放宽）
================================
* **不是** Wang 2014 原法的逐字复现。原文 §3.1（标题 TOA Estimation Based on 2D-MUSIC
  Algorithm）是**双天线**联合 (τ, ζ) 二维谱（Wang 式(7)—(15)）；§3.2 起才是 successive
  逐次一维。**旧 B3** 对应 successive（两 Rx），**与本模块不是同一项**。
* 本模块 = **Wang §3.1 的两参数联合二维谱准则** ＋ **Deng §3.1 的多阵元时延矩阵结构**
  （M 阵元 ULA、频变空间相位 Phi^{i-1}(theta)），在本硬件 CIR 数据上做 **4 虚拟通道平面波
  (τ, θ) 二维** 适配。论文侧同时引用 Wang §3.1 与 Deng §3.1，**不称 Wang 原法原样复现**。
* 导向矢量**只含平面波项**（原点双程时延 τ ＋ 通道线性相位），**绝不**使用本文主方法的
  exact Tx→target→Rx 双程球面几何流形；因此它是**传统对照**，**不是** B0（同流形校验）。

原文公式（本地 PDF 原页已核对：Wang_p01 封面、Wang_p04 = 期刊 p.2448、Wang_p05 = p.2449；
Deng_p04/p05 = Sensors p.4/p.5）
-----------------------------------------------------------------------------------------
Wang, Zhang & Wang, "Joint Estimation of TOA and DOA in IR-UWB System Using a Successive
MUSIC Algorithm", **Wireless Personal Communications (2014) 77:2445-2464**,
DOI 10.1007/s11277-014-1644-z (Springer)。§3.1::

    S in C^{N x N}：对角阵，元素 = 发射信号频率采样 S(w_n)
    E_tau = [e_1, ..., e_L] in C^{N x L}：**频率 x 路径**时延矩阵（不是 diag），
            列 e_l = [1, e^{-j dw tau_l}, ..., e^{-j (N-1) dw tau_l}]^T
    式(7)(8)  Y_1 = S E_tau B + W_1，Y_2 = S E_zeta B + W_2
    式(13)    Z = [Y_1; Y_2] = [S E_tau; S E_zeta] B + [W_1; W_2] in C^{2N x K}  <- 通道优先堆叠
    式(14)    Z = A(tau, zeta) B + W，A(tau,zeta) = [S E_tau; S E_zeta] in C^{2N x L}
    式(15)    R_hat = Z Z^H / K，EVD -> P_2D-MUSIC(tau,zeta) = 1 / ( a^H U_N U_N^H a )
    式(11)    Delta_tau_l = zeta_l - tau_l = d sin(theta_l) / c
    式(12)    theta_hat_l = arcsin(Delta_tau_l c / d)

Deng 2023（Sensors 23(14):6363）§3.1::

    式(6)(7) M 阵元：e_{tau_i l} = Phi^{i-1}(theta_l) e_{tau_1 l}，
             Phi(theta) = diag(e^{-j w_n d sin(theta) / c})
    式(8)    Z = [Y_1; ...; Y_M] = [S F_tau1; ...; S F_tauM] B + V = A(tau,theta) B + V
    式(9)    R_ZZ = Z Z^H / K
    式(10)   P_2D-MUSIC(tau,theta) = 1 / ( a^H U_N U_N^H a )  <- **(tau,theta) 真二维联合搜索**

本数据上的适配（逐条；偏离与理由一律申报在 diagnostics 与 methods/docs/B5_*.md）
----------------------------------------------------------------------------------
* **A1 观测接口**：原文是"发射 -> 多径 -> 阵列接收"的单向信道；本数据是 Tx -> 目标 -> Rx 的
  双程 CIR（**非被动雷达**）。本模块与 B1 用**同一平面波适配**：绝对虚拟坐标
  v_ch = tx[pair0] + rx[pair1]，导向
      a[f, ch] = S_f * exp(-j2pi f tau) * exp(+j2pi f v_x,ch sin(theta) / v_prop)
  即通道等效时延 tau_ch ≈ tau - v_x,ch sin(theta) / v_prop；tau 是**原点（v_x = 0, v_y = v_z = 0
  的坐标原点）双程等价到达时延**（含 system_delay）。**不实现、不冒充 exact two-way / r_ref 变体。**
* **A2 符号**：原文式(7) 印刷为 e^{-j w d sin(theta) / c}；本坐标（theta = atan2(x, -y)，
  视轴 -Y，theta > 0 为 +X 侧）由解析例给出**相对**相位 +2pi f (v_x,ch - v_x,ref) sin(theta)/v_prop
  （反号）。按本坐标实现，不照抄符号（与 B1 A2 同）。
* **A3 阵结构**：虚通道须沿 x **等间距**且 v_y / v_z **通道间全同**（原文 Phi^{m-1} 结构），
  否则 not_applicable（写明物理原因）。本批 4 通道：v_x = [-0.038, -0.019, 0, 0.019] m、
  v_y = 0、v_z = -0.02 m。
* **A4 v_y / v_z 共同分量的正确处理（不得含糊）**：
  远场一阶项为 u_theta · v_ch，u_theta = (sin(theta), -cos(theta), 0)（目标在 z = 0 平面）。
    - **v_z 共同分量**：对 u_theta·v_ch 的贡献恒为 0（因为 u_z = 0），即它**不进入**平面波
      一阶模型；本模块**不声称它被吸收进 tau**。它只经由二阶项影响精确双程几何，属已披露的
      平面波近似残差（见 A11）。
    - **v_y 共同分量**：u_y = -cos(theta) != 0，故非零共同 v_y **会**进入一阶项；本模块**未**实现
      该 u_y 项，因此对 **|v_y| > 1e-9 m 的窗显式 not_applicable**（不静默忽略）。本批 v_y = 0。
* **A5 频率顺序**：case X 为 (L, F*C) **频率优先**（index = f*C + ch）。本模块**不排序**，
  一律按 frequencies_hz 原始序（本批非单调）逐点构流形；bare J / 额外置换一律不做。
* **A6 堆叠置换与共轭（不得共轭错）**：原文式(13)(8) 是**通道优先**堆叠；本 case 是**频率优先**。
  两者只差一个固定行置换 P：Z_channel_major = P · Z_freq_major，Z_freq_major = X^T。
  故 R_hat = Z Z^H / K 在 case 口径下就是 R_hat = X^T X^* / L（X.T @ X.conj()），
  导向亦按 f*C + ch 排列 => 谱对 P 不变。
  **共轭口径（正式口径）**：设 a 为式(14)(8) 的导向**列**向量，则式(15)(10) 的谱分母是
      denom = a^H U_N U_N^H a = ||U_N^H a||^2，
  实现上用行向量 A 行 = a^T，取 proj = A @ U_N.conj()（其第 n 行 = (U_N^H a_n)^T，
  分量 (U_N^H a_n)_j = conj(a_n^H u_j)，u_j 为 U_N 第 j 列）。**写成 A @ U_N 即共轭错**
  （得到 ||U_N^T a||^2），由 methods/tests/test_b5_wang_full2d.py 的"共轭守卫"用例抓。
  独立验证必须用**标量**形式 conj(a) @ (U_N U_N^H) @ a。**注意**：main-style 的
  "Re Σ (A P) conj(A)"（A 行 = a^T）等于 ||U_N^T a||^2，与论文口径 ||U_N^H a||^2 在**同一个
  U_N** 下**一般不等价**（只在 U_N 为实数等特殊情形相同），故两种口径不得混用、也不得互相折算。
* **A7 协方差（只实现原生口径）**：原文式(15)(9) 为 R_hat = Z Z^H / K，**无前后向平均（FB）、
  无对角加载**。case 的 config.is_forward_backward = True 属**主方法选择**，本模块**不继承、
  不偷偷添加**、也**不提供** FB/对角加载开关（FB 专项已搁置），diagnostics 如实标注 fb_used = False。
* **A8 谱与取峰**：P(τ,θ) = 1/(a^H U_N U_N^H a)、噪声子空间维 M - K（M = F*C）。
  本模块把同一二维泛函**映射到公共 Cartesian 网格**上求值（"排除采样差异"，见 A9），
  返回**未归一**的谱 1/(denom + 1e-12)，并由**同一冻结取峰规则**在 estimate() 内取自身峰
  （与主方法计时口径一致：3x3 非严格极大 + 8 连通平台去重 + 0.30 线性相对阈值 +
  0.10 m 确定性 NMS + cap = K，来自 knownk_protocol.json 的 peaks 段）。
  评测仍走 **spectrum 路径**（诊断 eval_path='spectrum' + 显式 spectrum_grid），
  由运行器/评测器用同一规则重新取峰。
* **A9 二维搜索域与映射**：二维参数域是 (τ, θ)。公共 xy 网格 (x, y)（y < 0）与 (τ, θ)
  **在域内一一对应**：r = hypot(x,y)、theta = atan2(x, -y)、tau = 2r/v_prop + system_delay；
  反解 r = v_prop(tau - system_delay)/2、x = r sin(theta)、y = -r cos(theta)。故在公共 xy 网格上
  逐点求值**就是** (τ,θ) 二维搜索（同一二维域换采样），并天然复用公共域/取峰口径。
  **正式评估网格 = case 的公共 Cartesian 网格（grid_x/grid_y）**；FROZEN_PARAMS 里的
  tau_step_distance_m / angle_step_deg **只用于**导出函数 full2d_spectrum_native（原生矩形
  (τ,θ) 网格，仅供自测与公式核验），**不参与** estimate()，因此**不得**把本模块描述成
  "在均匀 τ/θ 网格上跑正式结果"。diagnostics 有 official_grid / native_selftest_grid 两栏区分。
* **A10 S（发射/共同响应谱）**：原文 S 是**发射谱对角阵**（元素 = 发射信号频率采样）。
  本数据的 CIR 里**并不已知**真实发射谱，故默认 **S = 1 是"通道间共同响应"假设**
  （不是"已知平坦真实发射谱"，也不是"已校准响应"）；该假设同时影响四通道（共同因子），
  并由 diagnostics 的 s_response_source 如实申报。**S 允许含零元素**（Deng 式(8) 本就只在
  非零带内频点构造 Z）：零元素只把该频点行整行置零，不改变其余结构。
  注意：B1 的输入解析含一条**降维（§4）专属**约束 S[reference_frequency_index] != 0；
  该约束**不是** full-2D 的要求。本模块复用 B1 解析时**自动**选一个 S 非零的频点作参考索引，
  并在 diagnostics 申报该索引**只用于通过 B1 解析、不参与 B5 的 full-2D 运算**（B1 代码不改）。
* **A11 计时**：estimate() 从 case['X'] 起算（协方差 -> EVD -> 导向 -> 谱 -> **自身取峰**），
  **不含**解码/重放/裁剪/FFT 构造（与主方法同口径）。
  平面波近似残差（一阶 u·v 之外的二阶项、以及 v_z 经精确几何的影响）**未**在计时内估计，
  只在 docs 中作披露，不构成任何性能声明。

不适用（not_applicable + diagnostics['reason']）
------------------------------------------------
* k = 0；F < 2；C < 2；K >= M = F*C（噪声子空间为空）；
* 虚通道不构成沿 x 等间距直线阵（v_y / v_z 通道间不恒定，或 x 间距不等）；
* **|v_y| 共同分量 > 1e-9 m**（本模块未实现 u_y 项，显式拒绝而不是静默忽略）；
* S 全零（无任何非零频点可用）。
失败（failed + diagnostics['reason']）：缺键 / 形状不符 / 含 NaN 或 Inf / S 长度不符等**显式输入错误**。

本模块只依赖 numpy，另按路径复用 b1_deng.parse_case 作**输入解析**（不调用其角度 1-D 或
条件 TOA 估计路径），并按路径复用 knownk_eval.peaks.nms_peaks 作**自身取峰**（与主方法同一规则）。
不读真值、不写历史产物、不改 common/ 与其它方法。
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

C0 = 299792458.0
TINY = 1e-300
#: v_y 共同分量的显式拒绝阈值（米）；本批 v_y = 0
VY_TOL_M = 1e-9

WS = Path(__file__).resolve().parents[4]
P0B_DRIVER = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "driver"
PROTOCOL_PATH = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "protocol" / "knownk_protocol.json"
if str(P0B_DRIVER) not in sys.path:
    sys.path.insert(0, str(P0B_DRIVER))

#: 冻结默认参数（写定，不按结果调整；实际取值一律在 diagnostics['params_used'] 申报）
#: **只影响导出函数 full2d_spectrum_native（native 自测/核验用矩形 (τ,θ) 网格）**，
#: 不进入 estimate() 的正式谱（正式谱在公共 Cartesian 网格求值）。取值与 B1/B2 的 grid5cm
#: 剖面对齐量一致（τ 步 = 2*0.05/v_prop、θ 步 = 0.6 deg），便于自测在同预算下核对。
FROZEN_PARAMS: dict[str, Any] = {
    "native_selftest_tau_step_distance_m": 0.05,
    "native_selftest_angle_step_deg": 0.6,
    "tau_grid_points_cap": 400000,
    "angle_grid_points_cap": 200001,
    "s_response": None,
}

B1_MODULE_NAME = "q6_b1_deng_reused_by_b5"
_B1_CACHE: dict[str, Any] = {}
_PEAK_CACHE: dict[str, Any] = {}


class MethodError(ValueError):
    """输入/实现层面的显式失败（调用方须记为 failed，并保留在指标分母）。"""


class NotApplicable(MethodError):
    """物理/结构原因导致该窗不可评估（记为 not_applicable，不计入指标分母）。"""


__all__ = [
    "C0",
    "VY_TOL_M",
    "FROZEN_PARAMS",
    "MethodError",
    "NotApplicable",
    "parse_case",
    "plane_wave_manifold",
    "noise_subspace",
    "full2d_spectrum_native",
    "spectrum_on_common_grid",
    "common_grid_mask",
    "estimate",
]


# --------------------------------------------------------------------------- #
# 复用：B1 输入解析（仅解析）与公共冻结取峰
# --------------------------------------------------------------------------- #
def _b1_module():
    """按文件路径装载 b1_deng（注册 sys.modules 以支持其 @dataclass），并缓存 sha256。"""
    if "mod" not in _B1_CACHE:
        path = Path(__file__).resolve().parent / "b1_deng.py"
        if not path.is_file():
            raise MethodError("找不到公共输入解析模块 %s（B5 复用其 parse_case；缺失即显式失败）" % path)
        spec = importlib.util.spec_from_file_location(B1_MODULE_NAME, str(path))
        if spec is None or spec.loader is None:
            raise MethodError("无法为 %s 建立模块 spec" % path)
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(B1_MODULE_NAME)
        sys.modules[B1_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if previous is None:
                sys.modules.pop(B1_MODULE_NAME, None)
            else:
                sys.modules[B1_MODULE_NAME] = previous
            raise
        _B1_CACHE.update({"mod": module, "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return _B1_CACHE["mod"], _B1_CACHE["sha256"], _B1_CACHE["path"]


def _peak_rule():
    """公共冻结取峰（knownk_protocol.json 的 peaks 段）——与主方法同一规则，仅加载一次。"""
    if "mod" not in _PEAK_CACHE:
        if not PROTOCOL_PATH.is_file():
            raise MethodError("找不到取峰协议 %s" % PROTOCOL_PATH)
        try:
            from knownk_eval import load_protocol          # noqa: PLC0415
            from knownk_eval import peaks as knownk_peaks  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise MethodError("公共冻结取峰不可用（%s: %s）：B5 拒绝私自实现一套取峰规则"
                              % (type(exc).__name__, exc)) from None
        proto = load_protocol(str(PROTOCOL_PATH))
        _PEAK_CACHE.update({"fn": knownk_peaks.nms_peaks, "rule": proto.peaks,
                            "protocol_path": str(PROTOCOL_PATH),
                            "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()})
    return _PEAK_CACHE["fn"], _PEAK_CACHE["rule"], _PEAK_CACHE["protocol_path"], _PEAK_CACHE["protocol_sha256"]


def _resolve_params(params: dict | None) -> dict:
    used = {key: value for key, value in FROZEN_PARAMS.items()}
    if params:
        for key, value in params.items():
            if key not in FROZEN_PARAMS:
                raise MethodError("未知参数 %r（本模块参数冻结，见 FROZEN_PARAMS）" % key)
            used[key] = value
    for key in ("native_selftest_tau_step_distance_m", "native_selftest_angle_step_deg"):
        if not (float(used[key]) > 0.0):
            raise MethodError("参数 %s 必须为正，收到 %r" % (key, used[key]))
    used["native_selftest_tau_step_distance_m"] = float(used["native_selftest_tau_step_distance_m"])
    used["native_selftest_angle_step_deg"] = float(used["native_selftest_angle_step_deg"])
    used["tau_grid_points_cap"] = int(used["tau_grid_points_cap"])
    used["angle_grid_points_cap"] = int(used["angle_grid_points_cap"])
    if used["s_response"] is not None:
        used["s_response"] = np.asarray(used["s_response"], dtype=np.complex128).reshape(-1)
    return used


def _resolve_s_vector(case: dict, params: dict | None, F: int) -> tuple[np.ndarray, str]:
    """S 解析（与 B1 同优先级：params > case.model.response_assumption.vector > S = 1 假设）。

    与 B1 的差别：**允许 S 含零元素**（full-2D 没有 e1 归一约束），只要不全零即可。
    """
    if params and params.get("s_response") is not None:
        S = np.asarray(params["s_response"], dtype=np.complex128).reshape(-1)
        src = "caller_params.s_response"
    else:
        model = case.get("model") or {}
        resp = (model.get("response_assumption") or {}) if isinstance(model, dict) else {}
        vec = resp.get("vector") if isinstance(resp, dict) else None
        if vec is not None:
            S = np.asarray(vec, dtype=np.complex128).reshape(-1)
            src = "case.model.response_assumption.vector"
        else:
            S = np.ones(F, dtype=np.complex128)
            src = "assumption_S_equals_1(shared_channel_response)"
    if S.size != F:
        raise MethodError("S 长度 %d != F=%d" % (S.size, F))
    if not np.all(np.isfinite(S)):
        raise MethodError("S 含 NaN/Inf")
    if not np.any(np.abs(S) > 0.0):
        raise NotApplicable("S 全零：没有任何非零频点可构造导向（Deng 式(8) 的带内非零频点集为空）")
    return S, src


def parse_case(case: dict, params: dict | None = None):
    """case -> (ParsedCase, params_used, parse_diag)；输入解析按 B1 口径复用（仅解析）。

    B1.parse_case 有一条**降维（§4）专属**约束 S[reference_frequency_index] != 0。它**不是**
    full-2D 的要求，故这里先自行解析 S、自动挑一个**非零**频点作为参考索引透传给 B1，
    并在 diagnostics 申报该索引只用于通过 B1 解析、不参与 B5 的 full-2D 运算（B1 代码不改）。
    """
    used = _resolve_params(params)
    b1, b1_sha, b1_path = _b1_module()
    freqs = np.asarray(case["frequencies_hz"], dtype=float).reshape(-1)
    S, s_src = _resolve_s_vector(case, params, int(freqs.size))
    nonzero = [int(i) for i in np.nonzero(np.abs(S) > 0.0)[0]]
    ref_idx = int(nonzero[0])
    passthrough: dict[str, Any] = {"reference_frequency_index": ref_idx}
    if params and params.get("s_response") is not None:
        passthrough["s_response"] = params["s_response"]
    try:
        pc, _b1_used, parse_diag = b1.parse_case(case, passthrough)
    except b1.NotApplicable as exc:
        raise NotApplicable(str(exc)) from None
    except b1.MethodError as exc:
        raise MethodError(str(exc)) from None
    if pc.C < 2:
        raise NotApplicable("Deng 多阵元结构需 >=2 个阵列通道；本 case C=%d" % pc.C)
    if pc.k == 0:
        raise NotApplicable("k=0：式(10) 的谱需要至少 1 个源，无目标窗不给候选")
    if pc.k >= pc.F * pc.C:
        raise NotApplicable("k=%d >= M=F*C=%d：噪声子空间为空，式(10) 不可用" % (pc.k, pc.F * pc.C))
    vy = np.asarray(pc.vcoord[:, 1], dtype=float)
    if float(np.max(np.abs(vy))) > VY_TOL_M:
        raise NotApplicable(
            "虚通道 y 分量非零（|max v_y| = %.3e m > %.1e m）：平面波一阶项含 u_y = -cos(theta)，"
            "本模块未实现该 u_y 项，故显式拒绝而不是静默忽略（本批 v_y = 0）"
            % (float(np.max(np.abs(vy))), VY_TOL_M))
    parse_diag = dict(parse_diag)
    parse_diag.update({
        "b5_input_parse_source": "reuse of b1_deng.parse_case (input parsing ONLY)",
        "b5_reused_module_path": b1_path,
        "b5_reused_module_sha256": b1_sha,
        "b5_s_response_source": s_src,
        "b5_s_zero_indices": [int(i) for i in np.nonzero(np.abs(S) <= 0.0)[0]],
        "b5_s_reference_index_forced": ref_idx,
        "b5_s_reference_index_note": (
            "B1 的 e1 归一约束（§4 降维）要求 S[ref] != 0；该项仅用于通过 B1 解析，"
            "**不参与** B5 的 full-2D 谱（B5 无该约束，允许 S 含零元素）。"),
        "b5_v_y_max_abs_m": float(np.max(np.abs(vy))),
        "b5_v_z_unique_m": sorted({float(v) for v in np.round(pc.vcoord[:, 2], 12)}),
        "b5_v_y_note": "非零共同 v_y 直接 not_applicable（未实现 u_y 项）",
        "b5_v_z_note": "共同 v_z 对 u_theta·v_ch 贡献为 0（u_z = 0），不声称吸收进 tau",
    })
    return pc, used, parse_diag


# --------------------------------------------------------------------------- #
# 平面波四通道导向矢量（Wang §3.1 准则 + Deng §3.1 多阵元结构）
# --------------------------------------------------------------------------- #
def plane_wave_manifold(pc, tau, sin_theta):
    """四通道平面波导向向量，形状 (N_points, F*C)，**频率优先**（index = f*C + ch）。

    逐点 a[f, ch] = S_f * exp(-j2pi f tau) * exp(+j2pi f v_x,ch sin(theta) / v_prop)。
    与 B1 conditional_toa_spectrum 的流形同源（同一平面波适配），但这里 tau / theta 可**联合变化**。
    """
    tau = np.asarray(tau, dtype=float).reshape(-1)
    s = np.asarray(sin_theta, dtype=float).reshape(-1)
    if tau.size != s.size:
        raise MethodError("tau 与 sin(theta) 长度不一致：%d vs %d" % (tau.size, s.size))
    freqs = np.asarray(pc.freqs, dtype=float)
    vx = np.asarray(pc.vcoord[:, 0], dtype=float)
    g = np.multiply.outer(freqs, vx)                                    # (F, C)
    phase_tau = np.exp(-2j * math.pi * np.multiply.outer(freqs, tau))    # (F, N)
    phase_spat = np.exp((2j * math.pi / pc.v_prop) * g[:, :, None] * s[None, None, :])  # (F,C,N)
    S = np.asarray(pc.S, dtype=np.complex128).reshape(-1, 1, 1)
    a = (S * phase_tau[:, None, :]) * phase_spat                         # (F, C, N)
    return a.reshape(pc.F * pc.C, tau.size).T.astype(np.complex128)


def noise_subspace(pc):
    """式(15)(9)：R_hat = X^T X^* / L（= Z Z^H / L），eigh -> 噪声子空间 U_N (M x (M-K))。

    **只实现原生口径**：无前后向平均、无对角加载（原文式(15)(9) 如此；FB 专项已搁置，不提供开关）。
    """
    X = np.asarray(pc.X, dtype=np.complex128)
    R = (X.T @ X.conj()) / float(pc.L)
    trace = float(np.real(np.trace(R)))
    if not math.isfinite(trace) or trace <= 0.0:
        raise MethodError("输入能量为零/非有限：R 的迹 = %r" % trace)
    w, V = np.linalg.eigh(0.5 * (R + R.conj().T))
    m = pc.F * pc.C
    U_N = V[:, : m - pc.k]
    cov_diag = {
        "shape": [int(m), int(m)],
        "trace": trace,
        "snapshots_L": int(pc.L),
        "source_count_K": int(pc.k),
        "noise_subspace_dim": int(m - pc.k),
        "covariance_row_convention": "R = X^T X^* / L = Z Z^H / L (Z = X^T, snapshots as columns)",
        "forward_backward_used": False,
        "diag_load_used": 0.0,
        "native_only_note": "只实现原文口径（无 FB、无对角加载）；不提供 FB/加载开关",
        "eig_smallest": float(w[0]),
        "eig_largest": float(w[-1]),
        "rank_bound_note": "R 秩 <= min(F*C, L)；噪声子空间维 = F*C - K",
    }
    return U_N, cov_diag


def _projection_denominator(A, U_N):
    """式(15)(10) 的分母 denom = ||U_N^H a||^2（逐行）。

    A 行 = a^T，故 proj = A @ U_N.conj() 的第 n 行 = (U_N^H a_n)^T，
    分量 (U_N^H a_n)_j = conj(a_n^H u_j)（u_j = U_N 第 j 列）；于是
    denom_n = Σ_j |(U_N^H a_n)_j|^2 = a_n^H U_N U_N^H a_n。
    **注意**：若写成 A @ U_N，则得到 ||U_N^T a||^2（共轭错），一般不等价。
    """
    proj = A @ U_N.conj()          # (N, M-K) = (U_N^H a)^T（行）
    denom = np.sum(np.abs(proj) ** 2, axis=1).real
    return np.where(np.isfinite(denom), denom, np.inf)


def _spectrum_from_manifold(A, U_N):
    denom = np.maximum(_projection_denominator(A, U_N), 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    return np.maximum(spectrum, 0.0), denom


def full2d_spectrum_native(pc, U_N, *, tau_grid_s=None, theta_grid_deg=None,
                           tau_step_distance_m: float = 0.05,
                           angle_step_deg: float = 0.6):
    """原生矩形 (tau, theta) 网格上的**联合**二维谱（**仅供自测/公式核验**）。

    **不进入** estimate()（正式谱在公共 Cartesian 网格求值）。步长取 5 cm 距离步 / 0.6 deg。
    """
    if tau_grid_s is None:
        step = 2.0 * float(tau_step_distance_m) / pc.v_prop
        lo, hi = float(pc.tau_span_s[0]), float(pc.tau_span_s[1])
        n = int(math.ceil((hi - lo) / step))
        if n + 1 > FROZEN_PARAMS["tau_grid_points_cap"]:
            raise MethodError("时延网格点数 %d 超过冻结上限" % (n + 1))
        tau_grid_s = np.linspace(lo, hi, n + 1)
    if theta_grid_deg is None:
        geom = math.degrees(math.atan2(float(np.max(np.abs(pc.grid_x))),
                                       max(float(np.min(np.abs(pc.grid_y))), 1e-12)))
        half = min(90.0, 0.5 * abs(pc.fov_deg) if pc.fov_deg else 90.0, geom)
        n = int(math.ceil(2.0 * half / float(angle_step_deg) - 1e-12)) + 1
        if n > FROZEN_PARAMS["angle_grid_points_cap"]:
            raise MethodError("角度网格点数 %d 超过冻结上限" % n)
        theta_grid_deg = np.linspace(-half, half, max(n, 3))
    tau = np.asarray(tau_grid_s, dtype=float).reshape(-1)
    th = np.asarray(theta_grid_deg, dtype=float).reshape(-1)
    sin = np.sin(np.deg2rad(th))
    values = np.empty((tau.size, th.size), dtype=float)
    for j in range(th.size):
        A = plane_wave_manifold(pc, tau, np.full(tau.size, sin[j]))
        values[:, j] = _spectrum_from_manifold(A, U_N)[0]
    return {"values": values, "tau_s": tau, "theta_deg": th,
            "objective": "P_2D-MUSIC(tau,theta) = 1 / ||U_N^H a(tau,theta)||^2",
            "grid_kind": "rectangular_tau_theta_native",
            "role": "selftest_only (NOT used by estimate())"}


def common_grid_mask(pc, grid_x=None, grid_y=None):
    """公共域掩膜（与主方法/公共域口径一致）：y < 0 且 |theta| <= fov/2 且 r 在距离环内。"""
    gx = np.asarray(pc.grid_x if grid_x is None else grid_x, dtype=float)
    gy = np.asarray(pc.grid_y if grid_y is None else grid_y, dtype=float)
    Xg, Yg = np.meshgrid(gx, gy)
    r = np.sqrt(Xg ** 2 + Yg ** 2)
    ang = np.arctan2(Xg, -Yg)
    half = np.deg2rad(0.5 * float(pc.fov_deg if pc.fov_deg else 180.0))
    mask = ((Yg < 0.0) & (np.abs(ang) <= half + 1e-12)
            & (r >= pc.ring_span_m[0] - 1e-9) & (r <= pc.ring_span_m[1] + 1e-9))
    return mask, Xg, Yg, r, ang


def spectrum_on_common_grid(pc, U_N, grid_x=None, grid_y=None):
    """把 (tau,theta) 二维泛函**逐点**映射到公共 Cartesian 网格，返回 (spectrum, info)。

    spectrum 形状 = (len(grid_y), len(grid_x))，域外像元为 0（与主方法同口径）；
    值为**未归一**的 1/(denom+1e-12)，denom = ||U_N^H a||^2。
    """
    gx = np.asarray(pc.grid_x if grid_x is None else grid_x, dtype=float)
    gy = np.asarray(pc.grid_y if grid_y is None else grid_y, dtype=float)
    mask, Xg, Yg, r, _ang = common_grid_mask(pc, gx, gy)
    image = np.zeros((gy.size, gx.size), dtype=np.float64)
    info = {
        "grid_kind": "full2d_tau_theta_on_common_cartesian_grid",
        "n_masked_grid_points": int(np.count_nonzero(mask)),
        "n_grid_points": int(mask.size),
        "mapping": "r=hypot(x,y); theta=atan2(x,-y); tau=2r/v_prop+system_delay",
        "inverse_mapping": "r=v_prop*(tau-system_delay)/2; x=r*sin(theta); y=-r*cos(theta)",
        "mask": "y<0 & |theta|<=fov/2 & r in distance_ring",
        "ring_span_m": [float(pc.ring_span_m[0]), float(pc.ring_span_m[1])],
        "spectrum_normalization": "identity 1/(denom+1e-12) (denom = ||U_N^H a||^2)",
    }
    if not mask.any():
        return image, info
    xv = Xg[mask]
    yv = Yg[mask]
    rv = r[mask]
    th = np.arctan2(xv, -yv)
    tau = 2.0 * rv / pc.v_prop + pc.sys_delay_s
    A = plane_wave_manifold(pc, tau, np.sin(th))
    spec, _denom = _spectrum_from_manifold(A, U_N)
    image[mask] = spec
    return image, info


# --------------------------------------------------------------------------- #
# 公共合同入口
# --------------------------------------------------------------------------- #
def _pack(predictions, status, diag, reason=""):
    diag = dict(diag)
    if reason:
        diag["reason"] = reason
    diag.setdefault("grid_kind", "full2d_tau_theta_on_common_cartesian_grid")
    diag.setdefault("n_candidates_evaluated", 0)
    diag.setdefault("dictionary_size", None)
    diag.setdefault("dictionary_size_note", "B5 非字典型：全网格扫描 (tau,theta) 二维泛函")
    return {"predictions": list(predictions), "status": str(status), "diagnostics": _py(diag)}


def estimate(case: dict, params: dict | None = None) -> dict:
    """B5 公共接口：estimate(case[, params]) -> {predictions, status, spectrum, diagnostics}。

    计时范围 = 协方差 -> EVD -> 导向 -> 公共网格谱 -> **自身冻结取峰**（从 case['X'] 起算，
    与主方法口径一致）。评测路径仍为 spectrum（diagnostics 申报 eval_path 与 spectrum_grid）。
    """
    t_start = time.perf_counter()
    diag: dict = {
        "method": "B5",
        "method_id": "b5_wang_full2d",
        "label": ("Wang 2014 §3.1 传统 full-2D (tau,theta) MUSIC 的四通道 CIR 扩展"
                  "（多阵元式参考 Deng 2023 §3.1）"),
        "source": ("Wang, Zhang & Wang, Wireless Personal Communications (2014) 77:2445-2464, "
                   "DOI 10.1007/s11277-014-1644-z, §3.1 式(7)-(15); "
                   "Deng et al., Sensors 23(14):6363 (2023) §3.1 式(6)-(10)"),
        "implemented_equations": "Wang (7)(8)(9)(10)(13)(14)(15) + Deng (6)(7)(8)(9)(10)",
        "structure": ("两参数 (tau,theta) **联合**二维 MUSIC（非角度 1-D + 条件 TOA 的降维两段式，"
                      "非两 Rx successive）"),
        "original_vs_adaptation": {
            "wang_original": ("双天线、通道优先堆叠、二维搜索 (tau,zeta) 两个 TOA；theta 由 "
                              "Delta_tau = zeta - tau 反解（式 11-12）"),
            "deng_reference": ("M 阵元 ULA、(tau,theta) 二维联合搜索、频变空间相位 "
                               "Phi^{i-1}(theta)（式 6-10）"),
            "this_adaptation": ("4 虚拟通道（v = tx + rx）平面波 (tau,theta) 适配；"
                                "不称 Wang 原法原样复现；不是旧 B3（successive 两 Rx）"),
        },
        "eval_path": "spectrum",
        "uses_exact_two_way_manifold": False,
    }
    try:
        pc, used, parse_diag = parse_case(case, params)
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001 - 显式失败，不静默
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
    diag["params_used"] = {
        k: (None if v is None else (list(np.asarray(v).tolist()) if k == "s_response" else v))
        for k, v in used.items()}
    diag["params_used_scope"] = {
        "s_response": "影响正式谱（导向矢量的频域因子）",
        "native_selftest_tau_step_distance_m": "**只影响** full2d_spectrum_native（自测），不进 estimate()",
        "native_selftest_angle_step_deg": "**只影响** full2d_spectrum_native（自测），不进 estimate()",
        "tau_grid_points_cap / angle_grid_points_cap": "只作用于 native 自测网格上限",
    }
    diag["parse"] = parse_diag
    diag["adaptation"] = {
        "virtual_channel_definition": "v_ch = tx[pair0] + rx[pair1]（绝对虚拟坐标）",
        "virtual_x_m": [float(v) for v in pc.vcoord[:, 0]],
        "v_y_max_abs_m": float(np.max(np.abs(pc.vcoord[:, 1]))),
        "v_z_unique_m": sorted({float(v) for v in np.round(pc.vcoord[:, 2], 12)}),
        "v_y_v_z_handling": ("共同 v_z 对 u_theta·v_ch 贡献恒为 0（u_z = 0），不声称吸收进 tau；"
                             "非零共同 v_y 会造成未实现的 u_y = -cos(theta) 项，故直接 not_applicable"),
        "spatial_phase": "exp(+j2pi f v_x,ch sin(theta) / v_prop)（本坐标符号，不照抄原文）",
        "feature_order": "frequency-major then channel (index = f*C + ch)，与 case['X'] 列序一致",
        "frequency_order": "raw case order (NOT sorted)",
        "covariance": "R = X^T X^* / L = Z Z^H / L（通道优先->频率优先只差固定置换 P，不改变谱）",
        "forward_backward_used": False,
        "diag_load_used": 0.0,
        "s_response_source": parse_diag["b5_s_response_source"],
        "s_response_is_flat": bool(np.allclose(pc.S, 1.0)),
        "s_response_zero_indices": parse_diag["b5_s_zero_indices"],
        "s_response_note": ("S = 1 是**通道间共同响应假设**（CIR 响应共同假设），"
                            "**不是**已知平坦的真实发射谱、也不是已校准响应；"
                            "原文 S 为**发射谱对角阵**（元素 = 发射信号频率采样）。"
                            "S 允许含零元素（Deng 式(8) 只在非零带内频点构造 Z）。"),
        "plane_wave_note": ("远场一阶平面波 u_theta·v_ch（u_theta 在 z=0 平面）；不含精确双程球面项。"
                            "r 为坐标原点双程等价半径，tau = 2r/v_prop + system_delay。"),
    }
    diag["channel_calibration"] = "X 内已含数据侧 exp(-1j*phi) 校准；本模块不重复施加"

    try:
        U_N, cov_diag = noise_subspace(pc)
        diag["covariance"] = cov_diag
        image, info = spectrum_on_common_grid(pc, U_N)
        peaks_fn, peak_rule, proto_path, proto_sha = _peak_rule()
        peaks, peak_status = peaks_fn(np.asarray(image, dtype=float),
                                      np.asarray(pc.grid_x, dtype=float),
                                      np.asarray(pc.grid_y, dtype=float),
                                      peak_rule, k=int(pc.k))
        predictions = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["value"])}
                       for p in peaks]
        diag.update({
            "official_grid": ("common Cartesian case grid_x/grid_y (NOT a uniform tau/theta grid); "
                              "candidates = %d masked grid points" % info["n_masked_grid_points"]),
            "native_selftest_grid": ("full2d_spectrum_native() only: step_tau=%.4g m distance -> "
                                     "2*step/v_prop s, step_theta=%.4g deg; NOT used by estimate()"
                                     % (used["native_selftest_tau_step_distance_m"],
                                        used["native_selftest_angle_step_deg"])),
            "grid_kind": info["grid_kind"],
            "spectrum_grid": {"grid_x": [float(v) for v in np.asarray(pc.grid_x, dtype=float)],
                              "grid_y": [float(v) for v in np.asarray(pc.grid_y, dtype=float)]},
            "n_candidates_evaluated": int(info["n_masked_grid_points"]),
            "n_grid_points_total": int(info["n_grid_points"]),
            "dictionary_size": None,
            "dictionary_size_note": "B5 非字典型：全网格扫描 (tau,theta) 二维泛函（映射到公共 Cartesian 网格）",
            "mapping": info["mapping"],
            "inverse_mapping": info["inverse_mapping"],
            "mask": info["mask"],
            "ring_span_m": info["ring_span_m"],
            "spectrum_normalization": info["spectrum_normalization"],
            "fb_used": False,
            "n_channels_used": int(pc.C),
            "M_features": int(pc.F * pc.C),
            "L_snapshots": int(pc.L),
            "peak_rule": ("统一冻结取峰（3x3 非严格极大 + 8 连通平台去重 + 0.30 线性相对阈值 "
                          "+ 0.10 m 确定性 NMS + cap=K）"),
            "peak_rule_source": proto_path,
            "peak_rule_protocol_sha256": proto_sha,
            "n_own_peaks": int(len(predictions)),
            "own_peak_status": {k: _py(v) for k, v in peak_status.items()},
            "timed_path": ("covariance_EVD_steering_spectrum_on_common_grid_own_peaks_from_case_X"
                           "（与主方法计时范围一致：含自身取峰）"),
            "evaluation_boundary": ("只给公共网格完整谱 + 自身冻结取峰；评测由集成层按 spectrum 路径"
                                    "重新取峰，本模块不读真值"),
        })
        diag["wall_seconds"] = time.perf_counter() - t_start
        return {"predictions": predictions, "status": "ok",
                "spectrum": np.asarray(image, dtype=float), "diagnostics": _py(diag)}
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))


def _py(obj):
    """把 numpy 标量/数组递归转成 JSON 可序列化对象（diagnostics 用）。"""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return obj
