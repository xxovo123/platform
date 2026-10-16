# -*- coding: utf-8 -*-
"""C1 硬件侧适配实现：van der Veen, Vanderveen & Paulraj 1998（SIJADE），IEEE TSP 46(2):405-418。

原文：A.-J. van der Veen, M. C. Vanderveen, A. Paulraj, 'Joint Angle and Delay Estimation
Using Shift-Invariance Techniques', DOI 10.1109/78.655425。
公式页：P405-P418（本执行侧渲染 hardware/logs/C1_pdf_pages/）。text 版公式大面积丢失，故以 PDF 为准。

逐项对应原文步骤（不是'二维 ESPRIT 风格'的泛称，也不是对 X 直接 EVD）
--------------------------------------------------------------------------------
* III-A（P409，式(8)(9)(10)）：把时延映射为频域相移的 CFR 矩阵 Hbar（M×N）左移叠加 m1 份，
    构成 Hankel 矩阵 H := [Hbar(1); ...; Hbar(m1)]，H = A B F，A = A_phi ◇ A_theta（Khatri-Rao），
    phi_i = e^{-j2pi*df*tau_i}（时延相位），theta_i = e^{+j2pi*D*sin(theta_i)}（阵列相位）；
    选择矩阵 J_xphi=[I_{m1-1} 0]⊗I, J_yphi=[0 I_{m1-1}]⊗I, J_xtheta=I⊗[I 0], J_ytheta=I⊗[0 I]。
* III-B（P409-410，式(11)(12)(13)）：① 二维平滑 (m1,m2) 的广义 H（式11）：行 = (phi 位移 j) 外
    ⊗ (阵元 a) 内，列 = (阵元位移 i) 外 ⊗ (频点位移 b) 内；② 前后向共轭增广
    H_e = [H, J H^{(c)}]（式12/13，J 行交换、^{(c)} 复共轭）。
* III-C（P410，式(14)）：H_e 的 SVD 取前 r 列 Uhat（**并做数值秩诊断**）；再对 U_xphi、U_xtheta
    做 QR，把 Q 因子作用到 U_yphi、U_ytheta（LS 投影），得 r×r 的 E_xphi/E_yphi/E_xtheta/E_ytheta。
* III-D（P410-411，式(15)）：**明确不做**。原文 P411 左栏自陈 Cayley 变换把实特征值映到 ±∞
    （tau 大或 alpha 接近 ±90 度）并写 numerically inadvisable to form
    (E_xphi^{-1}E_yphi) 与 (E_xtheta^{-1}E_ytheta)，故留在复平面走 III-E。
* III-E（P411，**选末段 'An alternative but similar way'**，不冒充完整 Method Q）：
    ① 独立求 E_xphi^{-1}E_yphi -> Lambda_phi/T_phi 与 E_xtheta^{-1}E_ytheta -> Lambda_theta/T_theta；
    ② M := T_theta^{-1}T_phi 的行 = **theta** 特征向量索引、列 = **phi** 索引（原文该段说
       M 在特征值互异时 should be close to a permutation matrix）；
    ③ 用**一对一 assignment**（K<=3 枚举全部置换，取行归一 |M| 之和最大，字典序确定性 tie）
       求 pi(theta->phi)，再取 **phi->theta 的逆映射** inv = pi^{-1}，输出配对
       (Lambda_phi[j], Lambda_theta[inv[j]])（j 为 phi 索引）。**不用**单行 argmax（会给出重复
       theta 索引；K=2 自逆置换会掩盖方向错误，K>=3 非自逆置换下方向错就整体配错）。
    ④ 纯度/偏差/条件数/特征值间距写入 diagnostics，用于**报告**原文自陈的局限（重根时特征向量
       不定 -> 配对不可靠）；不丢弃候选。
  排除其它 III-E 方法的理由（按原文归属）：Method T 需 III-D 实数变换，P413 临界情形下
  the eigenvector matrix T becomes complex so that Phi and Theta are no longer real valued；
  Method Q（ACMP）的局部 Jacobi 需检测 swap 并同步 Phi 标号，原文 P411 明确它 would fail in the
  theoretical case of **exactly repeated eigenvalues of the first matrix**（本实现不做局部旋转版）；
  Method QZ 迭代，原文 P411-412 说其 convergence properties have been disappointing、需要
  accurate starting point；P417 §VII-5 说通常 Q 与 QZ 差别不显著、T 在临界情形明显变差。

原文自用参数与本次定点
--------------------------------------------------------------------------------
* 原文 P415 Fig.5 图内注：P=2、M=2、m1=5、m2=1；角度 [-10,20] 度、延时 [0,1.1]T。
* 本机 M=4、N=17 -> 按原文**式(19)**：m1 = round((N+1)(2-sqrt2)) = round(10.544) = 11，
  m2 = round(M(sqrt2-1)) = round(1.657) = 2。
  可辨识条件（P412）：a) r <= (m1-1)(M-m2+1)=30；b) r <= m1(M-m2)=22；c) r <= 2m2(N-m1+1)=28。

本机适配（声明近似与必要口径）
--------------------------------------------------------------------------------
* A1 观测：原文是单向信道（P407 式(2)）；本数据是被动双程 CIR。只实现**原生平面波（虚阵）适配**：
    a[f,ch] = S_f * exp(-j2pi f tau_anchor) * exp(+j2pi f v_x,ch sin(theta)/v_prop)，
    tau_anchor 锚定**参考阵元** x = v_x_ref（本批 -0.038 m）。
* A2 符号与通道序：原文 A_theta 用 theta_i = e^{j2pi*D*sin(theta_i)}；本机 theta = atan2(x,-y)、
    视轴 -Y 下虚阵相对相位正是 +2pi f (v_x,ch - v_x_ref) sin(theta)/v_prop，故沿用原文符号。
    A_theta = [1,theta,...,theta^{M-1}] 要求 Hbar 的**通道维按 x 递增**：case 的 pairs 顺序任意，
    故构造 Hbar 时施加 v_order（置换写入 diagnostics）。
* A3 频率维 = 延时维（原文 II-B P407-408）：原文要求所选频点**按递增频率排列**
    （the final floor(LW/2) samples of ghat should be moved up front），这是原文的 J_g。
    case 的 X 是 fftfreq 原始序（7.9872->8.2372 GHz 后接 7.7372->7.95595 GHz），本模块内部
    按递增频率重排；**S 也随之按排序后的频点取值**（S[freq_order[k]]，不是 S[k]）。
* A4 S（发射谱）与去卷积：S = 1 为共同假设（case.model.response_assumption）时式(6) 除法为空
    操作；给出非平坦 S 时按式(6) 逐频点相除；S 在任一被选频点为 0 -> 显式 failed。
* A5 窄带空间相位（模型近似，不是代码 bug）：原文 A_theta 与频率无关（P405 条件 5），本机真实
    相位随频率变化（wideband freq coupling）。不改流形结构，按声明参考频率 f_ref =
    config.center_freq_si 换算：sin(theta_hat) = angle(theta_hat)/(2pi*D_ref)，D_ref = d*f_ref/v_prop
    （本批 0.506206）；相对带宽 6.26%、两端 lambda 差 6.4%。
* A6 相位唯一性（按**严格共同域**，不按 fov=180 一刀切）：D_ref = 0.506206 > 0.5，唯一解域
    |theta| <= asin(1/(2D_ref)) = 81.03 度；域内 theta_max = atan2(2.2, 0.55) = 75.96 度，
    D_ref*sin(theta_max) = 0.49109 < 0.5（余量 1.8%）-> 限定域内相位唯一。band-edge 风险
    （f_hi 时 D(f_hi)*sin(theta_max) = 0.5065 > 0.5）另列。
* A7 **ref-anchor 模型（明确不是 origin-carrier）**：tau_anchor = (2*r_origin -
    v_x_ref*sin(theta))/v_prop + system_delay；反解
    r_origin = v_prop*(tau_hat - system_delay)/2 + (v_x_ref/2)*sin(theta_hat)（本批 -0.019 m），
    x = r_origin*sin(theta_hat)、y = -r_origin*cos(theta_hat)。**域边界同样按锚点换算**：给定 theta，
      tau_lo/hi(theta) = system_delay + 2*r_min/v_prop - v_x_ref*sin(theta)/v_prop（hi 同式取 r_max）
    否则分支判定会带 v_x_ref*sin(theta)/v_prop（本批最多 0.063 ns）的整体偏移。
* A8 共同域 = **严格** bbox + y<0 + 距离环 + FoV，仅 DOMAIN_EPS = 1e-9 m 数值容差
    （与 B1 现行 _in_domain 的严格实现一致；**不做 half-cell 外延**，避免悄悄放大共同域）。
    时间比较用**独立** TIME_EPS = 1e-18 s：把 1e-9 当秒用等于外扩 ±1 ns ≈ ±0.15 m，属无理外延。
* A9 时延相位周期与分支：phi = e^{-j2pi*df*tau}，df = 31.25 MHz -> 相位周期 1/df = 32 ns，
    对应**双程距离周期 v_prop/(2*df) = 4.7967 m**（2.398 m 是相位主值半周期对应的 tau = 16 ns）。
    本批延时域跨度 = 2*(r_max - r_min)/v_prop = 30 ns <= 32 ns，故'取模后落域内'的分支唯一
    （余量 2 ns = 6.25%）。**若出现多解**：该 pair 记 ambiguous、**不计入成功**（记录全部分支），
    全部 pair 都 ambiguous 时整体 status='failed'——拒绝用'环中心'猜分支冒充结果。
* A10 多快照：L=128 慢时间快照作为 H_e 的**附加列**（同一 A、各自 F_t），不改行结构/配对。
* A11 不继承主方法前处理：不用 case.config.is_forward_backward / diag_load_factor；JADE 的
    前后向共轭增广（式12）是该方法自身原生步骤，独立实现并申报。
* A12 数值秩（III-C 真实性）：numerical_rank = #{s_i > rel_tol*s_max}，rel_tol = 1e-8 是**冻结的数值
    近似阈值**（人为选定、非物理判据）。其作用范围**仅限自测覆盖到的情形**（零能量输入；严格窄带模型下
    单源却声明 K=3），**不构成"真实秩确实不足"的一般断言** —— 本机宽带失配会以额外秩形式出现（A5），
    故 wideband 数据下该守卫不会、也不应触发。numerical_rank < k 时记 **failed**（不是 NA，也不许用
    噪声列捏造 subspace）；奇异值头、秩估计、gap 与所用阈值一并落 diagnostics 供主审判断。
* A13 pencil 求解：np.linalg.solve 在近退化 pencil 上偶发抛 LinAlgError -> 由 estimate() 统一记
    **failed**（可接受，只如实申报）；**不为近退化情形加特判/正则/优化**（不做过度工程）。

不适用：k=0；C<2；F<3；H 行数 < k；虚通道 x 不等间距或 y/z 跨通道不一致；频点非等间隔；r 超式(19)。
失败：缺键/形状不符/NaN/Inf/S 长度不符/S 在被选频点为 0/pairs 越界/数值秩 < k/全部 pair ambiguous。
允许 status='ok' 但零个候选（原生解全在共同域外）。

只依赖 numpy；不读真值、不写历史产物、不改 common/ 与其它方法。
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

C0 = 299792458.0
TINY = 1e-300
SHARED_NMS_MIN_DIST_M = 0.10
#: 空间数值容差（严格共同域，与 B1 _in_domain 一致）
DOMAIN_EPS = 1e-9
#: 时间比较容差（**独立**于空间：把 1e-9 当秒用等于外扩 ±1 ns ≈ ±0.15 m）
TIME_EPS = 1e-18

ORIGINAL = {
    "id": "C1",
    "method": "SIJADE (JADE)",
    "title": "Joint Angle and Delay Estimation Using Shift-Invariance Techniques",
    "authors": "A.-J. van der Veen, M. C. Vanderveen, A. Paulraj",
    "venue": "IEEE Trans. Signal Processing, 46(2):405-418, Feb. 1998",
    "doi": "10.1109/78.655425",
    "pdf": "Q1_交付/09_TOA_DOA重新定位_20260921/文献证据/OA/JADE1998_SIJADE_delft_repository.pdf",
    "text": "Q1_交付/10_Q1完整改稿_20260921/evidence/text/JADE1998TSP.txt",
    "formula_pages": "P405-P418（渲染件 hardware/logs/C1_pdf_pages/）",
    "equations_used": "(6)(7)(8)(9)(10)(11)(12)(13)(14)(16)(19)",
    "steps_implemented": "III-A, III-B, III-C(含数值秩), III-E 末段 alternative（独立特征向量+一对一置换）",
    "steps_not_implemented": "III-D real processing；III-E Method Q 的局部 Jacobi 版本（不冒充）",
    "original_self_params": "P=2, M=2, m1=5, m2=1（P415 Fig.5 图内注）",
    "final_step_choice": "III-E 末段：独立 T_phi/T_theta，一对一 assignment 求 theta->phi，"
                         "输出 phi->theta 逆映射配对",
}

#: 冻结声明（不接受覆盖；estimate() 收到非空 params 即显式失败）
FROZEN: dict[str, Any] = {
    "data_extension": "2d_smoothing_eq11_plus_forward_backward_eq12_13（均启用）",
    "rank_reduction": "svd_head_k_then_qr_ls_projection（式14）+ numerical_rank 诊断",
    "final_step": "III-E_alternative_independent_eig_plus_one_to_one_permutation_assignment",
    "real_processing_III_D": "not_used",
    "m1_m2": "auto_by_eq19_rounded（本批 F=17,M=4 -> m1=11,m2=2）",
    "reference_frequency": "config.center_freq_si（本批 7.9872 GHz）",
    "delay_reference_model": "ref_anchor（tau 锚定参考阵元 x=v_x_ref；域边界同式换算）",
    "s_response_policy": "case.model.response_assumption.vector 非空则按式(6) 逐频点相除（随频率排序取值），否则 S=1",
    "delay_branch_policy": "域内唯一才作成功；多解标 ambiguous 并记 dropped（全 ambiguous 才 failed，不用环中心猜）",
    "nms_min_dist_m": SHARED_NMS_MIN_DIST_M,
    "domain_tolerance": "strict bbox + ring + FoV，仅 DOMAIN_EPS=1e-9 m（与 B1 现行实现一致，无 half-cell）",
    "time_tolerance": "TIME_EPS=1e-18 s（时间比较专用）",
    "numerical_rank_rel_tol": 1e-8,
    "numerical_rank_rel_tol_note": ("1e-8 为**冻结的数值近似**阈值（人为选定、非物理判据）；"
                                    "作用范围仅限自测覆盖情形（零能量；严格窄带模型下单源却声明 "
                                    "K>rank），不代表对真实秩的一般断言"),
    "pencil_solve_failure_policy": ("np.linalg.solve 抛 LinAlgError（近退化 pencil）-> 显式 failed；"
                                    "不加特判/正则/优化"),
    "score": "per-ray explained energy ||G[l,:]||_2, G = A_nat^+ H_e（原生模型 LS，不用真值）",
    "degeneracy_policy": "只报告不丢弃（不得删不利样本）",
}

__all__ = [
    "C0",
    "TIME_EPS",
    "ORIGINAL",
    "FROZEN",
    "MethodError",
    "NotApplicable",
    "ParsedCase",
    "parse_case",
    "estimate",
]


class MethodError(ValueError):
    """输入或内部状态不合规：显式失败（status='failed'），不静默降级。"""


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
    """把 numpy / 复数递归转成 JSON 可序列化形式（diagnostics 落盘用）。"""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "c":
            return {"kind": "complex_array", "n": int(obj.size),
                    "real": [float(v) for v in np.real(obj).ravel()],
                    "imag": [float(v) for v in np.imag(obj).ravel()]}
        flat = np.asarray(obj).ravel()
        if flat.size > 256:
            return {"kind": "array_summary", "n": int(flat.size),
                    "shape": [int(v) for v in np.shape(obj)],
                    "min": float(np.min(flat)), "max": float(np.max(flat)),
                    "first8": [float(v) for v in flat[:8]]}
        return [float(v) for v in flat]
    if isinstance(obj, (complex, np.complexfloating)):
        return {"kind": "complex", "real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


_SHARED_NMS_CACHE: dict = {}


def _shared_nms_impl():
    """加载公共合同里的共享 NMS（仅加载一次）。不复制规则：公共合同缺失即显式失败。"""
    if "fn" not in _SHARED_NMS_CACHE:
        import importlib
        import sys

        common_dir = Path(__file__).resolve().parents[2] / "common"
        if not (common_dir / "q6_contract.py").is_file():
            raise MethodError("找不到公共合同 %s/q6_contract.py：本模块拒绝私自实现 NMS"
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
    stats["two_d_full_spectrum_threshold_applied"] = False
    return {"kept": kept, "stats": stats}


def _pack(predictions, status, diag, reason="") -> dict:
    diag = dict(diag)
    if reason:
        diag["reason"] = reason
    diag.setdefault("grid_kind", "closed_form_shift_invariance_no_grid")
    diag.setdefault("n_candidates_evaluated", len(predictions))
    diag.setdefault("dictionary_size", None)
    diag.setdefault("dictionary_size_note",
                    "C1 无字典/无网格谱：Phi、Theta 由平移不变 pencil 的闭式 rank-reducing numbers 给出")
    return {"predictions": list(predictions), "status": str(status), "diagnostics": _py(diag)}


# --------------------------------------------------------------------------- #
# case 解析（只读 case；不读真值）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedCase:
    case_id: str
    X: np.ndarray
    freqs_raw: np.ndarray
    freq_order: np.ndarray
    freqs_sorted: np.ndarray
    df_hz: float
    pairs: np.ndarray
    vcoord: np.ndarray
    v_order: np.ndarray
    v_x_ref: float
    spacing_m: float
    v_prop: float
    sys_delay_s: float
    k: int
    L: int
    F: int
    C: int
    m1: int
    m2: int
    n_rows: int
    n_ant: int
    n_freq_shift: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    fov_deg: float
    ring_m: tuple
    ring_span_m: tuple
    tau_span_prop_s: tuple
    tau_step_s: float
    S: np.ndarray
    S_source: str
    ref_freq_hz: float
    delta_ref: float
    theta_max_deg: float
    wrap_margin: float
    alias_sin_limit: float
    band_edge_delta: tuple
    ident: dict
    config_fb_flag: bool


def _band_order(freqs: np.ndarray, tol: float = 1e-6) -> tuple:
    """原文 J_g：把所选频点按**递增频率**排列，并检查等间隔（式(6)(7) 的 Vandermonde phi）。"""
    order = np.argsort(freqs, kind="stable")
    fs = np.asarray(freqs, dtype=float)[order]
    if fs.size < 2:
        raise NotApplicable("式(6)(7) 需要 >=2 个带内频点；本 case F=%d" % fs.size)
    d = np.diff(fs)
    if not np.all(d > 0.0):
        raise MethodError("频率排序后仍出现重复/非递增频点：f=%s" % (fs.tolist(),))
    spread = float(np.max(d) - np.min(d)) / float(np.max(d))
    if spread > tol:
        raise NotApplicable(
            "原文式(6)(7) 的 phi_i^k（frequency shift -> delay phase）要求所选频点**等间隔**；"
            "本 case 排序后相邻间距相对散布 %.3e > %.1e（间隔 %s Hz），结构不成立"
            % (spread, tol, np.array2string(d, precision=3)))
    return order, fs, float(d.mean()), spread


def _ula_layout(vcoord: np.ndarray) -> tuple:
    """原文 ULA：虚通道沿 x 等间距，且 y/z 跨通道一致（共同平移不影响平面波相位）。"""
    if vcoord.shape[0] < 2:
        raise NotApplicable("原文 ULA 需要 >=2 个阵列通道（虚通道 %s）" % (vcoord.shape[0],))
    order = np.argsort(vcoord[:, 0], kind="stable")
    v = vcoord[order]
    spans = []
    for axis, name in ((1, "y"), (2, "z")):
        span = float(np.ptp(v[:, axis]))
        spans.append(span)
        if span > 1e-9:
            raise NotApplicable(
                "原文 ULA（式(9)(10) 的 A_theta = [1,theta,...,theta^{M-1}]）要求虚通道 %s 坐标"
                "跨通道一致；实测极差 %.3e m，原生平面波适配不成立" % (name, span))
    dx = np.diff(v[:, 0])
    if not np.all(dx > 0.0):
        raise NotApplicable("虚通道 x 坐标存在重复（阵元间距为 0）：原文 ULA 结构不成立")
    spread = float(np.max(dx) - np.min(dx)) / float(np.max(dx))
    if spread > 1e-6:
        raise NotApplicable(
            "原文 ULA 要求虚通道沿 x **等间距**（theta^{m-1} 结构）；相对间距散布 %.3e > 1e-6，"
            "结构不成立" % spread)
    return order, float(dx.mean()), spans


def _resolve_s_response(case: dict, F: int) -> tuple:
    """S（发射谱）：case.model.response_assumption.vector 非空则用于式(6) 去卷积，否则 S=1。

    返回的 S 与 case 的**原始频率序**对齐（消费方必须用 freq_order 取下标）。
    """
    model = case.get("model") or {}
    resp = (model.get("response_assumption") or {}) if isinstance(model, dict) else {}
    vec = resp.get("vector") if isinstance(resp, dict) else None
    if vec is None:
        return np.ones(F, dtype=np.complex128), "assumption_S_equals_1(unknown_transmit_spectrum)"
    S = np.asarray(vec, dtype=np.complex128).reshape(-1)
    if S.size != F:
        raise MethodError("case.model.response_assumption.vector 长度 %d != F=%d" % (S.size, F))
    if not np.all(np.isfinite(S)):
        raise MethodError("case.model.response_assumption.vector 含 NaN/Inf")
    return S, "case.model.response_assumption.vector"


def _domain_theta_max(grid_x: np.ndarray, grid_y: np.ndarray, ring_m,
                      fov_deg: float) -> tuple:
    """**严格共同域**内 |theta| 的最大值（只用 case 给出的域；不读真值）。"""
    gx, gy = np.meshgrid(np.asarray(grid_x, float), np.asarray(grid_y, float))
    r = np.hypot(gx, gy)
    keep = (gy < -DOMAIN_EPS) & (r > DOMAIN_EPS)
    if ring_m is not None:
        keep &= (r >= ring_m[0] - DOMAIN_EPS) & (r <= ring_m[1] + DOMAIN_EPS)
    if fov_deg and fov_deg < 180.0:
        keep &= (np.abs(np.degrees(np.arctan2(gx, -gy))) <= 0.5 * abs(fov_deg) + DOMAIN_EPS)
    if not np.any(keep):
        return 0.0, 0
    theta = np.abs(np.degrees(np.arctan2(gx[keep], -gy[keep])))
    return float(np.max(theta)), int(np.count_nonzero(keep))


def _identifiability(M: int, N: int, m1: int, m2: int) -> dict:
    """原文 P412 式(19) 后 a)~c) 与同延时/同角度分支容量。"""
    cond_a = int((m1 - 1) * (M - m2 + 1))
    cond_b = int(m1 * (M - m2))
    cond_c = int(2 * m2 * (N - m1 + 1))
    return {
        "condition_a_r_le_(m1-1)(M-m2+1)": cond_a,
        "condition_b_r_le_m1(M-m2)": cond_b,
        "condition_c_r_le_2m2(N-m1+1)": cond_c,
        "r_max_conditions_min": int(min(cond_a, cond_b, cond_c)),
        "equal_delay_capacity": int(min(2 * m2, math.floor(2.0 * M / 3.0))),
        "equal_delay_condition": "原文 P410：m2 >= d/2 且 M >= 3d/2（d = 同延时条数）",
        "equal_angle_capacity": int(min(m1 - 1, math.floor(2.0 * N / 3.0))),
        "equal_angle_condition": "原文 P412：m1 >= d+1 且 LW >= 3d/2（d = 同角度条数）",
        "note": "容量只用于**报告**退化风险，不用于丢弃任何候选；K 由 case 给定（known-K）",
    }


def parse_case(case: dict) -> tuple:
    """case -> (ParsedCase, parse_diag)。只读 case，不读真值。"""
    if not isinstance(case, dict):
        raise MethodError("case 必须是 dict，收到 %r" % type(case).__name__)

    X = _finite_array(_require(case, "X"), "X", ndim=2)
    freqs = _finite_array(_require(case, "frequencies_hz"), "frequencies_hz"
                          ).reshape(-1).astype(float)
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
    L = int(X.shape[0])
    if pairs.shape[1] != 2:
        raise MethodError("pairs 必须是 (C,2)，收到 shape=%s" % (pairs.shape,))
    if tx.shape[1] != 3 or rx.shape[1] != 3:
        raise MethodError("tx_positions/rx_positions 必须是 (*,3)")
    if int(X.shape[1]) != F * C:
        raise MethodError("X 列数 %d != F*C = %d*%d（频率优先 f*C+c）" % (X.shape[1], F, C))
    bad_idx = (np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= tx.shape[0])
               or np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= rx.shape[0]))
    if bad_idx:
        raise MethodError("pairs 索引超出 tx_positions/rx_positions 范围")
    if k == 0:
        raise NotApplicable("k=0：无源，式(9) 的 H = A B F 秩为 0，平移不变对不存在")
    if C < 2:
        raise NotApplicable("原文 ULA 需要 >=2 个阵列通道（式(9)）；本 case C=%d" % C)
    if F < 3:
        raise NotApplicable("式(6)(7) 的频域平移结构需要 >=3 个带内频点；本 case F=%d" % F)
    if k >= F * C:
        raise NotApplicable("k=%d >= F*C=%d：秩削减后无冗余维，式(14) 不可用" % (k, F * C))

    m1 = int(round((F + 1) * (2.0 - math.sqrt(2.0))))
    m2 = int(round(C * (math.sqrt(2.0) - 1.0)))
    m1 = max(1, min(m1, F - 1))
    m2 = max(1, min(m2, C - 1))
    n_ant = C - m2 + 1
    n_rows = m1 * n_ant
    n_freq_shift = F - m1 + 1
    if n_ant < 1 or n_freq_shift < 1:
        raise NotApplicable("式(19) 定点后 M-m2+1=%d、N-m1+1=%d 出现空维" % (n_ant, n_freq_shift))
    ident = _identifiability(C, F, m1, m2)
    if k > ident["r_max_conditions_min"]:
        raise NotApplicable(
            "k=%d 超过原文式(19) 后可辨识条件上限 %d（a/b/c = %d/%d/%d，m1=%d,m2=%d,M=%d,N=%d）"
            % (k, ident["r_max_conditions_min"],
               ident["condition_a_r_le_(m1-1)(M-m2+1)"], ident["condition_b_r_le_m1(M-m2)"],
               ident["condition_c_r_le_2m2(N-m1+1)"], m1, m2, C, F))
    if n_rows < k:
        raise NotApplicable("H 的行数 %d < k=%d：式(14) 的 r×r pencil 需要 >=k 行" % (n_rows, k))

    freq_order, freqs_sorted, df_hz, df_spread = _band_order(freqs)
    vcoord = (tx[pairs[:, 0]] + rx[pairs[:, 1]]).astype(float)
    v_order, spacing, v_spans = _ula_layout(vcoord)
    v_x_ref = float(vcoord[v_order[0], 0])

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
            "既无 search.one_dimensional_budget.round_trip_tau_span_s，也无 "
            "config.distance_ring_m：没有可冻结的一维延时域，无法解 1/df 的取模分支")
    if not (span_prop[1] > span_prop[0]):
        raise MethodError("延时搜索域必须严格递增，收到 %r" % (span_prop,))
    ring_span_m = (0.5 * v_prop * span_prop[0], 0.5 * v_prop * span_prop[1])
    step = budget.get("suggested_one_d_tau_step_s")
    tau_step = float(step) if step is not None else 2.0 * 0.05 / v_prop
    if not (tau_step > 0.0):
        raise MethodError("一维时延步长必须为正，收到 %r" % tau_step)
    fov = float(cfg.get("fov_degrees", 180.0) or 180.0)

    S, s_src = _resolve_s_response(case, F)
    zero_idx = [int(i) for i in np.nonzero(np.abs(S) == 0.0)[0]]
    if zero_idx:
        raise MethodError(
            "S（发射谱/脉冲形状）在频点 %s 为 0：原文式(6) 的 ghat 除法在此点退化，"
            "去卷积不可定义（显式失败，不静默裁剪）" % zero_idx[:8])

    ref_freq = cfg.get("center_freq_si")
    if ref_freq is None:
        fr = cfg.get("freq_range_si") or [float(freqs_sorted[0]), float(freqs_sorted[-1])]
        ref_freq = 0.5 * (float(fr[0]) + float(fr[1]))
        ref_src = "band_midpoint_from_config.freq_range_si"
    else:
        ref_src = "config.center_freq_si"
    ref_freq = float(ref_freq)
    if not (ref_freq > 0.0):
        raise MethodError("参考频率必须为正，收到 %r" % ref_freq)

    delta_ref = float(spacing * ref_freq / v_prop)
    alias_sin_limit = 1.0 / (2.0 * delta_ref)
    theta_max_deg, n_domain_cells = _domain_theta_max(grid_x, grid_y, ring_span_m, fov)
    wrap_margin = 1.0 - float(delta_ref * math.sin(math.radians(theta_max_deg))) / 0.5
    band_edge_delta = (float(spacing * freqs_sorted[0] / v_prop),
                       float(spacing * freqs_sorted[-1] / v_prop))

    parse_diag = {
        "manifold": "plane_wave_virtual_array (v = tx + rx; 原生平面波适配，见 A1/A2)",
        "n_features": F * C,
        "feature_order_in_case": case.get("feature_order"),
        "frequency_order_raw": "case 原始序（fftfreq 序，非单调）",
        "frequency_order_increasing_indices": [int(i) for i in freq_order],
        "frequency_deconvolution": ("S=1 时为空操作；非平坦 S 按原文式(6) 逐频点相除，"
                                    "**下标随频率排序**（S[freq_order[k]]）（S 来源：%s）" % s_src),
        "df_hz": df_hz,
        "df_relative_spread": df_spread,
        "band_hz": [float(freqs_sorted[0]), float(freqs_sorted[-1])],
        "relative_bandwidth": float((freqs_sorted[-1] - freqs_sorted[0]) / ref_freq),
        "reference_frequency_hz": ref_freq,
        "reference_frequency_source": ref_src,
        "narrowband_note": ("原文 A_theta 与频率无关（P405 条件 5）；本机真实空间相位随频率变化，"
                            "按声明参考频率换算角度，属**模型近似**而非代码缺陷"),
        "virtual_x_m": [float(v) for v in vcoord[:, 0]],
        "virtual_x_sorted_m": [float(v) for v in vcoord[v_order, 0]],
        "virtual_channel_order": [int(i) for i in v_order],
        "channel_order_note": ("case 通道序 -> 按 x 递增的 ULA 序（A_theta = theta^{m-1} 要求）；"
                              "Hbar 构造时按该置换重排通道"),
        "virtual_yz_spread_m": [float(s) for s in v_spans],
        "reference_element_x_m": v_x_ref,
        "ula_spacing_m": spacing,
        "delta_ref_d_over_lambda": delta_ref,
        "alias_sin_limit_1_over_2delta": alias_sin_limit,
        "theta_max_domain_deg": theta_max_deg,
        "n_domain_cells_scanned": n_domain_cells,
        "wrap_margin_fraction": wrap_margin,
        "band_edge_delta_ref": [band_edge_delta[0], band_edge_delta[1]],
        "delay_reference_model": ("ref_anchor：tau 锚定参考阵元 x=v_x_ref；给定 theta 的域边界为 "
                                  "tau_lo/hi = system_delay + 2*r/v_prop - v_x_ref*sin(theta)/v_prop"),
        "propagation_speed_m_s": v_prop,
        "propagation_speed_source": v_src,
        "system_delay_si": sys_delay,
        "tau_span_propagation_s": [span_prop[0], span_prop[1]],
        "tau_span_source": span_src,
        "ring_span_m": [ring_span_m[0], ring_span_m[1]],
        "delay_phase_period_s": 1.0 / df_hz,
        "delay_two_way_span_period_m": float(v_prop / (2.0 * df_hz)),
        "delay_branch_unique_capacity": bool(
            (span_prop[1] - span_prop[0]) <= 1.0 / df_hz + TIME_EPS),
        "domain_policy": ("共同域 = **严格** bbox + y<0 + 距离环 + FoV，仅 DOMAIN_EPS=1e-9 m 容差"
                          "（与 B1 现行 _in_domain 的严格实现一致，无 half-cell 外延）"),
        "time_tolerance_s": TIME_EPS,
        "tau_step_s": tau_step,
        "fov_degrees_config": fov,
        "m1": m1, "m2": m2, "n_rows": n_rows, "n_ant": n_ant,
        "n_freq_shift_blocks": n_freq_shift,
        "m1_m2_rule": "原文式(19)：m1=round((N+1)(2-sqrt2))、m2=round(M(sqrt2-1))",
        "identifiability": ident,
        "s_response_source": s_src,
        "s_response_is_flat": bool(np.allclose(S, 1.0)),
        "case_config_is_forward_backward": bool(cfg.get("is_forward_backward", False)),
        "forward_backward_inherited_from_case": False,
        "forward_backward_used_by_this_method": True,
        "forward_backward_note": ("case.config.is_forward_backward 属**主方法**选择，本模块不继承；"
                                 "式(12)(13) 的共轭增广是 JADE 自身原生步骤（A11）"),
        "diag_load_used": 0.0,
        "channel_phase_calibration": "X 内已含数据侧 exp(-1j*phi) 校准；本模块不重复施加",
    }
    pc = ParsedCase(
        case_id=str(case.get("id", "?")), X=X, freqs_raw=freqs, freq_order=freq_order,
        freqs_sorted=freqs_sorted, df_hz=df_hz, pairs=pairs, vcoord=vcoord,
        v_order=v_order, v_x_ref=v_x_ref, spacing_m=spacing, v_prop=v_prop,
        sys_delay_s=sys_delay, k=k, L=L, F=F, C=C, m1=m1, m2=m2, n_rows=n_rows,
        n_ant=n_ant, n_freq_shift=n_freq_shift, grid_x=grid_x, grid_y=grid_y, fov_deg=fov,
        ring_m=ring_m, ring_span_m=ring_span_m, tau_span_prop_s=span_prop,
        tau_step_s=tau_step, S=S, S_source=s_src, ref_freq_hz=ref_freq, delta_ref=delta_ref,
        theta_max_deg=theta_max_deg, wrap_margin=wrap_margin, alias_sin_limit=alias_sin_limit,
        band_edge_delta=band_edge_delta, ident=ident,
        config_fb_flag=bool(cfg.get("is_forward_backward", False)))
    return pc, parse_diag


# --------------------------------------------------------------------------- #
# III-A/III-B：数据矩阵构造（式 6/7 -> 11 -> 12/13）
# --------------------------------------------------------------------------- #
def _hbar_from_case(pc: ParsedCase) -> np.ndarray:
    """原文式(6)(7) 的 Hbar（M×N，逐快照）。

    Hbar[t, a, k] = X[t, freq_order[k]*C + v_order[a]] / S[freq_order[k]]
      * 通道维按 **v_order**（x 递增）重排 —— A_theta = [1,theta,...,theta^{M-1}] 的前提；
      * 去卷积因子 S 的**下标随频率排序**（S 定义在 case 原始频率序上，A3）。
    """
    X = pc.X
    n_snap = int(X.shape[0])
    cube = np.empty((n_snap, pc.C, pc.F), dtype=np.complex128)
    for k, src in enumerate(pc.freq_order):
        block = X[:, int(src) * pc.C:(int(src) + 1) * pc.C]     # (L, C) case 通道序
        cube[:, :, k] = (block / pc.S[int(src)])[:, pc.v_order]  # -> ULA 序（x 递增）
    return cube


def _hankel_2d(cube: np.ndarray, m1: int, m2: int) -> np.ndarray:
    """原文式(11) 的广义 H（含式(8) 的频率左移）：

        H[(j,a), (i,b)] = Hbar[i + a, j + b]
        j=0..m1-1（phi 位移），a=0..C-m2（阵元），i=0..m2-1（阵元位移），b=0..F-m1（频点位移）

    返回 (n_rows, n_snap * n_cols_per_snapshot)（快照按列块拼接，A10）。
    """
    n_snap, C, F = cube.shape
    n_ant = C - m2 + 1
    n_freq_shift = F - m1 + 1
    n_rows = m1 * n_ant
    n_cols = m2 * n_freq_shift
    H = np.empty((n_snap, n_rows, n_cols), dtype=np.complex128)
    for j in range(m1):
        for a in range(n_ant):
            row = j * n_ant + a
            for i in range(m2):
                for b in range(n_freq_shift):
                    H[:, row, i * n_freq_shift + b] = cube[:, a + i, b + j]
    return H.transpose(1, 0, 2).reshape(n_rows, n_snap * n_cols)


def _forward_backward(H: np.ndarray) -> np.ndarray:
    """原文式(12)(13)：H_e = [H, J H^{(c)}]（J 行交换、^{(c)} 复共轭）。"""
    return np.concatenate([H, np.conj(H)[::-1, :]], axis=1)


def _selection_indices(m1: int, m2: int, C: int) -> dict:
    """原文式(10) 的选择矩阵（作用在 H 的行索引 (j,a) 上）。"""
    n_ant = C - m2 + 1
    idx = np.arange(m1 * n_ant).reshape(m1, n_ant)
    return {
        "x_phi": idx[0:m1 - 1, :].ravel(),
        "y_phi": idx[1:m1, :].ravel(),
        "x_theta": idx[:, 0:n_ant - 1].ravel(),
        "y_theta": idx[:, 1:n_ant].ravel(),
    }


# --------------------------------------------------------------------------- #
# III-C：SVD 秩削减（含数值秩诊断）+ QR/LS 投影（式14）
# --------------------------------------------------------------------------- #
def _rank_reduce(H_e: np.ndarray, k: int, rel_tol: float) -> tuple:
    """原文式(14) 前的 SVD 秩削减 + **数值秩诊断**（不捏造 subspace）。

    numerical_rank = #{s_i > rel_tol * s_max}；< k（零输入、有效源不足）时调用方必须 failed。
    """
    U, s, _Vh = np.linalg.svd(H_e, full_matrices=False)
    m = min(int(U.shape[1]), int(s.size))
    s_max = float(s[0]) if s.size else 0.0
    rank = int(np.count_nonzero(s > rel_tol * s_max)) if s_max > 0.0 else 0
    info = {
        "singular_values_head": [float(v) for v in s[: min(s.size, k + 2)]],
        "singular_values_count": int(s.size),
        "s_max": s_max,
        "rel_tol": float(rel_tol),
        "rel_tol_note": ("冻结的**数值近似**阈值（人为选定、非物理判据）；只在自测覆盖到的情形"
                         "（零能量、严格窄带模型下 K>rank）用于拦截，不构成对真实秩的一般断言"),
        "numerical_rank": rank,
        "k": int(k),
        "energy_zero": bool(s_max <= 0.0),
        "gap_ratio_s_k_over_s_0": (float(s[k] / s_max) if (s.size > k and s_max > 0) else None),
        "pencil_solve_policy": "solve 失败(LinAlgError) -> 整体 failed；不为近退化加优化",
    }
    if m < k:
        raise MethodError("H_e 的列空间只有 %d 维，无法取 k=%d 列信号子空间" % (m, k))
    return U[:, :k], s, info


def _qr_reduce(U_x: np.ndarray, U_y: np.ndarray) -> tuple:
    """原文式(14) 的 LS 投影：QR(U_x)=Q R -> E_x=R, E_y=Q^H U_y（r×r）。"""
    Q, R = np.linalg.qr(U_x)
    return R, Q.conj().T @ U_y


# --------------------------------------------------------------------------- #
# III-E（末段 alternative）：独立特征分解 + 一对一置换配对
# --------------------------------------------------------------------------- #
def _pencil_eig(E_x: np.ndarray, E_y: np.ndarray) -> tuple:
    """原文式(16)：独立求 E_x^{-1}E_y 的特征值/特征向量（此处不做任何配对）。"""
    A = np.linalg.solve(E_x, E_y)
    w, T = np.linalg.eig(A)
    return w, T, A


def _normalized_abs_m(T_phi: np.ndarray, T_theta: np.ndarray) -> tuple:
    """M = T_theta^{-1}T_phi：**行 = theta 索引、列 = phi 索引**；返回 (absM, 行归一 absM)。"""
    absM = np.abs(np.linalg.solve(T_theta, T_phi))
    row_max = np.max(absM, axis=1, keepdims=True)
    return absM, absM / np.maximum(row_max, TINY)


def _permutation_assignment(norm_abs_m: np.ndarray) -> tuple:
    """一对一 assignment（不是单行 argmax）：枚举全部置换，取行归一 |M| 之和最大者。

    返回 (perm, score)，perm[i] = **phi** 索引，对应 theta 索引 i（即 theta -> phi）。
    itertools.permutations 按字典序产出，"严格更优才替换"因此给出确定性 tie-break。
    """
    n = int(norm_abs_m.shape[0])
    if n <= 6:
        best_perm, best_score = None, -1.0
        for perm in itertools.permutations(range(n)):
            sc = float(sum(norm_abs_m[i, perm[i]] for i in range(n)))
            if sc > best_score + 1e-12:
                best_perm, best_score = list(perm), sc
        return best_perm, best_score
    order = sorted(range(n), key=lambda i: -float(np.max(norm_abs_m[i])))
    used, perm = set(), [0] * n
    for i in order:
        cols = [j for j in range(n) if j not in used]
        j = max(cols, key=lambda c: (float(norm_abs_m[i, c]), -c))
        perm[i] = j
        used.add(j)
    return perm, float(sum(norm_abs_m[i, perm[i]] for i in range(n)))


def _pairing_report(T_phi: np.ndarray, T_theta: np.ndarray,
                    lam_phi: np.ndarray, lam_theta: np.ndarray) -> dict:
    """原文 P411 末段：由 M = T_theta^{-1}T_phi 给出显式配对（方向已按主审修正）。

    M 行 = theta 索引、列 = phi 索引 => 行主元是 **theta -> phi**；本函数用一对一 assignment
    求 pi(theta->phi)，给出 **phi -> theta 逆映射** inv，配对为 (Lambda_phi[j], Lambda_theta[inv[j]])。
    """
    absM, norm = _normalized_abs_m(T_phi, T_theta)
    n = int(norm.shape[0])
    theta_to_phi, score = _permutation_assignment(norm)
    phi_to_theta = [0] * n
    for i, j in enumerate(theta_to_phi):
        phi_to_theta[j] = i
    naive_map = [int(c) for c in np.argmax(absM, axis=1)]
    row_purity = np.max(absM, axis=1) / np.maximum(np.sum(absM, axis=1), TINY)
    sep = (lambda v: float(np.min(np.abs(v[:, None] - v[None, :]) + np.eye(n) * 1e3))
           if n > 1 else None)
    return {
        "pairing_map_theta_to_phi": [int(v) for v in theta_to_phi],
        "pairing_map_phi_to_theta_inverse": [int(v) for v in phi_to_theta],
        "pairing_assignment_mean_purity": float(score / max(n, 1)),
        "pairing_row_purity_min": float(np.min(row_purity)),
        "pairing_method": ("one_to_one_enumeration_all_permutations_max_row_normalized_abs_sum"
                           "_lexicographic_tie" if n <= 6 else "greedy_by_row_max_deterministic"),
        "naive_row_argmax_map_would_be": naive_map,
        "naive_row_argmax_would_conflict": bool(len(set(naive_map)) < n),
        "cond_T_phi": float(np.linalg.cond(T_phi)),
        "cond_T_theta": float(np.linalg.cond(T_theta)),
        "eigenvalue_sep_phi_min_abs": sep(lam_phi),
        "eigenvalue_sep_theta_min_abs": sep(lam_theta),
        "limitation": ("原文 P411：T_theta^{-1}T_phi should be close to a permutation matrix "
                       "(in the case of distinct eigenvalues)；出现（近似）重根时特征向量不定，"
                       "配对纯度下降 —— 只报告，不丢弃候选；K=2 自逆置换会掩盖方向错误，"
                       "故两个方向都落盘"),
    }


def _joint_pairs(E_xphi: np.ndarray, E_yphi: np.ndarray,
                 E_xtheta: np.ndarray, E_ytheta: np.ndarray) -> dict:
    """III-E 末段 alternative：独立特征分解 + 一对一 assignment 给出**成对**结果。"""
    lam_phi, T_phi, _A_phi = _pencil_eig(E_xphi, E_yphi)
    lam_theta, T_theta, _A_theta = _pencil_eig(E_xtheta, E_ytheta)
    rep = _pairing_report(T_phi, T_theta, lam_phi, lam_theta)
    inv = rep["pairing_map_phi_to_theta_inverse"]
    n = int(lam_phi.size)
    phi_paired = np.asarray([lam_phi[j] for j in range(n)], dtype=np.complex128)
    theta_paired = np.asarray([lam_theta[inv[j]] for j in range(n)], dtype=np.complex128)
    rep.update({
        "pencil_matrix_cond_xphi": float(np.linalg.cond(E_xphi)),
        "pencil_matrix_cond_xtheta": float(np.linalg.cond(E_xtheta)),
        "phi_eigenvalues_raw": [complex(v) for v in lam_phi],
        "theta_eigenvalues_raw": [complex(v) for v in lam_theta],
        "phi_eigenvalues_paired": [complex(v) for v in phi_paired],
        "theta_eigenvalues_paired": [complex(v) for v in theta_paired],
        "pairing_rule": ("lambda_phi[j] <-> lambda_theta[inv[j]]（j = phi 索引）；"
                         "inv = pi^{-1}，pi 由行归一 |T_theta^{-1}T_phi| 的一对一 assignment 给出"),
    })
    return {"phi": phi_paired, "theta": theta_paired, "report": rep}


# --------------------------------------------------------------------------- #
# 参数换算：theta -> 角度、phi -> 延时 -> 位置
# --------------------------------------------------------------------------- #
def _theta_to_sin(theta_val: complex, delta_ref: float) -> tuple:
    """theta = e^{+j2pi*D_ref*sin(theta)} -> sin(theta)（主值），并检查是否超出唯一解域。"""
    ang = float(np.angle(theta_val))
    denom = 2.0 * math.pi * delta_ref
    sin_val = ang / denom
    if abs(sin_val) > 1.0:
        return float(np.clip(sin_val, -1.0, 1.0)), True
    return float(sin_val), False


def _delay_domain_for_theta(pc: ParsedCase, theta_deg: float) -> tuple:
    """A7：把 origin 距离环按给定 theta 映成 **ref-anchor** 的绝对延时域。"""
    shift = -pc.v_x_ref * math.sin(math.radians(theta_deg)) / pc.v_prop
    return (float(pc.sys_delay_s + pc.tau_span_prop_s[0] + shift),
            float(pc.sys_delay_s + pc.tau_span_prop_s[1] + shift))


def _delay_branch(phi_val: complex, df_hz: float, tau_lo: float, tau_hi: float) -> dict:
    """phi = e^{-j2pi*df*tau} -> 绝对延时 tau_hat（在 [tau_lo, tau_hi] 内解 1/df 取模分支）。

    时间比较用 **TIME_EPS=1e-18 s**（不是 DOMAIN_EPS=1e-9：后者当秒用等于外扩 ±1 ns ≈ ±0.15 m）。
    多解**不猜**：返回全部候选并标 ambiguous，调用方不得把该 pair 当成功样本。
    """
    period = 1.0 / df_hz
    tau_principal = -float(np.angle(phi_val)) / (2.0 * math.pi * df_hz)
    k_lo = int(math.floor((tau_lo - tau_principal) / period)) - 1
    k_hi = int(math.ceil((tau_hi - tau_principal) / period)) + 1
    in_domain = [(kk, tau_principal + kk * period) for kk in range(k_lo, k_hi + 1)
                 if tau_lo - TIME_EPS <= tau_principal + kk * period <= tau_hi + TIME_EPS]
    base = {"tau_principal_s": tau_principal, "tau_domain_s": [float(tau_lo), float(tau_hi)],
            "period_s": float(period)}
    if not in_domain:
        base.update({"tau_abs_s": None, "branch_index": None, "n_branches_in_domain": 0,
                     "ambiguous": False, "branch_candidates": [],
                     "reason": "delay_mod_period_outside_domain"})
        return base
    if len(in_domain) == 1:
        kk, cand = in_domain[0]
        base.update({"tau_abs_s": float(cand), "branch_index": int(kk),
                     "n_branches_in_domain": 1, "ambiguous": False,
                     "branch_candidates": [[int(kk), float(cand)]], "reason": ""})
        return base
    base.update({"tau_abs_s": None, "branch_index": None, "n_branches_in_domain": len(in_domain),
                 "ambiguous": True,
                 "branch_candidates": [[int(kk), float(cand)] for kk, cand in in_domain],
                 "reason": "delay_branch_ambiguous_in_domain"})
    return base


def _in_domain(pc: ParsedCase, x: float, y: float, r_m: float) -> tuple:
    """**严格**共同域：bbox + 半平面 y<0 + 距离环 + 正距离 + FoV（仅 DOMAIN_EPS 数值容差）。"""
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


def _native_model_matrix(phi_hat: np.ndarray, theta_hat: np.ndarray,
                         m1: int, n_ant: int) -> np.ndarray:
    """原生模型 A[(j,a), l] = phi_l^{j} * theta_l^{a}（式(9)(11)）。"""
    k = int(phi_hat.size)
    A = np.empty((m1 * n_ant, k), dtype=np.complex128)
    for l in range(k):
        ph = phi_hat[l] ** np.arange(m1)
        th = theta_hat[l] ** np.arange(n_ant)
        A[:, l] = (ph.reshape(m1, 1) * th.reshape(1, n_ant)).reshape(-1)
    return A


# --------------------------------------------------------------------------- #
# 公共接口
# --------------------------------------------------------------------------- #
def estimate(case: dict, params: dict | None = None) -> dict:
    """C1（SIJADE / JADE 1998）公共接口：estimate(case[, params]) -> predictions/status/diagnostics。

    params 仅接受 None 或空 dict：参数**冻结**（一致性优先），不接受覆盖以免按结果调参。
    """
    t_start = time.perf_counter()
    diag: dict = {
        "method": "C1",
        "source": ORIGINAL["venue"],
        "doi": ORIGINAL["doi"],
        "original": ORIGINAL,
        "structure": "III-A Hankel 频率左移 + III-B 二维平滑(m1,m2)与前后向共轭增广 + "
                     "III-C SVD/QR 秩削减(含数值秩) + III-E 末段 alternative 独立特征分解+置换配对",
        "frozen_params": FROZEN,
    }
    if params:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag,
                     "本模块参数冻结（FROZEN）：不接受 params 覆盖（收到 %s）"
                     % sorted(str(k) for k in params))
    try:
        pc, parse_diag = parse_case(case)
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001 - 显式失败，不静默
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
    diag["parse"] = parse_diag

    try:
        cube = _hbar_from_case(pc)
        H = _hankel_2d(cube, pc.m1, pc.m2)
        H_e = _forward_backward(H)
        U_hat, sv, rank_info = _rank_reduce(H_e, pc.k, float(FROZEN["numerical_rank_rel_tol"]))
        diag["rank_reduction"] = dict(rank_info)
        if rank_info["energy_zero"]:
            raise MethodError(
                "输入能量为零（H_e 最大奇异值 = 0）：无信号子空间可估计 -> 显式 failed"
                "（不是 NA，也不捏造 subspace）")
        if rank_info["numerical_rank"] < int(pc.k):
            raise MethodError(
                "H_e 的数值秩 %d < k=%d（s_max=%.6e, rel_tol=%.1e, s_head=%s）：给定 K 下不存在 "
                "k 维信号子空间 -> 显式 failed（不捏造 subspace）"
                % (rank_info["numerical_rank"], int(pc.k), rank_info["s_max"],
                   float(FROZEN["numerical_rank_rel_tol"]),
                   [float(v) for v in rank_info["singular_values_head"]]))
        sel = _selection_indices(pc.m1, pc.m2, pc.C)
        U_xphi, U_yphi = U_hat[sel["x_phi"], :], U_hat[sel["y_phi"], :]
        U_xtheta, U_ytheta = U_hat[sel["x_theta"], :], U_hat[sel["y_theta"], :]
        E_xphi, E_yphi = _qr_reduce(U_xphi, U_yphi)
        E_xtheta, E_ytheta = _qr_reduce(U_xtheta, U_ytheta)
        solved = _joint_pairs(E_xphi, E_yphi, E_xtheta, E_ytheta)
        phi_hat, theta_hat = solved["phi"], solved["theta"]

        diag["data_matrix"] = {
            "m1": pc.m1, "m2": pc.m2, "n_rows": int(pc.n_rows),
            "n_cols_per_snapshot": int(pc.m2 * pc.n_freq_shift),
            "n_cols_total_with_forward_backward": int(H_e.shape[1]),
            "n_snapshots": int(pc.L), "rank_k": int(pc.k),
            "forward_backward": True, "hankel_frequency_shift": True,
            "singular_values_head": [float(v) for v in sv[: min(sv.size, pc.k + 2)]],
            "row_index_order": "(j: phi 位移) outer x (a: 阵元) inner（式11）",
            "col_index_order": "(i: 阵元位移) outer x (b: 频点位移) inner（式11）",
            "single_snapshot_hbar_numerical_rank": int(
                np.linalg.matrix_rank(cube[0], tol=1e-10)),
            "narrowband_model_mismatch_note": (
                "本机真实空间相位随频率变化（A5），故单源 Hbar 也不是严格秩 1（本批实测单快照 "
                "rank(Hbar) = min(C, ...) 量级）：宽带回频耦合会以'额外秩'的形式出现，因此"
                "numerical_rank 守卫（1e-8 冻结近似阈值）只用于拦'零能量'与'严格窄带模型下 K>rank'"
                "这两类自测覆盖到的输入，不把宽带失配当秩不足、也不作真实秩的一般断言；"
                "K 仍按 known-K 取前 K 列（口径 §0.2），失配影响如实计入误差来源"),
        }
        diag["pencils"] = solved["report"]

        pairs_out, dropped = [], []
        n_ambiguous = 0
        for l in range(int(pc.k)):
            sin_theta, aliased = _theta_to_sin(complex(theta_hat[l]), pc.delta_ref)
            theta_deg = float(math.degrees(math.asin(float(np.clip(sin_theta, -1.0, 1.0)))))
            tau_lo, tau_hi = _delay_domain_for_theta(pc, theta_deg)
            branch = _delay_branch(complex(phi_hat[l]), pc.df_hz, tau_lo, tau_hi)
            row = {
                "pair_index": int(l),
                "theta_hat_deg": theta_deg,
                "sin_theta_hat": float(sin_theta),
                "theta_out_of_unambiguous_range": bool(aliased),
                "phi_hat": complex(phi_hat[l]),
                "theta_hat": complex(theta_hat[l]),
                "tau_principal_s": branch["tau_principal_s"],
                "tau_domain_ref_anchor_s": branch["tau_domain_s"],
                "branch_index": branch["branch_index"],
                "n_branches_in_domain": branch["n_branches_in_domain"],
                "branch_candidates": branch["branch_candidates"],
                "branch_ambiguous": bool(branch["ambiguous"]),
                "branch_note": branch["reason"],
            }
            if branch["ambiguous"]:
                n_ambiguous += 1
                row["drop_reason"] = "delay_branch_ambiguous_not_scored"
                dropped.append(row)
                continue
            if branch["tau_abs_s"] is None:
                row["drop_reason"] = "delay_mod_1_over_df_outside_domain"
                dropped.append(row)
                continue
            tau_abs = float(branch["tau_abs_s"])
            ref_corr = 0.5 * pc.v_x_ref * math.sin(math.radians(theta_deg))
            r_m = pc.v_prop * (tau_abs - pc.sys_delay_s) / 2.0 + ref_corr
            x = r_m * math.sin(math.radians(theta_deg))
            y = -r_m * math.cos(math.radians(theta_deg))
            row.update({"tau_abs_s": tau_abs, "r_hat_m": float(r_m),
                        "reference_element_correction_m": float(ref_corr),
                        "x": float(x), "y": float(y)})
            ok, why = _in_domain(pc, x, y, r_m)
            if ok:
                pairs_out.append(row)
            else:
                row["drop_reason"] = why
                dropped.append(row)

        A_nat = _native_model_matrix(phi_hat, theta_hat, pc.m1, pc.n_ant)
        G = np.linalg.lstsq(A_nat, H_e, rcond=None)[0]
        energies = np.linalg.norm(G, axis=1)
        for row in pairs_out:
            row["score"] = float(energies[row["pair_index"]])
        for row in dropped:
            row["score"] = float(energies[row["pair_index"]])

        nms = _shared_nms([{"x": p["x"], "y": p["y"], "score": p["score"]} for p in pairs_out],
                          pc.k, float(FROZEN["nms_min_dist_m"]))
        predictions = [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["score"])}
                       for p in nms["kept"]]

        phi_list = [complex(v) for v in phi_hat]
        min_phi_sep = None
        if len(phi_list) > 1:
            seps = [abs(float(np.angle(phi_list[i] / phi_list[j])))
                    for i in range(len(phi_list)) for j in range(i + 1, len(phi_list))]
            min_phi_sep = float(min(seps))
        theta_seps = [abs(float(np.angle(complex(theta_hat[i]) / complex(theta_hat[j]))))
                      for i in range(int(pc.k)) for j in range(i + 1, int(pc.k))]
        diag["identifiability_check"] = {
            "k": int(pc.k),
            "r_max_conditions_min": pc.ident["r_max_conditions_min"],
            "equal_delay_capacity": pc.ident["equal_delay_capacity"],
            "equal_angle_capacity": pc.ident["equal_angle_capacity"],
            "k_exceeds_equal_delay_capacity": bool(pc.k > pc.ident["equal_delay_capacity"]),
            "k_exceeds_equal_angle_capacity": bool(pc.k > pc.ident["equal_angle_capacity"]),
            "min_pairwise_phi_phase_sep_rad": min_phi_sep,
            "min_pairwise_phi_delay_sep_ns": (
                None if min_phi_sep is None
                else float(min_phi_sep / (2.0 * math.pi * pc.df_hz) * 1e9)),
            "min_pairwise_theta_phase_sep_rad": (min(theta_seps) if theta_seps else None),
            "degeneracy_policy": ("只报告不丢弃：同延时条数超过原文容量时 Phi 侧秩退化/配对不可靠；"
                                  "同角度条数超过容量时 Theta 侧同样退化"),
        }
        diag["delay_branch_ambiguity"] = {
            "n_pairs_ambiguous": int(n_ambiguous),
            "n_pairs_total": int(pc.k),
            "domain_span_s": float(pc.tau_span_prop_s[1] - pc.tau_span_prop_s[0]),
            "period_s": float(1.0 / pc.df_hz),
            "unique_capacity": bool(
                (pc.tau_span_prop_s[1] - pc.tau_span_prop_s[0]) <= 1.0 / pc.df_hz + TIME_EPS),
            "policy": ("域内唯一才作为成功候选；多解一律标 ambiguous 并记 dropped（记录全部分支），"
                       "**不用环中心猜**；全部 pair 都 ambiguous 时整体 failed"),
        }
        diag["domain"] = {
            "grid_x_m": [float(pc.grid_x[0]), float(pc.grid_x[-1])],
            "grid_y_m": [float(pc.grid_y[0]), float(pc.grid_y[-1])],
            "theta_max_domain_deg": pc.theta_max_deg,
            "delta_ref_d_over_lambda": pc.delta_ref,
            "wrap_criterion_delta_ref_sin_theta_max": float(
                pc.delta_ref * math.sin(math.radians(pc.theta_max_deg))),
            "wrap_criterion_threshold": 0.5,
            "wrap_criterion_ok": bool(
                pc.delta_ref * math.sin(math.radians(pc.theta_max_deg)) <= 0.5),
            "wrap_margin_fraction": pc.wrap_margin,
            "alias_sin_limit_1_over_2delta": pc.alias_sin_limit,
            "band_edge_delta_ref": [pc.band_edge_delta[0], pc.band_edge_delta[1]],
            "band_edge_wrap_criterion": [
                float(pc.band_edge_delta[0] * math.sin(math.radians(pc.theta_max_deg))),
                float(pc.band_edge_delta[1] * math.sin(math.radians(pc.theta_max_deg)))],
            "note": ("按**严格共同域**判相位唯一性（不按 fov=180 一刀切）：域内 "
                     "D_ref*sin(theta_max) < 0.5 即唯一；band-edge 外推风险另列"),
            "ring_m": None if pc.ring_m is None else [pc.ring_m[0], pc.ring_m[1]],
            "ring_span_m": [pc.ring_span_m[0], pc.ring_span_m[1]],
            "tau_span_propagation_s": [pc.tau_span_prop_s[0], pc.tau_span_prop_s[1]],
            "time_epsilon_s": TIME_EPS,
            "space_epsilon_m": DOMAIN_EPS,
            "delay_branch_period_s": float(1.0 / pc.df_hz),
            "delay_two_way_span_period_m": float(pc.v_prop / (2.0 * pc.df_hz)),
            "delay_branch_unique_capacity": bool(
                (pc.tau_span_prop_s[1] - pc.tau_span_prop_s[0]) <= 1.0 / pc.df_hz + TIME_EPS),
            "delay_branch_period_to_domain_span_ratio": float(
                (1.0 / pc.df_hz) / (pc.tau_span_prop_s[1] - pc.tau_span_prop_s[0])),
            "policy": "严格 bbox + ring + FoV，仅 1e-9 m 容差（与 B1 现行实现一致，无 half-cell）",
        }
        diag["pairs"] = pairs_out
        diag["dropped_pairs"] = dropped
        diag["postprocess"] = nms["stats"]
        diag["score_definition"] = FROZEN["score"]
        diag["n_predictions"] = len(predictions)
        diag["n_candidates_evaluated"] = int(len(pairs_out) + len(dropped))
        diag["n_native_pairs"] = int(pc.k)
        diag["n_domain_dropped"] = int(len(dropped))
        diag["evaluation_boundary"] = ("只给 <=K 个原生候选；评测（含真值）由集成层做，"
                                       "本模块不读真值")
        diag["wall_seconds"] = time.perf_counter() - t_start
        if n_ambiguous == int(pc.k) and int(pc.k) > 0:
            return _pack([], "failed", diag,
                         "全部 %d 个 pair 的 1/df 分支在延时域内不唯一（ambiguous）："
                         "拒绝用环中心猜分支冒充结果（见 dropped_pairs/delay_branch_ambiguity）"
                         % n_ambiguous)
        if not predictions:
            return _pack([], "ok", diag,
                         "无候选：全部原生解落在共同域外（见 dropped_pairs）——允许 <K")
        return _pack(predictions, "ok", diag, "")
    except NotApplicable as exc:
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "not_applicable", diag, str(exc))
    except Exception as exc:  # noqa: BLE001
        diag["wall_seconds"] = time.perf_counter() - t_start
        return _pack([], "failed", diag, "%s: %s" % (type(exc).__name__, exc))
