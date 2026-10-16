#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q6 硬件侧冻结“搜索剖面”（search profile）：把各基线的采样结构与本方法 5 cm 网格对齐。

为什么需要单独一层
------------------
* 输入 cache（`hardware/cache/cases/*.npz`）**冻结不改**；搜索步长不能靠改写 cache 里的
  `one_dimensional_budget` 悄悄生效（那会与 C2 的专用预算混淆，主审 2026-09-24 明确否决）。
* 因此把对齐后的步长写成**显式 profile → 方法 params 覆盖**，由运行器 `--search-profile`
  下发；方法内部优先级为 **显式 params > case 一维预算 > 冻结默认**，并在 diagnostics 申报来源。
* 本文件只给常量与映射，不做任何估计；不读数据、不读真值。

统一标准（与 `hardware/analysis_grid5cm/01_对齐方案_grid5cm_20260924.md` 一致）
----------------------------------------------------------------------------
参照 = 本方法 Cartesian 网格 dx = dy = 0.05 m（**不动**）。

* **S1（统一口径）**：各**离散搜索维**的相邻采样间距 <= 5 cm（换算到共同几何域）。
* 参照量（**不是**对所有方法的统一阈值）：参照 Cartesian 网格的**每轴**半格 2.5 cm，其欧氏合成
  为 5/sqrt(2) = 3.5355 cm。**不存在**“所有方法统一满足欧氏半格 <= 2.5 cm”的全局 S2。
* **C2 附加（保守设计）**：因逐通道 τ 量化经 2 参数 LS 耦合放大（Σ|a_c|/Σa_c² = 42.105·c），
  设计要求全域**欧氏**半格位移 <= 2.5 cm；这比参照网格自身的欧氏半格（3.5355 cm）略严格，
  属保守、且与参照**非完全等价**（已在方法与 profile 文本中明示）。
* 只描述**采样量化**，不含性能结论；不用真值；不按效果调参。

profile = `grid5cm` 的逐方法设定与依据
------------------------------------
* **B1/B2（角度 1-D + 条件距离 1-D）**：角度步 0.6 deg（端点对称 ±half、`n = ceil(2·half/step)+1`
  ⇒ 实际步长 <= 0.6 deg；本批 half = 75.9638 deg ⇒ n = 255、实际 0.5981398 deg，
  r_max = 4.7413 m 处切向间距 0.0494969 m、半格 0.0247485 m）；距离步 0.05 m ⇒ 双程
  τ 步 = 2·0.05/c = 333.5641 ps（时延域 [2, 32] ns ⇒ n = 90、实际 333.3333 ps ⇒ 距离 0.0499654 m）。
  旧值（角 0.05 deg / τ 3.1689 ps ⇒ 距离 0.000475 m）比参照细 24x/105x，属于**采样预算不一致**。
* **C2（逐通道 τ 网格 + 2 参数 LS）**：LS 把逐通道 ±dt/2 量化误差按 Σ|a_c|/Σa_c² = 42.105·c
  放大到 sinθ，再经全域几何放大（角点 r/cosθ = 9.35 m）产生位置扰动。
  独立数学审阅（`hardware/verification_grid5cm/`，16 PASS）给出 dt = 0.4 ps 时
  半格位置扰动 = 0.0246849 m <= 0.025 m（本代理精确有限界复算 0.024148 m，同一结论）。
  故 **dt = 0.4 ps**（距离步 0.05996 mm）；旧值 3.1689 ps 对应全域最坏 0.2371 m，远超预算。
* **C1（JADE，闭式平移不变）**：无采样网格（`pc.tau_step_s` 不进入估计路径），**不改代码**
  （保持 sha 与旧结果同源），也不需要补诊断。
* **C3（DFT-ESPRIT 的 CIR 适配版）**：原生 range bin 0.2821576 m（F = 17、df = 31.25 MHz）。
  取最小整数 zeropad = ceil(0.2821576/0.05) = **6** ⇒ Δτ = 0.3137255 ns、Δrange = 0.0470263 m、
  半格 0.0235131 m。zeropad 只是把同一 Dirichlet 核插值到更密格点，**不新增信息、不提高
  带宽真实分辨力、不改原算法顺序**。

使用::
    from q6_search_profiles import params_for, profile_record, PROFILE_NAMES
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

C0 = 299792458.0

#: 参照网格（本方法 Cartesian 步长，米；**不动**）
THIS_METHOD_GRID_M = 0.05
#: 参照网格的**每轴**半格（米）——统一口径的参照量，**不是**所有方法都必须满足的全域阈值
REFERENCE_EACH_AXIS_HALF_CELL_M = 0.5 * THIS_METHOD_GRID_M
#: 参照网格的**欧氏**半格（米）= 0.05/sqrt(2)（每轴 2.5 cm 对应的欧氏合成）
REFERENCE_EUCLIDEAN_HALF_CELL_M = THIS_METHOD_GRID_M / math.sqrt(2.0)
#: **仅 C2** 采用的保守附加设计阈值（米）：因逐通道 τ 量化经 2 参数 LS 耦合放大，
#: 要求全域欧氏半格位移 <= 2.5 cm。它比参照网格自身的欧氏半格（3.5355 cm）略严格，
#: 属**保守设计**，不代表“所有方法的统一 S2 阈值”。
C2_CONSERVATIVE_HALF_CELL_DISPLACEMENT_MAX_M = 0.025

#: profile 名 -> 逐方法 params（None = 不下发覆盖，方法走 case 预算/冻结默认）
PROFILES = {
    "native": {
        "kind": "q6-search-profile",
        "description": "现行行为（不下发任何覆盖）：B1/B2 走 case 一维预算，C2 走 case 预算，C3 原生 bin。"
                       "**保留旧参数**（case 预算/冻结默认/原生 bin），用于对照旧 full214 结果；"
                       "注意本轮把 B1/B2 的角度端点规则由 round 固定为 ceil（本批默认 0.05 度下 n = 3040"
                       "与旧值相同，故本批网格逐位一致），但**不承诺任意 case 与旧结果逐位一致**。",
        "params_by_method": {},
    },
    "grid5cm": {
        "kind": "q6-search-profile",
        "description": "各离散搜索维相邻采样 <= 5 cm；C2 另用保守的 LS 量化预算（全域欧氏半格 <= 2.5 cm）；"
                       "连续域/闭式方法保留原结构",
        "standard": {
            "reference_grid_m": THIS_METHOD_GRID_M,
            "S1_each_dimension_adjacent_sampling_max_m": THIS_METHOD_GRID_M,
            "reference_each_axis_half_cell_m": REFERENCE_EACH_AXIS_HALF_CELL_M,
            "reference_euclidean_half_cell_m": REFERENCE_EUCLIDEAN_HALF_CELL_M,
            "c2_conservative_half_cell_displacement_max_m": C2_CONSERVATIVE_HALF_CELL_DISPLACEMENT_MAX_M,
            "note": ("统一口径只有 S1（各离散搜索维相邻采样间距 <= 5 cm）。**不存在**“所有方法统一满足"
                     "欧氏半格 <= 2.5 cm”的全局 S2：参照 Cartesian 网格每轴半格 2.5 cm 对应欧氏 5/sqrt(2) "
                     "= 3.5355 cm；C2 因 LS 耦合另用保守的欧氏半格 <= 2.5 cm 设计（更严格、非完全等价）。"),
            "scope": "仅采样量化（网格取整）的最坏位置扰动；不是估计精度，不保证峰选择正确",
            "uses_truth": False,
        },
        "params_by_method": {
            # B1/B2：角度 0.6 deg（端点规则改 ceil，实际步长 <= 请求值）；距离 0.05 m -> 双程 τ = 2·0.05/c
            "b1_deng": {"angle_step_deg": 0.6,
                        "tau_step_distance_m": THIS_METHOD_GRID_M},
            "b2_kakouche": {"angle_step_deg": 0.6,
                            "tau_step_distance_m": THIS_METHOD_GRID_M},
            # C2：LS 放大后的全域量化预算要求（独立审阅 16 PASS）
            "c2_navarro": {"tau_step_s": 4.0e-13},
            # C3：目标距离步 -> zeropad = max(1, ceil(native_range_bin/target))
            "c3_kim": {"target_range_step_m": THIS_METHOD_GRID_M},
        },
        "derivations": {
            "b1_b2_angle": ("半格切向位移 = r_max·(Δθ_actual/2)，r_max = 4.7413 m、Δθ_actual = 0.5981398 deg "
                            "⇒ 0.0247485 m <= 0.025 m；若沿用 round 规则则 n = 254 ⇒ 实际 0.600504 deg > 0.6 deg（超请求）"),
            "b1_b2_range": "半格 = 0.05/2 = 0.025 m；实际步 0.0499654 m（含端点 linspace）",
            "c2": ("Σ|a_c|/Σa_c² = c·0.076/0.001805 = 1.262284e10 s⁻¹；δs = (dt/2)·ampl；"
                   "位移 <= max_bbox r·|asin(clip(s+δs))-asin(s)| + c·|δτ0|/2，δτ0 <= dt/2（w = [.1,.2,.3,.4]）；"
                   "dt = 0.4 ps ⇒ 0.024148 m（本代理精确复算）；独立审阅 0.0246849 m（更保守导数界），均 <= 0.025 m"),
            "c3": "native_range_bin = v·1/(F·df)/2 = 0.2821576 m；ceil(0.2821576/0.05) = 6 ⇒ 0.0470263 m、半格 0.0235131 m",
            "c1": "闭式平移不变，无网格；`pc.tau_step_s` 不进入估计路径（源码检索 0 次）",
        },
    },
}

PROFILE_NAMES = tuple(PROFILES)


class ProfileError(ValueError):
    """profile 名或内容不合规（显式失败，不静默降级）。"""


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ProfileError("未知 search profile %r；可用：%s" % (name, ", ".join(PROFILE_NAMES)))
    return PROFILES[name]


def params_for(name: str, method: str) -> dict | None:
    """返回该 profile 对某方法的 params 覆盖（无覆盖 -> None，表示按现行行为）。"""
    prof = get_profile(name)
    params = prof.get("params_by_method", {}).get(method)
    return None if not params else dict(params)


def profile_record(name: str) -> dict:
    """运行记录用：profile 名 + 本模块 sha256 + 常量 + 逐方法 params（写入 meta.json）。"""
    prof = get_profile(name)
    path = Path(__file__).resolve()
    return {
        "kind": "q6-search-profile-record",
        "name": name,
        "module": str(path),
        "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "description": prof.get("description"),
        "standard": prof.get("standard"),
        "params_by_method": {k: dict(v) for k, v in prof.get("params_by_method", {}).items()},
        "derivations": prof.get("derivations"),
    }


def b1b2_realized(angle_step_deg: float, half_deg: float, tau_span_s, v_prop: float,
                  distance_step_m: float) -> dict:
    """按冻结端点规则复算 B1/B2 实际步长（供 test/审计用；同样不读真值）。"""
    span = 2.0 * float(half_deg)
    n_ang = int(math.ceil(span / float(angle_step_deg) - 1e-12)) + 1
    step_tau_req = 2.0 * float(distance_step_m) / float(v_prop)
    n_tau = int(math.ceil((float(tau_span_s[1]) - float(tau_span_s[0])) / step_tau_req))
    step_tau_real = (float(tau_span_s[1]) - float(tau_span_s[0])) / n_tau
    return {"angle_n_points": n_ang, "angle_actual_step_deg": span / (n_ang - 1),
            "tau_step_requested_s": step_tau_req,
            "tau_grid_n_points": n_tau + 1, "tau_step_realized_s": step_tau_real,
            "range_step_realized_m": float(v_prop) * step_tau_real / 2.0}


def c3_zeropad_for(native_range_bin_m: float, target_range_step_m: float) -> int:
    return max(1, int(math.ceil(float(native_range_bin_m) / float(target_range_step_m) - 1e-12)))


if __name__ == "__main__":  # pragma: no cover - 人工查看用
    import json
    print(json.dumps(profile_record("grid5cm"), ensure_ascii=False, indent=1))
