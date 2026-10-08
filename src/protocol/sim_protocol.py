#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仿真实验协议（2026-09-26 冻结稿）：参数主档、统一静态域、频序、噪声、场景矩阵、方法表。

本模块只产出**可 JSON 序列化的协议字典**（不读真值、不读数据、不运行任何仿真）。
派生量（rmax、角步、时延步、C3 zeropad、C1 相位唯一角）由 sim_common 统一计算，避免硬编码。

场景矩阵（E1-E6）与冻结决策见 freeze_record.md；本文件与 freeze_record.md 一并冻结。
"""

from __future__ import annotations

import math

import sim_common as sc

PROTOCOL_VERSION = "20260929-bandfix-1"  # BANDLIMIT-FIX-20260929：隔离 bundle 修复 analytic 正向未调 band_mask 的缺陷

#: E2 基础几何（横向/径向/斜向）——base (r0, theta0)
E2_BASE_R0_M = 1.8
E2_BASE_THETA0_DEG = 12.0
#: E2 间距集合（m）；最小 0.15（取峰 NMS 0.10）
E2_SPACINGS_M = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
E2_SNR_DB = [0.0, 10.0, 20.0]
E2_TRIALS = 100
#: E2-B（FDTD）代表点
E2_B_SPACINGS_M = [0.30, 0.50, 1.00]
E2_B_SNR_DB = [10.0]
E2_B_TRIALS = 50

#: E1 SNR 档（-10:5:30，共 9 档）
E1_SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
E1_TRIALS = 200
E1_B_TRIALS = 50

#: E1 五个布局（S1-S5，极坐标 (r_m, theta_deg)）；离网格真值由统一微平移施加
E1_LAYOUTS = {
    "S1": [[1.8, 12.0]],
    "S2": [[1.8, 12.0], [1.8, -12.0]],
    "S3": [[1.5, 12.0], [2.1, 12.0]],
    "S4": [[1.5, -12.0], [2.1, 12.0]],
    "S5": [[1.5, -20.0], [1.8, 0.0], [2.1, 20.0]],
}
#: 对所有场景**统一施加**的离网格微平移（米）；不是按性能挑选
OFFGRID_SHIFT_M = [0.017, -0.013]

#: E4（快照敏感性）：同一 128 clean 序列的前 L 切片；L 档
E4_L_LIST = [10, 16, 32, 64, 128]
E4_BASE_LAYOUT = "S2"
E4_TRIALS = 100
E4_B_TRIALS = 50

#: E3 可选挑战（默认关闭）：功率差 x 目标间相关性
E3_OPT_POWER_DIFF_DB = [0.0, 6.0, 12.0]
E3_OPT_CORR = [0.0, 0.9, 1.0]
E3_OPT_TRIALS = 50

#: 主比较组（5 法）与仅验证组
MAIN_METHODS = ["this_method", "b1_deng", "b2_kakouche", "c1_jade", "c3_kim"]
REFERENCE_METHODS = ["b0_standard_music"]


def _geometry() -> dict:
    return {
        "kind": "1Tx4Rx_controlled",
        "tx_positions_m": sc.TX_POSITIONS_M.tolist(),
        "rx_positions_m": sc.RX_POSITIONS_M.tolist(),
        "pairs": sc.PAIRS.tolist(),
        "virtual_x_m": sc.VIRTUAL_X_M.tolist(),
        "spacing_m": float(sc.SPACING_M),
        "strictly_identical_to_hardware_physical_geometry": False,
        "historical_restoration": False,
        "note": sc.GEOMETRY_NOTE,
    }


def _frequency_plan() -> dict:
    idx = sc.in_band_indices()
    freqs = sc.frequencies_hz()
    return {
        "axis_rule": "np.fft.fftfreq(fft_length=32, d=TS) + fc_hz",
        "in_band_indices": [int(i) for i in idx],
        "frequencies_hz": [float(f) for f in freqs],
        "order": "fft_index_ascending_non_monotonic（原始频序，不排序）",
        "n_freq_bins": int(freqs.size),
        "feature_order": "frequency-major then channel (index = f*C + ch)",
        "note": "频点作为有序列表原样传递（FB 的 J 沿整条特征序翻转）；不得排序。",
    }


def _c1_phase_unique_deg() -> float:
    """C1（19 mm @ fc）相位唯一域 |theta| <= asin(1/(2*D_ref))，D_ref = d*fc/v。"""
    d_ref = float(sc.SPACING_M) * float(sc.FC_HZ) / float(sc.C0)
    return float(math.degrees(math.asin(min(1.0, 1.0 / (2.0 * d_ref)))))


def _search() -> dict:
    gx, gy = sc.build_grid()
    ring = sc.distance_ring_m()
    return {
        "kind": "cartesian_2d",
        "grid_x_spec": [float(sc.GRID_X_MIN), float(sc.GRID_X_MAX), float(sc.GRID_STEP_M)],
        "grid_y_spec": [float(sc.GRID_Y_MIN), float(sc.GRID_Y_MAX), float(sc.GRID_STEP_M)],
        "grid_n_points": [int(gx.size), int(gy.size)],
        "grid_step_m": float(sc.GRID_STEP_M),
        "fov_degrees": float(sc.FOV_DEGREES),
        "unified_static_domain": True,
        "not_scene_cropped": True,
        "distance_ring_m": [float(ring[0]), float(ring[1])],
        "ring_taps": [int(sc.CROP_START), int(sc.RAW_TAPS)],
        "half_plane": "y < 0",
        "rmax_m": float(sc.domain_rmax_m()),
        "b1b2_angle_step_deg": float(sc.angle_step_deg()),
        "b1b2_angle_actual": sc.b1b2_angle_actual(),
        "b1b2_tau_actual": sc.b1b2_tau_actual(),
        "c3_target_range_step_m": float(sc.GRID_STEP_M),
        "c3_native_range_bin_m": float(sc.native_range_bin_m()),
        "c3_zeropad_factor": int(sc.c3_zeropad_factor()),
        "c1_phase_unique_max_deg": _c1_phase_unique_deg(),
        "c1_domain_theta_max_deg": _c1_domain_theta_max_deg(),
        "note": ("统一静态域（bbox ±2.2 x [-3.2,-0.55] 5 cm + 硬件 fov=180 + 距离环 tap[2,32]）；"
                 "B1/B2 角步按共同实际 rmax 复算；C1 全域 |theta|max < 相位唯一角 => 无全域 NA。"),
    }


def _c1_domain_theta_max_deg() -> float:
    gx, gy = sc.build_grid()
    import numpy as np
    Xg, Yg = np.meshgrid(gx, gy)
    mask = sc.domain_mask(gx, gy)
    th = np.abs(np.degrees(np.arctan2(Xg[mask], -Yg[mask])))
    return float(np.max(th)) if th.size else 0.0


def _frontend() -> dict:
    return {
        "raw_taps": int(sc.RAW_TAPS),
        "crop_start": int(sc.CROP_START),
        "n_taps_after_crop": int(sc.N_TAPS_WINDOW),
        "fft_length": int(sc.FFT_LENGTH),
        "path": ("raw32 -> cropped=raw[:,:,crop_start:crop_start+30]（v2 起点 0，不裁前 2 物理 tap）"
                 "-> np.fft.fft(cropped, n=fft_length) -> in-band 17 bins"),
        "clean_npz_key": "raw_cir",
        "clean_npz_shape": "[L, 4, 32] complex",
        "time_origin_rule": ("主协议：clean raw 的 raw tap 0 = 物理零延迟（tau = 0）；目标回波写在 "
                             "raw tap = tau/Ts。FDTD 层 tap0 同为物理零，两层的延迟参考一致。"),
        "crop_offset_disclosure": (
            "2026-09-26 v2 口径：crop_start=0，**不裁前 2 个物理抽头**，数据等效延迟 = tau，"
            "与绝对 tau 导向流形之间无固定参考偏移（crop_offset_range_m = %.5f m）。"
            "v1 口径（crop_start=2）曾引入 -2*Ts ≈ -0.29979 m 双程参考偏移，已弃用；"
            "对照验证见 deployment/evidence/nocrop_verify_20260926/。"
            % sc.crop_offset_range_m()),
        "diagnostic_profile_compensated": (
            "adapter/analytic forward 的 compensate_crop 生成档（生成端以 tau+crop_start*Ts 造回波）；"
            "v2 下 crop_start=0 时为恒等（no-op），保留用于口径对照与 v1 复现。"),
        "raw32_vs_algo30_note": "raw_cir（32 tap）与 X（30 截窗 + FFT32 补零）分别命名、分别记录，不得混用。",
    }


def _noise() -> dict:
    return {
        "model": "complex_gaussian_same_variance",
        "normalization": "global_mean_square_over_full_L_x_4ch_x_30tap",
        "reference_position": "final FFT-before actual input CIR (30-tap window starting at tap 0, background-processed)",
        "per_signal": "each case has its own Ps computed over its own L x 4 x 30 noiseless input",
        "noise_prefix_policy": ("E4 快照敏感性：先按最大 L=128 生成 [128,4,30] 标准复高斯，"
                                "再取前 L 个快照（确定前缀）；每 L 的 sigma 按其实际输入重算。"),
        "not_rf_frontend_snr": True,
        "note": ("本口径是接收处理后 SNR（仿真侧统一协议），不是 RF 前端 SNR；"
                 "旧代码两套口径（逐 snapshot 整个 array / 仅(nrx,ntx,time)）均不沿用。"),
    }


def _scenes() -> dict:
    return {
        "E1_A": {
            "source": "analytic", "layer": "A", "name": "原论文任务重建（受控布局 + SNR 扫描）",
            "layouts": {k: [list(t) for t in v] for k, v in E1_LAYOUTS.items()},
            "offgrid_shift_m": list(OFFGRID_SHIFT_M),
            "snr_db": list(E1_SNR_DB), "trials": int(E1_TRIALS), "L": int(sc.SNAPSHOTS),
            "row_count_method_runs": len(E1_LAYOUTS) * len(E1_SNR_DB) * E1_TRIALS * len(MAIN_METHODS),
        },
        "E1_B": {
            "source": "fdtd", "layer": "B", "name": "FDTD 代表场景 + SNR 扫描（同一后端噪声评估）",
            "layouts": {k: [list(t) for t in v] for k, v in E1_LAYOUTS.items()},
            "offgrid_shift_m": list(OFFGRID_SHIFT_M),
            "snr_db": list(E1_SNR_DB), "trials": int(E1_B_TRIALS), "L": int(sc.SNAPSHOTS),
            "row_count_method_runs": len(E1_LAYOUTS) * len(E1_SNR_DB) * E1_B_TRIALS * len(MAIN_METHODS),
        },
        "E2_A": {
            "source": "analytic", "layer": "A", "name": "间距/可分性密扫（横向/径向/斜向）",
            "directions": ["transverse", "radial", "oblique"],
            "base_r0_m": float(E2_BASE_R0_M), "base_theta0_deg": float(E2_BASE_THETA0_DEG),
            "spacings_m": list(E2_SPACINGS_M), "snr_db": list(E2_SNR_DB), "trials": int(E2_TRIALS),
            "L": int(sc.SNAPSHOTS),
            "row_count_method_runs": 3 * len(E2_SPACINGS_M) * len(E2_SNR_DB) * E2_TRIALS * len(MAIN_METHODS),
        },
        "E2_B": {
            "source": "fdtd", "layer": "B", "name": "FDTD 代表点（3 方向 x 3 间距）",
            "directions": ["transverse", "radial", "oblique"],
            "base_r0_m": float(E2_BASE_R0_M), "base_theta0_deg": float(E2_BASE_THETA0_DEG),
            "spacings_m": list(E2_B_SPACINGS_M), "snr_db": list(E2_B_SNR_DB), "trials": int(E2_B_TRIALS),
            "L": int(sc.SNAPSHOTS),
            "row_count_method_runs": 3 * len(E2_B_SPACINGS_M) * len(E2_B_SNR_DB) * E2_B_TRIALS * len(MAIN_METHODS),
        },
        "E3": {
            "name": "配对与稳定性",
            "mandatory": "整场景成功率（共用 E1/E2 同一批 trial，不新增行）",
            "optional": {"power_diff_db": list(E3_OPT_POWER_DIFF_DB), "target_corr": list(E3_OPT_CORR),
                         "trials": int(E3_OPT_TRIALS), "default_enabled": False},
        },
        "E4_A": {
            "source": "analytic", "layer": "A", "name": "快照敏感性（同一 128 clean 序列前 L 切片）",
            "base_layout": E4_BASE_LAYOUT, "L_list": list(E4_L_LIST), "snr_db": list(E1_SNR_DB),
            "trials": int(E4_TRIALS),
            "row_count_method_runs": len(E4_L_LIST) * len(E1_SNR_DB) * E4_TRIALS * len(MAIN_METHODS),
        },
        "E4_B": {
            "source": "fdtd", "layer": "B", "name": "快照敏感性（FDTD 同后端噪声评估）",
            "base_layout": E4_BASE_LAYOUT, "L_list": list(E4_L_LIST), "snr_db": list(E1_SNR_DB),
            "trials": int(E4_B_TRIALS),
            "row_count_method_runs": len(E4_L_LIST) * len(E1_SNR_DB) * E4_B_TRIALS * len(MAIN_METHODS),
        },
        "E6": {
            "name": "复杂度/资源（cold/warm 计时）",
            "base_layout": E4_BASE_LAYOUT, "snr_db": 10.0,
            "definitions": {
                "warm": "从已构建字典/流形 + 共有 X 起算 -> pred",
                "cold": "从共有 X 起算，并计入本次所需的字典/流形构建 -> pred",
                "common_frontend": "通用前端 CIR->X（FFT/校准）warm/cold 均另列，不并入上述两段",
            },
            "report": "median 与 P95，及峰值内存",
            "note": "FDTD 求解时间不计入算法时间；warm 必须是真正预构字典（不可只跑两遍）。",
        },
    }


def _method_params() -> dict:
    """逐方法搜索剖面（grid5cm 对齐）；C1 闭式无参数。"""
    return {
        "this_method": {},
        "b0_standard_music": {},
        "b1_deng": {"angle_step_deg": float(sc.angle_step_deg()),
                    "tau_step_distance_m": float(sc.GRID_STEP_M)},
        "b2_kakouche": {"angle_step_deg": float(sc.angle_step_deg()),
                        "tau_step_distance_m": float(sc.GRID_STEP_M)},
        "c1_jade": {},
        "c3_kim": {"target_range_step_m": float(sc.GRID_STEP_M)},
    }


def build_protocol() -> dict:
    """构建协议字典（纯函数，确定性）。"""
    return {
        "kind": "sim-protocol",
        "protocol_id": "sim-20260926-unified",
        "protocol_version": PROTOCOL_VERSION,
        "band_limiting": {
            "marker": "BANDLIMIT-FIX-20260929",
            "primary_band_mode": "bandlimited_500mhz",
            "diagnostic_band_mode": "full_band_ideal",
            "implemented_in": "analytic/forward.py::raw_cir_from_coeffs",
            "mask": "|f-fc|<=B/2 -> 1, else 0（32 个 fftfreq bin 中 17 个带内 = 17/32）",
            "note": ("冻结源定义 band_mask 但从未调用（缺陷 BANDLIMIT-ANALYTIC-FORWARD-NO-MASK）；"
                     "隔离 bundle 内显式乘掩膜修复，并推高本版本号使旧 analytic clean 签名失配、不复用。"),
        },
        "created_at": "2026-09-26",
        "frozen": True,
        "purpose": ("恢复 V 章算法仿真：解析受控层（A）+ FDTD 实际散射层（B），单一协议、"
                    "单一后端、统一网格/噪声/指标；同时支撑 Q4/Q5/Q6。"),
        "fc_hz": float(sc.FC_HZ),
        "bandwidth_hz": float(sc.BANDWIDTH_HZ),
        "sample_interval_s": float(sc.SAMPLE_INTERVAL_S),
        "raw_taps": int(sc.RAW_TAPS),
        "crop_start": int(sc.CROP_START),
        "fft_length": int(sc.FFT_LENGTH),
        "snapshots": int(sc.SNAPSHOTS),
        "n_channels": int(sc.N_CHANNELS),
        "manifold_dim": int(sc.N_CHANNELS * sc.n_freq_bins()),
        "epsilon_r_background": float(sc.EPSILON_R_BACKGROUND),
        "system_delay_si": float(sc.SYSTEM_DELAY_SI),
        "diag_load_factor": float(sc.DIAG_LOAD_FACTOR),
        "is_forward_backward": bool(sc.IS_FORWARD_BACKWARD),
        "geometry": _geometry(),
        "frequency_plan": _frequency_plan(),
        "search": _search(),
        "frontend": _frontend(),
        "noise": _noise(),
        "tau_convention": {
            "model": "exact two-way spherical range per channel: tau = (||p-tx||+||p-rx||)/v_prop + system_delay",
            "response_assumption": "S_equals_1_pure_phase（共同假设，不是已校准接收响应）",
        },
        "coordinate_convention": {
            "polar_to_xy": "p = (r*sin(theta), -r*cos(theta))",
            "theta_sign": "theta>0 = +X 侧，视轴 -Y",
            "xy_to_polar": "r=hypot(x,y), theta=degrees(atan2(x,-y))",
        },
        "methods": {
            "main_comparison": list(MAIN_METHODS),
            "same_model_reference": list(REFERENCE_METHODS),
            "archived": [],
            "params_by_method": _method_params(),
            "note": "B0 仅作同模型正确性校验，不是性能对手；所有方法共享同一 X/噪声/域/known-K。",
        },
        "metrics": {
            "gate_m": float(sc.GATE_M), "ospa_c_m": float(sc.OSPA_C_M), "ospa_p": 2,
            "rms_ospa": "sqrt(mean_t OSPA_t^2)",
            "tp_conditional_errors": ["e_r", "e_theta", "e_xy"],
            "failures_counted_in_denominator": True,
            "e2_resolution_gate": "min(0.10, d/3)",
            "na_policy": "NA 不缩小主分母（由 metrics 负责；runner 传状态并保留全部行）",
        },
        "scenes": _scenes(),
        "e0": {
            "checks": ["numerical_consistency_b0_vs_this_method", "ideal_locality_diagnostic",
                       "crop_mismatch_disclosure", "zero_offset_cfr_diagnostic"],
            "note": "数值一致性与理想定位正确性分开；crop2/当前 FB 的可预期失配如实披露，不为准确率偷偷修补。",
        },
        "e6": {"threads": "BLAS/OMP 固定为 1", "timer": "time.perf_counter",
               "report": ["median", "p95", "peak_memory"]},
        "fdtd_defaults": {
            # 半径统一 = 1.5*lambda（lambda = c/fc ≈ 0.037551 m -> 0.056327 m）；
            # 本键是**唯一**来源，scene_export / fdtd 侧不得再用不同默认值。
            "scatterer_radius_m": float(1.5 * sc.C0 / sc.FC_HZ),
            "lambda_m": float(sc.C0 / sc.FC_HZ),
            "micro_motion_sigma_m": 0.002,
            "micro_motion_radius_m": 0.01, "resolution_px_per_m": 250.0,
            "run_time_meep": 20.0, "pml_thickness_lambda": 2.0,
            "background_mode": "empty_field",
            "note": ("FDTD（层 B）由 fdtd 代理负责（fdtd/api.py::generate_clean）。层 B 的 tap0 同为"
                     "物理零延迟；同一后端噪声评估由本 runner 施加（与层 A 完全同一噪声口径）。"
                     "250/350 网格收敛 + 跑时长前置验证由 fdtd 代理执行；非全量双档。"),
            "separation_rule": ("E1/E2 目标间距必须 >= 2*r_target + 2*r_traj + margin = %.3f m"
                                % (2 * 0.05 + 2 * 0.01 + 0.005)),
        },
        "decisions": {
            "E4_trials": int(E4_TRIALS),
            "E4_b_base_layout": E4_BASE_LAYOUT,
            "e1_clean_signal_fixed_per_case": True,
            "e1_trials_are_noise_only": True,
            "independent_source_mc_separately_labeled": True,
            "unified_static_domain": True,
        },
    }


PROTOCOL = build_protocol()


def protocol_hash(proto: dict | None = None) -> str:
    return sc.param_hash(proto if proto is not None else PROTOCOL)


if __name__ == "__main__":  # pragma: no cover - 人工查看
    import json
    print(json.dumps(PROTOCOL, ensure_ascii=False, indent=1))
