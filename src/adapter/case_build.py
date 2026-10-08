#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 X 装配成冻结后端 case（通过 q6_contract.validate_case 硬门控；真值不进入 case）。

case 结构与硬件侧一致（键集见 q6_contract.REQUIRED_CASE_KEYS）；本模块**只装配**，
不做 tap 补偿、不改频序、不排序、不用真值。
"""

from __future__ import annotations

import json
import math
import sys

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np

FEATURE_ORDER = "frequency-major then channel (index = f*C + ch)"


def one_dimensional_budget(v_prop: float) -> dict:
    ring = sc.distance_ring_m()
    span_prop = (2.0 * ring[0] / v_prop, 2.0 * ring[1] / v_prop)
    return {
        "round_trip_tau_span_s": [float(span_prop[0]), float(span_prop[1])],
        "suggested_one_d_tau_step_s": float(2.0 * sc.GRID_STEP_M / v_prop),
        "distance_ring_m": [float(ring[0]), float(ring[1])],
        "note": ("只用于披露原生一维网格粒度与代价；方法按原生结构搜索并在 diagnostics 申报点数。"
                 "B1/B2 一律用显式 params.tau_step_distance_m 覆盖为 5 cm 距离步。"),
    }


def build_config(*, k: int) -> dict:
    ring = sc.distance_ring_m()
    return {
        "fft_length": int(sc.FFT_LENGTH),
        "tap_interval_si": float(sc.SAMPLE_INTERVAL_S),
        "center_freq_si": float(sc.FC_HZ),
        "bandwidth_si": float(sc.BANDWIDTH_HZ),
        "epsilon_r_background": float(sc.EPSILON_R_BACKGROUND),
        "system_delay_si": float(sc.SYSTEM_DELAY_SI),
        "diag_load_factor": float(sc.DIAG_LOAD_FACTOR),
        "is_forward_backward": bool(sc.IS_FORWARD_BACKWARD),
        "fov_degrees": float(sc.FOV_DEGREES),
        "cir_window": [int(sc.CROP_START), int(sc.FFT_LENGTH)],
        "freq_range_si": [float(v) for v in sc.freq_range_si()],
        "deg_corr": None,
        "ant_calib_phase": None,
        "ant_calib_applied_in_X": "无（解析层无实测校准矢量）",
        "propagation_speed_m_s": float(sc.C0 / np.sqrt(sc.EPSILON_R_BACKGROUND)),
        "distance_ring_m": [float(ring[0]), float(ring[1])],
        "tap_handling": ("X 由 cropped tap（raw[:,:,crop_start:]）经 fft(n=fft_length) 构造；"
                         "cir_window 仅用于距离环掩膜；raw tap0 = 物理零延迟，crop 固定偏移如实披露、不补偿"),
        "mdl_max_targets": int(max(int(k), 3)),
        "crop_offset_range_m": float(sc.crop_offset_range_m()),
    }


def build_search(grid_x, grid_y) -> dict:
    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)
    return {
        "kind": "cartesian_2d",
        "grid_x_m": gx, "grid_y_m": gy,
        "grid_spec_source": [float(sc.GRID_X_MIN), float(sc.GRID_X_MAX), float(sc.GRID_Y_MIN),
                             float(sc.GRID_Y_MAX), float(sc.GRID_STEP_M)],
        "resolution_m": float(gx[1] - gx[0]),
        "n_grid_points": int(gx.size * gy.size),
        "half_plane": "y < 0（与硬件实现一致）",
        "one_dimensional_budget": one_dimensional_budget(float(sc.C0 / np.sqrt(sc.EPSILON_R_BACKGROUND))),
        "unified_static_domain": True,
        "fairness_note": ("候选域与分辨率预算统一披露；原生分步/连续方法按各自结构搜索，"
                          "不得声称所有方法在同一 Cartesian 点集上逐点比较。"),
    }


def build_model() -> dict:
    return {
        "geometry": ("exact two-way spherical range per channel: "
                     "tau = (||p - tx|| + ||p - rx||) / v_prop + system_delay"),
        "response_assumption": {
            "mode": "S_equals_1_pure_phase",
            "vector": None,
            "is_calibrated_response": False,
            "note": ("所有方法共用的纯相位（S=1）假设；不是已校准接收响应，也不是完全原论文复现；"
                     "误差同时受该模型适配影响。"),
        },
        "channel_calibration": {"ant_calib_en": False, "applied": "无", "deg_corr": None},
        "fb_note": ("FB（前后向平均）是方法自身的处理选择；case 只给未做 FB 的 X。"
                    "各方法按原文原生结构决定是否 FB。"),
        "frequency_grid": {
            "n": int(sc.n_freq_bins()),
            "frequencies_hz": sc.frequencies_hz(),
            "order": "fft_index_ascending_non_monotonic（原始频序）",
        },
        "time_origin": "raw tap0 = physical zero; crop_start offset retained & disclosed",
        "crop_offset_range_m": float(sc.crop_offset_range_m()),
    }


def build_case(*, case_id, clean_id, X, k, grid_x, grid_y, layer, source, scene, tags,
               snr_db, trial, seed, L, resolution_distance_m=None, extra_diag=None,
               geometry=None) -> dict:
    """装配方法可见 case。**不放入真值/几何先验**（resolution_distance_m / tags / 目标坐标等
    一律留在 runner row/provenance）；只保留非真值 generic id 与必要参数。

    X 冻结为只读（flags.writeable=False），防止方法就地修改；调用方在方法前后比对 case 指纹。
    geometry 选填：Q4 等专用输入几何可覆盖默认 1Tx4Rx（默认不变 = 协议主几何）。
    """
    geom = dict(geometry or {})
    tx_pos = np.asarray(geom.get("tx_positions_m", sc.TX_POSITIONS_M), dtype=float)
    rx_pos = np.asarray(geom.get("rx_positions_m", sc.RX_POSITIONS_M), dtype=float)
    pairs = np.asarray(geom.get("pairs", sc.PAIRS), dtype=np.int64)
    X = np.asarray(X, dtype=np.complex128)
    if X.flags.writeable:
        X = X.copy()
    X.flags.writeable = False
    case = {
        "case_schema_version": 1,
        "id": opaque_id("case", str(case_id)),
        "recording_id": opaque_id("rec", str(clean_id), int(trial)),
        "environment": ("analytic" if source == "analytic" else "fdtd"),
        "config_id": opaque_id("cfg", str(clean_id), _fmt_snr(snr_db), int(L)),
        "t_end_s": float(trial),
        "X": X,
        "frequencies_hz": sc.frequencies_hz(),
        "pairs": pairs,
        "tx_positions": tx_pos,
        "rx_positions": rx_pos,
        "k": int(k),
        "grid_x": np.asarray(grid_x, dtype=float),
        "grid_y": np.asarray(grid_y, dtype=float),
        "config": build_config(k=k),
        "search": build_search(grid_x, grid_y),
        "model": build_model(),
        "feature_order": FEATURE_ORDER,
        "n_snapshots": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_freq_bins": int(sc.n_freq_bins()),
        "n_channels": int(pairs.shape[0]),
        "snapshots_available": False,
    }
    return case


def _fmt_snr(snr_db) -> str:
    """SNR 标签：有限值用 snr%+.0f；None/inf/-inf 用稳定字串（避免 %d/%f 对非有限量报错）。"""
    if snr_db is None:
        return "snr=none"
    v = float(snr_db)
    if math.isfinite(v):
        return "snr%+.0f" % v
    return "snr=%s" % ("+inf" if v > 0 else "-inf")


def opaque_id(kind: str, *parts) -> str:
    """方法可见 id/config_id：不透明身份（哈希），不含 E2 d/方向等明文几何先验。

    明文 case_id / 方向 / 间距 / 真值只留在 runner 的 row（provenance/绘图用），不进 case。
    """
    return "opq_%s_%s" % (kind, sc.param_hash(*parts)[:16])


def validate_case(case: dict) -> None:
    """用 vendor q6_contract 硬门控（真值字段/形状/网格）。不可用即显式失败。"""
    if str(sc.VENDOR_DIR / "common") not in sys.path:
        sys.path.insert(0, str(sc.VENDOR_DIR / "common"))
    try:
        import q6_contract as contract
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("vendor q6_contract 不可导入：%s" % exc)
    contract.validate_case(case)
    leaks = contract.find_truth_leaks(case)
    if leaks:
        raise RuntimeError("case 含疑似真值字段：%s" % ", ".join(leaks[:8]))


def case_fingerprint(case: dict) -> str:
    """case 数据指纹（X 字节 + 关键参数）；用于方法调用前后比对，检测就地修改。"""
    if str(sc.VENDOR_DIR / "common") not in sys.path:
        sys.path.insert(0, str(sc.VENDOR_DIR / "common"))
    import q6_contract as contract
    return contract.case_fingerprint(case)


__all__ = ["build_case", "build_config", "build_search", "build_model",
           "one_dimensional_budget", "validate_case", "case_fingerprint", "FEATURE_ORDER"]
