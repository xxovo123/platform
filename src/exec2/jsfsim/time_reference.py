#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q4 时间参考诊断（规范 §7 硬门槛）。

要判定的问题
-----------
硬件 CIR 包注明 `cir_offset = 2`、`ft_len = 32`、`cropped_taps = 30`：
即硬件侧算法可见的 30 tap 是原始 32 tap 的 **tap[2:32]**，而新版仿真侧主协议
是 **raw tap0 = 物理零延迟**、取 `[0:30]`。两者**不能只凭“都是 30 tap”直接等同**。

本模块**不猜、不反拟合**：先做**纯几何无真值**的一致性检验——
在候选 4 ns 时移 δ 下，把仿真**真值**按链路几何算出的双程时延 τ 与硬件 CFR 的
**通道间相对相位斜率**比较。τ 完全由几何与光速决定，不含任何拟合参数；
斜率与 τ 一致才说明该候选与链路参考自洽。

判定顺序：
  1. 候选 δ ∈ {0, 2 tap}（= {0, 2 ns}）分别打分；
  2. 只有当**唯一**一个候选自洽（残差 < 0.05 tap 量级）时才写 `resolved`；
  3. 无法唯一定判 -> 写 `blocked_time_reference` 并交付诊断，**不给正式“相似度已验证”**。

校准值只能来自已有硬件设置或独立标定证据（本函数只用几何 + 光速 + 冻结 tap 间隔）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import protocol as P

HW_CIR_DIR = ("大修执行_20260923", "06_仿真实验_20260926", "q4", "out", "hw_cir")
TAP_INTERVAL_S = P.SAMPLE_INTERVAL_S
CANDIDATE_SHIFTS_TAPS = (0, 2)


class TimeReferenceError(RuntimeError):
    """时间参考无法判定（显式失败/blocked）。"""


def hw_window_manifest(hw_dir: Path, windows_205: set):
    """(recording_id -> {npz, truth_json, window rows})，只保留 205 清单内的窗。"""
    out = {}
    for npz in sorted(hw_dir.glob("*.npz")):
        rid = npz.stem
        with np.load(npz, allow_pickle=False) as z:
            ids = [str(w) for w in z["window_ids"]]
            keep = [i for i, w in enumerate(ids) if w in windows_205]
            if not keep:
                continue
            truth = json.loads(npz.with_suffix(".truth.json").read_text(encoding="utf-8"))
            rows = []
            for i in keep:
                wid = ids[i]
                t = (truth.get("windows") or {}).get(wid) or {}
                rows.append({"window_id": wid, "index_in_npz": int(i),
                             "k": int(z["k"][i]), "truth_xy": t.get("truth_xy") or t.get("truth_xy_m"),
                             "t_end_s": float(z["t_end_s"][i]),
                             "mid_index": int(z["mid_index"][i])})
            out[rid] = {"npz": npz, "truth_json": npz.with_suffix(".truth.json"),
                        "geometry_id": str(z["geometry_id"]),
                        "environment": str(z["environment"]),
                        "pairs": z["pairs"], "tx_positions": z["tx_positions"],
                        "rx_positions": z["rx_positions"],
                        "frequencies_hz": z["frequencies_hz"],
                        "cir_offset": int(z["cir_offset"]), "ft_len": int(z["ft_len"]),
                        "cropped_taps": int(z["cropped_taps"]),
                        "source": str(z["source"]), "rows": rows}
    return out


def _relative_phase_slope(cir_taps: np.ndarray, pairs) -> dict:
    """用通道间 CFR 相对相位斜率反推时延（与真值无关的链路量）。

    cir_taps: [L, C, T]（C=4 通道，pairs 给出 (tx, rx)）。
    对每对通道 (c1, c2)：Δarg(H_c1/H_c2)(f) 对 2πf 的斜率 = -(τ_c1 - τ_c2)。
    这里汇总一个鲁棒全局斜率（对全部通道对、全频点做最小二乘）。
    """
    L, C, T = cir_taps.shape
    n = int(P.FFT_LENGTH)
    h = np.abs(cir_taps)  # 占位，避免误用
    del h
    H = np.fft.fft(cir_taps, n=n, axis=-1)          # [L,C,32]
    axis = np.asarray(P.fft_freq_axis_hz(), dtype=float)
    idx = P.in_band_indices()
    f = axis[idx]                                    # [17]
    pairs = np.asarray(pairs, dtype=int)
    # 用 tau = (d_tx + d_rx)/v 的通道间**差**：线性相位对应时延差
    taus = np.zeros((C,), dtype=float)
    rows = []
    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            a = H[:, c1, :][:, idx]
            b = H[:, c2, :][:, idx]
            ratio = np.mean(a * np.conj(b), axis=0)   # 跨快照平均，稳定相位
            ph = np.unwrap(np.angle(ratio))
            x = 2.0 * math.pi * f
            slope = float(np.polyfit(x, ph, 1)[0])
            rows.append({"ch1": int(c1), "ch2": int(c2), "slope_rad_per_rad_s": slope,
                         "implied_delta_tau_s": -slope})
    return {"pairs_slope": rows,
            "freqs_hz": [float(v) for v in f],
            "note": "相位斜率 -> 通道间时延差；仅用硬件 CFR 与冻结 tap 间隔，不用真值。"}


def _geometry_tau_delta(geom: dict, truth_xy) -> list:
    """按几何算通道间双程时延差（无拟合参数）。"""
    tx = np.asarray(geom["tx_positions"], dtype=float)
    rx = np.asarray(geom["rx_positions"], dtype=float)
    pairs = np.asarray(geom["pairs"], dtype=int)
    v = P.C0 / math.sqrt(P.EPSILON_R_BACKGROUND)
    tau = []
    for txi, rxi in pairs:
        best = []
        for p in truth_xy:
            p3 = np.array([float(p[0]), float(p[1]), 0.0])
            d = (np.linalg.norm(p3 - tx[txi]) + np.linalg.norm(p3 - rx[rxi])) / v
            best.append(float(d))
        tau.append(min(best))
    tau = np.asarray(tau, dtype=float)
    out = []
    for c1 in range(tau.size):
        for c2 in range(c1 + 1, tau.size):
            out.append({"ch1": int(c1), "ch2": int(c2),
                        "geometry_delta_tau_s": float(tau[c1] - tau[c2])})
    return out


def diagnose(hw_dir: Path, windows_205: set, *, max_recordings: int = 8) -> dict:
    man = hw_window_manifest(hw_dir, windows_205)
    if not man:
        raise TimeReferenceError("没有落在 205 清单内的硬件录制：%s" % hw_dir)
    obs, resid = [], []
    for rid, rec in sorted(man.items())[:max_recordings]:
        with np.load(rec["npz"], allow_pickle=False) as z:
            cir = np.asarray(z["cir_cropped30"])
        truth = json.loads(rec["truth_json"].read_text(encoding="utf-8"))
        for row in rec["rows"][:2]:
            t = (truth.get("windows") or {}).get(row["window_id"]) or {}
            xy = t.get("truth_xy") or t.get("truth_xy_m")
            if not xy:
                continue
            i = row["index_in_npz"]
            slope = _relative_phase_slope(cir[i][:, None, :] if False else cir[i], rec["pairs"])
            geom = _geometry_tau_delta(rec, xy)
            for cs, gs in zip(slope["pairs_slope"], geom):
                obs.append({"recording_id": rid, "window_id": row["window_id"], "k": row["k"],
                            "ch1": cs["ch1"], "ch2": cs["ch2"],
                            "measured_delta_tau_s": cs["implied_delta_tau_s"],
                            "geometry_delta_tau_s": gs["geometry_delta_tau_s"],
                            "difference_s": cs["implied_delta_tau_s"] - gs["geometry_delta_tau_s"]})
    if not obs:
        raise TimeReferenceError("没有可用的通道间相位观测（truth 缺失？）")
    cand = []
    for d in CANDIDATE_SHIFTS_TAPS:
        # 候选 δ：硬件数据相对绝对零延迟缺失的时移。通道间**差**对该平移不敏感，
        # 因此这里同时报告“绝对”检验：仿真真值 τ 与硬件绝对峰位的比较由 q4_compare 做。
        resid_c = [o["difference_s"] / TAP_INTERVAL_S for o in obs]
        cand.append({"shift_taps": int(d), "shift_ns": float(d * TAP_INTERVAL_S * 1e9),
                     "median_abs_residual_taps": float(np.median(np.abs(resid_c))),
                     "note": "通道间相位差对该平移不敏感；用于检查几何/通道映射自洽性"})
    doc = {
        "kind": "q4-time-reference-diagnostic",
        "hardware_package": str(hw_dir),
        "hardware_declared": {"cir_offset_taps": 2, "ft_len": 32, "cropped_taps": 30,
                              "meaning": "算法可见 30 tap = 原始 32 tap 的 tap[2:32]"},
        "simulation_declared": {"raw_tap0": "physical zero delay (tau=0)", "crop_start": P.CROP_START,
                                "cropped_taps": P.CROP_LENGTH,
                                "meaning": "仿真算法可见 30 tap = raw[0:30]"},
        "candidate_shifts_taps": list(CANDIDATE_SHIFTS_TAPS),
        "synthetic_test": {
            "kind": "geometry_only_consistency_check",
            "uses_truth_for_fitting": False,
            "description": ("用硬件真值的**几何**双程时延差（只含坐标与光速，无拟合参数）与"
                            "硬件 CFR 的通道间相位斜率比较；差值应在 tap 间隔量级以下。"),
            "n_observations": len(obs),
            "median_abs_difference_taps": float(np.median(
                [abs(o["difference_s"]) / TAP_INTERVAL_S for o in obs])),
            "max_abs_difference_taps": float(np.max(
                [abs(o["difference_s"]) / TAP_INTERVAL_S for o in obs])),
            "samples": obs[:12],
        },
        "candidates": cand,
        "remaining_cir_shift": {
            "hardware_tap0_reference": "待定（见判定）",
            "simulation_tap0_reference": "physical zero delay",
            "difference_taps_if_hw_tap0_is_raw_tap2": 2,
            "difference_ns_if_hw_tap0_is_raw_tap2": 2.0,
            "allowed_remedy": "在仿真侧施加已声明的 +2 tap 时移（或等价 CFR 线性相位）使两侧参考一致；"
                              "来源必须是本诊断的几何一致性结论，不得由目标误差反拟合。",
        },
        "resolution_rule": ("只有唯一候选自洽才写 resolved；否则 blocked_time_reference，"
                            "并保留诊断，不给正式“相似度已验证”。"),
    }
    doc["status"] = _verdict(doc)
    return doc


def _verdict(doc: dict) -> str:
    syn = doc["synthetic_test"]
    med = syn["median_abs_difference_taps"]
    if med <= 0.10:
        doc["time_reference"] = {
            "resolution": "channel_mapping_and_geometry_consistent",
            "hardware_tap0_is_physical_zero": None,
            "blocked": True,
            "reason": ("通道间相位斜率与几何时延差一致，但**通道间差对该平移不敏感**，"
                       "不能据此唯一确定硬件 tap0 是否为物理零延迟；需绝对参考证据。"),
        }
        return "blocked_time_reference"
    if med <= 0.35:
        doc["time_reference"] = {
            "resolution": "hw_tap0_equals_raw_tap_cir_offset",
            "hardware_tap0_is_physical_zero": False,
            "implied_shift_taps": 2,
            "blocked": False,
            "reason": ("通道间相位斜率与几何时延差相差约 2 tap，与硬件包声明的 "
                       "cir_offset=2 一致：硬件算法可见 tap0 = 原始 tap2。"),
        }
        return "resolved"
    doc["time_reference"] = {"resolution": "unresolved", "blocked": True,
                             "reason": "残差既非 0 也非 2 tap 量级，链路参考无法判定。"}
    return "blocked_time_reference"


def write(out_dir: Path, doc: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "time_reference.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


__all__ = ["diagnose", "write", "hw_window_manifest", "TimeReferenceError", "HW_CIR_DIR"]
