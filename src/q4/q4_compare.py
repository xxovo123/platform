# -*- coding: utf-8 -*-
"""Q4 跨域比较：CIR 相似度（ρ 复数 / ρ_env 包络）+ 按同一真值身份的跨域峰位差 + 各自误差。

输入（每域独立）
----------------
* 仿真 CIR：npz，键 ``raw_cir`` 为 ``(L, 4, 32)``（或已 30 tap 的 ``(L,4,30)``）。
  若为 32 tap，按**固定** ``cir_offset`` 截为 30（**禁止 oracle shift**）。
* 硬件 CIR：``q4_export`` 产出的可移植包 npz（``cir_cropped30`` 或逐窗键）。
* 预测 json：``{window_id: [[x,y], ...]}`` 或 ``{window_id: [{"x":..,"y":..}, ...]}``。
* 真值 json（**独立**）：``{window_id: [[x,y], ...]}``。

输出
----
逐窗 ``rho_complex`` / ``rho_env``、按同一 identity 的 ``delta_sim_hw_peak_m`` 及
每域各自 ``e_xy/e_r/e_theta``、未配对计数；再给跨窗汇总。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .q4_common import (CROPPED_TAPS, DEFAULT_CIR_OFFSET, Q4Error, cross_domain_peak_delta,
                        load_json, mid_index, own_errors, raw32_to_cropped30,
                        window_similarity, write_json)


# --------------------------------------------------------------------------- #
# 输入装载
# --------------------------------------------------------------------------- #
def _to_xy_list(items) -> list:
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append([float(it["x"]), float(it["y"])])
        else:
            out.append([float(it[0]), float(it[1])])
    return out


def load_predictions(path) -> dict:
    doc = load_json(path)
    out = {}
    for wid, val in doc.items():
        if isinstance(val, dict):
            val = val.get("predictions", val.get("peaks", []))
        out[str(wid)] = _to_xy_list(val)
    return out


def load_truth(path) -> dict:
    doc = load_json(path)
    out = {}
    for wid, val in doc.items():
        if isinstance(val, dict):
            val = val.get("truth_xy", val.get("truth", []))
        out[str(wid)] = _to_xy_list(val)
    return out


def load_sim_cir(path, *, cir_offset=DEFAULT_CIR_OFFSET):
    """npz -> (L,4,30) 复数；raw_cir 为 32 tap 时按固定 offset 截窗。"""
    with np.load(path, allow_pickle=False) as z:
        if "raw_cir" in z.files:
            raw = np.asarray(z["raw_cir"])
        elif "cir_cropped30" in z.files:
            raw = np.asarray(z["cir_cropped30"])
        else:
            raise Q4Error("sim npz 缺 raw_cir / cir_cropped30：%s" % (z.files,))
        if raw.ndim != 3 or raw.shape[1] != 4:
            raise Q4Error("sim CIR 必须是 (L,4,T)，收到 shape=%s" % (raw.shape,))
        if raw.shape[-1] == CROPPED_TAPS:
            return raw.astype(np.complex64)
        return raw32_to_cropped30(raw, cir_offset=cir_offset).astype(np.complex64)


def load_hw_cir_window(path, window_id):
    """从可移植包 npz 取单窗 cropped30 ``(L,4,30)``。"""
    with np.load(path, allow_pickle=False) as z:
        if "cir_cropped30" in z.files and "window_ids" in z.files:
            wids = [str(w) for w in z["window_ids"]]
            if window_id not in wids:
                raise Q4Error("%s 不含 window %s" % (path, window_id))
            return np.asarray(z["cir_cropped30"][wids.index(window_id)], dtype=np.complex64)
        key = "cir_cropped30__%s" % window_id
        if key in z.files:
            return np.asarray(z[key], dtype=np.complex64)
        raise Q4Error("%s 缺 window %s 的 cropped30" % (path, window_id))


def hw_window_ids(path):
    with np.load(path, allow_pickle=False) as z:
        if "window_ids" in z.files:
            return [str(w) for w in z["window_ids"]]
        return sorted(k[len("cir_cropped30__"):] for k in z.files if k.startswith("cir_cropped30__"))


def sim_window_ids(path):
    """仿真 npz 的窗口清单：优先 ``raw_cir__<wid>``；否则单窗 ``raw_cir``/``cir_cropped30``。"""
    with np.load(path, allow_pickle=False) as z:
        ids = [k[len("raw_cir__"):] for k in z.files if k.startswith("raw_cir__")]
        if ids:
            return sorted(ids)
        if "raw_cir" in z.files or "cir_cropped30" in z.files:
            if "window_id" in z.files:
                return [str(z["window_id"])]
            if "window_ids" in z.files:
                return [str(w) for w in z["window_ids"]]
            return ["__single__"]
        return []


def load_sim_window_cir(path, window_id, *, cir_offset=DEFAULT_CIR_OFFSET):
    """取仿真单窗 cropped30 ``(L,4,30)``：支持 ``raw_cir__<wid>``、单窗 ``raw_cir``、``cir_cropped30``。"""
    with np.load(path, allow_pickle=False) as z:
        key = "raw_cir__%s" % window_id
        ckey = "cir_cropped30__%s" % window_id
        if key in z.files:
            raw = np.asarray(z[key])
        elif ckey in z.files:
            return np.asarray(z[ckey], dtype=np.complex64)
        elif "raw_cir" in z.files:
            raw = np.asarray(z["raw_cir"])
        elif "cir_cropped30" in z.files:
            raw = np.asarray(z["cir_cropped30"])
        else:
            raise Q4Error("%s 缺 window %s 的 sim CIR" % (path, window_id))
    if raw.ndim != 3 or raw.shape[1] != 4:
        raise Q4Error("sim CIR 必须是 (L,4,T)，收到 shape=%s" % (raw.shape,))
    if raw.shape[-1] == CROPPED_TAPS:
        return raw.astype(np.complex64)
    return raw32_to_cropped30(raw, cir_offset=cir_offset).astype(np.complex64)


# --------------------------------------------------------------------------- #
# 单窗比较
# --------------------------------------------------------------------------- #
def compare_window(window_id, sim_cir30, hw_cir30, sim_truth, sim_pred, hw_truth, hw_pred, *,
                   sim_status="ok", hw_status="ok") -> dict:
    """单窗比较。**方法状态保留**：非 ``ok``（含未知 ``None``）时 CIR ρ 仍算（与估计器无关），
    但 ``peak delta`` / ``own_errors`` 不得当成功。"""
    sim_cir30 = np.asarray(sim_cir30)
    hw_cir30 = np.asarray(hw_cir30)
    if sim_cir30.shape != hw_cir30.shape:
        raise Q4Error("sim/hw CIR shape 不一致：%s vs %s" % (sim_cir30.shape, hw_cir30.shape))
    sim = window_similarity(sim_cir30, hw_cir30)
    both_ok = (sim_status == "ok" and hw_status == "ok")
    if both_ok:
        cross = cross_domain_peak_delta(sim_truth, sim_pred, hw_truth, hw_pred)
    else:
        cross = {"n_common_identities": 0, "items": [],
                 "sim_unmatched_truth": [], "sim_unmatched_pred": [],
                 "hw_unmatched_truth": [], "hw_unmatched_pred": [],
                 "skipped": "method_status_not_ok(sim=%s,hw=%s)" % (sim_status, hw_status)}
    if sim_status == "ok":
        sim_own = own_errors(sim_truth, sim_pred)
    else:
        sim_own = {"status": sim_status, "ok": False, "errors": [], "tp": 0,
                   "fn": len(sim_truth), "fp": len(sim_pred),
                   "n_truth": len(sim_truth), "n_predictions": len(sim_pred)}
    if hw_status == "ok":
        hw_own = own_errors(hw_truth, hw_pred)
    else:
        hw_own = {"status": hw_status, "ok": False, "errors": [], "tp": 0,
                  "fn": len(hw_truth), "fp": len(hw_pred),
                  "n_truth": len(hw_truth), "n_predictions": len(hw_pred)}
    return {
        "window_id": window_id,
        "L": int(sim_cir30.shape[0]),
        "n_channels": int(sim_cir30.shape[1]),
        "taps": int(sim_cir30.shape[2]),
        "mid_index": int(mid_index(sim_cir30.shape[0])),
        "rho_complex": sim["rho_complex"],
        "rho_env": sim["rho_env"],
        "sim_method_status": sim_status, "hw_method_status": hw_status,
        "sim_own": sim_own, "hw_own": hw_own,
        "cross_domain": cross,
    }


def _mean(vals):
    arr = [v for v in vals if v is not None and math.isfinite(v)]
    return (sum(arr) / len(arr)) if arr else None


def summarize(per_window: list) -> dict:
    deltas = [it["delta_sim_hw_peak_m"] for w in per_window
              for it in w["cross_domain"]["items"]]
    sim_e = [it["sim_e_xy_m"] for w in per_window for it in w["cross_domain"]["items"]]
    hw_e = [it["hw_e_xy_m"] for w in per_window for it in w["cross_domain"]["items"]]
    return {
        "n_windows": len(per_window),
        "n_method_failed": sum(1 for w in per_window
                               if w.get("sim_method_status") != "ok"
                               or w.get("hw_method_status") != "ok"),
        "n_windows_peak_delta_ok": sum(1 for w in per_window
                                       if w.get("sim_method_status") == "ok"
                                       and w.get("hw_method_status") == "ok"),
        "n_common_identities": sum(w["cross_domain"]["n_common_identities"] for w in per_window),
        "mean_rho_complex": _mean([w["rho_complex"] for w in per_window]),
        "mean_rho_env": _mean([w["rho_env"] for w in per_window]),
        "n_rho_complex_undefined": sum(1 for w in per_window if w["rho_complex"] is None),
        "n_rho_env_undefined": sum(1 for w in per_window if w["rho_env"] is None),
        "mean_delta_sim_hw_peak_m": _mean(deltas),
        "mean_sim_e_xy_m": _mean(sim_e),
        "mean_hw_e_xy_m": _mean(hw_e),
        "sim_unmatched_truth_total": sum(len(w["cross_domain"]["sim_unmatched_truth"]) for w in per_window),
        "sim_unmatched_pred_total": sum(len(w["cross_domain"]["sim_unmatched_pred"]) for w in per_window),
        "hw_unmatched_truth_total": sum(len(w["cross_domain"]["hw_unmatched_truth"]) for w in per_window),
        "hw_unmatched_pred_total": sum(len(w["cross_domain"]["hw_unmatched_pred"]) for w in per_window),
    }


def run_compare(*, sim_cir_path, hw_cir_path, sim_pred_path, hw_pred_path,
                sim_truth_path, hw_truth_path, windows=None, out_path=None,
                cir_offset=DEFAULT_CIR_OFFSET, max_L=None) -> dict:
    sim_pred = load_predictions(sim_pred_path)
    hw_pred = load_predictions(hw_pred_path)
    sim_truth = load_truth(sim_truth_path)
    hw_truth = load_truth(hw_truth_path)
    sim_train = load_sim_cir(sim_cir_path, cir_offset=cir_offset)

    hw_ids = hw_window_ids(hw_cir_path)
    if windows:
        wids = [str(w) for w in windows]
    else:
        wids = [w for w in hw_ids if w in sim_pred and w in hw_pred]
    if not wids:
        raise Q4Error("没有共同可比较的 window（hw windows=%s）" % hw_ids[:5])

    per_window = []
    for wid in wids:
        if wid not in sim_truth or wid not in hw_truth:
            raise Q4Error("缺少 %s 的真值（sim/hw 各需独立真值）" % wid)
        hw = load_hw_cir_window(hw_cir_path, wid)
        # sim 单窗 raw_cir：npz 若只含单录制单窗，直接整体视作该窗
        sim = sim_train
        if max_L is not None:
            sim = sim[:int(max_L)]
            hw = hw[:int(max_L)]
        if sim.shape[0] != hw.shape[0]:
            L = min(sim.shape[0], hw.shape[0])
            sim, hw = sim[:L], hw[:L]
        per_window.append(compare_window(wid, sim, hw, sim_truth[wid], sim_pred[wid],
                                         hw_truth[wid], hw_pred[wid]))
    doc = {
        "kind": "q4-cross-domain-compare",
        "schema_version": 1,
        "inputs": {"sim_cir": str(sim_cir_path), "hw_cir": str(hw_cir_path),
                   "sim_pred": str(sim_pred_path), "hw_pred": str(hw_pred_path),
                   "sim_truth": str(sim_truth_path), "hw_truth": str(hw_truth_path)},
        "tap_rule": "sim raw 32 -> 固定 offset=%d 截 30；hw 已是 cropped30；无时移搜索" % cir_offset,
        "identity_rule": "各自按 truth 匹配到同 identity 后再算跨域峰位差；不做 sim-hw 最佳配对",
        "per_window": per_window,
        "summary": summarize(per_window),
    }
    if out_path:
        write_json(out_path, doc)
    return doc


# --------------------------------------------------------------------------- #
# 极简合成自测（明确的 test，非正式结果）
# --------------------------------------------------------------------------- #
def demo_synthetic(tmp_dir):
    rng = np.random.default_rng(20260926)
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    L, C, T = 8, 4, 32
    sim_raw = (rng.normal(size=(L, C, T)) + 1j * rng.normal(size=(L, C, T))).astype(np.complex64)
    hw_src = sim_raw[:, :, DEFAULT_CIR_OFFSET:DEFAULT_CIR_OFFSET + CROPPED_TAPS]
    hw = (hw_src + 0.05 * (rng.normal(size=hw_src.shape) + 1j * rng.normal(size=hw_src.shape))).astype(np.complex64)
    wid = "demo_t10"
    sim_npz = tmp / "sim_raw.npz"
    np.savez(sim_npz, raw_cir=sim_raw, mid_index=np.int64(mid_index(L)))
    hw_npz = tmp / "hw.npz"
    np.savez(hw_npz, window_ids=np.array([wid], dtype="U"),
             cir_cropped30=hw[None, ...], mid_index=np.array([mid_index(L)]),
             pairs=np.array([[0, 2], [0, 3], [1, 2], [1, 3]]), source="demo")
    truth = {wid: [[0.0, -2.0], [0.2, -2.0]]}
    sim_pred = {wid: [{"x": 0.05, "y": -2.0}, {"x": 0.25, "y": -2.0}]}
    hw_pred = {wid: [{"x": 0.08, "y": -2.0}, {"x": 0.28, "y": -2.0}]}
    for name, obj in (("truth.json", truth), ("sim_pred.json", sim_pred), ("hw_pred.json", hw_pred)):
        (tmp / name).write_text(json.dumps(obj), encoding="utf-8")
    return run_compare(sim_cir_path=sim_npz, hw_cir_path=hw_npz, sim_pred_path=tmp / "sim_pred.json",
                       hw_pred_path=tmp / "hw_pred.json", sim_truth_path=tmp / "truth.json",
                       hw_truth_path=tmp / "truth.json", out_path=tmp / "compare.json")
