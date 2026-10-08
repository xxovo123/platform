#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0 输入适配/一致性校验：把**数值一致性**与**理想定位正确性**分开，并披露 crop 失配。

四项检查
--------
1. numerical_consistency：同 X 下本方法 vs B0 的伪谱/峰是否逐格一致（数值正确性）。
2. ideal_locality：单目标无噪、理想定位正确性（DOA + 距离是否恢复；与数值一致性分开报告）。
3. crop_offset_disclosure：crop_start 引入的固定延迟参考偏移 / 距离偏移（只报告，不补偿）。
4. zero_offset_cfr_diagnostic：**独立零偏移解析 CFR**（直接用 17 个带内频点构 X，不经 30-tap
   截窗）-> 仅作方向/相位诊断。主协议不因此改动；compensated 生成 profile 默认关闭。
"""

from __future__ import annotations

import math

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np

from . import case_build, methods_loader, metrics_bridge


def _peaks_xy(result):
    return [(float(p["x"]), float(p["y"])) for p in (result.get("predictions") or [])]


def numerical_consistency(case: dict) -> dict:
    """本方法 vs B0（同 X、同网格、同 K）：谱差 / 谱余弦 / 峰集一致性。"""
    a = methods_loader.call("this_method", case)
    b = methods_loader.call("b0_standard_music", case)
    out = {"check": "numerical_consistency_b0_vs_this_method",
           "this_method_status": a["status"], "b0_status": b["status"],
           "peaks_identical": _peaks_xy(a) == _peaks_xy(b),
           "n_peaks_this": len(a["predictions"]), "n_peaks_b0": len(b["predictions"])}
    if a.get("spectrum") is not None and b.get("spectrum") is not None:
        sa, sb = np.asarray(a["spectrum"], float), np.asarray(b["spectrum"], float)
        scale = max(float(np.max(np.abs(sa))), 1e-300)
        denom = float(np.linalg.norm(sa.ravel()) * np.linalg.norm(sb.ravel()))
        out.update({"max_abs_diff": float(np.max(np.abs(sa - sb))),
                    "max_rel_diff": float(np.max(np.abs(sa - sb)) / scale),
                    "spectrum_cosine": (float(sa.ravel() @ sb.ravel() / denom) if denom > 0 else 1.0)})
    return out


def ideal_locality(case: dict, truth_xy) -> dict:
    """理想定位正确性（单目标无噪）：各法峰位相对真值的 e_xy / e_r / e_theta。"""
    from protocol.sim_protocol import PROTOCOL
    truth = [(float(x), float(y)) for x, y in truth_xy]
    rows = {}
    for method in methods_loader.METHOD_ORDER:
        params = methods_loader.params_for(method, PROTOCOL) or None
        res = methods_loader.call(method, case, params)
        preds = _peaks_xy(res)
        entry = {"status": res["status"], "n_predictions": len(preds)}
        if preds and truth:
            # 最近峰（诊断用；正式指标走 metrics）
            dists = [math.hypot(px - truth[0][0], py - truth[0][1]) for px, py in preds]
            j = int(np.argmin(dists))
            px, py = preds[j]
            r_t, th_t = sc.xy_to_polar(*truth[0])
            r_p, th_p = sc.xy_to_polar(px, py)
            entry.update({"e_xy_m": float(dists[j]), "e_r_m": float(abs(r_p - r_t)),
                          "e_theta_deg": float(abs(((th_p - th_t) + 180.0) % 360.0 - 180.0)),
                          "peak_xy_m": [px, py], "truth_xy_m": [truth[0][0], truth[0][1]]})
        rows[method] = entry
    return {"check": "ideal_locality_diagnostic", "by_method": rows,
            "note": ("理想定位正确性（无噪单目标）与数值一致性分开；正式指标由 metrics 计算，"
                     "本项只作适配诊断。")}


def crop_offset_disclosure() -> dict:
    """crop_start 引入的固定偏移（只报告）：时间偏移与双程距离偏移。"""
    return {
        "check": "crop_mismatch_disclosure",
        "raw_tap0": "physical zero delay（层 A 与层 B 一致）",
        "crop_start": int(sc.CROP_START),
        "offset_seconds": float(sc.crop_offset_seconds()),
        "offset_range_m": float(sc.crop_offset_range_m()),
        "consequence": ("裁剪把数据延迟参考整体改变 crop_start*Ts（2026-09-26 v2 口径 crop_start=0，"
                        "即不裁前 2 个物理抽头，本批偏移 %.5f m）；在**理想无限/循环"
                        "shift**解释下，v1（crop_start=2）相当于把参考平移约 -0.29979 m。但有限窗、FB 与精确球面"
                        "流形耦合的实际影响**未经证明**（不保证只径向、不保证不影响 DOA）；"
                        "实际偏差以诊断与逐场景结果为记录。"
                        % sc.crop_offset_range_m()),
        "ideal_infinite_shift_explanation_only": True,
        "no_guarantee_radial_only": True,
        "not_compensated": True,
    }


def direct_cfr_x(clean_id: str, targets_xy, k: int, L: int) -> np.ndarray:
    """独立零偏移解析 CFR：直接取 17 个带内频点构 X（不经 30-tap 截窗）。"""
    from analytic import forward
    rng = sc.rng_from(sc.clean_seed(clean_id))
    coeff = forward.coefficients_from_spec(int(k), int(L), rng=rng)
    tau = forward.delays_for_targets(targets_xy)                      # (K,C)
    freqs = sc.frequencies_hz()                                        # (F,) 原始频序
    phase = np.exp(-2j * math.pi * tau[:, :, None] * freqs[None, None, :])   # (K,C,F)
    H = np.einsum("lk,kcf->lcf", coeff, phase)                          # (L,C,F)
    return np.ascontiguousarray(H.transpose(0, 2, 1).reshape(int(L), -1))


def zero_offset_cfr_diagnostic(spec: dict, *, runid_prefix: str = "e0diag") -> dict:
    """零偏移 CFR 诊断：直接 CFR（无截窗）下方法能否恢复方向/距离（仅诊断，不改主协议）。"""
    gx, gy = sc.build_grid()
    k = int(spec["k"])
    L = int(spec.get("L", sc.SNAPSHOTS))
    X = direct_cfr_x(str(spec["clean_id"]), spec["targets_xy"], k, L)
    case = case_build.build_case(case_id="%s_zero_offset_cfr" % runid_prefix, clean_id=spec["clean_id"],
                                 X=X, k=k, grid_x=gx, grid_y=gy, layer="A", source="analytic",
                                 scene="E0", tags=[], snr_db=float("inf"), trial=0, seed=0, L=L)
    case_build.validate_case(case)
    out = {"check": "zero_offset_cfr_diagnostic", "clean_id": spec["clean_id"],
           "truth_xy_m": [[float(x), float(y)] for x, y in spec["targets_xy"]],
           "note": "直接 17-bin CFR（无 30-tap 截窗）；用于验证方向/相位正确性，非主协议。",
           "by_method": {}}
    for method in ("this_method", "b0_standard_music"):
        res = methods_loader.call(method, case)
        preds = _peaks_xy(res)
        entry = {"status": res["status"], "predictions_xy_m": [[p[0], p[1]] for p in preds]}
        if preds:
            e = []
            for tx, ty in spec["targets_xy"]:
                e.append(min(math.hypot(px - tx, py - ty) for px, py in preds))
            entry["nearest_truth_errors_m"] = [float(v) for v in e]
        out["by_method"][method] = entry
    return out


def run_all(case: dict, truth_xy, spec: dict | None = None, clean_case: dict | None = None) -> dict:
    """case=SNR 诊断用（含噪）；clean_case=理想/有限窗诊断用（无噪）。二者显式区分。"""
    report = {"six_method_run": six_method_run(case),
              "numerical_consistency": numerical_consistency(case),
              "ideal_locality": ideal_locality(clean_case if clean_case is not None else case, truth_xy),
              "ideal_locality_case": ("clean_noiseless" if clean_case is not None else "given_case"),
              "crop_offset_disclosure": crop_offset_disclosure()}
    if spec is not None:
        try:
            report["zero_offset_cfr_diagnostic"] = zero_offset_cfr_diagnostic(spec)
        except Exception as exc:  # noqa: BLE001
            report["zero_offset_cfr_diagnostic"] = {"available": False, "error": str(exc)}
    return report


def six_method_run(case: dict) -> dict:
    """E0 实际运行 6 法（5 主比较 + B0），不只验证装载成功。

    同模型口径：this_method 与 b0_standard_music 属同一模型（标准全二维 MUSIC），其
    数值一致性另见 numerical_consistency；B1/B2/C1/C3 为各自文献模型，**只要求实际运行**
    并如实记录 status/n_pred/reason（不作同模型声明）。
    """
    from protocol.sim_protocol import PROTOCOL
    by_method = {}
    for method in methods_loader.METHOD_ORDER:
        params = methods_loader.params_for(method, PROTOCOL) or None
        res = methods_loader.call(method, case, params)
        diag = res["diagnostics"] or {}
        by_method[method] = {
            "status": res["status"], "n_predictions": len(res["predictions"]),
            "grid_kind": diag.get("grid_kind"),
            "reason": None if res["status"] == "ok" else str(diag.get("reason"))[:160],
            "load_fail_markers": [k for k in ("module_load_failed", "interface_missing",
                                              "output_invalid", "estimate_exception")
                                  if diag.get("reason") == k],
        }
    all_ran = all(v["status"] in ("ok", "failed", "not_applicable") and not v["load_fail_markers"]
                  for v in by_method.values())
    no_hard_fail = all(v["status"] != "failed" for v in by_method.values())
    return {"check": "six_method_actual_run", "by_method": by_method,
            "all_six_executed_no_loadfail": bool(all_ran),
            "all_six_status_not_failed": bool(no_hard_fail),
            "same_model_methods": ["this_method", "b0_standard_music"],
            "other_model_methods": ["b1_deng", "b2_kakouche", "c1_jade", "c3_kim"],
            "note": ("E0 实际运行 6 法；同模型一致性与理想定位正确性分开报告。"
                     "自然 no-hit 允许，但不得是装载/接口/输出非法失败。")}


__all__ = ["numerical_consistency", "ideal_locality", "crop_offset_disclosure",
           "direct_cfr_x", "zero_offset_cfr_diagnostic", "six_method_run", "run_all"]
