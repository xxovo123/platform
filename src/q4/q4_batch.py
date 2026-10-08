# -*- coding: utf-8 -*-
"""Q4 全清单批量入口：由**已生成 inventory** 映射 rid->geometry_id，按 15 几何共享生成关联仿真。

修正（2026-09-26，替代首版）
---------------------------
1. ``n_geometries`` 取 **inventory 的 15 个 geometry_id**，不是录制数（32）。
2. 仿真按 **geometry_id** 关联（15 个几何共享生成），**不是按 recording_id** 逐录制造文件。
3. **L 严格相等**：sim/hw 形状不一致（尤其 L 不同）记 ``L_mismatch`` 单列，**绝不静默取 min 缩 L**
   （固定 midpoint/预算不可破坏）；不同 L 档须作为独立档显式列出。
4. 预测**按字段存在性**判定：窗口在预测集合里**缺失**记 ``*_prediction_missing``；
   **不复用** ``get(wid, [])`` 把“没跑”当成真实空输出。

硬件侧全自动：录制/窗口 ← 导出包；真值 ← ``truth_index.json``；冻结预测 ←
``results/<method>/per_method/``（谱 ``peaks`` / native ``errors+fp_items``）。
仿真侧（15 几何共享）文件名约定（``<sim_dir>/<geometry_id>/``）：

    clean_cir.npz    # raw_cir__<window_id> (L,4,32)，或单窗 raw_cir + window_id（跨窗共享）
    pred.json        # {window_id: [[x,y], ...]}
    truth.json       # {window_id: [[x,y], ...]}

也接受 ``<sim_dir>/<geometry_id>.npz`` 等平铺命名（见 ``sim_paths``）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .q4_common import (DEFAULT_TRUTH_INDEX, Q4Error, geometry_of, load_json, resolve_path,
                        workspace_root, write_json)
from .q4_compare import (compare_window, load_hw_cir_window, load_sim_window_cir, load_predictions,
                         load_truth, sim_window_ids, summarize)

DEFAULT_HW_CIR_DIR = "大修执行_20260923/06_仿真实验_20260926/q4/out/hw_cir"
DEFAULT_HW_RESULTS = "大修执行_20260923/04_Q6/hardware/results/grid5cm_full214_20260924"
DEFAULT_INVENTORY = "大修执行_20260923/06_仿真实验_20260926/q4/manifest/q4_inventory.json"
DEFAULT_METHOD = "this_method"


# --------------------------------------------------------------------------- #
# 几何映射（来自已生成 inventory）
# --------------------------------------------------------------------------- #
def geometry_index(inventory_json=DEFAULT_INVENTORY) -> dict:
    """返回 ``{"rid2geom": {...}, "geometries": {...}, "source": str}``。

    ``rid2geom`` 用 inventory 的 ``recordings`` 生成（只取 valid）；inventory 缺失时回退
    ``geometry_of``（命名规则）并标注 fallback。
    """
    path = resolve_path(inventory_json, workspace_root())
    if Path(path).is_file():
        doc = load_json(path)
        rid2geom = {r["recording_id"]: r["geometry_id"] for r in doc.get("recordings", [])
                    if r.get("valid") is True}
        geoms = {g["geometry_id"]: {"n_targets": g.get("n_targets"),
                                    "truth_xy": g.get("truth_xy"),
                                    "fdtd_geometry": g.get("fdtd_geometry")}
                 for g in doc.get("geometries", [])}
        return {"rid2geom": rid2geom, "geometries": geoms, "source": str(path)}
    return {"rid2geom": {}, "geometries": {}, "source": "fallback:geometry_of(%s 缺失)" % path}


def _geom_of(rid, idx) -> str:
    return idx["rid2geom"].get(rid) or geometry_of(rid)


# --------------------------------------------------------------------------- #
# 硬件侧装载
# --------------------------------------------------------------------------- #
def hw_recordings(hw_dir) -> dict:
    d = Path(hw_dir)
    if not d.is_dir():
        raise Q4Error("硬件 CIR 包目录不存在：%s（先跑 export --source cache）" % d)
    out = {}
    for p in sorted(d.glob("*.npz")):
        out[p.name[:-len(".npz")]] = hw_window_ids_file(p)
    if not out:
        raise Q4Error("硬件 CIR 包目录没有任何 *.npz：%s" % d)
    return out


def hw_window_ids_file(path) -> list:
    with np.load(path, allow_pickle=False) as z:
        if "window_ids" in z.files:
            return [str(w) for w in z["window_ids"]]
        return sorted(k[len("cir_cropped30__"):] for k in z.files if k.startswith("cir_cropped30__"))


def hw_window_lengths(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if "cir_cropped30" in z.files and "window_ids" in z.files:
            arr = z["cir_cropped30"]
            return {str(w): int(arr.shape[1]) if arr.ndim == 4 else int(arr.shape[0])
                    for w in z["window_ids"]}
        return {k[len("cir_cropped30__"):]: int(z[k].shape[0])
                for k in z.files if k.startswith("cir_cropped30__")}


def hw_truth_windows(truth_index) -> dict:
    idx = load_json(truth_index)
    return {str(wid): {"recording_id": str(v["recording_id"]),
                       "truth_xy": [[float(a), float(b)] for a, b in v["truth_xy"]],
                       "k": int(v.get("k") if v.get("k") is not None else v.get("n_truth", 0))}
            for wid, v in (idx.get("windows") or {}).items()}


def reconstruct_hw_predictions(results_dir, method=DEFAULT_METHOD) -> dict:
    path = Path(results_dir) / method / "per_method" / ("%s_per_window.json" % method)
    if not path.is_file():
        raise Q4Error("冻结预测缺失：%s" % path)
    doc = load_json(path)
    out = {}
    for wid, rec in doc.items():
        if rec is None:
            continue
        if rec.get("status") not in (None, "ok"):
            out[str(wid)] = []
            continue
        if rec.get("peaks"):
            out[str(wid)] = [[float(p["x"]), float(p["y"])]
                             for p in sorted(rec["peaks"], key=lambda r: int(r.get("rank", 0)))]
            continue
        slots = {}
        for e in (rec.get("errors") or []):
            slots[int(e["pred_index"])] = [float(e["estimated_xy"][0]), float(e["estimated_xy"][1])]
        for f in (rec.get("fp_items") or []):
            slots[int(f["pred_index"])] = [float(f["prediction_xy"][0]), float(f["prediction_xy"][1])]
        out[str(wid)] = [slots[i] for i in sorted(slots)]
    return out


def reconstruct_hw_predictions_status(results_dir, method=DEFAULT_METHOD) -> tuple:
    """冻结 hw：返回 (preds, status)。逐窗 ``status`` 取记录字段；该冻结集文档为全 ok。"""
    path = Path(results_dir) / method / "per_method" / ("%s_per_window.json" % method)
    if not path.is_file():
        raise Q4Error("冻结预测缺失：%s" % path)
    doc = load_json(path)
    preds, status = {}, {}
    for wid, rec in doc.items():
        if rec is None:
            continue
        status[str(wid)] = str(rec.get("status", "ok"))      # 冻结 hw 文档：全 ok
        preds[str(wid)] = [] if status[str(wid)] != "ok" else \
            ([[float(p["x"]), float(p["y"])]
              for p in sorted(rec.get("peaks") or [], key=lambda r: int(r.get("rank", 0)))]
             if rec.get("peaks") else _slots_of(rec))
    return preds, status


def _slots_of(rec) -> list:
    slots = {}
    for e in (rec.get("errors") or []):
        slots[int(e["pred_index"])] = [float(e["estimated_xy"][0]), float(e["estimated_xy"][1])]
    for f in (rec.get("fp_items") or []):
        slots[int(f["pred_index"])] = [float(f["prediction_xy"][0]), float(f["prediction_xy"][1])]
    return [slots[i] for i in sorted(slots)]


def load_sim_predictions_status(path) -> tuple:
    """sim 预测 JSON -> (preds, status)；缺 ``status`` 记 **None（unknown，不默补 ok）**。"""
    doc = load_json(path)
    preds, status = {}, {}
    for wid, val in doc.items():
        if isinstance(val, dict):
            status[str(wid)] = val.get("status")
            body = val.get("predictions", val.get("peaks", []))
        else:
            status[str(wid)] = None
            body = val
        preds[str(wid)] = [[float(p["x"]), float(p["y"])] if isinstance(p, dict)
                           else [float(p[0]), float(p[1])] for p in (body or [])]
    return preds, status


# --------------------------------------------------------------------------- #
# 仿真侧（按 geometry 共享）
# --------------------------------------------------------------------------- #
def sim_paths(sim_dir, geometry_id, method=DEFAULT_METHOD) -> dict:
    d = Path(sim_dir)
    base = d / geometry_id
    return {
        "cir": [base / "clean_cir.npz", base / (geometry_id + ".npz"), d / (geometry_id + ".npz")],
        "pred": [base / (method + ".pred.json"), base / "pred.json",
                 base / (geometry_id + ".pred.json"), d / (geometry_id + ".pred.json")],
        "truth": [base / "truth.json", base / (geometry_id + ".truth.json"), d / (geometry_id + ".truth.json")],
    }


def _first(paths):
    for p in paths:
        if Path(p).is_file():
            return Path(p)
    return None


def sim_window_lengths(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k[len("raw_cir__"):]: int(z[k].shape[0]) for k in z.files if k.startswith("raw_cir__")}
        out.update({k[len("cir_cropped30__"):]: int(z[k].shape[0])
                    for k in z.files if k.startswith("cir_cropped30__")})
        if out:
            return out
        if "raw_cir" in z.files or "cir_cropped30" in z.files:
            return {"__single__": int(np.asarray(z["raw_cir"] if "raw_cir" in z.files
                                                  else z["cir_cropped30"]).shape[0])}
        return {}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan_batch(*, sim_dir=None, hw_dir=DEFAULT_HW_CIR_DIR, truth_index=DEFAULT_TRUTH_INDEX,
               results_dir=DEFAULT_HW_RESULTS, inventory_json=DEFAULT_INVENTORY,
               method=DEFAULT_METHOD, allow_truth_fallback=False) -> dict:
    hw_dir = resolve_path(hw_dir, workspace_root())
    truth_index = resolve_path(truth_index, workspace_root())
    results_dir = resolve_path(results_dir, workspace_root())
    idx = geometry_index(inventory_json)
    hw = hw_recordings(hw_dir)
    truth = hw_truth_windows(truth_index)
    try:
        hw_pred, hw_status = reconstruct_hw_predictions_status(results_dir, method)
        pred_error = None
    except Q4Error as exc:
        hw_pred, hw_status, pred_error = {}, {}, str(exc)

    sim_cache = {}

    def sim_info(geom):
        if geom not in sim_cache:
            paths = sim_paths(sim_dir, geom, method) if sim_dir else {"cir": [], "pred": [], "truth": []}
            cir_p, pred_p, truth_p = _first(paths["cir"]), _first(paths["pred"]), _first(paths["truth"])
            sim_cache[geom] = {
                "cir": cir_p, "pred": pred_p, "truth": truth_p,
                "lens": sim_window_lengths(cir_p) if cir_p else {},
                "pred_wids": set(load_predictions(pred_p)) if pred_p else set(),
                "pred_status": (load_sim_predictions_status(pred_p)[1] if pred_p else {}),
                "truth_wids": set(load_truth(truth_p)) if truth_p else set(),
            }
        return sim_cache[geom]

    hw_lens_cache = {}
    windows, missing = [], []
    n_avail = 0
    for rid, wids in hw.items():
        geom = _geom_of(rid, idx)
        si = sim_info(geom)
        hw_npz = hw_dir / (rid + ".npz")
        if rid not in hw_lens_cache:
            hw_lens_cache[rid] = hw_window_lengths(hw_npz)
        for wid in wids:
            reasons = []
            if wid not in truth:
                reasons.append("hw_truth_missing")
            if wid not in hw_pred:
                reasons.append("hw_prediction_missing")
            if si["cir"] is None:
                reasons.append("sim_cir_missing(%s)" % geom)
            else:
                sim_lens = si["lens"]
                if wid not in sim_lens and "__single__" not in sim_lens:
                    reasons.append("sim_cir_window_missing")
                else:
                    sim_L = sim_lens.get(wid, sim_lens.get("__single__"))
                    hw_L = hw_lens_cache[rid].get(wid)
                    if hw_L is not None and sim_L is not None and sim_L != hw_L:
                        reasons.append("L_mismatch(sim=%s,hw=%s)" % (sim_L, hw_L))
            if si["pred"] is None:
                reasons.append("sim_prediction_missing")
            elif wid not in si["pred_wids"] and "__single__" not in si["pred_wids"]:
                reasons.append("sim_prediction_window_missing")
            if si["truth"] is None and not allow_truth_fallback:
                reasons.append("sim_truth_missing")
            elif (si["truth"] is not None and wid not in si["truth_wids"]
                  and "__single__" not in si["truth_wids"] and not allow_truth_fallback):
                reasons.append("sim_truth_window_missing")
            sim_st = (si["pred_status"].get(wid, si["pred_status"].get("__single__"))
                      if si["pred"] else None)
            entry = {"recording_id": rid, "geometry_id": geom, "window_id": wid,
                     "sim_method_status": sim_st,
                     "hw_method_status": hw_status.get(wid, "ok"),
                     "available": not reasons, "reasons": reasons}
            windows.append(entry)
            if reasons:
                missing.append(entry)
            else:
                n_avail += 1
    geoms_all = sorted({_geom_of(rid, idx) for rid in hw})
    return {
        "kind": "q4-batch-plan", "schema_version": 2,
        "inputs": {"sim_dir": str(sim_dir) if sim_dir else None, "hw_dir": str(hw_dir),
                   "truth_index": str(truth_index), "results_dir": str(results_dir),
                   "inventory_json": idx["source"], "method": method,
                   "allow_truth_fallback": bool(allow_truth_fallback)},
        "hw_prediction_error": pred_error,
        "n_recordings": len(hw),
        "n_geometries": len(geoms_all),
        "geometries": geoms_all,
        "n_windows_total": len(windows), "n_windows_available": n_avail,
        "n_windows_missing": len(missing),
        "windows": windows, "missing": missing,
        "note": "plan 不算任何指标；sim 缺失/字段缺失/L 不符一律 missing，不假结果、不缩 L",
    }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run_batch(*, sim_dir, out_dir, hw_dir=DEFAULT_HW_CIR_DIR, truth_index=DEFAULT_TRUTH_INDEX,
              results_dir=DEFAULT_HW_RESULTS, inventory_json=DEFAULT_INVENTORY,
              method=DEFAULT_METHOD, allow_truth_fallback=False) -> dict:
    hw_dir = resolve_path(hw_dir, workspace_root())
    truth_index = resolve_path(truth_index, workspace_root())
    results_dir = resolve_path(results_dir, workspace_root())
    plan = plan_batch(sim_dir=sim_dir, hw_dir=hw_dir, truth_index=truth_index,
                      results_dir=results_dir, inventory_json=inventory_json, method=method,
                      allow_truth_fallback=allow_truth_fallback)
    truth = hw_truth_windows(truth_index)
    ok_pred = plan["hw_prediction_error"] is None
    if ok_pred:
        hw_pred, hw_status = reconstruct_hw_predictions_status(results_dir, method)
    else:
        hw_pred, hw_status = {}, {}

    per_window, failures = [], [dict(e, stage="plan") for e in plan["missing"]]
    for entry in [e for e in plan["windows"] if e["available"]]:
        rid, wid, geom = entry["recording_id"], entry["window_id"], entry["geometry_id"]
        paths = sim_paths(sim_dir, geom, method)
        cir_p, pred_p, truth_p = _first(paths["cir"]), _first(paths["pred"]), _first(paths["truth"])
        try:
            sim30 = load_sim_window_cir(cir_p, wid)
            hw30 = load_hw_cir_window(hw_dir / (rid + ".npz"), wid)
            if sim30.shape != hw30.shape:   # 严格，不缩 L
                raise Q4Error("shape_mismatch sim=%s hw=%s" % (sim30.shape, hw30.shape))
            sim_pred_all = load_predictions(pred_p)
            sim_pred = sim_pred_all.get(wid, sim_pred_all.get("__single__"))
            if sim_pred is None:
                raise Q4Error("sim_prediction_window_missing")
            sim_status_all = load_sim_predictions_status(pred_p)[1]
            sim_st = sim_status_all.get(wid, sim_status_all.get("__single__"))
            hw_preds = hw_pred[wid] if wid in hw_pred else None
            if hw_preds is None:
                raise Q4Error("hw_prediction_missing")
            if truth_p is not None:
                sim_truth_all = load_truth(truth_p)
                sim_truth = sim_truth_all.get(wid, sim_truth_all.get("__single__"))
                if sim_truth is None:
                    raise Q4Error("sim_truth_window_missing")
            elif allow_truth_fallback:
                sim_truth = truth[wid]["truth_xy"]
            else:
                raise Q4Error("sim_truth_missing")
            rec = compare_window(wid, sim30, hw30, sim_truth, sim_pred, truth[wid]["truth_xy"],
                                 hw_preds, sim_status=(sim_st if sim_st is not None else "unknown"),
                                 hw_status=hw_status.get(wid, "ok"))
            rec["recording_id"] = rid
            rec["geometry_id"] = geom
            per_window.append(rec)
        except Q4Error as exc:
            failures.append({"recording_id": rid, "geometry_id": geom, "window_id": wid,
                             "available": False, "reasons": ["run_failed"], "detail": str(exc)})

    doc = {
        "kind": "q4-batch-compare", "schema_version": 2,
        "inputs": plan["inputs"],
        "n_geometries": plan["n_geometries"], "n_recordings": plan["n_recordings"],
        "n_windows_planned": plan["n_windows_total"],
        "n_windows_compared": len(per_window), "n_windows_missing": len(failures),
        "missing": failures, "per_window": per_window,
        "summary": summarize(per_window) if per_window else None,
        "n_method_failed": (summarize(per_window)["n_method_failed"] if per_window else 0),
        # 未全量可比较即 incomplete（不把 unknown/全 0 available 当 success）
        "completion_status": ("complete" if (per_window and not failures) else "incomplete"),
        "note": "Q4 批量跨域比较；缺字段/L 不符单列，不缩 L、不假结果；有 missing 即 incomplete",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "q4_batch.json", doc)
    with (out_dir / "q4_batch_per_window.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "geometry_id", "window_id", "sim_method_status",
                    "hw_method_status", "rho_complex", "rho_env",
                    "n_common_identities", "mean_delta_sim_hw_peak_m"])
        for it in per_window:
            deltas = [x["delta_sim_hw_peak_m"] for x in it["cross_domain"]["items"]]
            w.writerow([it["recording_id"], it["geometry_id"], it["window_id"],
                        it.get("sim_method_status"), it.get("hw_method_status"),
                        it["rho_complex"], it["rho_env"],
                        it["cross_domain"]["n_common_identities"],
                        (sum(deltas) / len(deltas)) if deltas else None])
    return doc
