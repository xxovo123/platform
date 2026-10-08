#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总：把已保存的逐次结果聚合成论文可用表格（只消费已保存结果，不重跑）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import metrics
from . import protocol as P


def _all_rows(output_root, runs=("E1_A", "E1_B", "E2_A", "E2_B", "E4_A", "E4_B")) -> list:
    base = Path(output_root) / "runs"
    out = []
    if not base.is_dir():
        return out
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        rows_file = run_dir / "rows.jsonl"
        if not rows_file.is_file():
            continue
        if runs and not any(run_dir.name.startswith(r) for r in runs):
            continue
        for line in rows_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_csv(path: Path, rows: list, cols=None) -> None:
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cond(r):
    return {"task": r["task"], "layer": r["layer"], "scene": r["scene"], "snr_db": r["snr_db"],
            "L": r["L"], "layout": r.get("layout"), "direction": r.get("direction"),
            "d_nominal_m": r.get("d_nominal_m")}


def summarize(output_root) -> int:
    out_root = Path(output_root)
    rows = _all_rows(out_root)
    if not rows:
        print("没有可汇总的 rows（先 run）")
        return 1
    groups = {}
    for r in rows:
        key = json.dumps(_cond(r), ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, []).append(r)

    by_condition = []
    for key, sub in sorted(groups.items()):
        c = _cond(sub[0])
        agg = metrics.aggregate(sub)
        pooled, dr, succ = agg["rmse"], agg["dr_summary"], agg["success"]
        boot = pooled.get("rmse_bootstrap") or {}
        res = agg.get("resolution") or {}
        by_condition.append({
            "task": c["task"], "layer": c["layer"], "scene": c["scene"], "snr_db": c["snr_db"],
            "L": c["L"], "layout": c["layout"], "direction": c["direction"],
            "d_nominal_m": c["d_nominal_m"],
            "n_trials": pooled["n_trials"], "n_trials_with_pairs": pooled["n_trials_with_pairs"],
            "pairs_total": pooled["pairs_total"], "missing_total": pooled["missing_total"],
            "spurious_total": pooled["spurious_total"],
            "rmse_pooled_m": pooled["rmse_pooled_m"],
            "rmse_ci95_lo": boot.get("ci95_lo"), "rmse_ci95_hi": boot.get("ci95_hi"),
            "trial_rmse_sd_m": pooled["trial_rmse_sd_m"],
            "pair_coverage": pooled["pair_coverage"],
            "tp": dr["tp"], "fn": dr["fn"], "fp": dr["fp"], "dr": dr["dr"],
            "success_rate": succ["rate"], "success_ci95_lo": succ["lo"], "success_ci95_hi": succ["hi"],
            "resolution_success_rate": (res.get("success") or {}).get("rate"),
            "resolution_gate_m": res.get("gate_min_m"),
            "rms_ospa_m": agg["ospa"]["rms_ospa_m"],
            "n_ok": agg["n_ok"], "n_empty_or_failed": agg["n_empty_or_failed"],
        })

    by_layer = _by_layer(rows)
    by_scene = _by_scene(rows)
    e2_table = [r for r in by_condition if r["scene"] == "E2"]
    e4_table = [r for r in by_condition if r["scene"] == "E4"]
    totals = {
        "n_rows": len(rows),
        "by_task": {}, "by_status": {},
    }
    for r in rows:
        totals["by_task"][r["task"]] = totals["by_task"].get(r["task"], 0) + 1
        totals["by_status"][r["status"]] = totals["by_status"].get(r["status"], 0) + 1

    summary = {"kind": "jsf-summary", "output_root": str(out_root), "n_rows": len(rows),
               "n_conditions": len(by_condition), "totals": totals,
               "by_condition": by_condition, "by_layer": by_layer, "by_scene": by_scene,
               "e2_spacing_table": e2_table, "e4_L_table": e4_table,
               "note": ("RMSE 为池化值（Σsse/Σpairs）；未配对目标不计入 RMSE，单列 missing；"
                        "DR 单独 0.30 m 门限；pairs/n_truth 是配对覆盖率，不是 DR。")}
    sc = _sc()
    out_json = out_root / "summary_all.json"
    sc.atomic_write_json(out_json, summary)
    _write_csv(out_root / "summary_by_condition.csv", by_condition)
    _write_csv(out_root / "summary_by_layer.csv", by_layer)
    _write_csv(out_root / "summary_by_scene.csv", by_scene)
    print(json.dumps({"n_rows": len(rows), "n_conditions": len(by_condition),
                      "by_status": totals["by_status"], "out": str(out_json)},
                     ensure_ascii=False, indent=1))
    return 0


def _sc():
    from . import paths
    return paths.sim_common()


def _by_layer(rows) -> list:
    out = []
    for layer in ("A", "B"):
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        agg = metrics.aggregate(sub)
        out.append({"layer": layer, "n_rows": len(sub),
                    "rmse_pooled_m": agg["rmse"]["rmse_pooled_m"],
                    "pairs_total": agg["rmse"]["pairs_total"],
                    "missing_total": agg["rmse"]["missing_total"],
                    "dr": agg["dr_summary"]["dr"], "tp": agg["dr_summary"]["tp"],
                    "fn": agg["dr_summary"]["fn"], "fp": agg["dr_summary"]["fp"],
                    "rmse_ci95_lo": (agg["rmse"].get("rmse_bootstrap") or {}).get("ci95_lo"),
                    "rmse_ci95_hi": (agg["rmse"].get("rmse_bootstrap") or {}).get("ci95_hi"),
                    "note": "A/B 不得混成一个总 RMSE；本表仅为分层概览。"})
    return out


def _by_scene(rows) -> list:
    out = []
    for scene in sorted({r["scene"] for r in rows}):
        for layer in ("A", "B"):
            sub = [r for r in rows if r["scene"] == scene and r["layer"] == layer]
            if not sub:
                continue
            agg = metrics.aggregate(sub)
            out.append({"scene": scene, "layer": layer, "n_rows": len(sub),
                        "rmse_pooled_m": agg["rmse"]["rmse_pooled_m"],
                        "success_rate": agg["success"]["rate"],
                        "dr": agg["dr_summary"]["dr"],
                        "missing_total": agg["rmse"]["missing_total"]})
    return out


__all__ = ["summarize"]
