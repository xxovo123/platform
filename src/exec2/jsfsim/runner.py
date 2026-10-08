#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行器：逐次落盘、断点续跑、规范 §11 的 run 产物布局。

run 目录（`<output-root>/runs/<run_id>/`）必须含：
    config.json  source_manifest.json  tasks.jsonl  truth.json  rows.jsonl
    pairs.csv    summary.csv           checks.json  status.json
逐次包含：task/family/layer/scene/clean_id/L/SNR/trial/seed/status/预测/SSE/配对数/TPFNFP/耗时/输入hash。
CSV/JSON 中未定义用空值/null，**禁止伪零**；不根据性能删除行。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

from . import data as data_mod
from . import estimate, metrics, paths
from . import protocol as P


class RunError(RuntimeError):
    """运行不合规（显式失败，非零退出）。"""


ROW_FIELDS = [
    "run_id", "task", "family", "layer", "scene", "clean_id", "k", "snr_db", "trial", "L",
    "L_clean", "seed", "status", "n_truth", "n_predictions", "pairs", "missing", "spurious",
    "sse_m2", "rmse_trial_m", "pair_coverage", "gate_m", "tp", "fn", "fp", "dr", "success",
    "resolution_d_m", "resolution_gate_m", "resolution_success", "ospa_m", "elapsed_s",
    "input_hash", "truth_sha256", "layout", "direction", "d_nominal_m", "d_actual_m",
    "snr_realised_db", "Ps", "sigma", "spectrum_saved", "fail_reason",
]


def _sc():
    return paths.sim_common()


def _hash_payload(obj) -> str:
    return hashlib.sha256(P.canonical_json(obj).encode("utf-8")).hexdigest()


def _input_hash(rec: dict, *, trial: int, protocol_hash: str) -> str:
    return _hash_payload({
        "protocol_hash": protocol_hash, "clean_id": rec["clean_id"],
        "task_id": rec["task_id"], "L": int(rec["L"]), "snr_db": float(rec["snr_db"]),
        "trial": int(trial), "noise_seed": P.noise_seed(rec["clean_id"], trial),
        "targets_xy": rec["targets_xy"], "methods": [estimate.ONLY_METHOD],
    })


def run_paths(output_root, run_id: str) -> dict:
    base = Path(output_root) / "runs" / str(run_id)
    return {"run_dir": base, "rows": base / "rows.jsonl", "pairs": base / "pairs.csv",
            "summary": base / "summary.csv", "checks": base / "checks.json",
            "status": base / "status.json", "config": base / "config.json",
            "tasks": base / "tasks.jsonl", "truth": base / "truth.json",
            "source_manifest": base / "source_manifest.json", "samples": base / "samples",
            "logs": base / "logs"}


def source_manifest(protocol_hash: str) -> dict:
    import numpy as np
    sc = _sc()
    bundle = paths.frozen_bundle_dir()
    fp = sc.numerical_source_fingerprint(root=bundle)
    return {"kind": "jsf-source-manifest", "protocol_version": P.PROTOCOL_VERSION,
            "protocol_hash": protocol_hash, "frozen_bundle": str(bundle),
            "frozen_source_aggregate": fp["aggregate"], "n_frozen_files": fp["n_files"],
            "estimator_module": estimate.load_estimator().MODULE_FILES.get(estimate.ONLY_METHOD),
            "estimator_module_sha256": estimate.module_hash(),
            "new_package": str(paths.EXEC2_DIR),
            "new_package_files": _new_package_hashes(),
            "numpy": np.__version__, "python": __import__("sys").version.split()[0],
            "band_fix": "BANDLIMIT-FIX-20260929（analytic/forward.py 显式乘 band_mask）",
            "offgrid_shift_m": list(P.OFFGRID_SHIFT_M),
            "note": "冻结数值包只读复用；新版包 hash 逐文件记录。"}


def _new_package_hashes() -> dict:
    out = {}
    for p in sorted((paths.EXEC2_DIR / "jsfsim").glob("*.py")):
        out["jsfsim/%s" % p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------- #
# clean 缓存
# --------------------------------------------------------------------------- #
def prepare_cleans(specs, *, root, protocol_hash: str, source_aggregate: str, layer: str,
                   resume: bool, workers: int, resolution_overrides=None, geometry_by_clean=None,
                   max_states=None):
    """按 clean_id 去重生成/校验 clean；返回 (metas, blocked, counts)。"""
    seen, metas, blocked = {}, {}, {}
    for rec in specs:
        cid = rec["clean_id"]
        if cid in seen:
            continue
        seen[cid] = rec
    for cid, rec in seen.items():
        try:
            if rec["source"] == "analytic":
                out = data_mod.build_clean_analytic(rec, root=root, protocol_hash=protocol_hash,
                                                    source_aggregate=source_aggregate)
            else:
                numerical = (resolution_overrides or {}).get(cid)
                if isinstance(numerical, dict):
                    resolution = numerical.get("resolution_px_per_m")
                    run_time = numerical.get("run_time_meep")
                else:
                    resolution, run_time = numerical, None
                out = data_mod.build_clean_fdtd(
                    rec, root=root, protocol_hash=protocol_hash, source_aggregate=source_aggregate,
                    resume=resume, workers=workers,
                    resolution_px_per_m=resolution, run_time_meep=run_time,
                    geometry=(geometry_by_clean or {}).get(cid),
                    max_states=max_states)
            metas[cid] = out["meta"]
        except Exception as exc:  # noqa: BLE001
            blocked[cid] = {"clean_id": cid, "source": rec["source"],
                            "reason": "%s: %s" % (type(exc).__name__, exc)}
    return metas, blocked, {"planned_cleans": len(seen), "completed_cleans": len(metas),
                            "blocked_cleans": len(blocked)}


def _clean_map_for(specs) -> dict:
    out = {}
    for rec in specs:
        out.setdefault(rec["clean_id"], rec)
    return out


# --------------------------------------------------------------------------- #
# 样例保存规则（规范 §9：固定 trial=0、SNR=10 dB；E2 各间距与 E4 各 L 全保留）
# --------------------------------------------------------------------------- #
def is_sample(rec: dict, trial: int) -> bool:
    if int(trial) != 0:
        return False
    if rec["scene"] == "E4":
        return abs(float(rec["snr_db"]) - 10.0) < 1e-9
    return abs(float(rec["snr_db"]) - 10.0) < 1e-9


def save_sample(sdir: Path, rec: dict, row: dict, result: dict, X) -> dict:
    """保存谱与网格（NPZ）+ 元数据。无谱方法不伪造 heatmap。"""
    import numpy as np
    sc = _sc()
    sdir.mkdir(parents=True, exist_ok=True)
    stem = "%s__%s__L%03d__trial%04d" % (rec["task_id"], rec["clean_id"][:16], int(rec["L"]), int(row["trial"]))
    preds = np.asarray([[float(p["x"]), float(p["y"]), float(p.get("score", 0.0))]
                        for p in result["predictions"]], dtype=float).reshape(-1, 3)
    gx, gy = P.build_grid()
    arrays = {"predictions": preds,
              "grid_x": np.asarray(gx, dtype=float), "grid_y": np.asarray(gy, dtype=float),
              "truth_xy": np.asarray(rec["targets_xy"], dtype=float).reshape(-1, 2)}
    if result.get("spectrum") is not None:
        arrays["spectrum"] = np.asarray(result["spectrum"], dtype=float)
    sc.atomic_write_npz(sdir / ("%s.npz" % stem), **arrays)
    meta = {"task_id": rec["task_id"], "clean_id": rec["clean_id"], "scene": rec["scene"],
            "k": int(rec["k"]), "snr_db": float(rec["snr_db"]), "trial": int(row["trial"]),
            "L": int(rec["L"]), "seed": int(row["seed"]), "status": row["status"],
            "targets_xy_m": rec["targets_xy"], "prediction": preds.tolist(),
            "n_predictions": int(preds.shape[0]), "spectrum_available": bool(result.get("spectrum") is not None),
            "grid_note": "grid_x/grid_y = 协议统一静态域 89x54；spectrum 在时保存",
            "tags": rec["tags"], "layout": rec.get("layout"), "direction": rec.get("direction"),
            "d_nominal_m": rec.get("d_nominal_m"), "d_actual_m": rec.get("d_actual_m"),
            "note": "样例固定 trial=0、SNR=10 dB；不另挑 seed。"}
    sc.atomic_write_json(sdir / ("%s.json" % stem), meta)
    return meta


# --------------------------------------------------------------------------- #
# 主执行
# --------------------------------------------------------------------------- #
def _trial_worker(payload: dict) -> dict:
    """trial 级并行 worker（模块级函数保证可 pickle）。

    纪律：不改数学、不改 seed、不改噪声口径——只是把同一 trial 的
    「加噪 → 装配 case → 调用冻结 JSF-MUSIC → 评估 → 落样例」放到独立进程执行。
    每个子进程导入时 `sim_common` 会把 BLAS/OMP 线程固定为 1（与父进程同一口径）。
    """
    import numpy as np
    sc = _sc()
    rec = payload["rec"]
    raw = np.asarray(payload["raw"])
    gx, gy = P.build_grid()
    row = {k: None for k in ROW_FIELDS}
    t0 = time.perf_counter()
    row.update({"run_id": payload["run_id"], "task": rec["task_id"], "family": rec["family"],
                "layer": payload["layer"], "scene": rec["scene"], "clean_id": rec["clean_id"],
                "k": int(rec["k"]), "snr_db": float(rec["snr_db"]), "trial": int(payload["trial"]),
                "L": int(rec["L"]), "L_clean": int(rec["L_clean"]),
                "seed": int(P.noise_seed(rec["clean_id"], payload["trial"])),
                "truth_sha256": payload["truth_sha"], "layout": rec.get("layout"),
                "direction": rec.get("direction"), "d_nominal_m": rec.get("d_nominal_m"),
                "d_actual_m": rec.get("d_actual_m"),
                "input_hash": _input_hash(rec, trial=int(payload["trial"]),
                                          protocol_hash=payload["protocol_hash"])})
    try:
        built = data_mod.build_x(raw, L_requested=int(rec["L"]), snr_db=float(rec["snr_db"]),
                                 physical_id=rec["clean_id"], trial=int(payload["trial"]))
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "blocked", "fail_reason": "%s: %s" % (type(exc).__name__, exc),
                    "elapsed_s": time.perf_counter() - t0})
        return row
    X = built["X"]
    row.update({"Ps": built["diag"]["Ps"], "sigma": built["diag"]["sigma"],
                "snr_realised_db": built["diag"]["snr_realised_db"]})
    case = _build_case(rec, X, grid_x=gx, grid_y=gy, trial=int(payload["trial"]), seed=row["seed"])
    result = estimate.run_single(case)
    m = metrics.evaluate_row(rec["targets_xy"],
                             [(p["x"], p["y"]) for p in result["predictions"]],
                             resolution_distance_m=rec["resolution_distance_m"],
                             status=result["status"])
    row.update({
        "status": str(m["status_used"] if result["status"] == "ok" else result["status"]),
        "n_truth": int(m["n_truth"]), "n_predictions": int(m["n_predictions"]),
        "pairs": int(m["pairs"]), "missing": int(m["missing"]), "spurious": int(m["spurious"]),
        "sse_m2": float(m["sse_m2"]),
        "rmse_trial_m": (math.sqrt(float(m["sse_m2"]) / int(m["pairs"]))
                         if int(m["pairs"]) else None),
        "pair_coverage": m["pair_coverage"], "gate_m": float(m["gate_m"]),
        "tp": int(m["tp"]), "fn": int(m["fn"]), "fp": int(m["fp"]), "dr": m["dr"],
        "success": bool(m["success"]),
        "resolution_d_m": rec["resolution_distance_m"],
        "resolution_gate_m": (m["resolution"] or {}).get("gate_m"),
        "resolution_success": (m["resolution"] or {}).get("success"),
        "ospa_m": float(m["ospa_m"]), "elapsed_s": float(time.perf_counter() - t0),
        "fail_reason": (None if result["status"] == "ok"
                        else str(result["diagnostics"].get("reason"))[:200]),
        "_errors": m["errors"],
        "_predictions": [[float(p["x"]), float(p["y"]), float(p.get("score", 0.0))]
                         for p in result["predictions"]],
        "spectrum_saved": bool(result.get("spectrum") is not None),
    })
    if payload.get("save_sample") and result.get("status") == "ok":
        save_sample(Path(payload["samples_dir"]), rec, row, result, X)
    return row


def execute_task(task: str, *, mode: str, output_root, run_id: str, resume: bool,
                 workers: int = 4, resolution_overrides=None, max_states=None,
                 dry_run: bool = False, limit_specs=None,
                 trial_workers: int = 1, fdtd_validation=None, numerical_waiver=None) -> dict:
    if mode not in ("smoke", "pilot", "formal"):
        raise RunError("mode 必须是 smoke|pilot|formal，收到 %r" % mode)
    if mode == "formal" and (limit_specs is not None or max_states is not None):
        raise RunError("正式任务不能用 limit/max_states 缩短；请用 pilot 或 smoke")
    validation = None
    if task.endswith("_B") and (mode == "formal" or numerical_waiver):
        from . import numerical_validation as NV
        if fdtd_validation and numerical_waiver:
            raise RunError("请选择严格数值验收报告或用户明确放行决定，不能混用")
        validation = (NV.verify_user_waiver(numerical_waiver, profile="controlled") if numerical_waiver
                      else NV.verify(fdtd_validation, profile="controlled", output_root=output_root))
        selected = validation["selected_numerics"]
        requested = resolution_overrides or {}
        chosen = {"resolution_px_per_m": selected["resolution_px_per_m"],
                  "run_time_meep": selected["run_time_meep"]}
        for override in requested.values():
            if override != chosen:
                raise RunError("请求的数值参数与通过验收的档位不同")
        resolution_overrides = {r["clean_id"]: dict(chosen) for r in P.expand_task(task)}
    if mode == "smoke" and max_states is None and task.endswith("_B"):
        max_states = 2
    estimate.assert_methods(None)

    proto = P.build_protocol()
    protocol_hash = hashlib.sha256(P.canonical_json(proto).encode("utf-8")).hexdigest()
    specs = P.expand_task(task)
    if mode == "smoke":
        # smoke：固定首场景 + 10 dB + 2 trial（formal=false；不由 --limit 缩短冒充正式）
        specs = [s for s in specs if abs(float(s["snr_db"]) - 10.0) < 1e-9][:1]
        for s in specs:
            s["trials"] = 2
    elif mode == "pilot":
        for s in specs:
            s["trials"] = 1
    if limit_specs is not None:
        specs = specs[: int(limit_specs)]
    if not specs:
        raise RunError("任务 %s 在 mode=%s 下没有可执行 spec" % (task, mode))

    rp = run_paths(output_root, run_id)
    sc = _sc()
    layer = task[-1]
    sm = source_manifest(protocol_hash)
    identity = {"task": task, "mode": mode, "protocol_hash": protocol_hash,
                "specs": specs, "methods": [estimate.ONLY_METHOD],
                "frozen_source_aggregate": sm["frozen_source_aggregate"],
                "new_package_files": sm["new_package_files"],
                "resolution_overrides": dict(resolution_overrides or {}),
                "max_states": max_states,
                "numerical_validation_sha256": validation["sha256"] if validation else None}
    run_fingerprint = _hash_payload(identity)
    if rp["config"].is_file():
        existing = json.loads(rp["config"].read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RunError("同 run_id 已有不同或旧版配置；拒绝覆盖，请使用新的 run_id")
        if not resume:
            raise RunError("已有 run_id 必须显式 resume；拒绝覆盖原结果")
    elif rp["rows"].is_file():
        raise RunError("已有 rows 却缺 config；拒绝复用")
    rp["run_dir"].mkdir(parents=True, exist_ok=True)
    sc.atomic_write_json(rp["source_manifest"], sm)
    config = {
        "kind": "jsf-run-config", "run_id": run_id, "task": task, "mode": mode,
        "formal": bool(mode == "formal"), "layer": layer,
        "run_fingerprint": run_fingerprint,
        "numerical_validation": validation,
        "numerical_convergence_verified": bool(validation and validation.get("passed") is True),
        "numerical_gate_policy": "user_waiver" if numerical_waiver else "strict_validation",
        "methods": [estimate.ONLY_METHOD],
        "protocol_id": P.PROTOCOL_ID, "protocol_version": P.PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "n_specs": len(specs), "n_trials_total": int(sum(s["trials"] for s in specs)),
        "n_method_runs_total": int(sum(s["trials"] for s in specs)),
        "snr_db": sorted({float(s["snr_db"]) for s in specs}),
        "L": sorted({int(s["L"]) for s in specs}),
        "output_root": str(Path(output_root)), "run_dir": str(rp["run_dir"]),
        "workers": int(workers), "max_states": max_states,
        "resolution_overrides": dict(resolution_overrides or {}),
        "budget_upper_bound": P.BATCH_UPPER_BOUND,
        "created_at": sc.now_iso(),
        "provenance": sc.provenance_block(script="jsfsim.runner"),
    }
    sc.atomic_write_json(rp["config"], config)
    with rp["tasks"].open("w", encoding="utf-8") as fh:
        for rec in specs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    truth = {"kind": "jsf-truth", "task": task,
             "cases": {rec["task_id"]: {"clean_id": rec["clean_id"], "k": rec["k"],
                                        "targets_xy_m": rec["targets_xy"],
                                        "targets_polar": rec["targets_polar"],
                                        "snr_db": rec["snr_db"], "L": rec["L"],
                                        "resolution_distance_m": rec["resolution_distance_m"]}
                       for rec in specs}}
    sc.atomic_write_json(rp["truth"], truth)
    truth_sha = sc.sha256_file(rp["truth"])

    if dry_run:
        status = {"task": task, "mode": mode, "dry_run": True, "n_specs": len(specs),
                  "n_method_runs": int(sum(s["trials"] for s in specs)),
                  "run_dir": str(rp["run_dir"])}
        sc.atomic_write_json(rp["status"], status)
        print(json.dumps(status, ensure_ascii=False, indent=1))
        return status

    source_aggregate = sm["frozen_source_aggregate"]
    print("[runner] 构建/校验 clean 缓存（%d 个 clean）..." % len({s['clean_id'] for s in specs}))
    metas, blocked, ccounts = prepare_cleans(
        specs, root=output_root, protocol_hash=protocol_hash,
        source_aggregate=source_aggregate, layer=layer, resume=resume, workers=workers,
        resolution_overrides=resolution_overrides, max_states=max_states)
    print("[runner] clean：planned=%d completed=%d blocked=%d"
          % (ccounts["planned_cleans"], ccounts["completed_cleans"], ccounts["blocked_cleans"]))

    rows = _read_rows(rp["rows"]) if resume else []
    done = {(r["task"], int(r["trial"])) for r in rows}
    n_new = 0
    total_trials = int(sum(s["trials"] for s in specs))
    # ---- 组装待执行 trial 清单（跳过已完成的；blocked clean 全跳过）----
    jobs, skipped = [], 0
    for rec in specs:
        if rec["clean_id"] in blocked:
            skipped += int(rec["trials"])
            continue
        loaded = data_mod.load_clean(output_root, layer, rec["clean_id"])
        raw = loaded["raw_cir"]
        for trial in range(int(rec["trials"])):
            if (rec["task_id"], trial) in done:
                continue
            jobs.append({"rec": rec, "raw": raw, "trial": trial, "layer": layer,
                         "protocol_hash": protocol_hash, "truth_sha": truth_sha,
                         "run_id": run_id, "samples_dir": str(rp["samples"]),
                         "save_sample": is_sample(rec, trial)})
    print("[runner] 待执行 trial=%d（trial_workers=%d）；跳过 blocked=%d"
          % (len(jobs), int(trial_workers), skipped))
    tw = max(1, int(trial_workers))
    if tw == 1 or len(jobs) <= 1:
        for payload in jobs:
            rows.append(_trial_worker(payload))
            n_new += 1
            if n_new % 200 == 0:
                _flush_rows(rp["rows"], rows)
                print("[runner] %d/%d method-runs" % (len(rows), total_trials))
    else:
        import concurrent.futures as _cf
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        with _cf.ProcessPoolExecutor(max_workers=tw, mp_context=ctx) as ex:
            for row in ex.map(_trial_worker, jobs, chunksize=4):
                rows.append(row)
                n_new += 1
                if n_new % 200 == 0:
                    _flush_rows(rp["rows"], rows)
                    print("[runner] %d/%d method-runs（并行 %d workers）"
                          % (len(rows), total_trials, tw))
    # 稳定排序，便于跨运行对齐（不改任何数值）
    rows.sort(key=lambda r: (str(r.get("task")), str(r.get("clean_id")), int(r.get("L") or 0),
                             int(r.get("trial") or 0)))
    _flush_rows(rp["rows"], rows)
    _write_pairs_csv(rp["pairs"], rows)
    summary_rows = _write_summary_csv(rp["summary"], rows)
    checks = _checks(rows, specs, blocked, proto, metas, rp)
    status = {
        "task": task, "mode": mode, "formal": bool(mode == "formal"),
        "run_id": run_id, "run_dir": str(rp["run_dir"]),
        "expected_method_runs": int(sum(s["trials"] for s in specs)),
        "completed_method_runs": int(len([r for r in rows if r.get("status") != "blocked"])),
        "failed_rows": int(len([r for r in rows if r.get("status") == "failed"])),
        "empty_rows": int(len([r for r in rows if r.get("status") == "empty"])),
        "blocked_rows": int(len([r for r in rows if r.get("status") == "blocked"])),
        "blocked_cleans": blocked,
        "clean_counts": ccounts,
        "summary_conditions": len(summary_rows),
        "checks": checks,
        "completion_status": ("complete"
                              if ccounts["blocked_cleans"] == 0
                              and len(rows) == int(sum(s["trials"] for s in specs))
                              else "incomplete"),
        "finished_at": sc.now_iso(),
    }
    sc.atomic_write_json(rp["checks"], checks)
    sc.atomic_write_json(rp["status"], status)
    print(json.dumps({k: v for k, v in status.items()
                      if k not in ("blocked_cleans", "checks")}, ensure_ascii=False, indent=1))
    return status


def _build_case(rec: dict, X, *, grid_x, grid_y, trial: int, seed: int, geometry=None):
    cb = paths.adapter("case_build")
    return cb.build_case(case_id=rec["task_id"], clean_id=rec["clean_id"], X=X, k=int(rec["k"]),
                         grid_x=grid_x, grid_y=grid_y, layer=rec["layer"], source=rec["source"],
                         scene=rec["scene"], tags=rec["tags"], snr_db=float(rec["snr_db"]),
                         trial=int(trial), seed=int(seed), L=int(X.shape[0]),
                         resolution_distance_m=rec["resolution_distance_m"],
                         geometry=geometry or {})


def _read_rows(p: Path) -> list:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _flush_rows(p: Path, rows) -> None:
    sc = _sc()
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    sc.atomic_write_bytes(p, payload.encode("utf-8"))


def _write_pairs_csv(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "task", "layer", "scene", "clean_id", "snr_db", "L", "trial",
            "truth_index", "pred_index", "e_xy", "e_r", "e_theta", "in_gate_0.30m"]
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            for e in r.get("_errors") or []:
                w.writerow({"run_id": r["run_id"], "task": r["task"], "layer": r["layer"],
                            "scene": r["scene"], "clean_id": r["clean_id"], "snr_db": r["snr_db"],
                            "L": r["L"], "trial": r["trial"],
                            "truth_index": e["truth_index"], "pred_index": e["pred_index"],
                            "e_xy": e["e_xy"], "e_r": e["e_r"], "e_theta": e["e_theta"],
                            "in_gate_0.30m": bool(float(e["e_xy"]) <= metrics.GATE_M + 1e-12)})


def _condition_of(r: dict) -> dict:
    return {"task": r["task"], "layer": r["layer"], "scene": r["scene"],
            "snr_db": r["snr_db"], "L": r["L"], "layout": r.get("layout"),
            "direction": r.get("direction"), "d_nominal_m": r.get("d_nominal_m")}


def _write_summary_csv(p: Path, rows) -> list:
    groups = {}
    for r in rows:
        key = json.dumps(_condition_of(r), ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, []).append(r)
    out = []
    for key, sub in sorted(groups.items()):
        cond = _condition_of(sub[0])
        agg = metrics.aggregate(sub)
        pooled = agg["rmse"]
        boot = pooled.get("rmse_bootstrap") or {}
        out.append({
            "task": cond["task"], "layer": cond["layer"], "scene": cond["scene"],
            "snr_db": cond["snr_db"], "L": cond["L"], "layout": cond["layout"],
            "direction": cond["direction"], "d_nominal_m": cond["d_nominal_m"],
            "n_trials": pooled["n_trials"], "n_trials_with_pairs": pooled["n_trials_with_pairs"],
            "pairs_total": pooled["pairs_total"], "missing_total": pooled["missing_total"],
            "spurious_total": pooled["spurious_total"],
            "rmse_pooled_m": pooled["rmse_pooled_m"],
            "rmse_ci95_lo": boot.get("ci95_lo"), "rmse_ci95_hi": boot.get("ci95_hi"),
            "trial_rmse_sd_m": pooled["trial_rmse_sd_m"],
            "pair_coverage": pooled["pair_coverage"],
            "tp": agg["dr_summary"]["tp"], "fn": agg["dr_summary"]["fn"],
            "fp": agg["dr_summary"]["fp"], "dr": agg["dr_summary"]["dr"],
            "success_rate": agg["success"]["rate"],
            "success_ci95_lo": agg["success"]["lo"], "success_ci95_hi": agg["success"]["hi"],
            "resolution_success_rate": (agg.get("resolution") or {}).get("success", {}).get("rate"),
            "resolution_gate_m": (agg.get("resolution") or {}).get("gate_min_m"),
            "rms_ospa_m": agg["ospa"]["rms_ospa_m"],
            "n_ok": agg["n_ok"], "n_empty_or_failed": agg["n_empty_or_failed"],
        })
    cols = list(out[0].keys()) if out else ["task"]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in out:
            w.writerow(row)
    return out


def _checks(rows, specs, blocked, proto, metas, rp) -> dict:
    n_expected = int(sum(s["trials"] for s in specs))
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    band_fractions = [float(m.get("band_check", {}).get("oob_fraction_before_crop") or 0.0)
                      for m in metas.values() if m.get("band_check")]
    samples = sorted(p.name for p in Path(rp["samples"]).glob("*.npz")) \
        if Path(rp["samples"]).is_dir() else []
    return {
        "kind": "jsf-run-checks",
        "methods_only_this_method": True,
        "no_offgrid_shift": list(proto["offgrid_shift_m"]) == [0.0, 0.0],
        "expected_method_runs": n_expected,
        "completed_rows": int(len(rows)),
        "by_status": by_status,
        "blocked_cleans": sorted(blocked),
        "analytic_band_oob_max": (max(band_fractions) if band_fractions else None),
        "analytic_band_oob_gate": 1e-12,
        "analytic_band_gate_passed": (max(band_fractions) <= 1e-12 if band_fractions else None),
        "n_samples": len(samples),
        "sample_names": samples[:40],
        "checks_passed": bool(
            len(rows) == n_expected and not blocked
            and (not band_fractions or max(band_fractions) <= 1e-12)),
        "notes": ["CSV/JSON 未定义用空值，禁止伪零。",
                  "所有 trial 保留在分母；失败/空输出不删行。",
                  "rmse_trial_m 为 null 表示该 trial 无配对（不是 0）。"],
    }


__all__ = ["RunError", "execute_task", "run_paths", "prepare_cleans", "ROW_FIELDS",
           "is_sample", "source_manifest"]
