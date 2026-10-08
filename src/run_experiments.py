#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仿真实验统一 CLI（2026-09-26）。

安全纪律
--------
* --help 与 plan 只规划，不仿真（不建 cache、不跑方法）。
* run 协议：--limit N、--dryrun、--resume。**未给 --resume 且 rows 已存在 -> 拒绝覆盖**。
* 层 B（E1_B/E2_B/E4_B）**必须先通过 FDTD 收敛前检**：--fdtd-validation <convergence.json>
  （verdict=passed 且与 scene 物理/数值口径匹配；不匹配即拒绝并返回非零）。
  tiny smoke 用 --validation-only（自动标非正式，不得作 main-run 证据）。
* cache 严格匹配；runid 单一名字 + containment；已有不同 config_hash -> 拒覆。
* 完成度：per-clean_id blocked 记录 + planned/completed/missing 计数 + completion_status；
  不静默丢行伪完整；incomplete 时返回非零。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import sim_common as sc  # noqa: E402  先于 numpy：线程 env 先固定

import numpy as np  # noqa: E402

FAMILIES = ("E1_A", "E2_A", "E4_A", "HW_A", "HW_B", "E1_B", "E2_B", "E4_B", "E3", "ALL_A", "ALL_B")
B_FAMILIES = ("E1_B", "E2_B", "E4_B", "HW_B", "ALL_B")


def _protocol():
    import protocol
    return protocol.PROTOCOL, protocol.protocol_hash()


def _methods(spec_arg):
    from adapter import methods_loader as ml
    tokens = [t.strip() for t in (spec_arg or "").split(",") if t.strip()]
    if not tokens or tokens == ["main"]:
        return list(ml.MAIN_METHODS)
    if tokens == ["all"]:
        return list(ml.METHOD_ORDER)
    unknown = [t for t in tokens if t not in ml.METHOD_ORDER]
    if unknown:
        raise SystemExit("未知方法：%s（可选 %s）" % (unknown, list(ml.METHOD_ORDER)))
    return tokens


def _config_hash(protocol_dict, extra: dict) -> str:
    return sc.config_hash_for(protocol_dict, extra_config=extra)


def _provenance_dump(paths, protocol_dict, cfg_hash, tag, extra=None):
    """把冻结协议 JSON + 源指纹 + env 全量落 provenance（不只哈希）。"""
    prov = paths["provenance"]
    sc.atomic_write_json(prov / ("protocol_%s.json" % tag), protocol_dict)
    sc.atomic_write_json(prov / ("source_fingerprint_%s.json" % tag), sc.source_fingerprint())
    env = sc.provenance_block()
    env.update({"config_hash": cfg_hash, "protocol_json_sha256": sc.sha256_file(prov / ("protocol_%s.json" % tag))})
    if extra:
        env.update(extra)
    sc.atomic_write_json(prov / ("env_%s.json" % tag), env)


def _group_fields(spec):
    tags = list(spec.get("tags") or [])
    layout = next((t.split("=", 1)[1] for t in tags if t.startswith("layout=")), None)
    direction = next((t.split("=", 1)[1] for t in tags if t.startswith("direction=")), None)
    return {"tags": tags, "layout": layout, "direction": direction,
            "d_nominal_m": spec.get("d_nominal_m"), "d_actual_m": spec.get("d_actual_m"),
            "resolution_distance_m": spec.get("resolution_distance_m")}


ROW_COLUMNS = ["run_id", "family", "layer", "source", "scene", "case_id", "clean_id", "k",
               "snr_db", "trial", "L", "L_requested", "L_actual", "L_available", "formal",
               "method", "status", "n_predictions", "tp", "fn", "fp",
               "ospa", "success", "estimate_s", "noise_seed", "tags", "layout", "direction",
               "d_nominal_m", "d_actual_m", "resolution_distance_m", "spectrum_available",
               "case_fingerprint_before", "case_fingerprint_after", "case_modified",
               "truth_sha256", "failures"]


def _build_clean_all(specs, paths, ph, *, resume=True, verbose=True, fdtd=None):
    """构建/加载 clean 缓存。返回 (clean_map, metas, blocked, counts)。

    blocked 逐 clean_id 记录（**不是**按 source 一刀切跳过）。counts 含 planned/completed/missing。
    fdtd 选填 dict{validation_only, max_states, formal, geometry_by_clean}。
    """
    import analytic.scenes as scenes
    from adapter import sources as sources_mod
    agg = sc.source_fingerprint()["aggregate"]
    clean_map = scenes.resolve_clean_specs(specs)
    metas, blocked = {}, {}
    fdtd = dict(fdtd or {})
    geom_by_clean = dict(fdtd.get("geometry_by_clean") or {})
    for cid, rec in clean_map.items():
        src = sources_mod.get_source(rec["source"])
        try:
            ok, reason = src.available()
        except Exception as exc:
            ok, reason = False, "%s: %s" % (type(exc).__name__, exc)
        if not ok:
            blocked[cid] = {"clean_id": cid, "source": rec["source"], "reason": reason}
            if verbose:
                print("  [blocked] %s source=%s：%s" % (cid, rec["source"], reason))
            continue
        if verbose:
            print("  [clean] %s source=%s k=%d L_clean=%d" % (cid, rec["source"], rec["k"],
                                                              rec["L_clean"]))
        try:
            if rec["source"] == "fdtd":
                metas[cid] = src.build_clean(rec, cache_dir=paths["cache"], protocol_hash=ph,
                                             source_aggregate=agg, resume=resume,
                                             max_states=fdtd.get("max_states"),
                                             formal=bool(fdtd.get("formal", True)),
                                             workers=int(fdtd.get("workers", 1) or 1),
                                             res_overrides=fdtd.get("res_overrides"),
                                             geometry=geom_by_clean.get(cid))
            else:
                metas[cid] = src.build_clean(rec, cache_dir=paths["cache"], protocol_hash=ph,
                                             source_aggregate=agg, resume=resume)
        except Exception as exc:
            blocked[cid] = {"clean_id": cid, "source": rec["source"],
                            "reason": "build_clean 失败：%s: %s" % (type(exc).__name__, exc)}
            if verbose:
                print("  [blocked] %s：%s" % (cid, blocked[cid]["reason"]))
    planned = len(clean_map)
    completed = len(metas)
    counts = {"planned_clean": planned, "completed_clean": completed,
              "missing_clean": planned - completed,
              "planned_specs": len(specs),
              "blocked_specs": sum(1 for s in specs if s["clean_id"] in blocked),
              "completion_status": ("complete" if not blocked else "incomplete")}
    return clean_map, metas, blocked, counts


def _atomic_flush_rows(path, rows):
    payload = "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows)
    sc.atomic_write_bytes(path, payload.encode("utf-8"))


def _read_rows_strict(path):
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise SystemExit("rows 文件第 %d 行 JSON 解析失败（拒绝静默丢弃）：%s: %s"
                             % (lineno, type(exc).__name__, exc))
    return rows


def _truth_file(paths, family, specs, truth_by_case):
    payload = {"kind": "sim-truth", "family": family,
               "cases": {cid: rec for cid, rec in sorted(truth_by_case.items())}}
    p = paths["results"] / ("truth_%s.json" % family)
    sc.atomic_write_json(p, payload)
    sha = sc.sha256_file(p)
    (paths["results"] / ("truth_%s.sha256" % family)).write_text("%s  %s%s" % (sha, p.name, chr(10)),
                                                                encoding="utf-8")
    return sha


SCHEMA = {
    "kind": "sim-run-schema", "version": 2,
    "rows_file": "results/rows_<family>.jsonl", "rows_csv": "results/rows_<family>.csv",
    "truth_file": "results/truth_<family>.json (+ .sha256)",
    "summary": "results/summary_<family>.json",
    "row_columns": list(ROW_COLUMNS),
    "row_extra_keys": ["metrics", "predictions"],
    "grouping": "layer + method + scene + L + snr_db（另有 tags/layout/direction/d_nominal_m/d_actual_m）",
    "metrics_contract": {"provider": "metrics 包", "api": "evaluate_predictions/aggregate",
                         "failures_counted": True},
    "samples_layout": {"dir": "results/samples",
                       "pattern": "<family>__<case_id>__trial<trial:04d>__<method>.npz",
                       "keys": ["predictions (N,3)", "points (=predictions)", "grid_x", "grid_y",
                                "truth_xy (K,2, 样例侧)", "spectrum (ny,nx) 或不存在"],
                       "grid_note": "无谱方法（C1）用统一显示域 grid_x/grid_y 画 native 点；不伪造 heatmap。",
                       "trial_policy": "固定 trial==0 样例",
                       "c1_note": "连续域方法（C1）无网格谱 -> 不产生 spectrum，不伪造 heatmap。"},
    "artifact_dirs": ["cache/", "results/", "figures/", "logs/", "provenance/"],
    "provenance": ["protocol_<tag>.json", "source_fingerprint_<tag>.json", "env_<tag>.json"],
    "truth_isolation": "case 不含真值；真值在 truth_<family>.json（带 sha256），rows 存 predictions+truth_sha256",
}


def _save_sample(paths, row, res, spec, trial):
    if int(trial) != 0:
        return
    from adapter import methods_loader as _ml
    sdir = paths["results"] / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    stem = "%s__%s__trial%04d__%s" % (spec["family"], spec["case_id"], int(trial), row["method"])
    preds = np.asarray([[float(p["x"]), float(p["y"]), float(p["score"])]
                        for p in res["predictions"]], dtype=float).reshape(-1, 3)
    gx, gy = sc.build_grid()
    truth_xy = np.asarray([[float(x), float(y)] for x, y in spec["targets_xy"]], dtype=float).reshape(-1, 2)
    # points 与 predictions 一致（fig native-plot 兼容；连续域方法无谱也画统一显示域上的点）
    arrays = {"predictions": preds, "points": preds,
              "grid_x": np.asarray(gx, dtype=float), "grid_y": np.asarray(gy, dtype=float),
              "truth_xy": truth_xy}
    sp_avail = res.get("spectrum") is not None
    if sp_avail:
        arrays["spectrum"] = np.asarray(res["spectrum"], dtype=float)
    sc.atomic_write_npz(sdir / ("%s.npz" % stem), **arrays)
    sc.atomic_write_json(sdir / ("%s.json" % stem), {
        "family": spec["family"], "case_id": spec["case_id"], "clean_id": spec["clean_id"],
        "scene": spec["scene"], "method": row["method"], "status": row["status"],
        "trial": int(trial), "k": int(spec["k"]), "snr_db": float(spec["snr_db"]), "L": int(spec["L"]),
        "tags": list(spec.get("tags") or []),
        "layout": row.get("layout"), "direction": row.get("direction"),
        "d_nominal_m": row.get("d_nominal_m"), "d_actual_m": row.get("d_actual_m"),
        "resolution_distance_m": row.get("resolution_distance_m"),
        "spectrum_available": bool(sp_avail),
        "grid_source": "unified_static_domain（±2.2 x [-3.2,-0.55] 5 cm；无谱方法亦用它做显示域）",
        "truth_xy_m": [[float(x), float(y)] for x, y in spec["targets_xy"]],
        "predictions_equals_points": True,
        "n_predictions": int(preds.shape[0]),
        "params": _ml.params_for(row["method"], _protocol()[0]) or {},
        "case_fingerprint": row["case_fingerprint_before"], "noise_seed": int(row["noise_seed"]),
        "note": ("native points/predictions 一致并落盘；grid_x/grid_y = 统一显示域；truth_xy 只在样例"
                 "（不进 case）；spectrum 仅当方法产出网格谱时保存（C1 无谱，不伪造 heatmap）。")})


def cmd_plan(args) -> int:
    protocol_dict, ph = _protocol()
    import analytic.scenes as scenes
    methods = _methods(args.methods)
    out = {"protocol_hash": ph, "families": {}}
    for fam in ([args.family] if args.family else list(FAMILIES)):
        try:
            specs = scenes.specs_for(fam, trials=args.trials,
                                     include_e3_optional=args.include_e3_optional)
        except Exception as exc:
            out["families"][fam] = {"error": str(exc)}
            continue
        shots = sum(s["trials"] for s in specs)
        out["families"][fam] = {
            "n_specs": len(specs), "n_clean_sequences": len({s["clean_id"] for s in specs}),
            "n_noise_trials": shots, "n_method_runs": shots * len(methods),
            "snr_db": sorted({s["snr_db"] for s in specs}), "L": sorted({s["L"] for s in specs}),
            "requires_fdtd_validation": fam in B_FAMILIES}
    out["search"] = protocol_dict["search"]
    out["noise_policy"] = protocol_dict["noise"]["noise_prefix_policy"]
    out["note"] = "plan 只规划，不仿真；层 B 正式 run 必须先通过 --fdtd-validation。"
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def _load_res_overrides(path):
    """加载场景级分辨率覆盖 {clean_id: res}；None/缺省 → {}。"""
    if not path:
        return {}
    import json as _json
    from pathlib import Path as _Path
    doc = _json.loads(_Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not all(isinstance(v, (int, float)) and v > 0
                                            for v in doc.values()):
        raise SystemExit("--fdtd-scene-res-overrides 必须是 {clean_id: 正数分辨率} JSON")
    return {str(k): float(v) for k, v in doc.items()}


def _fdtd_gate(args, family, specs, ph):
    """层 B 门禁：返回 (gate_result, fdtd_opts)。非 B 直接返回 (None, {})。

    workers 只进调度（fdtd_opts），不进 config_hash/extra（不影响数值与 run 身份）。
    """
    workers = max(1, int(getattr(args, "fdtd_workers", 1) or 1))
    res_overrides = _load_res_overrides(getattr(args, "fdtd_scene_res_overrides", None))
    if family not in B_FAMILIES:
        return None, {}
    from adapter import scene_export, fdtd_validation, sources
    protocol_dict = _protocol()[0]
    if args.validation_only:
        return ({"passed": True, "mode": "validation_only_smoke", "formal": False,
                 "physical_validation": False,
                 "note": "tiny smoke，非正式；不得作为 main-run 证据。"},
                {"validation_only": True, "max_states": int(args.fdtd_max_states or 2),
                 "formal": False, "workers": workers, "res_overrides": res_overrides})
    if not args.fdtd_validation:
        raise SystemExit("层 B 正式 run 必须先通过 FDTD 收敛前检：请先运行 fdtd convergence，"
                         "再用 --fdtd-validation <convergence.json>；tiny smoke 用 --validation-only。")
    src = sources.get_source("fdtd")
    ok, reason = src.available()
    if not ok:
        raise SystemExit("层 B provider 不可用：%s" % reason)
    scenes_map = scene_export.scenes_for_specs(specs, res_overrides=res_overrides)
    gate = fdtd_validation.gate_scenes(scenes_map, args.fdtd_validation, protocol=protocol_dict)
    if gate.get("blocked"):
        raise SystemExit("FDTD 门禁不可用（helper 未就绪，正式 B 暂不可跑）：%s" % gate.get("reason"))
    if not gate["passed"]:
        raise SystemExit("FDTD 收敛门禁未通过（拒跑正式 B）：%s" % "; ".join(gate["reasons"]))
    return gate, {"validation_only": False, "max_states": None, "formal": True,
                  "workers": workers, "res_overrides": res_overrides}


def cmd_run(args) -> int:
    protocol_dict, ph = _protocol()
    from adapter import case_build, frontend, methods_loader, metrics_bridge, sources
    import analytic.scenes as scenes
    methods = _methods(args.methods)
    family = args.family
    extra = {"family": family, "methods": list(methods), "trials": args.trials,
             "limit": args.limit, "e3_optional": bool(args.include_e3_optional),
             "validation_only": bool(args.validation_only),
             "fdtd_validation": args.fdtd_validation}
    include_ids = None
    if getattr(args, "fdtd_include_clean_ids", None):
        import json as _json
        from pathlib import Path as _Path
        _doc = _json.loads(_Path(args.fdtd_include_clean_ids).read_text(encoding="utf-8"))
        include_ids = sorted(str(x) for x in _doc)
        if not include_ids:
            raise SystemExit("--fdtd-include-clean-ids 过滤列表为空")
        extra["fdtd_include_clean_ids"] = include_ids
    cfg_hash = _config_hash(protocol_dict, extra)
    runid = args.runid or sc.make_runid(cfg_hash)
    data_root = Path(args.data_root) if args.data_root else sc.default_data_root()
    sc.ensure_run_dir(data_root, runid, config_hash=cfg_hash, extra=extra)
    paths = sc.run_paths(data_root, runid)
    sc.atomic_write_json(paths["results"] / "SCHEMA.json", dict(SCHEMA, runid=runid, family=family))
    specs = scenes.specs_for(family, trials=args.trials,
                             include_e3_optional=args.include_e3_optional)
    if include_ids is not None:
        _n0 = len(specs)
        specs = [s for s in specs if str(s["clean_id"]) in set(include_ids)]
        print("[run] fdtd-include-clean-ids 子集过滤：%d -> %d specs（剔除场景另行披露）"
              % (_n0, len(specs)))
        if not specs:
            raise SystemExit("fdtd-include-clean-ids 过滤后 specs 为空")
    if args.limit is not None:
        specs = specs[: int(args.limit)]
    if args.validation_only:
        # validation-only：先把 fdtd spec 的 L_clean 改成 max_states（cache/noise 签名一致），
        # 保留 L_requested=L(128) 以便 row 如实标注 requested/actual。
        _ms = int(args.fdtd_max_states or 2)
        for _s in specs:
            if _s["source"] == "fdtd":
                _s["L_requested"] = int(_s["L"])
                _s["L_clean"] = min(int(_s["L_clean"]), _ms)
    _provenance_dump(paths, protocol_dict, cfg_hash, family, extra)
    print("[run] runid=%s family=%s methods=%s specs=%d data_root=%s"
          % (runid, family, methods, len(specs), data_root))

    if args.dryrun:
        plan = {"runid": runid, "family": family, "methods": methods, "n_specs": len(specs),
                "n_trials_total": sum(s["trials"] for s in specs),
                "n_method_runs": sum(s["trials"] for s in specs) * len(methods),
                "clean_ids": sorted({s["clean_id"] for s in specs}),
                "protocol_hash": ph, "config_hash": cfg_hash,
                "requires_fdtd_validation": family in B_FAMILIES,
                "note": "dryrun：未建 cache、未跑方法。"}
        sc.atomic_write_json(paths["results"] / ("dryrun_%s.json" % family), plan)
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        return 0

    # 先做拒覆检查（在任何 cache 构建/FDTD 之前），避免误重跑昂贵求解
    rows_path = paths["results"] / ("rows_%s.jsonl" % family)
    if rows_path.is_file() and not args.resume:
        raise SystemExit("rows 已存在且未给 --resume：拒绝覆盖 %s（避免误重跑；请用 --resume 续跑或换 --runid）"
                         % rows_path)
    gate, fdtd_opts = _fdtd_gate(args, family, specs, ph)
    if gate is not None:
        sc.atomic_write_json(paths["provenance"] / ("fdtd_gate_%s.json" % family), gate)
        print("[run] FDTD 门禁：%s" % json.dumps(gate, ensure_ascii=False)[:300])

    print("[run] building clean caches ...")
    clean_map, clean_metas, blocked, counts = _build_clean_all(
        specs, paths, ph, resume=args.resume, fdtd=fdtd_opts)
    if blocked:
        sc.atomic_write_json(paths["provenance"] / ("blocked_%s.json" % family), blocked)

    rows = _read_rows_strict(rows_path) if args.resume else []
    done = {(r["case_id"], int(r["trial"]), r["method"]) for r in rows}
    truth_by_case, skipped_blocked = {}, set(blocked)
    short_blocked = {}
    n_run = 0
    for spec in specs:
        if spec["clean_id"] in skipped_blocked or spec["clean_id"] in short_blocked:
            continue
        raw = np.asarray(sources.get_source(spec["source"])
                         .load_clean(paths["cache"], spec["clean_id"])["raw_cir"])
        truth_xy = [[float(x), float(y)] for x, y in spec["targets_xy"]]
        truth_by_case[spec["case_id"]] = {
            "clean_id": spec["clean_id"], "k": int(spec["k"]), "truth_xy_m": truth_xy,
            "resolution_distance_m": spec.get("resolution_distance_m"), "scene": spec["scene"],
            "snr_db": float(spec["snr_db"]), "L": int(spec["L"])}
        gf = _group_fields(spec)
        L_req = int(spec.get("L_requested", spec["L"]))
        allow_shorter = bool(args.validation_only and spec["source"] == "fdtd")
        for trial in range(int(spec["trials"])):
            todo = [m for m in methods if (spec["case_id"], trial, m) not in done]
            if not todo:
                continue
            seed = sc.trial_seed(spec["clean_id"], trial)
            try:
                X, L_actual, ldiag = sc.assemble_noisy_x(raw, L_req, spec["snr_db"], seed,
                                                         allow_shorter=allow_shorter,
                                                         validation_only=bool(args.validation_only))
            except sc.SimError as exc:
                short_blocked[spec["clean_id"]] = {
                    "clean_id": spec["clean_id"], "source": spec["source"], "reason": str(exc),
                    "L_requested": L_req, "L_available": int(np.asarray(raw).shape[0])}
                print("  [blocked] %s：%s" % (spec["clean_id"], exc))
                break
            gx, gy = sc.build_grid()
            case = case_build.build_case(
                case_id=spec["case_id"], clean_id=spec["clean_id"], X=X, k=spec["k"],
                grid_x=gx, grid_y=gy, layer=spec["layer"], source=spec["source"],
                scene=spec["scene"], tags=spec["tags"], snr_db=spec["snr_db"], trial=trial,
                seed=seed, L=L_actual, resolution_distance_m=spec["resolution_distance_m"])
            case_build.validate_case(case)
            for method in todo:
                params = methods_loader.params_for(method, protocol_dict) or None
                fp0 = case_build.case_fingerprint(case)
                t0 = time.perf_counter()
                res = methods_loader.call(method, case, params)
                est_s = time.perf_counter() - t0
                fp1 = case_build.case_fingerprint(case)
                preds = [[float(p["x"]), float(p["y"]), float(p["score"])] for p in res["predictions"]]
                metrics = metrics_bridge.evaluate_predictions(
                    truth_xy, [(p[0], p[1]) for p in preds],
                    resolution_distance_m=spec["resolution_distance_m"], status=res["status"])
                row = {"run_id": runid, "family": spec["family"], "layer": spec["layer"],
                       "source": spec["source"], "scene": spec["scene"],
                       "case_id": spec["case_id"], "clean_id": spec["clean_id"], "k": int(spec["k"]),
                       "snr_db": float(spec["snr_db"]), "trial": int(trial), "L": L_actual,
                       "L_requested": L_req, "L_actual": L_actual,
                       "L_available": int(ldiag["L_available"]), "formal": bool(ldiag["formal"]),
                       "method": method, "status": res["status"], "n_predictions": len(preds),
                       "tp": metrics["tp"], "fn": metrics["fn"], "fp": metrics["fp"],
                       "ospa": metrics["ospa"], "success": bool(metrics["success"]),
                       "estimate_s": float(est_s), "noise_seed": int(seed),
                       "spectrum_available": res.get("spectrum") is not None,
                       "case_fingerprint_before": fp0, "case_fingerprint_after": fp1,
                       "case_modified": bool(fp0 != fp1),
                       "truth_sha256": None,
                       "failures": (None if res["status"] == "ok"
                                    else str(res["diagnostics"].get("reason"))[:200]),
                       "metrics": metrics, "predictions": preds, **gf}
                rows.append(row)
                _save_sample(paths, row, res, spec, trial)
                n_run += 1
            if n_run and n_run % 100 == 0:
                _atomic_flush_rows(rows_path, rows)
                print("[run] %d method-runs done" % n_run)
    blocked.update(short_blocked)
    if short_blocked:
        sc.atomic_write_json(paths["provenance"] / ("blocked_short_%s.json" % family), short_blocked)
    truth_sha = _truth_file(paths, family, specs, truth_by_case)
    for r in rows:
        r["truth_sha256"] = truth_sha
    _atomic_flush_rows(rows_path, rows)

    planned_runs = sum(s["trials"] for s in specs) * len(methods)
    runnable_runs = sum(s["trials"] for s in specs if s["clean_id"] not in skipped_blocked) * len(methods)
    completed_runs = len(rows)
    counts.update({"planned_method_runs": planned_runs, "runnable_method_runs": runnable_runs,
                   "completed_method_runs": completed_runs,
                   "missing_method_runs": max(0, planned_runs - completed_runs),
                   "blocked_method_runs": max(0, planned_runs - runnable_runs),
                   "planned_note": "planned=全部 specs 的 method-runs（含 blocked，不缩分母）",
                   "n_blocked_clean": len(blocked)})
    if blocked or completed_runs < planned_runs:
        counts["completion_status"] = "incomplete"
    summary = {"runid": runid, "family": family, "methods": methods, "protocol_hash": ph,
               "config_hash": cfg_hash, "n_rows": len(rows), "counts": counts,
               "completion_status": counts["completion_status"],
               "fdtd_gate": gate, "blocked": blocked,
               "overall": metrics_bridge.aggregate(rows),
               "case_modified_any": any(r.get("case_modified") for r in rows),
               "truth_sha256": truth_sha, "by_method": {},
               "note": ("incomplete 只说明依赖/未完成，不等同算法失败；但主 summary 不代表 full。"
                        if counts["completion_status"] != "complete" else "complete")}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        if sub:
            summary["by_method"][m] = metrics_bridge.aggregate(sub)
    sc.atomic_write_json(paths["results"] / ("summary_%s.json" % family), summary)
    with (paths["results"] / ("rows_%s.csv" % family)).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_COLUMNS + ["predictions"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[run] rows=%d completion=%s -> %s" % (len(rows), counts["completion_status"], rows_path))
    print(json.dumps({"runid": runid, "n_rows": len(rows),
                      "completion_status": counts["completion_status"], "counts": counts},
                     ensure_ascii=False))
    return 0 if counts["completion_status"] == "complete" else 2


def _rep_cases(spec, paths, ph):
    """返回 (clean_case, noisy_case, ldiag)：ideal 诊断用 clean，SNR 诊断用 noisy。

    诊断档允许快照不足（allow_shorter=True），但噪声与数据**严格同形状**，不做 broadcast 复制快照。
    """
    from adapter import case_build, frontend, sources
    _build_clean_all([spec], paths, ph, resume=True, verbose=False)
    raw = np.asarray(sources.get_source(spec["source"])
                     .load_clean(paths["cache"], spec["clean_id"])["raw_cir"])
    L_req = int(spec.get("L_requested", spec["L"]))
    X_noisy, L_actual, ldiag = sc.assemble_noisy_x(
        raw, L_req, spec["snr_db"], sc.trial_seed(spec["clean_id"], 0), allow_shorter=True)
    cc = frontend.cropped_from_raw(np.asarray(raw)[:L_actual])
    gx, gy = sc.build_grid()
    X_clean = frontend.x_from_cropped(cc)
    mk = lambda X, tag: case_build.build_case(  # noqa: E731
        case_id="rep_%s_%s" % (spec["case_id"], tag), clean_id=spec["clean_id"], X=X,
        k=spec["k"], grid_x=gx, grid_y=gy, layer="A", source="analytic", scene="rep",
        tags=spec["tags"], snr_db=spec["snr_db"], trial=0, seed=0, L=L_actual)
    clean_case, noisy_case = mk(X_clean, "clean"), mk(X_noisy, "noisy")
    case_build.validate_case(clean_case)
    case_build.validate_case(noisy_case)
    return clean_case, noisy_case, ldiag


def cmd_e6(args) -> int:
    protocol_dict, ph = _protocol()
    from adapter import methods_loader, warmcold, frontend, sources
    import analytic.scenes as scenes
    methods = _methods(args.methods)
    spec = [s for s in scenes.e1_specs("A") if "S2" in s["case_id"] and s["snr_db"] == 10.0][0]
    extra = {"family": "E6", "methods": list(methods), "reps": int(args.reps)}
    cfg_hash = _config_hash(protocol_dict, extra)
    runid = args.runid or sc.make_runid(cfg_hash, stamp="e6")
    data_root = Path(args.data_root) if args.data_root else sc.default_data_root()
    sc.ensure_run_dir(data_root, runid, config_hash=cfg_hash, extra=extra)
    paths = sc.run_paths(data_root, runid)
    _provenance_dump(paths, protocol_dict, cfg_hash, "E6", extra)
    clean_case, noisy_case, ldiag = _rep_cases(spec, paths, ph)
    raw_rep = np.asarray(sources.get_source("analytic")
                         .load_clean(paths["cache"], spec["clean_id"])["raw_cir"][: int(spec["L"])])
    cc_rep = frontend.cropped_from_raw(raw_rep)
    t0 = time.perf_counter()
    for _ in range(5):
        frontend.x_from_cropped(cc_rep)
    frontend_s = (time.perf_counter() - t0) / 5.0
    timings = {}
    for m in methods:
        timings[m] = warmcold.timed_cold_warm(m, noisy_case,
                                             methods_loader.params_for(m, protocol_dict) or None,
                                             n_reps=int(args.reps))
    mem = {}
    tmp = paths["provenance"] / "_e6_case"
    sc.atomic_write_npz(tmp.with_suffix(".npz"), X=np.asarray(noisy_case["X"]))
    sc.atomic_write_json(tmp.with_suffix(".json"),
                         {"case_id": "E6", "clean_id": spec["clean_id"], "k": spec["k"],
                          "grid_x": [float(v) for v in noisy_case["grid_x"]],
                          "grid_y": [float(v) for v in noisy_case["grid_y"]],
                          "snr_db": spec["snr_db"], "L": int(spec["L"]), "scene": "E6"})
    for m in methods:
        params = methods_loader.params_for(m, protocol_dict)
        mem[m] = {mode: warmcold.measure_peak_memory(m, mode, tmp.with_suffix(".npz"),
                                                     tmp.with_suffix(".json"), params)
                  for mode in ("cold", "warm")}
    out = {"runid": runid, "protocol_hash": ph, "config_hash": cfg_hash, "L_diag": ldiag,
           "frontend_crop_fft_s": frontend_s, "timing": timings, "peak_memory": mem,
           "warm_supported_methods": [m for m in methods if warmcold.warm_support(m)["supported"]],
           "warm_not_applicable": {m: warmcold.warm_support(m)["reason"] for m in methods
                                   if not warmcold.warm_support(m)["supported"]},
           "timing_report": ("cold/warm median+p95；逐次 status 落盘；失败不静默过滤；"
                             "峰值内存为隔离子进程 RSS；前端单独计时；E6 不宣称 5 法都有 warm"),
           **sc.provenance_block()}
    sc.atomic_write_json(paths["results"] / "e6_timing.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
    return 0


def cmd_verify(args) -> int:
    protocol_dict, ph = _protocol()
    from adapter import e0_checks, methods_loader
    import analytic.scenes as scenes
    family = args.family or "E1_A"
    spec = scenes.specs_for(family, trials=args.trials)[0]
    extra = {"family": "VERIFY", "verify_family": family, "methods": list(methods_loader.MAIN_METHODS),
             "trials": args.trials}
    cfg_hash = _config_hash(protocol_dict, extra)
    runid = args.runid or sc.make_runid(cfg_hash, stamp="verify")
    data_root = Path(args.data_root) if args.data_root else sc.default_data_root()
    sc.ensure_run_dir(data_root, runid, config_hash=cfg_hash, extra=extra)
    paths = sc.run_paths(data_root, runid)
    _provenance_dump(paths, protocol_dict, cfg_hash, "VERIFY", extra)
    clean_case, noisy_case, ldiag = _rep_cases(spec, paths, ph)
    report = e0_checks.run_all(noisy_case, spec["targets_xy"], spec=spec, clean_case=clean_case)
    report["clean_vs_noisy"] = {
        "note": "ideal_locality 用 clean X（无噪，理想/有限窗诊断）；numerical_consistency 用 noisy（SNR 诊断）。",
        "L_diag": ldiag,
        "clean_case_fingerprint": __import__("adapter.case_build", fromlist=["x"]).case_fingerprint(clean_case),
        "noisy_case_fingerprint": __import__("adapter.case_build", fromlist=["x"]).case_fingerprint(noisy_case)}
    sc.atomic_write_json(paths["results"] / "e0_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:5000])
    nc = report["numerical_consistency"]
    ok = (nc.get("this_method_status") == "ok" and nc.get("b0_status") == "ok"
          and nc.get("spectrum_cosine", 0.0) >= 1.0 - 1e-9)
    return 0 if ok else 3


def cmd_protocol(args) -> int:
    protocol_dict, ph = _protocol()
    out_dir = PKG / "protocol"
    sc.atomic_write_json(out_dir / "sim_protocol.json", protocol_dict)
    sha = sc.sha256_file(out_dir / "sim_protocol.json")
    (out_dir / "sim_protocol.sha256").write_text("%s  sim_protocol.json%s" % (sha, chr(10)),
                                                 encoding="utf-8")
    print(json.dumps({"protocol_hash": ph, "json_sha256": sha,
                      "path": str(out_dir / "sim_protocol.json")}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_experiments",
                                 description="仿真实验统一 CLI（plan/run/e6/verify/protocol）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "e6", "verify", "protocol"):
        p = sub.add_parser(name)
        p.add_argument("--family", default="E1_A", choices=list(FAMILIES))
        p.add_argument("--methods", default=None)
        p.add_argument("--trials", type=int, default=None)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--data-root", dest="data_root", default=None)
        p.add_argument("--runid", default=None)
        p.add_argument("--include-e3-optional", dest="include_e3_optional", action="store_true")
        if name == "run":
            p.add_argument("--dryrun", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--fdtd-validation", dest="fdtd_validation", default=None,
                           help="层 B 正式 run 必填：fdtd convergence 报告 JSON 路径")
            p.add_argument("--validation-only", dest="validation_only", action="store_true",
                           help="tiny smoke（非正式；不得作 main-run 证据）")
            p.add_argument("--fdtd-max-states", dest="fdtd_max_states", type=int, default=2)
            p.add_argument("--fdtd-workers", dest="fdtd_workers", type=int, default=1,
                           help="层 B 状态求解并行 worker 数（>1 = spawn 多进程分片，每进程一个"
                                " Meep 实例；仅调度并行，不影响数值与 config_hash）")
            p.add_argument("--fdtd-scene-res-overrides", dest="fdtd_scene_res_overrides",
                           default=None,
                           help="层 B 场景级分辨率覆盖 JSON：{clean_id: res}。网格收敛逐场景选口径；"
                                "进 cache key 与门禁 want_res，不改协议冻结文件")
            p.add_argument("--fdtd-include-clean-ids", dest="fdtd_include_clean_ids",
                           default=None,
                           help="层 B 场景白名单 JSON：[clean_id,...]（网格收敛门禁子集路线；"
                                "未通过场景剔除并另行披露）。进 config_hash（run 身份含子集事实）")
        if name == "e6":
            p.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "e6": cmd_e6, "verify": cmd_verify,
            "protocol": cmd_protocol}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
