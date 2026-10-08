#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fdtd CLI：plan / generate / convergence / smoke。

纪律：
* --help 与 plan 都**不导入 Meep**（求解只在 generate/smoke 真正开始时惰性导入）；
* convergence 的比较判据**先写盘再跑**，允许结论为 not_passed（不修改判据）；
* smoke 明确 physical_validation=false（低分辨率/小域只证明 API 与链路）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import api, protocol, scenes as scenes_mod
from .protocol import atomic_write_json, read_json

DEFAULT_CONVERGENCE_CRITERIA = {
    "max_peak_tap_shift": 1,
    "min_abs_correlation": 0.90,
    "max_rel_l2": 0.30,
    "_note": "判据在运行前写入本文件（criteria_frozen_before_run=true）；不满足即 not_passed，不得改写",
}


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _progress(msg: str) -> None:
    print("[fdtd] %s" % msg, flush=True)


def _load_protocol(path: str | None):
    return None if not path else read_json(path)


def cmd_plan(args) -> int:
    plan = api.plan_run(_load_protocol(args.protocol), args.scene, max_states=args.max_states,
                        background_mode=args.background_mode)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "plan.json", plan)
        _progress("plan 写入 %s" % (out / "plan.json"))
    _print({k: plan[k] for k in ("scene_id", "state_count", "cache_key", "covered_range_m",
                                 "max_two_way_path_m", "cost", "channel_names")})
    _print({"geometry_rules": plan["geometry_rules"], "protocol_summary": {
        "fc_hz": plan["protocol"]["fc_hz"], "bandwidth_hz": plan["protocol"]["bandwidth_hz"],
        "sample_interval_s": plan["protocol"]["sample_interval_s"], "raw_taps": plan["protocol"]["raw_taps"],
        "crop_start": plan["protocol"]["crop_start"], "fft_length": plan["protocol"]["fft_length"],
        "snapshots": plan["protocol"]["snapshots"], "background_mode": plan["protocol"]["background_mode"],
        "domain_size_m": plan["protocol"]["domain"]["size_m"], "resolution_px_per_m": plan["protocol"]["meep"]["resolution_px_per_m"]}})
    return 0


def cmd_generate(args) -> int:
    res = api.generate_clean(_load_protocol(args.protocol), args.scene, args.out,
                             cache_root=args.cache_root, resume=not args.no_resume,
                             max_states=args.max_states, background_mode=args.background_mode,
                             dry_run=args.dry_run, diagnostics_taps=args.diagnostics_taps,
                             block_states=args.block_states, workers=int(args.workers),
                             progress=(None if args.quiet else _progress))
    _print({k: res[k] for k in ("outdir", "raw_cir_shape", "cache_key", "cache_dir", "background_mode")
            if k in res})
    if "stats" in res:
        _print({"stats": res["stats"]})
    return 0


def _compare_cir(a: np.ndarray, b: np.ndarray) -> dict:
    """两条 raw_cir 的定量差异（形状必须一致）。"""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        raise ValueError("比较对象形状不一致：%s vs %s" % (a.shape, b.shape))
    va = a.reshape(-1)
    vb = b.reshape(-1)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    corr = float(abs(np.vdot(va, vb)) / (na * nb)) if na > 0 and nb > 0 else None
    rel_l2 = float(np.linalg.norm(va - vb) / na) if na > 0 else None
    pa = np.abs(a).argmax(axis=-1)
    pb = np.abs(b).argmax(axis=-1)
    shift = np.abs(pa - pb)
    return {"abs_correlation": corr, "rel_l2": rel_l2, "max_peak_tap_shift": int(shift.max()),
            "mean_peak_tap_shift": float(shift.mean()),
            "peak_abs_a": float(np.abs(a).max()), "peak_abs_b": float(np.abs(b).max())}


def cmd_convergence(args) -> int:
    """受控前检（委托 convergence 模块）：只变 res/runtime；判据冻结；not_passed -> 非零退出码。"""
    from . import convergence as conv

    if getattr(args, "dryrun", False):
        plan = conv.dryrun_plan(_load_protocol(args.protocol), args.scene, args.out, args.res, args.run_time,
                                args.max_states, args.background_mode, args.criteria)
        _print({"dryrun": True, "n_runs": plan["n_runs"], "controlled_hash": plan["controlled_hash"],
                "plan": str(Path(args.out) / "convergence_plan.json"), "criteria": plan["criteria"]})
        return 0
    report, code = conv.run_convergence(_load_protocol(args.protocol), args.scene, args.out, args.res,
                                       args.run_time, args.max_states, args.background_mode, args.criteria,
                                       quiet=bool(getattr(args, "quiet", False)))
    _print({"verdict": report["verdict"], "n_comparisons": len(report["comparisons"]),
            "report": str(Path(args.out) / "convergence.json")})
    return int(code)


def cmd_gate(args) -> int:
    """runner B 闸门：收敛证据必须“适用且 passed”，否则非零退出（禁止无前检开跑 128/B 档）。"""
    from . import convergence as conv

    res = conv.check_evidence(args.evidence, scene=args.scene, protocol_path=_load_protocol(args.protocol))
    _print(res)
    return 0 if (res.get("applicable") and res.get("passed")) else 3


def _legacy_convergence_unused(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    criteria = dict(DEFAULT_CONVERGENCE_CRITERIA)
    if args.criteria:
        criteria = dict(read_json(args.criteria)) if Path(args.criteria).is_file() else json.loads(args.criteria)
    criteria["criteria_frozen_before_run"] = True
    atomic_write_json(out / "criteria.json", criteria)
    _progress("判据先冻结写入 %s" % (out / "criteria.json"))
    res_list = [int(x) for x in args.res]
    rt_list = [float(x) for x in args.run_time]
    if getattr(args, "dryrun", False):
        items = []
        for res in res_list:
            for rt in rt_list:
                scene = dict(scenes_mod.load_scene(args.scene))
                scene["meep"] = dict(scene["meep"])
                scene["meep"]["resolution_px_per_m"] = res
                scene["meep"]["run_time_meep"] = rt
                scene.pop("domain", None)
                tag = "res%d_rt%g" % (res, rt)
                scene_path = out / ("scene_%s.json" % tag)
                atomic_write_json(scene_path, scene)
                pl = api.plan_run(_load_protocol(args.protocol), str(scene_path),
                                  max_states=args.max_states, background_mode=args.background_mode)
                items.append({"tag": tag, "resolution_px_per_m": res, "run_time_meep": rt,
                              "scene_file": str(scene_path), "cost": pl["cost"],
                              "delay_reference_s": 2.0 * 3.0 / 2.99792458e8,
                              "delay_note": "3 m 双程约 20 ns；run_time_meep 需覆盖 20 ns + 脉冲 4 ns"})
        plan = {"scene": args.scene, "criteria": criteria,
                "baseline": "res%d_rt%g" % (res_list[0], rt_list[0]),
                "items": items, "n_runs": len(items), "dryrun": True,
                "note": "本文件只是 250/350 与时长对照清单与预算；正式运行不在本轮",
                "generated_at": api._now_iso()}
        atomic_write_json(out / "convergence_plan.json", plan)
        _print({"dryrun": True, "n_runs": len(items),
                "plan": str(out / "convergence_plan.json"), "criteria": criteria})
        return 0
    runs = {}
    pairs = []
    for res in res_list:
        for rt in rt_list:
            tag = "res%d_rt%g" % (res, rt)
            scene = scenes_mod.load_scene(args.scene)
            scene = json.loads(json.dumps(scene)) if isinstance(scene, dict) else scene
            scene["meep"]["resolution_px_per_m"] = res
            scene["meep"]["run_time_meep"] = rt
            scene_path = out / ("scene_%s.json" % tag)
            scene.pop("domain", None)
            atomic_write_json(scene_path, scene)
            run_dir = out / tag
            api.generate_clean(_load_protocol(args.protocol), scene_path, run_dir, resume=True,
                               max_states=args.max_states, background_mode=args.background_mode,
                               progress=(None if args.quiet else _progress))
            with np.load(run_dir / "clean_cir.npz", allow_pickle=False) as npz:
                runs[tag] = npz["raw_cir"]
    base = "res%d_rt%g" % (res_list[0], rt_list[0])
    results = {}
    for tag, cir in runs.items():
        if tag == base:
            continue
        diff = _compare_cir(runs[base], cir)
        ok = ((diff["max_peak_tap_shift"] <= criteria["max_peak_tap_shift"])
              and (diff["abs_correlation"] is not None and diff["abs_correlation"] >= criteria["min_abs_correlation"])
              and (diff["rel_l2"] is not None and diff["rel_l2"] <= criteria["max_rel_l2"]))
        results["%s_vs_%s" % (tag, base)] = {"diff": diff, "passed": bool(ok)}
        pairs.append(ok)
    verdict = "passed" if pairs and all(pairs) else "not_passed"
    report = {"scene": args.scene, "criteria": criteria, "baseline": base,
              "comparisons": results, "verdict": verdict,
              "note": "本命令为收敛/时长前检入口；verdict=not_passed 必须如实报告，不得改判据",
              "generated_at": api._now_iso()}
    atomic_write_json(out / "convergence.json", report)
    _print({"verdict": verdict, "n_comparisons": len(results), "report": str(out / "convergence.json")})
    return 0


def cmd_smoke(args) -> int:
    """tiny smoke：API/链路/续跑/多目标同场（不是物理验证）。"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene_id = args.scene or "smoke_tiny_2target"
    checks = {}
    plan = api.plan_run(_load_protocol(args.protocol), scene_id, max_states=args.max_states,
                        background_mode=args.background_mode)
    checks["plan_k_targets"] = int(plan["n_targets"])
    checks["plan_ok"] = True
    r1 = api.generate_clean(_load_protocol(args.protocol), scene_id, out / "run1",
                            max_states=args.max_states, background_mode=args.background_mode,
                            progress=(None if args.quiet else _progress))
    with np.load(Path(r1["clean_npz"]), allow_pickle=False) as npz:
        cir = npz["raw_cir"]
        keys = list(npz.files)
        t_origin = float(npz["t_origin_s"])
        bg_mode = str(npz["background_mode"])
    checks["raw_cir_shape"] = list(cir.shape)
    checks["raw_cir_finite"] = bool(np.all(np.isfinite(cir)))
    checks["raw_cir_nonzero"] = bool(np.abs(cir).max() > 0)
    checks["clean_case_has_no_truth_keys"] = True
    protocol.assert_clean_case(keys)
    checks["t_origin_s"] = t_origin
    checks["background_mode"] = bg_mode
    checks["solves_actual_run1"] = int(r1["stats"]["solves_actual"])
    r2 = api.generate_clean(_load_protocol(args.protocol), scene_id, out / "run1",
                            max_states=args.max_states, background_mode=args.background_mode,
                            progress=(None if args.quiet else _progress))
    checks["solves_actual_run2_resume"] = int(r2["stats"]["solves_actual"])
    checks["cache_hits_run2"] = r2["stats"]["cache_hits"]
    checks["resume_skips_solves"] = bool(int(r2["stats"]["solves_actual"]) == 0)
    outdir_conflict_refused = False
    try:
        api.generate_clean(_load_protocol(args.protocol), "main_1tx4rx_2target", out / "run1",
                           max_states=args.max_states, background_mode=args.background_mode, progress=None)
    except Exception as exc:  # 预期：不同 cache key 被开头拒绝
        outdir_conflict_refused = "cache key" in str(exc)
    checks["outdir_conflict_refused"] = bool(outdir_conflict_refused)
    report = {"scene": scene_id, "physical_validation": False,
              "note": "tiny/低分辨率小域只证明 API 与链路；本文件不构成物理或性能结论",
              "checks": checks, "generated_at": api._now_iso(),
              "pass": bool(checks["raw_cir_finite"] and checks["raw_cir_nonzero"]
                           and checks["resume_skips_solves"] and checks["outdir_conflict_refused"]
                           and int(checks["plan_k_targets"]) >= 2)}
    atomic_write_json(out / "smoke_report.json", report)
    _print(report)
    return 0 if report["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fdtd", description="层 B FDTD 生成器（plan/generate/convergence/smoke）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--protocol", default=None, help="协议 JSON 路径（可覆盖默认物理参数）")
        sp.add_argument("--scene", default=None, help="场景 JSON 路径或内建 scene_id")
        sp.add_argument("--out", default=None, help="输出目录")
        sp.add_argument("--max-states", type=int, default=None, help="只跑前 N 个状态（smoke 用）")
        sp.add_argument("--background-mode", default=None, choices=["empty_field", "ema_windowed"])
        sp.add_argument("--quiet", action="store_true")

    sp = sub.add_parser("plan", help="只出清单与预算（不导入 Meep）")
    common(sp); sp.add_argument("--measured-updates-per-s", type=float, default=None)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("generate", help="生成 clean case（真求解，需要 Meep）")
    common(sp)
    sp.add_argument("--cache-root", default=None)
    sp.add_argument("--no-resume", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--diagnostics-taps", type=int, default=64)
    sp.add_argument("--block-states", type=int, default=8)
    sp.add_argument("--workers", type=int, default=1,
                    help=">1 时 spawn 多进程并行预填状态缓存（每进程一个 Meep 实例；"
                         "受核数与峰值内存约束；数值路径与串行一致）")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("convergence", help="分辨率/时长前检（判据先冻结，允许 not_passed）")
    common(sp)
    sp.add_argument("--res", type=int, nargs="+", default=[250, 350])
    sp.add_argument("--run-time", type=float, nargs="+", default=[20.0, 40.0])
    sp.add_argument("--criteria", default=None, help="判据 JSON 路径或内联 JSON")
    sp.add_argument("--dryrun", "--dry-run", dest="dryrun", action="store_true",
                    help="只生成 250/350 与时长对照清单与预算")
    sp.set_defaults(func=cmd_convergence)

    sp = sub.add_parser("gate", help="runner B 闸门：检查收敛证据是否适用且 passed")
    common(sp)
    sp.add_argument("--evidence", required=True, help="convergence 证据目录（含 convergence.json/plan）")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("smoke", help="tiny smoke（API/链路/续跑；非物理验证）")
    common(sp)
    sp.set_defaults(func=cmd_smoke)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "out", None) is None and args.cmd != "plan":
        args.out = str(Path.cwd() / ("fdtd_out_%s" % args.cmd))
    if getattr(args, "scene", None) is None:
        args.scene = "smoke_tiny_2target" if args.cmd in ("smoke",) else "main_1tx4rx_2target"
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
