# -*- coding: utf-8 -*-
"""Q4 统一 CLI：inventory / export / compare / demo。

示例::

    py=.venv/Scripts/python.exe
    & $py -B 大修执行_20260923/06_仿真实验_20260926/q4/q4_cli.py inventory \
        --out 大修执行_20260923/06_仿真实验_20260926/q4/manifest
    & $py -B .../q4/q4_cli.py export --source cache --out .../q4/out/hw_cir
    & $py -B .../q4/q4_cli.py export --source bin   --out .../q4/out/hw_cir_raw
    & $py -B .../q4/q4_cli.py compare --sim-cir sim.npz --hw-cir hw/indoor_G-370_N1_S1_001.npz \
        --sim-pred sim_pred.json --hw-pred hw_pred.json --sim-truth sim_truth.json --hw-truth hw_truth.json \
        --out compare.json
    & $py -B .../q4/q4_cli.py demo
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from q4 import (build_manifest, demo_synthetic, eval_clean, eval_plan, export_from_bin,  # noqa: E402
                export_from_cache, export_scene_specs, generate, plan_batch, run_batch,
                run_compare, write_manifest)
from q4.q4_batch import geometry_index  # noqa: E402
from q4.q4_scenes import Q4_BACKGROUND_MODE  # noqa: E402
from q4.q4_common import (DEFAULT_CACHE_DIR, DEFAULT_CIR_OFFSET,  # noqa: E402
                          DEFAULT_DATA_DIR, DEFAULT_TRUTH_INDEX, Q4Error, write_json)
from q4.q4_export import DEFAULT_B0_DIR  # noqa: E402
from q4.q4_batch import (DEFAULT_HW_CIR_DIR, DEFAULT_HW_RESULTS, DEFAULT_INVENTORY,  # noqa: E402
                         DEFAULT_METHOD)


def _cmd_inventory(args) -> int:
    manifest = build_manifest(data_dir=args.data_dir)
    written = write_manifest(manifest, args.out)
    print(json.dumps({"n_geometries": manifest["n_geometries"],
                      "n_recordings_total": manifest["n_recordings_total"],
                      "n_recordings_valid": manifest["n_recordings_valid"],
                      "n_recordings_invalid": manifest["n_recordings_invalid"],
                      "valid_recordings_by_k": manifest["valid_recordings_by_k"],
                      "excluded_invalid": manifest["excluded_invalid"],
                      "written": written}, ensure_ascii=False, indent=2))
    return 0


def _cmd_export(args) -> int:
    if args.source == "cache":
        doc = export_from_cache(args.cache_dir, args.out, truth_index=args.truth_index,
                                recordings=args.recordings, force=args.force)
    else:
        doc = export_from_bin(args.data_dir, args.out, b0_dir=args.b0_dir, cache_dir=args.cache_dir,
                              truth_index=args.truth_index, recordings=args.recordings, force=args.force)
    print(json.dumps({"source": doc["source"], "n_recordings": doc["n_recordings"],
                      "out": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


def _cmd_compare(args) -> int:
    doc = run_compare(sim_cir_path=args.sim_cir, hw_cir_path=args.hw_cir,
                      sim_pred_path=args.sim_pred, hw_pred_path=args.hw_pred,
                      sim_truth_path=args.sim_truth, hw_truth_path=args.hw_truth,
                      windows=args.windows.split(",") if args.windows else None,
                      out_path=args.out, cir_offset=args.cir_offset)
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=2))
    return 0


def _cmd_demo(args) -> int:
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="q4_demo_"))
    doc = demo_synthetic(out)
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=2))
    print("demo output: %s" % (Path(out) / "compare.json"))
    return 0


def _batch_kwargs(args) -> dict:
    return {"sim_dir": args.sim_dir, "hw_dir": args.hw_dir, "truth_index": args.truth_index,
            "results_dir": args.results_dir, "method": args.method,
            "allow_truth_fallback": args.allow_truth_fallback}


def _cmd_batch_plan(args) -> int:
    plan = plan_batch(**_batch_kwargs(args))
    if args.out:
        write_json(args.out, plan)
    print(json.dumps({"n_recordings": plan["n_recordings"],
                      "n_geometries": plan["n_geometries"],
                      "n_windows_total": plan["n_windows_total"],
                      "n_windows_available": plan["n_windows_available"],
                      "n_windows_missing": plan["n_windows_missing"],
                      "hw_prediction_error": plan["hw_prediction_error"],
                      "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def _cmd_batch_run(args) -> int:
    if not args.sim_dir:
        raise Q4Error("batch-run 需要 --sim-dir（仿真 CIR/预测/真值目录）")
    doc = run_batch(out_dir=args.out, **_batch_kwargs(args))
    print(json.dumps({"n_windows_planned": doc["n_windows_planned"],
                      "n_windows_compared": doc["n_windows_compared"],
                      "n_windows_missing": doc["n_windows_missing"],
                      "completion_status": doc["completion_status"],
                      "summary": doc["summary"], "out": args.out}, ensure_ascii=False, indent=2))
    # 有 missing（未全量可比较）-> 非零；不把 incomplete 当 success
    return 0 if doc["completion_status"] == "complete" else 1


def _cmd_scenes_export(args) -> int:
    doc = export_scene_specs(inventory_json=args.inventory, out_dir=args.out,
                             background_mode=args.background_mode)
    print(json.dumps({"n_geometries": doc["n_geometries"], "background_mode": doc["background_mode"],
                      "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def _cmd_generate(args) -> int:
    plan = generate(scene_dir=args.scene_dir, out_dir=args.out, run=args.run,
                    background_mode=args.background_mode, evidence_dir=args.evidence_dir,
                    workers=int(getattr(args, "fdtd_workers", 1) or 1))
    print(json.dumps({"mode": plan["mode"], "n_scenes": plan["n_scenes"],
                      "background_mode": plan["background_mode"], "out": args.out},
                     ensure_ascii=False, indent=2))
    if not args.run:
        print("plan_only：只做 fdtd.api.plan_run（无 Meep）；如需真跑请显式 --run")
    return 0


def _cmd_eval_plan(args) -> int:
    doc = eval_plan(inventory_json=args.inventory)
    if args.out:
        write_json(args.out, doc)
    print(json.dumps({"n_geometries": doc["n_geometries"], "deps_ready": doc["deps"]["ready"],
                      "deps": doc["deps"], "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def _cmd_eval_clean(args) -> int:
    idx = geometry_index(args.inventory)
    gid = args.geometry_id
    if gid not in idx["geometries"]:
        raise Q4Error("geometry_id=%r 不在 inventory（可用 %s）" % (gid, sorted(idx["geometries"])[:5]))
    geom = idx["geometries"][gid]
    block = dict(geom.get("fdtd_geometry") or {})
    block["truth_xy"] = geom.get("truth_xy")
    doc = eval_clean(clean_dir=args.clean_dir, out_dir=args.out, geometry_block=block,
                     geometry_id=gid, k=geom.get("n_targets"))
    print(json.dumps({"scene_id": doc["scene_id"], "n_methods": doc["n_methods"],
                      "status": doc["status"], "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Q4 清单 / 可移植 CIR 包 / 跨域比较")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory", help="15 几何 / 32 有效录制 manifest")
    pi.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    pi.add_argument("--out", default="大修执行_20260923/06_仿真实验_20260926/q4/manifest")
    pi.set_defaults(func=_cmd_inventory)

    pe = sub.add_parser("export", help="导出可移植硬件 CIR 包")
    pe.add_argument("--source", choices=("cache", "bin"), default="cache")
    pe.add_argument("--out", required=True)
    pe.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    pe.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    pe.add_argument("--b0-dir", default=DEFAULT_B0_DIR)
    pe.add_argument("--truth-index", default=DEFAULT_TRUTH_INDEX)
    pe.add_argument("--recordings", nargs="*", default=None)
    pe.add_argument("--force", action="store_true", help="允许用不同 source 覆盖已有输出包")
    pe.set_defaults(func=_cmd_export)

    pc = sub.add_parser("compare", help="跨域 CIR 相似度 + 身份 peak delta")
    pc.add_argument("--sim-cir", required=True)
    pc.add_argument("--hw-cir", required=True)
    pc.add_argument("--sim-pred", required=True)
    pc.add_argument("--hw-pred", required=True)
    pc.add_argument("--sim-truth", required=True)
    pc.add_argument("--hw-truth", required=True)
    pc.add_argument("--windows", default=None, help="逗号分隔 window_id；默认取 sim/hw 预测交集")
    pc.add_argument("--cir-offset", type=int, default=DEFAULT_CIR_OFFSET)
    pc.add_argument("--out", default=None)
    pc.set_defaults(func=_cmd_compare)

    pd = sub.add_parser("demo", help="极简合成自测（test 非正式结果）")
    pd.add_argument("--out", default=None)
    pd.set_defaults(func=_cmd_demo)

    bp = sub.add_parser("batch-plan", help="Q4 全清单缺口检查（不计算、不造假）")
    bp.add_argument("--sim-dir", default=None)
    bp.add_argument("--hw-dir", default=DEFAULT_HW_CIR_DIR)
    bp.add_argument("--truth-index", default=DEFAULT_TRUTH_INDEX)
    bp.add_argument("--results-dir", default=DEFAULT_HW_RESULTS)
    bp.add_argument("--method", default=DEFAULT_METHOD)
    bp.add_argument("--allow-truth-fallback", action="store_true")
    bp.add_argument("--out", default=None)
    bp.set_defaults(func=_cmd_batch_plan)

    br = sub.add_parser("batch-run", help="Q4 全清单批量比较（缺失窗单列）")
    br.add_argument("--sim-dir", required=True)
    br.add_argument("--hw-dir", default=DEFAULT_HW_CIR_DIR)
    br.add_argument("--truth-index", default=DEFAULT_TRUTH_INDEX)
    br.add_argument("--results-dir", default=DEFAULT_HW_RESULTS)
    br.add_argument("--method", default=DEFAULT_METHOD)
    br.add_argument("--allow-truth-fallback", action="store_true")
    br.add_argument("--out", required=True)
    br.set_defaults(func=_cmd_batch_run)

    se = sub.add_parser("scenes-export", help="由 15 几何 inventory 导出 FDTD 场景 JSON（不跑）")
    se.add_argument("--inventory", default=DEFAULT_INVENTORY)
    se.add_argument("--background-mode", default=Q4_BACKGROUND_MODE)
    se.add_argument("--out", required=True)
    se.set_defaults(func=_cmd_scenes_export)

    ge = sub.add_parser("generate", help="fdtd.api.plan_run 校验 15 scene（默认不跑 Meep）")
    ge.add_argument("--scene-dir", required=True)
    ge.add_argument("--out", required=True)
    ge.add_argument("--background-mode", default=Q4_BACKGROUND_MODE)
    ge.add_argument("--evidence-dir", default=None, help="正式前检证据目录（convergence.check_evidence；--run 时必需且须 passed）")
    ge.add_argument("--run", action="store_true", help="显式真跑 fdtd.api.generate_clean（Meep）")
    ge.add_argument("--fdtd-workers", dest="fdtd_workers", type=int, default=1,
                    help=">1 时 generate_clean 内 spawn 多进程并行预填状态缓存（每进程一个 Meep 实例）")
    ge.set_defaults(func=_cmd_generate)

    ep = sub.add_parser("eval-plan", help="Q4 五法评估依赖检查（显式 adapter + PROTOCOL）")
    ep.add_argument("--inventory", default=DEFAULT_INVENTORY)
    ep.add_argument("--out", default=None)
    ep.set_defaults(func=_cmd_eval_plan)

    ec = sub.add_parser("eval-clean", help="clean_cir -> 五法 -> <geom>/<method>.pred.json + truth.json")
    ec.add_argument("--clean-dir", required=True)
    ec.add_argument("--out", required=True)
    ec.add_argument("--geometry-id", required=True)
    ec.add_argument("--inventory", default=DEFAULT_INVENTORY)
    ec.set_defaults(func=_cmd_eval_clean)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Q4Error as exc:
        print("Q4Error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
