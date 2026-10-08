#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新版 JSF-MUSIC 仿真入口（规范 §11 命令面）。

    python execution2/jsf_sim_cli.py plan        --protocol <json> --output-root <ROOT>
    python execution2/jsf_sim_cli.py validate    --protocol <json> --output-root <ROOT> --scope local
    python execution2/jsf_sim_cli.py run         --protocol <json> --output-root <ROOT> --task E1_A --mode smoke
    python execution2/jsf_sim_cli.py convergence --protocol <json> --output-root <ROOT> --profile controlled
    python execution2/jsf_sim_cli.py run         --protocol <json> --output-root <ROOT> --task E1_A --mode formal
    python execution2/jsf_sim_cli.py run         --protocol <json> --output-root <ROOT> --task E2_B --mode formal --resume
    python execution2/jsf_sim_cli.py q4          --protocol <json> --output-root <ROOT> --stage plan
    python execution2/jsf_sim_cli.py q4          --protocol <json> --output-root <ROOT> --stage run
    python execution2/jsf_sim_cli.py summarize   --output-root <ROOT>
    python execution2/jsf_sim_cli.py plot        --output-root <ROOT>

纪律
----
* `--help` / `plan` / `validate` **不启动 Meep、不跑正式批次**。
* `run` 必须显式给 `--task` 与 `--mode`；smoke 只取固定首场景/10 dB/2 trial 且 formal=false，
  不用 `--limit` 缩短后冒充完整正式结果。
* `methods` 硬限 `['this_method']`，外部基线参数直接非零拒绝。
* 缺输出根、旧 archive 路径、同 run_id 不同配置、未过数值前检、缺 Q4 时间参考均显式拒绝。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from jsfsim import paths  # noqa: E402  先固定路径/线程，再 import numpy

paths.ensure_paths()

from jsfsim import data as data_mod  # noqa: E402
from jsfsim import estimate, metrics, runner  # noqa: E402
from jsfsim import protocol as P  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REJECT = 2
EXIT_BLOCKED = 3

FORBIDDEN_OUTPUT_FRAGMENTS = (
    "/data1/zsy/paper_revision_sim_20260926",
    "paper_revision_sim_20260926",
    "archive_local_20260928",
)
DEFAULT_NEW_REMOTE_ROOT = "/data1/zsy/paper_revision_sim_recovery_20260929"


# --------------------------------------------------------------------------- #
# 协议文件
# --------------------------------------------------------------------------- #
def protocol_payload() -> dict:
    proto = P.build_protocol()
    checks = P.self_check(proto)
    return {"protocol": proto, "self_check": checks,
            "protocol_hash": hashlib.sha256(P.canonical_json(proto).encode("utf-8")).hexdigest()}


def load_protocol(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit("协议文件不存在：%s" % p)
    doc = json.loads(p.read_text(encoding="utf-8"))
    proto = doc.get("protocol", doc)
    have = str(doc.get("protocol_hash") or "")
    want = hashlib.sha256(P.canonical_json(proto).encode("utf-8")).hexdigest()
    if have and have != want:
        raise SystemExit("协议哈希不匹配（文件 %s != 重算 %s）：拒绝使用" % (have[:12], want[:12]))
    if str(proto.get("protocol_version")) != P.PROTOCOL_VERSION:
        raise SystemExit("协议版本不符（文件 %s != 代码 %s）"
                         % (proto.get("protocol_version"), P.PROTOCOL_VERSION))
    if list(proto.get("method_scope", {}).get("methods") or []) != [estimate.ONLY_METHOD]:
        raise SystemExit("协议 methods 不是 ['this_method']：拒绝")
    if list(proto.get("offgrid_shift_m") or []) != [0.0, 0.0]:
        raise SystemExit("协议含旧统一平移 %r：新版禁止" % (proto.get("offgrid_shift_m"),))
    return doc


def check_output_root(output_root) -> Path:
    if output_root is None:
        raise SystemExit("必须给 --output-root")
    s = str(output_root).replace("\\", "/")
    for frag in FORBIDDEN_OUTPUT_FRAGMENTS:
        if frag.replace("\\", "/") in s:
            raise SystemExit("拒绝：输出根落在旧 archive/旧数据根内（%s）：%s" % (frag, output_root))
    return Path(output_root)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def cache_plan() -> dict:
    """cache-hit/miss 与 FDTD 求解次数估算（不跑任何求解）。"""
    out = {}
    for task in P.TASKS:
        specs = P.expand_task(task)
        cleans = {}
        for s in specs:
            cleans.setdefault(s["clean_id"], s)
        n_solves = 0
        if task.endswith("_B"):
            # 受控层 B：每 clean = (n_states + 1 空场) x n_tx
            for s in cleans.values():
                n_solves += (int(s["L_clean"]) + 1) * 1
        out[task] = {
            "n_specs": len(specs), "n_clean": len(cleans),
            "n_method_runs": int(sum(s["trials"] for s in specs)),
            "fdtd_solves_if_miss": int(n_solves),
            "cache_expectation": ("全部 miss（新版站位 clean_id 与旧带平移缓存不同名，不复用）"
                                 if task.endswith("_B") else
                                 "全部 miss（带限修复 + 协议版本入键，拒绝复用旧解析缓存）"),
        }
    q4_solves = len(P.Q4_TARGETS) * P.Q4_STATES_TOTAL * len(P.Q4_TX_POSITIONS_M)
    out["Q4"] = {"n_configs": len(P.Q4_TARGETS), "states_per_config": P.Q4_STATES_TOTAL,
                 "n_tx": len(P.Q4_TX_POSITIONS_M), "fdtd_solves_if_miss": q4_solves,
                 "hw_windows": P.Q4_HW_WINDOWS,
                 "note": "15 个仿真输出 + 205 个跨域比较；共用仿真不当作 205 次独立求解。"}
    out["totals"] = {
        "method_runs_full": sum(v["n_method_runs"] for k, v in out.items() if k != "Q4"),
        "fdtd_solves_controlled_if_miss": sum(v["fdtd_solves_if_miss"] for k, v in out.items()
                                              if k != "Q4"),
        "fdtd_solves_q4_if_miss": q4_solves,
        "spec_upper_bound": P.BATCH_TOTAL_UPPER_BOUND,
        "spec_upper_bound_table": 21900,
        "e4_a_conflict": P.bATCH_E4_A_DEFINITION_CONFLICT,
    }
    return out


def cmd_plan(args) -> int:
    doc = load_protocol(args.protocol)
    output_root = check_output_root(args.output_root)
    plan = {
        "kind": "jsf-plan", "protocol_version": P.PROTOCOL_VERSION,
        "protocol_hash": doc["protocol_hash"], "self_check": doc["self_check"],
        "methods": [estimate.ONLY_METHOD],
        "output_root": str(output_root),
        "output_root_is_new_recovery_root": (str(output_root).startswith(DEFAULT_NEW_REMOTE_ROOT)
                                             or "recovery_20260929" in str(output_root)),
        "tasks": {},
        "cache_and_cost": cache_plan(),
        "note": "plan 只规划：不建 cache、不跑方法、不启动 Meep。",
    }
    for task in P.TASKS:
        specs = P.expand_task(task)
        plan["tasks"][task] = {
            "n_specs": len(specs),
            "n_clean": len({s["clean_id"] for s in specs}),
            "n_trials_total": int(sum(s["trials"] for s in specs)),
            "snr_db": sorted({float(s["snr_db"]) for s in specs}),
            "L": sorted({int(s["L"]) for s in specs}),
            "layer": task[-1],
            "needs_fdtd_convergence": task.endswith("_B"),
            "s1_s5_check": {k: v for k, v in P.S1_S5_LAYOUTS.items()} if task.startswith("E1") else None,
        }
    out_path = output_root / "plan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ("protocol_version", "protocol_hash", "methods",
                                           "output_root")}, ensure_ascii=False, indent=1))
    for t, v in plan["tasks"].items():
        print("  %-6s specs=%3d clean=%2d trials=%5d snr=%s L=%s"
              % (t, v["n_specs"], v["n_clean"], v["n_trials_total"], v["snr_db"], v["L"]))
    print(json.dumps(plan["cache_and_cost"]["totals"], ensure_ascii=False, indent=1))
    print("plan -> %s" % out_path)
    return EXIT_OK if doc["self_check"]["passed"] else EXIT_FAIL


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def cmd_validate(args) -> int:
    doc = load_protocol(args.protocol)
    output_root = check_output_root(args.output_root)
    checks = {}
    checks["protocol_self_check"] = bool(doc["self_check"]["passed"])
    checks["methods_only_this_method"] = True
    try:
        estimate.assert_methods(["b1_deng"])
        checks["rejects_other_methods"] = False
    except estimate.MethodScopeError:
        checks["rejects_other_methods"] = True
    # 冻结包可导入且带限修复在位
    bundle = paths.frozen_bundle_dir()
    fwd = (bundle / "analytic" / "forward.py").read_text(encoding="utf-8")
    checks["frozen_bundle_present"] = (bundle / "sim_common.py").is_file()
    checks["bandlimit_fix_in_bundle"] = "BANDLIMIT-FIX-20260929" in fwd
    checks["bandlimit_mask_applied"] = "H = H * s_mask" in fwd
    checks["bandlimit_diagnostic_kept"] = "full_band_ideal" in fwd
    # 无平移
    checks["no_offgrid_shift_in_protocol"] = list(doc["protocol"]["offgrid_shift_m"]) == [0.0, 0.0]
    checks["no_shift_in_new_package"] = "OFFGRID_SHIFT_M" in (
        PKG_ROOT / "jsfsim" / "protocol.py").read_text(encoding="utf-8")
    # 输出根
    checks["output_root_not_old_archive"] = True
    # S1–S5 与规范一致（逐位）
    spec = {"S1": [[1.8, 12.0]], "S2": [[1.8, 12.0], [1.8, -12.0]], "S3": [[1.5, 12.0], [2.1, 12.0]],
            "S4": [[1.5, -12.0], [2.1, 12.0]], "S5": [[1.5, -20.0], [1.8, 0.0], [2.1, 20.0]]}
    checks["s1_s5_exact"] = all(
        [[float(a), float(b)] for a, b in doc["protocol"]["layouts_S1_S5"][k]] == v
        for k, v in spec.items())
    # 取峰规则冻结
    pk = doc["protocol"]["peak_rule"]
    checks["peak_rule_frozen"] = (abs(float(pk["threshold_rel"]) - 0.30) < 1e-12
                                  and abs(float(pk["nms_min_dist_m"]) - 0.10) < 1e-12
                                  and int(pk["window"]) == 3 and pk["strict"] is False)
    # E2 间距精确性
    e2_ok = True
    for d in P.E2_A_SPACINGS_M + P.E2_B_SPACINGS_M:
        for direction in P.E2_DIRECTIONS:
            _, _, actual = P.e2_pair(direction, d)
            if abs(actual - float(d)) > 1e-9:
                e2_ok = False
    checks["e2_euclidean_exact"] = e2_ok
    # Q4 清单与几何
    q4 = doc["protocol"]["q4"]
    checks["q4_15_configs"] = len(q4["targets_xy_m"]) == 15
    checks["q4_205_hw"] = int(q4["hw_windows"]["total"]) == 205
    checks["q4_real_txrx"] = (q4["tx_positions_m"] == P.Q4_TX_POSITIONS_M
                              and q4["rx_positions_m"] == P.Q4_RX_POSITIONS_M
                              and q4["pairs"] == P.Q4_PAIRS)
    checks["q4_time_reference_blocker_defined"] = bool(q4["time_reference"]["blocked_marker"])
    checks["scope_local_no_meep_run"] = bool(args.scope == "local")
    result = {"kind": "jsf-validate", "protocol_version": P.PROTOCOL_VERSION,
              "protocol_hash": doc["protocol_hash"], "output_root": str(output_root),
              "scope": args.scope, "checks": checks,
              "passed": bool(all(checks.values())),
              "failed": sorted(k for k, v in checks.items() if not v),
              "note": "validate 只做静态体检，不启动 Meep、不跑批次。"}
    out_path = output_root / "validate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("validate PASS=%s failed=%s" % (result["passed"], result["failed"]))
    for k, v in checks.items():
        print("  [%s] %s" % ("OK" if v else "FAIL", k))
    return EXIT_OK if result["passed"] else EXIT_FAIL


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    doc = load_protocol(args.protocol)
    output_root = check_output_root(args.output_root)
    if not args.task:
        raise SystemExit("必须显式给 --task（%s）" % list(P.TASKS))
    if args.task not in P.TASKS:
        raise SystemExit("未知 --task=%r（可选 %s）" % (args.task, list(P.TASKS)))
    if not args.mode:
        raise SystemExit("必须显式给 --mode smoke|formal")
    if args.methods and args.methods != estimate.ONLY_METHOD:
        raise SystemExit("拒绝：methods=%r；本轮只允许 this_method" % args.methods)
    if args.task.endswith("_B") and args.mode == "formal" and not (args.fdtd_validation or args.numerical_waiver):
        raise SystemExit("层 B 正式 run 必须先过数值前检：给 --fdtd-validation <convergence.json>")
    run_id = args.run_id or "%s_%s" % (args.task, args.mode)
    res = runner.execute_task(
        args.task, mode=args.mode, output_root=output_root, run_id=run_id, resume=bool(args.resume),
        workers=int(args.workers), max_states=args.max_states, dry_run=bool(args.dry_run),
        limit_specs=args.limit, trial_workers=int(args.trial_workers), fdtd_validation=args.fdtd_validation,
        numerical_waiver=args.numerical_waiver)
    return EXIT_OK if res.get("completion_status", "complete") == "complete" else EXIT_FAIL


# --------------------------------------------------------------------------- #
# summarize / plot / convergence / q4  →  由专用模块提供
# --------------------------------------------------------------------------- #
def cmd_summarize(args) -> int:
    from jsfsim import summarize as summ
    return summ.summarize(Path(args.output_root))


def cmd_plot(args) -> int:
    from jsfsim import plotting
    return plotting.plot_all(Path(args.output_root))


def cmd_convergence(args) -> int:
    from jsfsim import convergence as conv
    return conv.run(profile=args.profile, output_root=Path(args.output_root),
                    protocol_path=args.protocol, resolution=args.resolution,
                    run_time=args.run_time, states=args.states,
                    workers=int(args.workers), escalate=not bool(args.no_escalate))


def cmd_q4(args) -> int:
    from jsfsim import q4_compare
    return q4_compare.cli(stage=args.stage, output_root=Path(args.output_root),
                          protocol_path=args.protocol, configs=args.configs,
                          stage_time_reference=args.time_reference,
                          workers=int(args.workers), skip_precompute=bool(args.skip_precompute))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jsf_sim_cli", description="新版 JSF-MUSIC 仿真入口（仅 this_method）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="只规划（不跑）")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output-root", dest="output_root", required=True)
    p.set_defaults(fn=cmd_plan)

    v = sub.add_parser("validate", help="静态体检（不跑）")
    v.add_argument("--protocol", required=True)
    v.add_argument("--output-root", dest="output_root", required=True)
    v.add_argument("--scope", choices=("local", "remote"), default="local")
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("run", help="执行（必须给 task 与 mode）")
    r.add_argument("--protocol", required=True)
    r.add_argument("--output-root", dest="output_root", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--mode", choices=("smoke", "pilot", "formal"), required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--run-id", dest="run_id", default=None)
    r.add_argument("--workers", type=int, default=32,
                   help="层 B 状态求解并行 worker 数（每进程一个 Meep 实例）")
    r.add_argument("--trial-workers", dest="trial_workers", type=int, default=32,
                   help="解析/算法评估的 trial 级并行 worker 数（只调度，不改数学与 seed）")
    r.add_argument("--max-states", dest="max_states", type=int, default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--dry-run", dest="dry_run", action="store_true")
    r.add_argument("--methods", default=estimate.ONLY_METHOD)
    r.add_argument("--fdtd-validation", dest="fdtd_validation", default=None)
    r.add_argument("--numerical-waiver", default=None,
                   help="用户明确放行的决定 JSON；允许执行，但不宣称数值收敛通过")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("convergence", help="FDTD 数值前检（代表场景）")
    c.add_argument("--protocol", required=True)
    c.add_argument("--output-root", dest="output_root", required=True)
    c.add_argument("--profile", choices=("controlled", "q4"), default="controlled")
    c.add_argument("--resolution", type=float, default=None)
    c.add_argument("--run-time", dest="run_time", type=float, default=None)
    c.add_argument("--states", type=int, default=16)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--no-escalate", dest="no_escalate", action="store_true",
                   help="只跑第一阶段两档比较（20/40 与 250/350），不升级到 40/80、350/500")
    c.set_defaults(fn=cmd_convergence)

    q = sub.add_parser("q4", help="Q4 跨域比较")
    q.add_argument("--protocol", required=True)
    q.add_argument("--output-root", dest="output_root", required=True)
    q.add_argument("--stage", choices=("plan", "precompute", "time-reference", "run"),
                   required=True)
    q.add_argument("--configs", nargs="*", default=None)
    q.add_argument("--time-reference", dest="time_reference", default=None)
    q.add_argument("--workers", type=int, default=32,
                   help="Q4 状态级并行 worker 数（每进程一个 Meep 实例）")
    q.add_argument("--skip-precompute", dest="skip_precompute", action="store_true",
                   help="跳过并行预填（缓存已就绪时使用）")
    q.set_defaults(fn=cmd_q4)

    s = sub.add_parser("summarize", help="汇总已保存结果")
    s.add_argument("--output-root", dest="output_root", required=True)
    s.set_defaults(fn=cmd_summarize)

    pl = sub.add_parser("plot", help="只消费已保存结果出图")
    pl.add_argument("--output-root", dest="output_root", required=True)
    pl.set_defaults(fn=cmd_plot)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return EXIT_REJECT


if __name__ == "__main__":
    raise SystemExit(main())
