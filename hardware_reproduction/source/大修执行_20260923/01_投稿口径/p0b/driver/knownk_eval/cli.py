# -*- coding: utf-8 -*-
"""P0b driver 命令行入口（真实可用；失败返回非零退出码）。

子命令：
  verify-protocol   校验协议 canonical SHA256（漂移即失败）
  replay            用历史缓存谱 + 网格 + 已知 K + 真值完整重评，并与 P0a 底层表逐窗/聚合核对
  demo-predictions  演示连续域（原生 x,y）预测 API（合成输入，仅接口自检，非论文证据）
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

from . import sources
from .evaluate import evaluate_predictions, evaluate_windows, summarise
from .manifest import MANIFEST_NAME, build_manifest
from .metrics import STATS_COLUMNS, DuplicateWindowError, error_stats
from .protocol import ProtocolError, canonical_sha256, load_protocol, sha256_file

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILURE = 2
STATS_TOL = 1e-9
PEAK_TOL = 1e-12


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
                    encoding="utf-8")
    return path


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _window_rows(results):
    rows = []
    for res in results:
        first_failure = res.failures[0]["error"] if res.failures else ""
        rows.append({
            "window_id": res.window_id,
            "recording_id": res.recording_id,
            "environment": res.environment,
            "config_id": res.config_id,
            "n_truth": res.n_truth,
            "k": res.k,
            "input_kind": res.input_kind,
            "status": res.status,
            "spectrum_state": (res.spectrum_state or {}).get("spectrum_state", ""),
            "n_predictions": res.n_predictions,
            "tp": res.tp,
            "fn": res.fn,
            "fp": res.fp,
            "failure": first_failure,
            "wall_seconds": round(res.wall_seconds, 6),
        })
    return rows


def _reference_stats_from_p0a(data_dir):
    targets = sources.load_outcome_targets(data_dir)
    e_r, e_theta, e_xy = [], [], []
    tp = fn = 0
    for _wid, items in targets.items():
        for item in items:
            if item["outcome"] == "TP":
                tp += 1
                e_r.append(item["e_r_m"])
                e_theta.append(item["e_theta_deg"])
                e_xy.append(item["e_xy_m"])
            elif item["outcome"] == "FN":
                fn += 1
    windows = sources.load_per_window_branch(data_dir)
    fp = sum(int(w["fp"]) for w in windows.values())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "n_windows": len(windows),
        "e_r_m": error_stats(e_r),
        "e_theta_deg": error_stats(e_theta),
        "e_xy_m": error_stats(e_xy),
    }


def _compare_to_p0a(results, protocol, data_dir, out_dir):
    ref = _reference_stats_from_p0a(data_dir)
    ref_windows = sources.load_per_window_branch(data_dir)
    checks = []
    mismatches = []
    for res in results:
        ref_row = ref_windows.get(res.window_id)
        if ref_row is None:
            mismatches.append({"window_id": res.window_id, "issue": "P0a 无该窗记录"})
            continue
        for key in ("tp", "fn", "fp"):
            if int(getattr(res, key)) != int(ref_row[key]):
                mismatches.append({"window_id": res.window_id, "field": key,
                                   "driver": int(getattr(res, key)),
                                   "p0a": int(ref_row[key])})
    summary = summarise(results, protocol)
    overall = summary["overall"]
    for key in ("tp", "fn", "fp"):
        ok = int(overall[key]) == int(ref[key])
        checks.append({"check": "aggregate.%s" % key, "driver": int(overall[key]),
                       "p0a": int(ref[key]), "pass": bool(ok)})
    n_truth = ref["tp"] + ref["fn"]
    expected = {
        "dr": ref["tp"] / n_truth if n_truth else None,
        "miss_rate": ref["fn"] / n_truth if n_truth else None,
        "fp_per_estimate": ref["fp"] / ref["n_windows"] if ref["n_windows"] else None,
    }
    for key, value in expected.items():
        actual = overall[key]
        ok = (actual is not None and value is not None
              and abs(float(actual) - float(value)) <= STATS_TOL)
        checks.append({"check": "aggregate.%s" % key, "driver": actual, "p0a": value,
                       "pass": bool(ok)})
    for name in ("e_r_m", "e_theta_deg", "e_xy_m"):
        for stat in ("n", "mae", "rmse", "median", "p95", "std"):
            actual = overall["%s_%s" % (name, stat)]
            expected_value = ref[name][stat]
            if stat == "n":
                ok = int(actual) == int(expected_value)
            else:
                ok = (actual is not None and expected_value is not None
                      and abs(float(actual) - float(expected_value)) <= STATS_TOL)
            checks.append({"check": "aggregate.%s.%s" % (name, stat), "driver": actual,
                           "p0a": expected_value, "pass": bool(ok)})
    payload = {
        "kind": "p0b-vs-p0a-comparison",
        "generated_at": _now(),
        "reference": "Q2 数据采集/q2q3_eval_20260922/data/{per_window.csv, outcome_targets.csv} (oracle+nms)",
        "n_windows_compared": len(results),
        "per_window_mismatches": mismatches,
        "aggregate_checks": checks,
        "n_aggregate_checks": len(checks),
        "n_aggregate_pass": sum(1 for c in checks if c["pass"]),
        "tolerances": {"counts": "exact", "stats_abs": STATS_TOL},
        "pass": (not mismatches) and all(c["pass"] for c in checks),
    }
    _write_json(Path(out_dir) / "compare_p0a.json", payload)
    return payload


def _compare_recorded_peaks(results, data_dir, out_dir):
    recorded = sources.load_recorded_predictions(data_dir)
    mismatches = []
    compared = 0
    for res in results:
        want = recorded.get(res.window_id)
        if want is None:
            mismatches.append({"window_id": res.window_id, "issue": "历史无该窗预测"})
            continue
        compared += 1
        got = sorted(res.peaks, key=lambda p: p.get("rank", 0))
        if len(got) != len(want):
            mismatches.append({"window_id": res.window_id, "issue": "峰数不一致",
                               "driver": len(got), "p0a": len(want)})
            continue
        for idx, (g, w) in enumerate(zip(got, want)):
            if (abs(g["x"] - w["x"]) > PEAK_TOL or abs(g["y"] - w["y"]) > PEAK_TOL
                    or (w["value"] is not None and abs(g["value"] - w["value"]) > PEAK_TOL)):
                mismatches.append({"window_id": res.window_id, "peak_rank": idx,
                                   "driver": [g["x"], g["y"], g["value"]],
                                   "p0a": [w["x"], w["y"], w["value"]]})
    payload = {
        "kind": "p0b-vs-recorded-peaks",
        "generated_at": _now(),
        "reference": "outcome_predictions.csv (oracle+nms)",
        "n_windows_compared": compared,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:200],
        "tolerance_m": PEAK_TOL,
        "pass": not mismatches,
    }
    _write_json(Path(out_dir) / "recorded_peak_check.json", payload)
    return payload


def cmd_verify_protocol(args) -> int:
    protocol = load_protocol(args.protocol, expected_sha256=args.expect_sha256)
    payload = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    recomputed = canonical_sha256(payload)
    out = {
        "kind": "p0b-protocol-verification",
        "generated_at": _now(),
        "path": str(Path(args.protocol)),
        "schema": payload.get("schema"),
        "protocol_id": payload.get("protocol_id"),
        "protocol_revision": payload.get("protocol_revision"),
        "recorded_canonical_sha256": payload.get("protocol_canonical_sha256"),
        "recomputed_canonical_sha256": recomputed,
        "file_sha256": sha256_file(args.protocol),
        "peaks": {
            "threshold_rel": protocol.peaks.threshold_rel,
            "min_dist_m": protocol.peaks.min_dist_m,
            "cap": protocol.peaks.count_cap,
        },
        "matching": {
            "gate_m": protocol.matching.gate_m,
            "gate_tolerance_m": protocol.matching.gate_tolerance_m,
        },
        "pass": recomputed == payload.get("protocol_canonical_sha256"),
        "semantic_fields_checked": True,
        "behavioral_self_check": True,
    }
    if args.out:
        _write_json(Path(args.out) / "protocol_verification.json", out)
    print("[protocol] %s revision=%s canonical=%s file=%s"
          % (out["protocol_id"], out["protocol_revision"], recomputed, out["file_sha256"]))
    return EXIT_OK if out["pass"] else EXIT_FAILURE


def cmd_replay(args) -> int:
    protocol = load_protocol(args.protocol)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = list(sources.iter_cached_windows(
        args.spectra_dir, args.data_dir,
        windows=None if args.windows in (None, "all") else [w for w in args.windows.split(",") if w],
        recording_ids=None if not args.recordings else [r for r in args.recordings.split(",") if r],
    ))
    if args.limit:
        windows = windows[: int(args.limit)]
    if not windows:
        print("[replay] 没有匹配到任何窗（检查 --spectra-dir/--data-dir/--windows）")
        return EXIT_FAILURE
    print("[replay] windows=%d protocol=%s canonical=%s"
          % (len(windows), protocol.protocol_id, protocol.canonical_sha256_value))
    results, rows = evaluate_windows(windows, protocol, progress=50)
    summary = summarise(results, protocol)
    summary["generated_at"] = _now()
    summary["kind"] = "p0b-knownk-replay-summary"
    summary["inputs"] = {
        "spectra_dir": str(Path(args.spectra_dir)),
        "data_dir": str(Path(args.data_dir)) if args.data_dir else None,
        "n_windows_requested": len(windows),
    }
    _write_csv(out_dir / "aggregate.csv", STATS_COLUMNS, rows)
    window_rows = _window_rows(results)
    _write_csv(out_dir / "windows.csv", list(window_rows[0].keys()), window_rows)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "windows.json", [r.to_dict() for r in results])
    compare = None
    recorded = None
    if args.data_dir:
        compare = _compare_to_p0a(results, protocol, args.data_dir, out_dir)
        print("[compare] p0a per-window mismatches=%d aggregate %d/%d pass"
              % (len(compare["per_window_mismatches"]), compare["n_aggregate_pass"],
                 compare["n_aggregate_checks"]))
        if args.compare_recorded:
            recorded = _compare_recorded_peaks(results, args.data_dir, out_dir)
            print("[compare] recorded peaks windows=%d mismatches=%d"
                  % (recorded["n_windows_compared"], recorded["n_mismatches"]))
    overall = summary["overall"]
    print("[replay] TP=%d FN=%d FP=%d windows=%d failed=%d DR=%s fp/est=%s"
          % (overall["tp"], overall["fn"], overall["fp"], overall["n_windows"],
             overall["n_windows_failed"],
             None if overall["dr"] is None else round(overall["dr"], 6),
             None if overall["fp_per_estimate"] is None else round(overall["fp_per_estimate"], 6)))
    failed = overall["n_windows_failed"]
    checks_ok = (failed == 0
                 and (compare is None or compare["pass"])
                 and (recorded is None or recorded["pass"]))
    log = {
        "generated_at": _now(),
        "command": [Path(sys.argv[0]).name] + sys.argv[1:],
        "windows": len(windows),
        "failed_windows": summary["failed_windows"],
        "compare_p0a_pass": None if compare is None else compare["pass"],
        "recorded_peak_pass": None if recorded is None else recorded["pass"],
        "exit_code": EXIT_OK if checks_ok else EXIT_FAILURE,
        "manifest": MANIFEST_NAME,
    }
    _write_json(out_dir / "replay_log.json", log)
    manifest = build_manifest(
        out_dir, protocol,
        command=[Path(sys.argv[0]).name] + sys.argv[1:],
        extra_files=[protocol.path])
    print("[manifest] %s artifacts=%d driver_files=%d"
          % (out_dir / MANIFEST_NAME, manifest["n_artifacts"], manifest["n_driver_files"]))
    return EXIT_OK if checks_ok else EXIT_FAILURE


def cmd_demo_predictions(args) -> int:
    """连续域 API 演示：合成 3 真值 + 2 原生预测（仅接口自检，非论文证据）。"""
    protocol = load_protocol(args.protocol)
    truth = [(0.0, -2.0), (-1.286, -1.532), (1.286, -1.532)]
    preds = [(-0.05, -2.02), (1.30, -1.50)]
    res = evaluate_predictions(truth, preds, protocol, k=3, window_id="demo_synthetic",
                               source="demo_synthetic_native_xy")
    payload = {
        "kind": "p0b-native-prediction-api-demo",
        "generated_at": _now(),
        "warning": "合成输入，仅证明连续域 API 可用；不得作为论文证据或性能结果",
        "result": res.to_dict(),
        "pass": res.status == "ok" and res.tp == 2 and res.fn == 1 and res.fp == 0,
    }
    if args.out:
        _write_json(Path(args.out) / "demo_predictions.json", payload)
    print("[demo] status=%s tp=%d fn=%d fp=%d pass=%s"
          % (res.status, res.tp, res.fn, res.fp, payload["pass"]))
    return EXIT_OK if payload["pass"] else EXIT_FAILURE


def build_parser():
    parser = argparse.ArgumentParser(description="P0b known-K 统一评估 driver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify-protocol", help="校验协议 canonical SHA256")
    p_verify.add_argument("--protocol", required=True)
    p_verify.add_argument("--expect-sha256", default=None)
    p_verify.add_argument("--out", default=None)
    p_verify.set_defaults(func=cmd_verify_protocol)

    p_replay = sub.add_parser("replay", help="缓存谱完整重评并与 P0a 核对")
    p_replay.add_argument("--protocol", required=True)
    p_replay.add_argument("--spectra-dir", required=True)
    p_replay.add_argument("--data-dir", default=None)
    p_replay.add_argument("--out", required=True)
    p_replay.add_argument("--windows", default="all")
    p_replay.add_argument("--recordings", default=None)
    p_replay.add_argument("--limit", type=int, default=None)
    p_replay.add_argument("--compare-recorded", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_demo = sub.add_parser("demo-predictions", help="连续域原生 (x, y) 预测 API 演示")
    p_demo.add_argument("--protocol", required=True)
    p_demo.add_argument("--out", default=None)
    p_demo.set_defaults(func=cmd_demo_predictions)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ProtocolError as exc:
        print("[protocol-error] %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    except DuplicateWindowError as exc:
        print("[duplicate-window-error] %s" % exc, file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
