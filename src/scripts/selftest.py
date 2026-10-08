#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本包自检（本地 .venv 可跑）：协议/前端/适配/E0/方法装载/warmcold/metrics 全链路最小验证。

要求五法（this_method, b1_deng, b2_kakouche, c1_jade, c3_kim）**装载 OK 且运行正常**
（自然 no-hit 允许；模块导入失败 / 接口缺失 / 输出非法不算通过）。

用法::
    & .venv/Scripts/python.exe -B 大修执行_20260923/06_仿真实验_20260926/scripts/selftest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import sim_common as sc  # noqa: E402  先于 numpy：线程 env 先固定

import numpy as np  # noqa: E402

FAILURES = []
BLOCKED = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, (" :: " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return bool(cond)


def blocked(name, detail=""):
    print("[BLOCKED] %s :: %s" % (name, detail))
    BLOCKED.append({"check": name, "detail": str(detail)})


def main() -> int:
    import protocol
    from protocol import PROTOCOL
    import analytic.scenes as scenes
    from adapter import case_build, frontend, methods_loader, metrics_bridge, sources, warmcold, e0_checks

    print("== protocol ==")
    ph = protocol.protocol_hash()
    print("protocol_hash", ph[:16], "F", sc.n_freq_bins(), "rmax",
          round(sc.domain_rmax_m(), 6), "angstep", round(sc.angle_step_deg(), 6))
    check("protocol_frozen", bool(PROTOCOL["frozen"]))
    check("unified_static_domain", bool(PROTOCOL["search"]["unified_static_domain"]))
    check("c1_no_domain_NA", PROTOCOL["search"]["c1_domain_theta_max_deg"]
          < PROTOCOL["search"]["c1_phase_unique_max_deg"])

    # --- 一个代表 spec：E1 S2 @ +10 dB ---
    specs = [s for s in scenes.e1_specs("A") if s["scene"] == "E1" and "S2" in s["case_id"]
             and s["snr_db"] == 10.0]
    check("spec_found", len(specs) == 1, specs[0]["case_id"] if specs else "")
    spec = specs[0]

    data_root = sc.default_data_root()
    cfg_hash = sc.config_hash_for(PROTOCOL, extra_config={"selftest": True})
    runid = "selftest_%s_%s" % (ph[:6], cfg_hash[:6])
    sc.ensure_run_dir(data_root, runid, config_hash=cfg_hash, extra={"kind": "selftest"})
    paths = sc.run_paths(data_root, runid)
    cache_dir = paths["cache"]
    src = sources.get_source("analytic")
    src_agg = sc.source_fingerprint()["aggregate"]

    print("== clean cache (layer A) ==")
    clean = src.build_clean(spec, cache_dir=cache_dir, protocol_hash=ph, source_aggregate=src_agg)
    loaded = src.load_clean(cache_dir, spec["clean_id"], expect_signature=None)
    truth = src.load_truth(cache_dir, spec["clean_id"])
    check("clean_shape", list(loaded["raw_cir"].shape) == [sc.SNAPSHOTS, 4, 32],
          loaded["raw_cir"].shape)
    check("truth_separate", "targets_xy_m" in truth and truth["k"] == spec["k"])
    npz_keys = set()
    with np.load(clean["npz_path"], allow_pickle=False) as z:
        npz_keys = set(z.files)
    check("npz_no_truth_fields",
          not any(("truth" in kk.lower() or "target" in kk.lower()) for kk in npz_keys), sorted(npz_keys))

    print("== frontend + case ==")
    # 关键正确性：快照不足不得广播复制（raw1 -> X 必须只有 1 行）
    raw1 = np.asarray(loaded["raw_cir"])[:1]
    X1a, L1a, d1a = sc.assemble_noisy_x(raw1, 128, 10.0, 12345, allow_shorter=True)
    check("no_snapshot_broadcast_short_ok", int(X1a.shape[0]) == 1 and L1a == 1
          and d1a["L_requested"] == 128 and d1a["formal"] is False, d1a)
    try:
        sc.assemble_noisy_x(raw1, 128, 10.0, 12345, allow_shorter=False)
        check("no_snapshot_broadcast_formal_refuses", False, "应当因快照不足抛错")
    except sc.SimError as exc:
        check("no_snapshot_broadcast_formal_refuses", True, str(exc)[:80])
    # E4 噪声前缀：L=10 与 L=128 的前 10 快照 unit 必须逐格相同（且与 sigma 无关）
    u10 = sc.noise_unit_for(128, 10, 777)
    u128 = sc.noise_unit_for(128, 128, 777)
    check("e4_noise_prefix_L10_eq_L128_head10", bool(np.array_equal(u10, u128[:10])),
          {"u10": list(u10.shape), "u128": list(u128.shape)})
    Xd, Ld, dd = sc.assemble_noisy_x(np.asarray(loaded["raw_cir"]), 10, 10.0, 777,
                                     validation_only=False)
    check("assemble_formal_flag_not_validation", dd["formal"] is True and Ld == 10, dd)
    raw_L = loaded["raw_cir"][: int(spec["L"])]
    cropped_clean = frontend.cropped_from_raw(raw_L)
    rng = sc.rng_from(sc.trial_seed(spec["clean_id"], 0))
    noise_unit = sc.complex_gaussian(
        (int(spec["L_clean"]), 4, int(sc.N_TAPS_WINDOW)), rng)[: int(spec["L"])]
    sigma = sc.noise_sigma(cropped_clean, spec["snr_db"])
    X = frontend.x_from_cropped(cropped_clean + sigma * noise_unit)
    v = frontend.verify_vs_b0(cropped_clean + sigma * noise_unit)
    check("frontend_matches_b0", v.get("available") and v.get("max_abs_diff", 1) < 1e-12, v)

    gx, gy = sc.build_grid()
    case = case_build.build_case(case_id=spec["case_id"], clean_id=spec["clean_id"], X=X,
                                 k=spec["k"], grid_x=gx, grid_y=gy, layer="A", source="analytic",
                                 scene="E1", tags=spec["tags"], snr_db=spec["snr_db"], trial=0,
                                 seed=sc.trial_seed(spec["clean_id"], 0), L=int(spec["L"]))
    case_build.validate_case(case)
    check("case_valid", True)
    check("case_X_readonly", not case["X"].flags.writeable)
    check("case_no_truth_meta", "sim_meta" not in case)

    print("== methods (5 main + B0) ==")
    boot = methods_loader.bootstrap()
    print("preimported:", {k: (v.split("\\")[-1] if isinstance(v, str) else v)
                           for k, v in boot["preimported"].items()})
    statuses = {}
    for method in methods_loader.MAIN_METHODS + methods_loader.REFERENCE_METHODS:
        fp0 = case_build.case_fingerprint(case)
        params = methods_loader.params_for(method, PROTOCOL) or None
        res = methods_loader.call(method, case, params)
        fp1 = case_build.case_fingerprint(case)
        statuses[method] = res["status"]
        check("method_%s_ran" % method, res["status"] in ("ok", "failed", "not_applicable"),
              "%s n_pred=%d reason=%s" % (res["status"], len(res["predictions"]),
                                          str(res["diagnostics"].get("reason"))[:60]))
        check("case_unmodified_%s" % method, fp0 == fp1)
        check("method_%s_not_loadfail" % method,
              res["diagnostics"].get("reason") not in ("module_load_failed", "interface_missing"),
              res["diagnostics"].get("detail") or "")
    n_ok = sum(1 for s in statuses.values() if s in ("ok", "not_applicable"))
    check("five_main_loaded_and_ran", all(statuses[m] != "failed" for m in methods_loader.MAIN_METHODS),
          statuses)

    print("== E0 ==")
    e0 = e0_checks.run_all(case, spec["targets_xy"], spec=spec)
    nc = e0["numerical_consistency"]
    check("e0_numerical_consistency", nc.get("spectrum_cosine", 0) > 1 - 1e-9, nc)
    check("e0_six_methods_actually_ran", e0["six_method_run"]["all_six_executed_no_loadfail"]
          and e0["six_method_run"]["all_six_status_not_failed"],
          {k: v["status"] for k, v in e0["six_method_run"]["by_method"].items()})
    check("e0_crop_disclosed", e0["crop_offset_disclosure"]["not_compensated"] is True)
    check("case_id_opaque", case["id"].startswith("opq_") and case["config_id"].startswith("opq_"),
          [case["id"], case["config_id"]])
    zc = e0["zero_offset_cfr_diagnostic"]
    zok = zc.get("by_method", {}).get("this_method", {}).get("status") == "ok"
    check("e0_zero_offset_cfr_ran", zok, str(zc)[:160])
    print("  ideal_locality:", json.dumps(e0["ideal_locality"]["by_method"], ensure_ascii=False)[:300])
    print("  zero_offset:", json.dumps(zc, ensure_ascii=False)[:300])

    print("== warmcold verify ==")
    for method in warmcold.WARM_SUPPORTED:
        rep = warmcold.verify(method, case, methods_loader.params_for(method, PROTOCOL) or None)
        check("warm_%s_equivalent" % method, rep.get("equivalent") is True, rep)
    for method, reason in warmcold.WARM_NOT_APPLICABLE.items():
        check("warm_%s_declared_NA" % method, warmcold.warm_support(method)["supported"] is False,
              reason[:60])

    print("== metrics bridge ==")
    preds = methods_loader.call("this_method", case)["predictions"]
    m = metrics_bridge.evaluate_predictions(spec["targets_xy"], [(p["x"], p["y"]) for p in preds],
                                            resolution_distance_m=spec["resolution_distance_m"])
    check("metrics_row", "ospa2" in m and "tp" in m, {k: m[k] for k in ("tp", "fn", "fp", "ospa")})
    agg = metrics_bridge.aggregate([{"metrics": m, "n_truth": spec["k"]},
                                    {"metrics": metrics_bridge.evaluate_failed(spec["targets_xy"])}])
    check("metrics_aggregate", isinstance(agg, dict), list(agg.keys())[:8])

    print("== layer B interface (scene export + fdtd.api.plan_run, no Meep) ==")
    from adapter import scene_export
    b_specs = scenes.e1_specs("B")
    clean_map = scenes.resolve_clean_specs(b_specs)
    check("E1B_clean_dedup_across_snr", len(clean_map) == 5, len(clean_map))
    try:
        smap = scene_export.scenes_for_specs(b_specs, protocol=PROTOCOL)
        check("E1B_scene_export_unique", len(smap) == 5, sorted(smap))
        k0 = list(smap)[0]
        check("scene_has_no_case_id_in_physics",
              "sim_case_id" not in smap[k0] and smap[k0]["scene_id"] == k0)
    except Exception as exc:  # noqa: BLE001
        check("fdtd_scene_export", False, "%s: %s" % (type(exc).__name__, exc))
    try:
        import fdtd.api as fdtd_api
        plans = [fdtd_api.plan_run(None, smap[cid]) for cid in list(smap)[:2]]
        check("fdtd_plan_run_no_meep_ok", all("cost" in p and "cache_key" in p for p in plans),
              {p["scene_id"]: (p["cost"].get("n_cells") if isinstance(p.get("cost"), dict) else None)
               for p in plans})
    except Exception as exc:  # noqa: BLE001
        blocked("fdtd_plan_run_no_meep_ok", "外部模块 fdtd/ 不可用（非本包；另有代理在修）：%s: %s"
                % (type(exc).__name__, exc))

    report = {"protocol_hash": ph, "runid": runid, "spec": spec["case_id"],
              "statuses": statuses, "e0": e0,
              "warm": {mm: warmcold.warm_support(mm) for mm in methods_loader.METHOD_ORDER},
              "metrics_row": m, "metrics_aggregate_keys": sorted(agg.keys()),
              "failures": list(FAILURES), "blocked_external": list(BLOCKED)}
    out = paths["results"] / "selftest_report.json"
    sc.atomic_write_json(out, report)
    print("\n== summary ==")
    print("failures:", FAILURES if FAILURES else "none")
    print("blocked(external):", BLOCKED if BLOCKED else "none")
    print("report:", out)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
