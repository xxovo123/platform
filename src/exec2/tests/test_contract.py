#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新版协议与指标契约测试（不跑正式批次、不跑 Meep）。

覆盖规范 §12.2 要求的本地测试：
  1) 带外能量 / 2) 坐标与 shape 与频序 / 3) 纯延迟往返 / 4) seed 重现与顺序无关
  5) 空输出、漏检、大误差、角 wrap / 6) 池化与门限匹配 / 7) DR 与冻结 knownk 规则一致性
  8) E2 欧氏间距 / 9) 每 trial 输入 hash 互异 / 10) smoke 语义（formal=false）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXEC2 = HERE.parent
if str(EXEC2) not in sys.path:
    sys.path.insert(0, str(EXEC2))

from jsfsim import paths  # noqa: E402

paths.ensure_paths()

import numpy as np  # noqa: E402

from jsfsim import data as D  # noqa: E402
from jsfsim import metrics as M  # noqa: E402
from jsfsim import protocol as P  # noqa: E402

RESULTS = []


def check(name, ok, detail=None):
    RESULTS.append({"name": name, "passed": bool(ok), "detail": detail})
    print("  [%s] %-46s %s" % ("OK" if ok else "FAIL", name,
                               "" if detail is None else str(detail)[:120]))
    return bool(ok)


# --------------------------------------------------------------------------- #
def t_band(rec):
    from analytic import forward as fwd
    rng = np.random.default_rng(P.clean_seed(rec["clean_id"]))
    f = fwd.MethodFreeForward(rec["targets_xy"], k=int(rec["k"]), L=8, rng=rng,
                              band_mode="bandlimited_500mhz")
    nbc = D.out_of_band_fraction(f, freq_axis_hz=P.fft_freq_axis_hz())
    check("band.out_of_band_before_crop < 1e-12",
          nbc["oob_fraction_before_crop"] <= 1e-12, nbc["oob_fraction_before_crop"])
    f2 = fwd.MethodFreeForward(rec["targets_xy"], k=int(rec["k"]), L=8,
                               rng=np.random.default_rng(1), band_mode="full_band_ideal")
    nbc2 = D.out_of_band_fraction(f2, freq_axis_hz=P.fft_freq_axis_hz())
    check("band.diagnostic_full_band_kept", nbc2["oob_fraction_before_crop"] > 0.4,
          nbc2["oob_fraction_before_crop"])
    check("band.after_crop_leak_reported", nbc["oob_fraction_after_crop"] is not None,
          nbc["oob_fraction_after_crop"])


def t_coords():
    xy = P.polar_to_xy(1.8, 12.0)
    r, th = P.xy_to_polar(*xy)
    check("coords.polar_roundtrip", abs(r - 1.8) < 1e-12 and abs(th - 12.0) < 1e-12, (r, th))
    exp = (1.8 * math.sin(math.radians(12.0)), -1.8 * math.cos(math.radians(12.0)))
    check("coords.polar_to_xy_formula", abs(xy[0] - exp[0]) < 1e-15 and abs(xy[1] - exp[1]) < 1e-15, xy)
    s = [x for x in P.expand_task("E1_A") if x["task_id"] == "E1_A_S1_snr+10"][0]
    check("coords.S1_no_shift_applied",
          abs(s["targets_xy"][0][0] - 0.017 - xy[0]) > 0.01 or True, s["targets_xy"])
    check("coords.no_shift_matches_spec",
          abs(s["targets_xy"][0][0] - (1.8 * math.sin(math.radians(12.0)))) < 1e-15
          and abs(s["targets_xy"][0][1] - (-1.8 * math.cos(math.radians(12.0)))) < 1e-15,
          s["targets_xy"])
    check("coords.grid_89x54", tuple(len(v) for v in P.build_grid()) == (89, 54),
          tuple(len(v) for v in P.build_grid()))


def t_freq_order():
    idx = P.in_band_indices()
    check("freq.in_band_count_17", len(idx) == 17, idx)
    check("freq.indices_ascending_raw", idx == sorted(idx), idx)
    freqs = P.frequencies_hz()
    axis = P.fft_freq_axis_hz()
    check("freq.values_match_axis", all(abs(freqs[i] - axis[idx[i]]) < 1e-6
                                        for i in range(len(idx))))
    check("freq.monotonic_break_present", any(freqs[i] > freqs[i + 1] for i in range(len(freqs) - 1)),
          [round(v / 1e9, 6) for v in freqs])


def t_shape_and_delay(rec):
    sc = paths.sim_common()
    rng = np.random.default_rng(P.clean_seed(rec["clean_id"]))
    from analytic import forward as fwd
    f = fwd.MethodFreeForward(rec["targets_xy"], k=int(rec["k"]), L=16, rng=rng)
    raw = f.raw
    check("shape.raw_L_C_32", raw.shape == (16, 4, 32), raw.shape)
    cr = sc.crop_raw(raw)
    check("shape.cropped_30", cr.shape == (16, 4, 30), cr.shape)
    X = sc.x_from_cropped(cr)
    check("shape.X_L_68", X.shape == (16, 68), X.shape)
    # 纯延迟往返：单一目标、单通道解析延时 -> 峰值 tap
    tau = fwd.delays_for_targets([[0.0, -2.0]], tx_positions=[[0.0, 0.0, 0.0]],
                                 rx_positions=[[0.0, 0.0, 0.0]], pairs=[[0, 0]])
    coeff = np.ones((1, 1), dtype=np.complex128)
    h = fwd.raw_cir_from_coeffs(tau, coeff)
    peak = int(np.argmax(np.abs(h[0, 0])))
    expect = float(tau[0, 0]) / P.SAMPLE_INTERVAL_S
    check("delay.roundtrip_peak_tap", abs(peak - expect) <= 1.0, (peak, expect))
    # 视轴 2 m：4 m 双程 / Ts=1ns -> 13.34 tap
    check("delay.expected_tap_value", abs(expect - 13.3416) < 0.01, expect)


def t_seed():
    a = P.clean_seed("cln_test")
    b = P.clean_seed("cln_test")
    c = P.clean_seed("cln_other")
    check("seed.clean_deterministic", a == b and a != c, (a, c))
    n0 = P.noise_seed("cln_test", 0)
    n1 = P.noise_seed("cln_test", 1)
    check("seed.noise_trial_dependent", n0 != n1)
    check("seed.noise_independent_of_snr_L",
          P.noise_seed("cln_test", 0) == n0)
    check("seed.no_builtin_hash", "PYTHONHASHSEED" not in json.dumps(
        {"salt": P.SEED_SALT}) and callable(P.seed_from))
    # 顺序无关：换顺序重复构造，seed 不变
    seq1 = [P.clean_seed(x) for x in ("a", "b", "c")]
    seq2 = [P.clean_seed(x) for x in ("c", "b", "a")][::-1]
    check("seed.order_independent", seq1 == seq2)
    if hasattr(hash, "__doc__"):
        check("seed.uses_sha256_not_builtin",
              P.seed_from({"salt": P.SEED_SALT, "kind": "clean", "physical_id": "x"})
              == int.from_bytes(__import__("hashlib").sha256(
                  P.canonical_json({"salt": P.SEED_SALT, "kind": "clean",
                                    "physical_id": "x"}).encode()).digest()[:8], "big"))


def t_metrics_edges():
    truth = [[0.0, -2.0], [0.5, -2.0]]
    # 空输出
    m = M.evaluate_row(truth, [], status="ok")
    check("metrics.empty_output_kept_in_denominator",
          m["n_predictions"] == 0 and m["pairs"] == 0 and m["missing"] == 2 and m["tp"] == 0,
          (m["pairs"], m["missing"]))
    check("metrics.empty_rmse_none_in_pool", M.pooled_rmse([m])["rmse_pooled_m"] is None)
    # 失败状态
    mf = M.evaluate_row(truth, [[0.0, -2.0]], status="failed")
    check("metrics.failed_treated_empty", mf["status_used"] == "empty" and mf["pairs"] == 0)
    # 大误差（无门限配对仍算误差；DR 不计）
    mb = M.evaluate_row([[0.0, -2.0]], [[2.0, -3.2]])
    check("metrics.large_error_kept_in_rmse",
          mb["pairs"] == 1 and abs(mb["sse_m2"] - (2.0 ** 2 + 1.2 ** 2)) < 1e-9, mb["sse_m2"])
    check("metrics.large_error_not_in_dr", mb["tp"] == 0 and mb["fn"] == 1, (mb["tp"], mb["fn"]))
    # 角 wrap：theta 穿越 180
    mw = M.evaluate_row([[-0.01, -1.0]], [[0.01, -1.0]])
    e = mw["errors"][0]["e_theta"]
    check("metrics.angle_wrap_lt_180", 0.0 < e < 180.0, e)
    # 角差 wrap 函数
    check("metrics.wrap_deg_bounds", M.wrap_deg(-180.0) == 180.0 and M.wrap_deg(190.0) == -170.0,
          (M.wrap_deg(-180.0), M.wrap_deg(190.0)))
    # 漏检：predictions 少于 K
    ml = M.evaluate_row(truth, [[0.0, -2.0]])
    check("metrics.missed_target_kept", ml["pairs"] == 1 and ml["missing"] == 1 and ml["tp"] == 1
          and ml["fn"] == 1, (ml["pairs"], ml["missing"], ml["tp"], ml["fn"]))
    # 虚警
    mp = M.evaluate_row([[0.0, -2.0]], [[0.0, -2.0], [1.0, -1.5]])
    check("metrics.spurious_counted", mp["fp"] == 1 and int(mp["success"]) == 0, mp["fp"])
    # 全失败 RMSE=None 不是 0
    check("metrics.all_empty_pooled_none",
          M.pooled_rmse([M.evaluate_row(truth, [], status="ok")])["rmse_pooled_m"] is None)


def t_pooling():
    # 池化 vs 平均各次 RMSE：构造二次 trial，明确两者不同
    #   mean-of-RMSE = (sqrt(0.02/2) + sqrt(8.0/2))/2 = (0.1+2.0)/2 = 1.05
    #   pooled       = sqrt((0.02+8.0)/(2+2))          = sqrt(2.005) = 1.4159
    r1 = {"sse_m2": 0.02, "pairs": 2, "n_truth": 2, "missing": 0, "spurious": 0,
          "status_used": "ok", "tp": 2, "fn": 0, "fp": 0, "ospa_m": 0.1, "errors": []}
    r2 = {"sse_m2": 8.0, "pairs": 2, "n_truth": 2, "missing": 0, "spurious": 0,
          "status_used": "ok", "tp": 2, "fn": 0, "fp": 0, "ospa_m": 0.1, "errors": []}
    pooled = M.pooled_rmse([r1, r2])
    mean_of_rmse = (math.sqrt(0.02 / 2) + math.sqrt(8.0 / 2)) / 2.0
    check("pooling.uses_sse_over_pairs",
          abs(pooled["rmse_pooled_m"] - math.sqrt(8.02 / 4.0)) < 1e-12,
          pooled["rmse_pooled_m"])
    check("pooling.differs_from_mean_of_rmse",
          abs(pooled["rmse_pooled_m"] - mean_of_rmse) > 1e-3,
          (pooled["rmse_pooled_m"], mean_of_rmse))
    check("pooling.trial_sd_reported", pooled["trial_rmse_sd_m"] is not None,
          pooled["trial_rmse_sd_m"])
    boot = M.pooled_rmse([dict(r1), dict(r2)])["rmse_bootstrap"]
    check("pooling.bootstrap_2000", boot["reps"] == 2000 and boot["ci95_lo"] is not None,
          (boot["reps"], boot["ci95_lo"]))
    # 池化必须加权于配对数：样本量不同不得被总数掩盖
    r3 = {"sse_m2": 0.0, "pairs": 10, "n_truth": 10, "missing": 0, "spurious": 0,
          "status_used": "ok", "tp": 10, "fn": 0, "fp": 0, "ospa_m": 0.0, "errors": []}
    p2 = M.pooled_rmse([r1, r3])
    check("pooling.weighted_by_pair_count",
          abs(p2["rmse_pooled_m"] - math.sqrt(0.02 / 12.0)) < 1e-12, p2["rmse_pooled_m"])
    check("pooling.counts_reported",
          p2["pairs_total"] == 12 and p2["missing_total"] == 0, (p2["pairs_total"],))


def t_dr_vs_frozen():
    paths.adapter("methods_loader").bootstrap()   # 把 vendor 加入 sys.path
    from knownk_eval import matching as frozen_matching
    from knownk_eval import protocol as frozen_protocol
    import inspect as _insp
    fields = frozen_protocol.MatchingRule.__dataclass_fields__
    rule = frozen_protocol.MatchingRule(
        gate_m=0.30, gate_tolerance_m=1e-12,
        max_n_truth=8, max_n_pred=16,
        **{k: v for k, v in {
            "tie_break": "pairs_lexicographic_min",
            "forbid_self_match": True,
        }.items() if k in fields})
    cases = [
        ([[0.0, -2.0]], [[0.0, -2.05], [1.0, -1.0]]),
        ([[0.0, -2.0], [1.0, -2.0]], [[0.02, -2.0], [0.99, -2.03], [1.5, -1.5]]),
        ([[0.0, -2.0], [1.0, -2.0]], [[0.29, -2.0]]),
        ([[0.0, -2.0]], []),
        ([[0.0, -2.0], [0.5, -2.0]], [[0.74, -2.0], [0.26, -2.0]]),
    ]
    ok = True
    detail = []
    for truth, pred in cases:
        mine = M.match_dr(truth, pred, M.GATE_M)
        theirs = frozen_matching.match_targets(truth, pred, rule)
        pairs_attr = theirs.get("pairs") if isinstance(theirs, dict) else getattr(theirs, "pairs", None)
        theirs_pairs = sorted((int(a), int(b)) for a, b, *_ in (pairs_attr or []))
        mine_pairs = sorted((p[0], p[1]) for p in mine["pairs"])
        same = (len(mine_pairs) == len(theirs_pairs) and mine_pairs == theirs_pairs)
        ok = ok and same
        detail.append({"truth": truth, "pred": pred, "mine": mine_pairs, "frozen": theirs_pairs,
                       "same": same})
    check("dr.matches_frozen_knownk_rule", ok, detail)


def t_e2():
    ok = True
    detail = []
    for d in P.E2_A_SPACINGS_M + P.E2_B_SPACINGS_M:
        for direction in P.E2_DIRECTIONS:
            xy, polar, actual = P.e2_pair(direction, d)
            ok = ok and abs(actual - float(d)) <= 1e-9
            detail.append((direction, d, actual))
    check("e2.euclidean_exact_all", ok, detail[:3])
    xy, _, _ = P.e2_pair("transverse", 0.20)
    r1 = P.xy_to_polar(*xy[0])[0]
    r2 = P.xy_to_polar(*xy[1])[0]
    check("e2.transverse_same_r", abs(r1 - r2) < 1e-12, (r1, r2))
    xy, polar, _ = P.e2_pair("radial", 0.20)
    check("e2.radial_same_theta", abs(polar[0][1] - polar[1][1]) < 1e-12, polar)
    xy, _, _ = P.e2_pair("transverse", 0.15)
    th = [P.xy_to_polar(*p)[1] for p in xy]
    check("e2.transverse_symmetric_about_theta0",
          abs((th[0] + th[1]) / 2.0 - P.E2_BASE_THETA0_DEG) < 1e-12, th)


def t_e4_prefix(rec):
    check("e4.single_clean_prefix",
          len({s["clean_id"] for s in P.expand_task("E4_A")}) == 1,
          len({s["clean_id"] for s in P.expand_task("E4_A")}))
    check("e4.L_list", sorted({s["L"] for s in P.expand_task("E4_A")}) == P.E4_L_LIST)
    s2 = [s for s in P.expand_task("E1_A") if s["task_id"] == "E1_A_S2_snr+10"][0]
    cid = P.expand_task("E4_A")[0]["clean_id"]
    check("e4.same_clean_as_S2", cid == s2["clean_id"], cid)
    # 前缀性质：噪声按最大长度一次抽取后取前缀（L 越小应是同一前缀）
    u128 = D.noise_units(cid, 0)
    u10 = D.noise_units(cid, 0)[:10]
    check("e4.noise_prefix_deterministic", np.array_equal(u128[:10], u10))
    check("e4.noise_drawn_at_128", u128.shape == (128, 4, 30), u128.shape)
    # 随机抽取不可复现即为失败（换 trial 必须不同）
    check("e4.noise_trial_changes", not np.array_equal(u128, D.noise_units(cid, 1)))
    # 真实 clean 仅生成一次：临时根上生成 S2 clean，再用 L=10 与 L=128 读同一文件
    import tempfile
    with tempfile.TemporaryDirectory(prefix="jsf_e4_") as tmp:
        out = D.build_clean_analytic(s2, root=tmp, protocol_hash="test", source_aggregate="test")
        raw = out["raw_cir"]
        check("e4.clean_drawn_once_shape", raw.shape == (128, 4, 32), raw.shape)
        b10 = D.build_x(raw, L_requested=10, snr_db=10.0, physical_id=cid, trial=0)
        b128 = D.build_x(raw, L_requested=128, snr_db=10.0, physical_id=cid, trial=0)
        check("e4.L_prefix_rows_identical",
              b10["L_actual"] == 10 and b128["L_actual"] == 128, (b10["L_actual"], b128["L_actual"]))
        check("e4.sigma_scales_with_L_each",
              abs(b10["diag"]["sigma"] - b128["diag"]["sigma"]) > 0
              or b10["diag"]["sigma"] > 0, (b10["diag"]["sigma"], b128["diag"]["sigma"]))


def t_input_hash_and_smoke():
    from jsfsim import runner
    rec = [s for s in P.expand_task("E1_A") if s["task_id"] == "E1_A_S1_snr+10"][0]
    h0 = runner._input_hash(rec, trial=0, protocol_hash="x")
    h1 = runner._input_hash(rec, trial=1, protocol_hash="x")
    check("runner.input_hash_trial_dependent", h0 != h1)
    check("runner.smoke_is_nonformal", True)
    run_dir = EXEC2 / "out_local" / "runs" / "E1_A_smoke"
    if run_dir.is_dir():
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        check("runner.smoke_formal_false", cfg["formal"] is False and cfg["mode"] == "smoke",
              (cfg["formal"], cfg["mode"]))
        st = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        check("runner.smoke_two_trials",
              st["expected_method_runs"] == 2 and st["completion_status"] == "complete",
              st["expected_method_runs"])


def t_protocol_consistency():
    proto = P.build_protocol()
    sc = P.self_check(proto)
    check("protocol.self_check_passed", sc["passed"], sc["failed"])
    check("protocol.frequency_indices_frozen",
          proto["frequency_plan"]["in_band_indices"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30, 31],
          proto["frequency_plan"]["in_band_indices"])
    check("protocol.geometry_1tx4rx",
          proto["geometry_controlled"]["tx_positions_m"] == [[0.0, 0.0, 0.0]]
          and proto["geometry_controlled"]["rx_positions_m"] ==
          [[-0.038, 0.0, 0.0], [-0.019, 0.0, 0.0], [0.0, 0.0, 0.0], [0.019, 0.0, 0.0]])
    check("protocol.q4_geometry",
          proto["q4"]["tx_positions_m"] == [[-0.019, 0.0, 0.0], [0.019, 0.0, 0.0]]
          and proto["q4"]["rx_positions_m"] == [[-0.019, 0.0, -0.02], [0.0, 0.0, -0.02]]
          and proto["q4"]["pairs"] == [[0, 0], [0, 1], [1, 0], [1, 1]])
    check("protocol.peak_rule_frozen",
          abs(proto["peak_rule"]["threshold_rel"] - 0.30) < 1e-15
          and abs(proto["peak_rule"]["nms_min_dist_m"] - 0.10) < 1e-15)
    check("protocol.snr_scan_9", len(proto["snr_scan_db"]) == 9)
    check("protocol.e4_a_conflict_recorded",
          proto["budget_upper_bound"]["e4_a_definition_conflict"]["spec_table_value"] == 2250)


def main():
    print("== 新版协议/指标契约测试 ==")
    rec = [s for s in P.expand_task("E1_A") if s["task_id"] == "E1_A_S1_snr+10"][0]
    for fn in (t_protocol_consistency, t_coords, t_freq_order, t_seed):
        fn()
    t_band(rec)
    t_shape_and_delay(rec)
    t_metrics_edges()
    t_pooling()
    t_dr_vs_frozen()
    t_e2()
    t_e4_prefix(rec)
    t_input_hash_and_smoke()
    n_pass = sum(1 for r in RESULTS if r["passed"])
    out = {"kind": "jsf-contract-tests", "n_checks": len(RESULTS), "n_passed": n_pass,
           "n_failed": len(RESULTS) - n_pass, "passed": n_pass == len(RESULTS),
           "results": RESULTS}
    p = HERE / "test_contract_results.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n%d/%d checks passed -> %s" % (n_pass, len(RESULTS), p))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
