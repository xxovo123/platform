#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线单元测试（**不需要 Meep**）：脉冲/链路/对齐/轨迹/缓存键/真值隔离。

运行： ``python tests/test_chain.py``（在 fdtd/ 目录内，或设 PYTHONPATH 指向其父目录）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG_PARENT = Path(__file__).resolve().parent.parent.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from fdtd import cache as cache_mod  # noqa: E402
from fdtd import protocol, pulse, rf_chain, trajectory  # noqa: E402

C0 = 2.99792458e8
FC = 7.9872e9
B = 5.0e8


def _full_rate_axis(res_px_per_m: float = 250.0, run_time_meep: float = 20.0):
    """与 Meep 2D 同口径的全速率轴（dt = Courant/res，单位长度 1 m）。"""
    dt_meep = (1.0 / np.sqrt(2.0)) / res_px_per_m
    dt_s = dt_meep / C0
    n = int(run_time_meep / dt_meep) + 1
    return np.arange(n, dtype=float) * dt_s, dt_s


def test_pulse_formula_matches_legacy():
    """源与 MF 参考同公式：逐点对比旧实现的字面公式。"""
    t = np.linspace(0.0, 8e-9, 4001)
    duration = pulse.duration_from_bandwidth(B)
    assert abs(duration - 4e-9) < 1e-18
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    t0 = duration / 2.0
    t_rel = t - t0 + duration / 2.0
    inside = np.where((t >= 0) & (t <= duration), 1, 0)
    arg = 2 * np.pi * t_rel / duration
    w = (a0 - a1 * np.cos(arg) + a2 * np.cos(2 * arg) - a3 * np.cos(3 * arg)) * inside
    dw = (2 * np.pi / duration) * (a1 * np.sin(arg) - 2 * a2 * np.sin(2 * arg) + 3 * a3 * np.sin(3 * arg)) * inside
    carrier = np.cos(2 * np.pi * FC * (t - t0))
    carrier_deriv = -2 * np.pi * FC * np.sin(2 * np.pi * FC * (t - t0))
    legacy = dw * carrier + w * carrier_deriv
    ours = pulse.bh_derivative_passband(t, FC, duration)
    err = float(np.max(np.abs(ours - legacy)))
    scale = float(np.max(np.abs(legacy)))
    rel = err / scale
    # 公式逐项一致；残差只来自浮点结合顺序（O(ULP)），不做“逐位相同”的假声明
    assert rel < 1e-12, "与旧公式相对偏差过大：max|diff|=%.3e (rel=%.3e)" % (err, rel)
    # 标量输入（Meep CustomSource 需要）
    assert isinstance(pulse.bh_derivative_passband(1e-9, FC, duration), float)
    return {"max_abs_diff_vs_legacy": err, "rel_diff_vs_legacy": rel, "same_formula_terms": True}


def test_closed_form_reference_matches_hilbert():
    """闭式解析基带与 Hilbert 路径的量化差异（不假定一致）。"""
    t, _ = _full_rate_axis(run_time_meep=8.0)
    duration = pulse.duration_from_bandwidth(B)
    pb = pulse.bh_derivative_passband(t, FC, duration)
    via_hilbert = rf_chain.hilbert_iq(pb, t, FC)
    closed = pulse.bh_derivative_analytic_baseband(t, FC, duration)
    scale = float(np.max(np.abs(closed)))
    rel = float(np.max(np.abs(closed - via_hilbert))) / scale
    assert rel < 0.05, "闭式与 Hilbert 相对偏差过大：%.4f" % rel
    return {"rel_max_abs_diff": rel}


def test_exact_1ns_axis_and_legacy_drift():
    """新版精确 1 ns 轴；旧链路步长相对偏差约 1%（换算到 tap 上是 0.01·k）。"""
    t_full, dt_s = _full_rate_axis()
    t_out = rf_chain.output_time_axis(t_full, 1e-9)
    assert np.allclose(np.diff(t_out), 1e-9, rtol=1e-12, atol=0.0)
    assert t_out[0] == 0.0

    legacy_T = float(t_full[-1])
    legacy_n = int(legacy_T * 1e9)
    legacy_axis = np.linspace(0.0, legacy_T, legacy_n, endpoint=False)
    legacy_dt = float(legacy_axis[1] - legacy_axis[0])
    rel_err = legacy_dt / 1e-9 - 1.0
    tap32 = rel_err * 32
    assert 0.005 < rel_err < 0.02, "旧步长相对偏差应在 ~1%% 量级，实测 %.4f" % rel_err
    assert 0.2 < tap32 < 0.5, "旧偏差换算到 k=32 应约 0.32 tap，实测 %.3f" % tap32
    return {
        "legacy_n_samples": legacy_n,
        "legacy_dt_s": legacy_dt,
        "legacy_relative_error": rel_err,
        "legacy_error_at_tap32": tap32,
        "new_axis_exact_1ns": True,
    }


def test_matched_filter_multidim_delay():
    """多维（4D）输入 + 1D 参考：按物理 filter lag 对齐，峰位 = 注入时延。"""
    rng = np.random.default_rng(0)
    n_ref = 8
    ref = (rng.normal(size=n_ref) + 1j * rng.normal(size=n_ref)).astype(complex)
    n_time = 40
    delay = 11
    x = np.zeros((2, 3, 4, n_time), dtype=complex)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                x[i, j, k, delay : delay + n_ref] = ref * (1.0 + 0.1 * (i + j + k))
    aligned, lag = rf_chain.matched_filter_align(x, ref)
    assert lag == n_ref - 1
    assert aligned.shape == (2, 3, 4, n_time)
    peaks = np.abs(aligned).argmax(axis=-1)
    assert np.all(peaks == delay), "多维对齐峰位应为 %d，实测 %s" % (delay, peaks.ravel()[:8])
    return {"lag_samples": lag, "peak_index": int(delay)}


def test_antialias_rejection_and_passband():
    """显式抗混叠：带内 200 MHz 通过；900 MHz（> cutoff 400 MHz、> Nyquist 500 MHz）被压制。"""
    t_full, dt_s = _full_rate_axis()
    fs_in = 1.0 / dt_s
    f_pass, f_stop = 2e8, 9e8
    x = np.exp(2j * np.pi * f_pass * t_full) + 0.9 * np.exp(2j * np.pi * f_stop * t_full)
    t_out = rf_chain.output_time_axis(t_full, 1e-9)
    y = rf_chain.band_limited_resample(x, fs_in, t_out, 4e8, 2.0, t_in0_s=float(t_full[0]))
    # 复数单音：幅度 = |mean(y·e^{-j2πft})|（不是实余弦的 2 倍）
    amp_pass = float(np.abs(np.mean(y * np.exp(-2j * np.pi * f_pass * t_out))))
    amp_stop = float(np.abs(np.mean(y * np.exp(-2j * np.pi * f_stop * t_out))))
    stop_db = 20 * np.log10(max(amp_stop, 1e-300) / 0.9)
    assert 0.9 < amp_pass < 1.1, "带内幅度应≈1，实测 %.4f" % amp_pass
    assert stop_db < -40.0, "900 MHz 应被压制到 ≤ -40 dB，实测 %.2f dB" % stop_db
    return {"pass_amplitude": amp_pass, "stop_attenuation_db": stop_db}


def test_chain_param_validation():
    """fs / Nyquist / 截止参数非法时显式报错。"""
    base = dict(protocol.PHYSICS_DEFAULTS)
    rf_chain.validate_chain_params(base, fs_in_hz=1e11)
    bad1 = dict(base)
    bad1["aa_cutoff_hz"] = 5.5e8  # >= fs_out/2 = 500 MHz
    try:
        rf_chain.validate_chain_params(bad1, fs_in_hz=1e11)
        raise AssertionError("aa_cutoff >= Nyquist 应报错")
    except ValueError:
        pass
    bad2 = dict(base)
    bad2["aa_cutoff_hz"] = 0.6e8
    try:
        rf_chain.validate_chain_params(bad2, fs_in_hz=1.0e8)  # 输入 Nyquist 只有 50 MHz
        raise AssertionError("cutoff >= fs_in/2 应报错")
    except ValueError:
        pass
    bad3 = dict(base)
    bad3["sample_interval_s"] = -1e-9
    try:
        rf_chain.validate_chain_params(bad3)
        raise AssertionError("非正 sample_interval_s 应报错")
    except ValueError:
        pass
    return {"raised": 3}


def test_trajectory_reproducible_unique_128():
    """seed 可复现 + 128 个状态两两不同（不复制快照）。"""
    spec = {"model": trajectory.MODEL_ID, "seed": 20260926, "sigma_m": 0.01, "radius_m": 0.10, "L": 128}
    centers = [[-0.374, 1.761], [0.374, 1.761]]
    traj_a = trajectory.build_trajectory(spec, centers)
    traj_b = trajectory.build_trajectory(spec, centers)
    assert np.array_equal(traj_a, traj_b), "同 seed 必须逐位可复现"
    hashes = trajectory.assert_unique_states(traj_a)
    assert len(set(hashes)) == 128
    spec2 = dict(spec, seed=20260927)
    traj_c = trajectory.build_trajectory(spec2, centers)
    assert not np.array_equal(traj_a, traj_c), "不同 seed 应给出不同轨迹"
    assert np.max(np.linalg.norm(traj_a - np.asarray(centers)[None, :, :], axis=-1)) <= 0.10 + 1e-12
    return {"n_states": 128, "unique": True}


def _toy_scene():
    return {
        "scene_id": "toy",
        "dimensions": 2,
        "length_unit_m": 1.0,
        "geometry": {
            "tx_positions_m": [[0.0, 0.0, 0.0]],
            "rx_positions_m": [[-0.038, 0.0, 0.0], [-0.019, 0.0, 0.0], [0.0, 0.0, 0.0], [0.019, 0.0, 0.0]],
            "pairs": [[0, 0], [0, 1], [0, 2], [0, 3]],
        },
        "targets": {"nominal_centers_m": [[0.0, 1.5, 0.0]], "radius_m": 0.056, "target_scale_m": 0.056},
        "meep": {"resolution_px_per_m": 250, "run_time_meep": 20.0, "pml_thickness_lambda": 2.0},
    }


def test_cache_key_sensitivity_including_subpicosecond():
    """cache key 必须对**微小但物理上不同**的参数敏感，包括亚皮秒时间与 fc 尾数。"""
    scene = _toy_scene()
    cfg = dict(protocol.PHYSICS_DEFAULTS)
    k0 = cache_mod.cache_key(cfg, scene, "traj-hash-A", "empty_field")
    assert k0 == cache_mod.cache_key(dict(cfg), dict(scene), "traj-hash-A", "empty_field")
    assert k0 != cache_mod.cache_key(cfg, scene, "traj-hash-B", "empty_field")
    assert k0 != cache_mod.cache_key(cfg, scene, "traj-hash-A", "ema_windowed")
    tiny = dict(cfg, sample_interval_s=1e-13)  # 若做过 round(...,12) 会被归零并碰撞
    zero = dict(cfg, sample_interval_s=0.0)
    assert cache_mod.cache_key(tiny, scene, "traj-hash-A", "empty_field") != cache_mod.cache_key(
        zero, scene, "traj-hash-A", "empty_field"
    )
    assert protocol.sha256_hex({"dt": 1e-13}) != protocol.sha256_hex({"dt": 0.0})
    assert protocol.sha256_hex({"fc": 7.9872e9}) != protocol.sha256_hex({"fc": 7.9872e9 + 1e-3})
    assert protocol.sha256_hex({"r": 1e-13}) != protocol.sha256_hex({"r": 1e-13 * (1 + 1e-15)})
    return {"key_prefix": k0[:16]}


def test_clean_case_whitelist():
    """clean case 不得含真值/诊断字段。"""
    protocol.assert_clean_case(["raw_cir", "fc_hz", "t_origin_s", "background_mode"])
    for bad in ("truth_positions", "targets_xy_m", "trajectory_xy_m", "diagnostic_peak_tap", "expected_delay_s"):
        try:
            protocol.assert_clean_case(["raw_cir", bad])
        except ValueError:
            continue
        raise AssertionError("应拒绝真值/诊断键：%s" % bad)
    return {"rejected_markers": 5}


def test_records_to_aligned_cir_end_to_end():
    """端到端（无 Meep）：注入已知时延的回波，raw32 峰位 = 时延；keep_taps 不足须报错。"""
    t_full, dt_s = _full_rate_axis(run_time_meep=20.0)
    cfg = dict(protocol.PHYSICS_DEFAULTS)
    cfg["aa_cutoff_hz"] = 4e8
    delay_s = 12e-9
    duration = pulse.duration_from_bandwidth(B)
    echo = pulse.bh_derivative_passband(t_full - delay_s, FC, duration)
    records = echo.reshape(1, 1, 1, -1)  # [L=1, n_tx=1, n_rx=1, n_time]
    aligned, info = rf_chain.records_to_aligned_cir(
        records, None, t_full, cfg, keep_taps=32, background_mode="empty_field", ema_alpha=None
    ) if False else (None, None)
    # empty_field 需要 bg；用 0 背景等价于原始记录
    bg = np.zeros_like(records[0])
    cir, info = rf_chain.records_to_aligned_cir(records, bg, t_full, cfg, keep_taps=32)
    assert cir.shape == (1, 1, 1, 32)
    peak = int(np.abs(cir[0, 0, 0]).argmax())
    assert 11 <= peak <= 13, "回波时延 %.0f ns 应对齐到 tap≈12，实测 %d" % (delay_s * 1e9, peak)
    assert info["mf_lag_samples"] == info["mf_ref_len_samples"] - 1
    try:
        rf_chain.records_to_aligned_cir(records, bg, t_full, cfg, keep_taps=999)
        raise AssertionError("keep_taps 超过可用 tap 数时应报错，而不是静默截短")
    except ValueError:
        pass
    return {"peak_tap": peak, "n_out": info["n_out"], "mf_lag": info["mf_lag_samples"]}


def test_state_ema_matches_hardware_semantics():
    """EMA 语义与 b0.cfr_replay 一致：首样本=原值，其后 cur-ref，ref 更新用 alpha。"""
    cir = np.arange(1, 7, dtype=float).reshape(6, 1, 1)
    alpha = 0.25
    out = rf_chain.apply_state_ema(cir, alpha, first_raw=True, burnin=0)
    assert out.shape[0] == 6
    assert out[0, 0, 0] == 1.0
    ref = cir[0].copy()
    expect = []
    for l in range(1, 6):
        expect.append(cir[l] - ref)
        ref = (1 - alpha) * ref + alpha * cir[l]
    assert np.allclose(np.asarray(expect)[:, 0, 0], out[1:, 0, 0])
    out_b = rf_chain.apply_state_ema(cir, alpha, first_raw=True, burnin=2)
    assert out_b.shape[0] == 4 and np.allclose(out_b, out[2:])
    try:
        rf_chain.apply_state_ema(cir, alpha, first_raw=True, burnin=6)
        raise AssertionError("burnin >= L 应报错")
    except ValueError:
        pass
    return {"alpha": alpha}


def test_resample_absolute_time_nonzero_t0():
    """非零输入起点：重采样必须按**绝对时间**求值（相位用 t_out - t_in0）。"""
    dt_s = 1.0 / 1e11            # fs_in = 100 GHz
    t0 = 1e-9                    # 首样本绝对时间 = 1 ns（模拟 at_every 首回调 t=dt）
    n = 4000
    t_full = t0 + np.arange(n) * dt_s
    f = 2e8                      # 200 MHz，低于 cutoff(400 MHz) 与 Nyquist
    x = np.exp(2j * np.pi * f * t_full)
    # 取值必须落在**记录时间跨度内**（t0..t0+40 ns）：带限插值只在记录区间内有效；
    # 取 10–29 ns 的中段，避开记录两端的截断边缘（那是有限记录的固有性质，与 t0 处理无关）。
    t_out = t0 + (10 + np.arange(20)) * 1e-9
    y = rf_chain.band_limited_resample(x, 1.0 / dt_s, t_out, 4e8, 2.0, t_in0_s=t0)
    expect = np.exp(2j * np.pi * f * t_out)
    rel = float(np.max(np.abs(y - expect)) / np.max(np.abs(expect)))
    y_bad = rf_chain.band_limited_resample(x, 1.0 / dt_s, t_out, 4e8, 2.0, t_in0_s=0.0)
    rel_bad = float(np.max(np.abs(y_bad - expect)) / np.max(np.abs(expect)))
    # 修正后：仅剩有限记录/带限截断的百分之几残差；
    # 若相位漏掉 -t_in0（t_in0_s=0.0）：整体多移 t0，相对误差 ~1（2π f t0 ≈ 1.26 rad）。
    assert rel < 0.05, "按绝对时间求值失败：rel=%.4f" % rel
    assert rel_bad > 0.5, "对照（不改 t0）应出现明显偏差，实测 rel_bad=%.4f" % rel_bad
    return {"t0_s": t0, "f_hz": f, "rel_err_fixed": rel, "rel_err_without_t0_correction": rel_bad}


def test_mf_lag_nonzero_t0_known_integer_delay():
    """非零 RF 起点 + 已知整数时延：MF 对齐后 tap0 = zero lag，峰位 = 注入的整数延迟。"""
    t_full0, dt_s = _full_rate_axis(run_time_meep=20.0)
    t0 = 1e-9
    t_full = t_full0 + t0
    cfg = dict(protocol.PHYSICS_DEFAULTS)
    d_taps = 12
    duration = pulse.duration_from_bandwidth(B)
    echo = pulse.bh_derivative_passband(t_full - d_taps * 1e-9, FC, duration)
    records = echo.reshape(1, 1, 1, -1)
    bg = np.zeros_like(records[0])
    cir, info = rf_chain.records_to_aligned_cir(records, bg, t_full, cfg, keep_taps=32)
    peak = int(np.abs(cir[0, 0, 0]).argmax())
    assert peak == d_taps, "峰位应为注入的整数延迟 %d，实测 %d（不得按预期峰移轴）" % (d_taps, peak)
    assert abs(info["rf_absolute_start_s"] - t0) < 1e-15
    assert abs(info["output_grid_start_absolute_s"] - t0) < 1e-15
    return {"peak_tap": peak, "rf_absolute_start_s": info["rf_absolute_start_s"]}


def test_outdir_guard_rejects_other_key():
    """outdir 保护：已有不同 cache key 产物时开头即拒（不在结尾覆盖 clean）。"""
    from fdtd import api

    import shutil

    # 写在工作区内的临时目录（不依赖系统 temp，避免沙箱/权限差异）
    tmp_root = Path(__file__).resolve().parent / "_tmp_outdir_guard"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    outdir = tmp_root / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    key_a, key_b = "a" * 64, "b" * 64
    api.assert_outdir_available(outdir, key_a)
    (outdir / ".cache_key").write_text(key_a + "\n", encoding="utf-8")
    api.assert_outdir_available(outdir, key_a)
    raised = False
    try:
        api.assert_outdir_available(outdir, key_b)
    except RuntimeError:
        raised = True
    assert raised, "不同 cache key 必须开头即拒"
    shutil.rmtree(tmp_root, ignore_errors=True)
    return {"same_key_allowed": True, "other_key_refused": True}


TESTS = [
    ("pulse_formula_matches_legacy", test_pulse_formula_matches_legacy),
    ("closed_form_vs_hilbert_reference", test_closed_form_reference_matches_hilbert),
    ("exact_1ns_axis_and_legacy_drift", test_exact_1ns_axis_and_legacy_drift),
    ("matched_filter_multidim_delay", test_matched_filter_multidim_delay),
    ("antialias_rejection_and_passband", test_antialias_rejection_and_passband),
    ("chain_param_validation", test_chain_param_validation),
    ("trajectory_reproducible_unique_128", test_trajectory_reproducible_unique_128),
    ("cache_key_sensitivity_subps", test_cache_key_sensitivity_including_subpicosecond),
    ("clean_case_whitelist", test_clean_case_whitelist),
    ("records_to_aligned_cir_end_to_end", test_records_to_aligned_cir_end_to_end),
    ("state_ema_hardware_semantics", test_state_ema_matches_hardware_semantics),
    ("outdir_guard_rejects_other_key", test_outdir_guard_rejects_other_key),
    ("resample_absolute_time_nonzero_t0", test_resample_absolute_time_nonzero_t0),
    ("mf_lag_nonzero_t0_integer_delay", test_mf_lag_nonzero_t0_known_integer_delay),
]


def main() -> int:
    failures = []
    for name, fn in TESTS:
        try:
            info = fn()
            print("PASS %-38s %s" % (name, info))
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)))
            print("FAIL %-38s %s" % (name, repr(exc)))
    print("\n%d/%d passed" % (len(TESTS) - len(failures), len(TESTS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
