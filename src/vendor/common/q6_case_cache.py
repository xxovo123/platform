#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公共硬件 case 缓存：32 录制一次扫描 → 214 窗 → (X, frequencies, pairs, grid, config)。

口径（全部经实测确认，见 hardware/logs/tap_convention_check.json）
------------------------------------------------------------------------
* **每录制一次扫描**：b0.replay_windows 一次读完整个 .bin，同时抓取该录制的所有目标窗
  （与 B0 逐格一致的重放实现；背景 EMA 跨窗连续，不每窗重置）。
* **tap 约定**：先按 meta.algo_params['2D-MUSIC'].cir_offset 截 tap
  （cropped = arr[:, :, cir_offset:ft_len]），再把绝对 cir_window=(cir_offset, ft_len)
  交给硬件函数（后者只用它做**距离环掩膜** r_start/r_end）。这一"截 tap + 掩膜"组合已用
  历史缓存逐格核对：rel=0.0 逐格相同；不截 tap 则差 0.78（三窗一致）。
  → 因此不能把 cir_offset 说成"只搜索掩膜、不截 tap"。
* **无额外延时补偿**：system_delay_si 用 meta 值（本批 0.0），不额外加偏移。
* **X 定义**：(L, F*C) 复数观测矩阵，**频率优先、通道其次**（index = f*C + ch），
  由 cropped CIR 经 fft(n=fft_length) 取带内频点、并在**数据侧**乘 exp(-1j*phi)
  完成通道相位校准（与硬件实现同一选择，FB 下严格正确）。
* **真值隔离**：真值只写入 hardware/truth/truth_index.json，**不进入 case**；
  方法可见的 case 由 load_case 显式构造，且 q6_contract.validate_case 会拒绝真值字段。

用法::

    py = .venv/Scripts/python.exe
    & $py -B 大修执行_20260923/04_Q6/common/q6_case_cache.py build --out hardware/cache
    & $py -B 大修执行_20260923/04_Q6/common/q6_case_cache.py check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# BLAS 线程固定：必须在 import numpy 之前设置（同 CPU/线程口径）
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

WS = Path(__file__).resolve().parents[3]
B0_DIR = WS / "大修执行_20260923" / "04_Q6" / "B0_同模型校验"
P0B_DIR = WS / "大修执行_20260923" / "01_投稿口径" / "p0b"
P0B_DRIVER = P0B_DIR / "driver"
PROTOCOL = P0B_DIR / "protocol" / "knownk_protocol.json"
SRC_DIR = WS / "Q2 数据采集" / "uwb-occupancy-detection-main"
DATA_DIR = WS / "Q2 数据采集" / "data_in_adult"
EVAL_DATA = WS / "Q2 数据采集" / "q2q3_eval_20260922" / "data"
HARDWARE_DIR = WS / "大修执行_20260923" / "04_Q6" / "hardware"

for _p in (str(B0_DIR), str(P0B_DRIVER), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b0  # noqa: E402
from b0 import standard_music  # noqa: E402

FEATURE_ORDER = "frequency-major then channel (index = f*C + ch)"
RING = 128


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blas_env() -> dict:
    return {v: os.environ.get(v) for v in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def _jsonable(obj):
    """把 case 里的 numpy 值转成可 JSON 序列化形式（只用于索引/配置落盘）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def p0a_windows() -> dict:
    """P0a 底层 oracle+nms 分支的窗清单（214 窗），按 window_id 索引。"""
    from knownk_eval.sources import load_per_window_branch

    return load_per_window_branch(EVAL_DATA)


def prepare_recording(recording_id: str) -> dict:
    """录制的静态上下文：meta / 真值 / 通道 / 配置 / 网格 / 搜索域。"""
    bin_path = DATA_DIR / ("%s.bin" % recording_id)
    meta = b0.load_meta(bin_path)
    collection = meta.get("collection") or {}
    truth = [tuple(float(v) for v in pt) for pt in (collection.get("truth_points") or [])]
    k = len(truth)
    mp = ((meta.get("algo_params") or {}).get("2D-MUSIC") or {})
    pairs, _vpos = b0.virtual_pairs(meta["tx_positions"], meta["rx_positions"],
                                    mp.get("virt_tx_idx") or [0, 1],
                                    mp.get("virt_rx_idx") or [2, 3],
                                    use_phase_center_convention=False)
    cfg = b0.config_from_meta(meta, k=k, pairs=pairs)
    spec = meta["imaging_grid_spec"]
    grid_x = np.linspace(float(spec[0]), float(spec[1]), int(spec[2]))
    grid_y = np.linspace(float(spec[3]), float(spec[4]), int(spec[5]))
    axis, low, high, freq_idx = standard_music.in_band_indices(cfg)
    return {
        "recording_id": recording_id,
        "bin_path": bin_path,
        "meta": meta,
        "environment": collection.get("environment"),
        "config_id": collection.get("config_id"),
        "truth": truth,
        "k": int(k),
        "pairs": [tuple(int(v) for v in p) for p in pairs],
        "cfg": cfg,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "frequencies_hz": np.asarray(axis[freq_idx], dtype=float),
        "freq_range_si": [float(low), float(high)],
        "freq_axis_full": np.asarray(axis, dtype=float),
        "flip_lr": bool(mp.get("flip_lr")),
        "ant_calib_en": bool(mp.get("ant_calib_en")),
        "mdl_max_targets": int(mp.get("mdl_max_targets") or 3),
    }


def _one_dimensional_budget(ctx: dict) -> dict:
    """1-D 预算（供原生分步/连续方法对其自有网格；**不是要求所有方法用同一网格**）。

    推导（写定，不按结果调整）：
      * 两 Rx 孔径（本批 RX2/RX3，同 TX）d = |p_rx2 - p_rx3| = 0.019 m；
        最大 TDOA = d / c ≈ 63.4 ps。
      * 5 cm 距离格 ↔ 双程 2*0.05/c ≈ 333.6 ps —— 比孔径 TDOA 尺度粗约 5 倍，
        因此在 5 cm 格上做 1-D TOA/τ 搜索**无法承载角度信息**。
      * 共同角度颗粒度预算取"5 cm 空间格在 r = 1 m 处的横向步长"对应的角度
        Δθ = atan(0.05/1) ≈ 2.86°，对应差分时延 Δτ = d*Δ(sinθ)/c ≈ 3.17 ps。
      * 于是 1-D τ/TOA 搜索若要与二维候选域的角度颗粒度同量级，步长须 <= ~3.2 ps
        （约 1/105 的 5 cm 距离格步长）；这**不**代表精度会因此提高，只是网格不再成为限制。
    """
    c0 = 299792458.0
    rx = np.asarray(ctx["meta"]["rx_positions"], dtype=float)
    pairs = np.asarray(ctx["pairs"], dtype=int)
    same_tx = {}
    for tx_idx, rx_idx in pairs:
        same_tx.setdefault(int(tx_idx), []).append(int(rx_idx))
    best = None
    for tx_idx, rx_list in sorted(same_tx.items()):
        if len(rx_list) < 2:
            continue
        rx_list = sorted(set(rx_list))
        for i in range(len(rx_list)):
            for j in range(i + 1, len(rx_list)):
                d = float(np.linalg.norm(rx[rx_list[i]] - rx[rx_list[j]]))
                cand = {"tx_idx": tx_idx, "rx_idx_list": [rx_list[i], rx_list[j]],
                        "baseline_m": d, "max_tdoa_s": d / c0}
                if best is None or cand["baseline_m"] > best["baseline_m"]:
                    best = cand
    dist_per_tap = (float(ctx["cfg"].tap_interval_si) * c0) / 2.0
    r_start = float(ctx["cfg"].cir_window[0]) * dist_per_tap
    r_end = float(ctx["cfg"].cir_window[1]) * dist_per_tap
    return {
        "two_rx_baseline": best,
        "range_grid_5cm_two_way_tau_s": 2.0 * 0.05 / c0,
        "angle_granularity_budget_deg_at_r1m": float(np.degrees(np.arctan2(0.05, 1.0))),
        "differential_tau_budget_s": (best["baseline_m"] * (0.05 / 1.0) / c0) if best else None,
        "suggested_one_d_tau_step_s": (best["baseline_m"] * (0.05 / 1.0) / c0) if best else None,
        "distance_ring_m": [r_start, r_end],
        "round_trip_tau_span_s": [2.0 * r_start / c0, 2.0 * r_end / c0],
        "note": "本预算只用于**披露**原生一维网格的粒度与代价；不要求所有方法使用同一网格，"
                "也不据此宣布精度差异；方法须按原生结构搜索并在 diagnostics 申报实际点数。",
    }


def build_case_dict(ctx: dict, window_id: str, t_end_s: float, X: np.ndarray,
                    snapshots: list) -> dict:
    cfg = ctx["cfg"]
    n_snapshots, n_features = int(X.shape[0]), int(X.shape[1])
    n_freq = int(ctx["frequencies_hz"].size)
    n_ch = len(ctx["pairs"])
    spec = ctx["meta"]["imaging_grid_spec"]
    budget = _one_dimensional_budget(ctx)
    config = {
        "fft_length": int(cfg.fft_length),
        "tap_interval_si": float(cfg.tap_interval_si),
        "center_freq_si": float(cfg.center_freq_si),
        "bandwidth_si": float(cfg.bandwidth_si),
        "epsilon_r_background": float(cfg.epsilon_r_background),
        "system_delay_si": float(cfg.system_delay_si),
        "diag_load_factor": float(cfg.diag_load_factor),
        "is_forward_backward": bool(cfg.is_forward_backward),
        "fov_degrees": float(cfg.fov_degrees),
        "cir_window": [int(v) for v in cfg.cir_window],
        "freq_range_si": list(ctx["freq_range_si"]),
        "deg_corr": None if cfg.deg_corr is None else list(cfg.deg_corr),
        "ant_calib_phase": (None if cfg.ant_calib_phase is None
                            else [float(v) for v in cfg.ant_calib_phase]),
        "ant_calib_applied_in_X": "data side exp(-1j*phi)（与硬件实现同一选择，FB 下严格正确）",
        "propagation_speed_m_s": 299792458.0 / float(np.sqrt(float(cfg.epsilon_r_background))),
        "distance_ring_m": budget["distance_ring_m"],
        "tap_handling": "X 由 cropped tap（arr[:,:,cir_offset:ft_len]）经 fft(n=fft_length) 构造；"
                        "cir_window 仅用于距离环掩膜（已实测与历史缓存逐格一致）",
        "flip_lr": bool(ctx["flip_lr"]),
    }
    search = {
        "kind": "cartesian_2d",
        "grid_x_m": np.asarray(ctx["grid_x"], dtype=float),
        "grid_y_m": np.asarray(ctx["grid_y"], dtype=float),
        "grid_spec_source": list(spec),
        "resolution_m": float(ctx["grid_x"][1] - ctx["grid_x"][0]),
        "n_grid_points": int(ctx["grid_x"].size * ctx["grid_y"].size),
        "half_plane": "y < 0（与硬件实现一致）",
        "one_dimensional_budget": budget,
        "fairness_note": "候选域与分辨率预算统一披露；原生分步/连续方法按各自结构搜索，"
                         "不得声称所有方法在同一 Cartesian 点集上逐点比较。",
    }
    model = {
        "geometry": "exact two-way spherical range per channel: "
                    "tau = (||p - tx|| + ||p - rx||) / v_prop + system_delay",
        "response_assumption": {
            "mode": "S_equals_1_pure_phase",
            "vector": None,
            "is_calibrated_response": False,
            "note": "所有方法共用的纯相位（S=1）假设；**不是**已校准接收响应，"
                    "**不是**完全原论文复现；误差同时受该模型适配影响。"
                    "若方法自带已知非平坦 S/白化，须在 diagnostics 里声明并保持数据与导向一致。",
        },
        "channel_calibration": {
            "ant_calib_en": bool(ctx["ant_calib_en"]),
            "applied": "X 内已按数据侧 exp(-1j*phi) 校准",
            "deg_corr": None,
        },
        "fb_note": "FB（前后向平均）是**方法自身的处理选择**；case 只给未做 FB 的 X。"
                   "各方法按原文原生结构决定是否 FB；若使用 FB，须说明 J 与该方法自身的特征序一致"
                   "（X 特征序 = 频率优先、通道其次，J = 沿特征序翻转），且 L 仍表示原始慢时间快照数。",
        "frequency_grid": {
            "n": n_freq,
            "frequencies_hz": np.asarray(ctx["frequencies_hz"], dtype=float),
            "uniform": bool(np.allclose(np.diff(ctx["frequencies_hz"]),
                                        np.diff(ctx["frequencies_hz"])[0])),
            "spacing_hz": float(np.diff(ctx["frequencies_hz"])[0]),
            "axis": "np.fft.fftfreq(fft_length, tap_interval_si) + center_freq_si 的带内子集",
        },
    }
    return {
        "case_schema_version": 1,
        "id": window_id,
        "recording_id": ctx["recording_id"],
        "environment": ctx["environment"],
        "config_id": ctx["config_id"],
        "t_end_s": float(t_end_s),
        "X": np.asarray(X, dtype=np.complex128),
        "frequencies_hz": np.asarray(ctx["frequencies_hz"], dtype=float),
        "pairs": np.asarray(ctx["pairs"], dtype=np.int64),
        "tx_positions": np.asarray(ctx["meta"]["tx_positions"], dtype=float),
        "rx_positions": np.asarray(ctx["meta"]["rx_positions"], dtype=float),
        "k": int(ctx["k"]),
        "grid_x": np.asarray(ctx["grid_x"], dtype=float),
        "grid_y": np.asarray(ctx["grid_y"], dtype=float),
        "config": config,
        "search": search,
        "model": model,
        "feature_order": FEATURE_ORDER,
        "n_snapshots": n_snapshots,
        "n_features": n_features,
        "n_freq_bins": n_freq,
        "n_channels": n_ch,
        "snapshots_available": True,
        "snapshots": snapshots,
        "notes": [
            "case 不含真值（真值在 hardware/truth/truth_index.json，仅 evaluation 使用）",
            "k = 真值人数（known-K 口径，路线 §0.2）",
        ],
    }


def build_recording(recording_id: str, t_list: list, *, with_snapshots: bool = True):
    """一次扫描该录制，返回 (cases_by_window_id, npz_payload, build_info)。"""
    t0 = time.perf_counter()
    ctx = prepare_recording(recording_id)
    t_prepare = time.perf_counter() - t0
    t0 = time.perf_counter()
    captures = b0.replay_windows(ctx["bin_path"], ctx["meta"], t_list, ring=RING)
    t_replay = time.perf_counter() - t0
    t0 = time.perf_counter()
    cases, x_times = {}, []
    for t_end_s in t_list:
        capture = captures.get(float(t_end_s))
        if capture is None or capture.get("unavailable"):
            raise RuntimeError("录制 %s 的窗 t=%g 不可用：%s"
                               % (recording_id, t_end_s,
                                  (capture or {}).get("detail", "未出现在扫描结果中")))
        ts = time.perf_counter()
        cropped, n_taps_used = b0.cropped_snapshots(capture["arr"], int(ctx["cfg"].cir_window[0]))
        X, _info = standard_music.build_observation_matrix(cropped, ctx["cfg"])
        x_times.append(time.perf_counter() - ts)
        window_id = "%s__t%g" % (recording_id, float(t_end_s))
        case = build_case_dict(ctx, window_id, float(t_end_s), X,
                               cropped if with_snapshots else None)
        cases[window_id] = case
    t_features = time.perf_counter() - t0
    payload = {
        "grid_x": np.asarray(ctx["grid_x"], dtype=float),
        "grid_y": np.asarray(ctx["grid_y"], dtype=float),
        "frequencies_hz": np.asarray(ctx["frequencies_hz"], dtype=float),
        "freq_axis_full": np.asarray(ctx["freq_axis_full"], dtype=float),
        "pairs": np.asarray(ctx["pairs"], dtype=np.int64),
        "tx_positions": np.asarray(ctx["meta"]["tx_positions"], dtype=float),
        "rx_positions": np.asarray(ctx["meta"]["rx_positions"], dtype=float),
        "window_ids": np.array(sorted(cases), dtype=object),
        "t_end_s": np.array([cases[w]["t_end_s"] for w in sorted(cases)], dtype=float),
        "k": np.array([cases[w]["k"] for w in sorted(cases)], dtype=np.int64),
        "config_json": np.array(json.dumps(_jsonable(cases[sorted(cases)[0]]["config"]),
                                           ensure_ascii=False)),
        "model_json": np.array(json.dumps(_jsonable(cases[sorted(cases)[0]]["model"]),
                                          ensure_ascii=False)),
        "search_json": np.array(json.dumps(
            _jsonable({k: v for k, v in cases[sorted(cases)[0]]["search"].items()
                       if k not in ("grid_x_m", "grid_y_m")}), ensure_ascii=False)),
    }
    for window_id, case in cases.items():
        payload["X__%s" % window_id] = np.asarray(case["X"], dtype=np.complex128)
        payload["n_snapshots__%s" % window_id] = np.int64(case["n_snapshots"])
        payload["n_features__%s" % window_id] = np.int64(case["n_features"])
        if with_snapshots:
            payload["snapshots__%s" % window_id] = np.asarray(
                np.stack([np.asarray(s, dtype=np.complex64) for s in case["snapshots"]], axis=-1))
    info = {
        "recording_id": recording_id,
        "n_windows": len(cases),
        "prepare_s": round(t_prepare, 4),
        "replay_s": round(t_replay, 4),
        "features_s": round(t_features, 4),
        "x_build_s_per_window_mean": round(float(np.mean(x_times)), 6) if x_times else None,
        "bin_bytes": int(ctx["bin_path"].stat().st_size),
        "n_taps_used": int(cases[sorted(cases)[0]]["snapshots"][0].shape[-1])
        if with_snapshots else None,
    }
    return cases, payload, info, ctx


def _npz_path(case_dir: Path, recording_id: str) -> Path:
    return case_dir / ("%s.npz" % recording_id)


def build_cache(cache_dir, truth_path, *, recordings=None, force=False,
                with_snapshots=True, verbose=True):
    cache_dir = Path(cache_dir)
    case_dir = cache_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    truth_path = Path(truth_path)
    truth_path.parent.mkdir(parents=True, exist_ok=True)

    branch = p0a_windows()
    by_recording: dict[str, list] = {}
    for window_id, row in branch.items():
        by_recording.setdefault(row["recording_id"], []).append(float(row["t_end_s"]))
    if recordings:
        want = set(recordings)
        by_recording = {r: t for r, t in by_recording.items() if r in want}
    rec_infos, index_windows, truth_index = {}, {}, {}
    built_at = _now()
    for recording_id in sorted(by_recording):
        t_list = sorted(by_recording[recording_id])
        npz = _npz_path(case_dir, recording_id)
        if npz.is_file() and not force:
            with np.load(npz, allow_pickle=True) as z:
                have = sorted(str(v) for v in z["window_ids"])
            if have == sorted("%s__t%g" % (recording_id, t) for t in t_list):
                cases, payload = None, None
                info = {"recording_id": recording_id, "n_windows": len(have),
                        "reused": True, "sha256": sha256_file(npz)}
                ctx = None
                if verbose:
                    print("  [cache] %s reused (%d windows)" % (recording_id, len(have)))
            else:
                npz.unlink()
        if not npz.is_file():
            cases, payload, info, ctx = build_recording(recording_id, t_list,
                                                       with_snapshots=with_snapshots)
            np.savez_compressed(npz, **payload)
            info = {"recording_id": recording_id, "n_windows": len(cases),
                    "prepare_s": info["prepare_s"], "replay_s": info["replay_s"],
                    "features_s": info["features_s"],
                    "x_build_s_per_window_mean": info["x_build_s_per_window_mean"],
                    "bin_bytes": info["bin_bytes"], "n_taps_used": info["n_taps_used"],
                    "reused": False, "sha256": sha256_file(npz)}
            if verbose:
                print("  [cache] %s built: %d windows replay=%.2fs features=%.3fs npz=%.1fMB"
                      % (recording_id, len(cases), info["replay_s"], info["features_s"],
                         npz.stat().st_size / 1e6))
        rec_infos[recording_id] = dict(info, npz=str(npz.relative_to(WS)).replace("\\", "/"))
        with np.load(npz, allow_pickle=True) as z:
            window_ids = [str(w) for w in z["window_ids"]]
            t_arr = np.asarray(z["t_end_s"], dtype=float)
            k_arr = np.asarray(z["k"], dtype=np.int64)
        for window_id, t_val, k_val in zip(window_ids, t_arr, k_arr):
            row = branch[window_id]
            index_windows[window_id] = {
                "recording_id": recording_id,
                "t_end_s": float(t_val),
                "environment": row["environment"],
                "config_id": row["config_id"],
                "k": int(k_val),
                "npz": rec_infos[recording_id]["npz"],
            }
            truth_index[window_id] = {
                "recording_id": recording_id,
                "t_end_s": float(t_val),
                "environment": row["environment"],
                "config_id": row["config_id"],
                "n_truth": int(row["n_truth"]),
                "k": int(k_val),
                "truth_xy": [list(p) for p in row["truth_points"]],
                "truth_source": "P0a per_window.csv（oracle+nms 分支）truth_points；"
                                "与 meta.collection.truth_points 交叉核对",
            }

    index = {
        "kind": "q6-hardware-case-cache",
        "generated_at": built_at,
        "case_schema_version": 1,
        "feature_order": FEATURE_ORDER,
        "ring": RING,
        "blas_env": blas_env(),
        "source": {
            "data_dir": str(DATA_DIR.relative_to(WS)).replace("\\", "/"),
            "eval_data": str(EVAL_DATA.relative_to(WS)).replace("\\", "/"),
            "p0a_branch": "oracle+nms（P0a per_window.csv）",
            "p0b_protocol": str(PROTOCOL.relative_to(WS)).replace("\\", "/"),
            "b0_package": str(B0_DIR.relative_to(WS)).replace("\\", "/"),
            "tap_convention_evidence": "hardware/logs/tap_convention_check.json"
                                       "（截 tap 变体与历史缓存逐格相同 rel=0.0）",
        },
        "n_windows": len(index_windows),
        "n_recordings": len(rec_infos),
        "windows": index_windows,
        "recordings": rec_infos,
        "truth_index_path": str(truth_path.relative_to(WS)).replace("\\", "/"),
        "note": "case 文件不含真值；真值单独落 truth_index.json，仅 evaluation 读取。",
    }
    (cache_dir / "cache_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    truth_payload = {
        "kind": "q6-hardware-truth-index",
        "generated_at": built_at,
        "usage": "仅 evaluation 阶段读取；**绝不**传入估计器（case 里没有真值）",
        "source": "P0a per_window.csv（oracle+nms 分支）",
        "n_windows": len(truth_index),
        "windows": truth_index,
    }
    truth_path.write_text(json.dumps(truth_payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return index, truth_payload


def load_index(cache_dir):
    return json.loads((Path(cache_dir) / "cache_index.json").read_text(encoding="utf-8"))


def load_truth(truth_path):
    return json.loads(Path(truth_path).read_text(encoding="utf-8"))["windows"]


def load_recording_cases(cache_dir, recording_id, *, with_snapshots=False, configs=None):
    """按录制加载 case；返回 {window_id: case}。**不加载真值**。"""
    cache_dir = Path(cache_dir)
    npz = _npz_path(cache_dir / "cases", recording_id)
    out = {}
    with np.load(npz, allow_pickle=True) as z:
        window_ids = [str(w) for w in z["window_ids"]]
        config = json.loads(str(z["config_json"]))
        model = json.loads(str(z["model_json"]))
        search_extra = json.loads(str(z["search_json"]))
        for window_id in window_ids:
            n_snapshots = int(z["n_snapshots__%s" % window_id])
            n_features = int(z["n_features__%s" % window_id])
            X = np.asarray(z["X__%s" % window_id], dtype=np.complex128)
            snapshots = None
            if with_snapshots and ("snapshots__%s" % window_id) in z.files:
                stacked = np.asarray(z["snapshots__%s" % window_id], dtype=np.complex64)
                snapshots = [stacked[..., i] for i in range(stacked.shape[-1])]
            search = dict(search_extra)
            search["grid_x_m"] = np.asarray(z["grid_x"], dtype=float)
            search["grid_y_m"] = np.asarray(z["grid_y"], dtype=float)
            idx = window_ids.index(window_id)
            case = {
                "case_schema_version": 1,
                "id": window_id,
                "recording_id": recording_id,
                "environment": None,
                "config_id": None,
                "t_end_s": float(np.asarray(z["t_end_s"], dtype=float)[idx]),
                "X": X,
                "frequencies_hz": np.asarray(z["frequencies_hz"], dtype=float),
                "pairs": np.asarray(z["pairs"], dtype=np.int64),
                "tx_positions": np.asarray(z["tx_positions"], dtype=float),
                "rx_positions": np.asarray(z["rx_positions"], dtype=float),
                "k": int(np.asarray(z["k"], dtype=np.int64)[idx]),
                "grid_x": np.asarray(z["grid_x"], dtype=float),
                "grid_y": np.asarray(z["grid_y"], dtype=float),
                "config": config,
                "search": search,
                "model": model,
                "feature_order": FEATURE_ORDER,
                "n_snapshots": n_snapshots,
                "n_features": n_features,
                "n_freq_bins": int(np.asarray(z["frequencies_hz"]).size),
                "n_channels": int(np.asarray(z["pairs"]).shape[0]),
                "snapshots_available": bool(snapshots is not None or configs is not None),
                "snapshots": snapshots,
            }
            out[window_id] = case
    return out


def iter_cases(cache_dir, *, windows=None, recordings=None, with_snapshots=False,
               truth_index=None):
    """按 (recording_id, t_end_s) 顺序产出 (case, truth_info)；case 内不含真值。"""
    index = load_index(cache_dir)
    want = None if windows is None else set(windows)
    rec_want = None if recordings is None else set(recordings)
    for recording_id in sorted(index["recordings"]):
        if rec_want is not None and recording_id not in rec_want:
            continue
        cases = load_recording_cases(cache_dir, recording_id, with_snapshots=with_snapshots)
        rec_info = index["recordings"][recording_id]
        for window_id in sorted(cases, key=lambda w: cases[w]["t_end_s"]):
            if want is not None and window_id not in want:
                continue
            case = cases[window_id]
            row = index["windows"][window_id]
            case["environment"] = row["environment"]
            case["config_id"] = row["config_id"]
            case["cache_npz"] = rec_info["npz"]
            info = None if truth_index is None else truth_index.get(window_id)
            yield case, info


def subset_case(case, keep_channels, *, label=None):
    """按通道子集派生 case（例如 B3/B4 的"同 TX + RX2/3"两 Rx 预算）。

    * X 子集 = 取每个频点块内的保留通道列（频率优先顺序不变）；
    * config.pairs / pairs 同步收窄；FB 若使用，J 必须按**该方法自己的特征序**构造
      （本函数不改 config.is_forward_backward —— 那是方法的选择）；
    * L（n_snapshots）语义不变：仍表示原始慢时间快照数，不因 FB 或子集而改名。
    """
    keep = sorted(int(c) for c in keep_channels)
    X = np.asarray(case["X"])
    n_freq = int(case["n_freq_bins"])
    n_ch = int(case["n_channels"])
    if keep and (keep[0] < 0 or keep[-1] >= n_ch):
        raise ValueError("keep_channels 超界：%s（n_channels=%d）" % (keep, n_ch))
    idx = [f * n_ch + c for f in range(n_freq) for c in keep]
    out = dict(case)
    out["X"] = np.ascontiguousarray(X[:, idx])
    out["pairs"] = np.asarray(case["pairs"])[keep]
    out["n_channels"] = len(keep)
    out["n_features"] = int(out["X"].shape[1])
    out["feature_order"] = "frequency-major then channel-subset (index = f*|S| + s)"
    out["config"] = dict(case["config"])
    out["search"] = dict(case["search"])
    out["model"] = dict(case["model"])
    out["id"] = "%s__subset%s" % (case["id"], label or ("ch" + "-".join(map(str, keep))))
    out["derived_from"] = case["id"]
    out["subset"] = {
        "keep_channels": keep,
        "pairs": np.asarray(out["pairs"]).tolist(),
        "n_channels": len(keep),
        "note": "同通道预算对照用；本方法必须在该子集上跑同一评测口径",
    }
    if int(out["k"]) >= int(out["n_features"]):
        raise ValueError("子集后 M=%d <= K=%d，该方法在此预算下不可评估"
                         % (int(out["n_features"]), int(out["k"])))
    return out


def verify_cache(cache_dir, *, sample=3, verbose=True):
    """复算检查：X 是否等于由 snapshots 重建的观测矩阵（真实一致性判据）。"""
    index = load_index(cache_dir)
    picks = []
    for recording_id in sorted(index["recordings"]):
        picks.append(recording_id)
        if len(picks) >= sample:
            break
    rows = []
    for recording_id in picks:
        cases = load_recording_cases(cache_dir, recording_id, with_snapshots=True)
        ctx_cfg = None
        for window_id in sorted(cases, key=lambda w: cases[w]["t_end_s"])[:2]:
            case = cases[window_id]
            cfg = _cfg_from_case(case)
            X_new, _info = standard_music.build_observation_matrix(case["snapshots"], cfg)
            max_abs = float(np.max(np.abs(X_new - case["X"])))
            rel = max_abs / max(float(np.max(np.abs(case["X"]))), 1e-300)
            row = {"window_id": window_id, "X_shape": list(case["X"].shape),
                   "max_abs_diff": max_abs, "max_rel_diff": rel,
                   "identical": bool(np.array_equal(X_new, case["X"]))}
            rows.append(row)
            if verbose:
                print("  [verify] %s X vs rebuild: identical=%s rel=%.3e"
                      % (window_id, row["identical"], rel))
    return {"kind": "q6-case-cache-verify", "verified_at": _now(),
            "n_checked": len(rows), "windows": rows,
            "pass": all(r["identical"] for r in rows)}


def _cfg_from_case(case):
    cfg = case["config"]
    return standard_music.MusicConfig(
        fft_length=int(cfg["fft_length"]),
        tap_interval_si=float(cfg["tap_interval_si"]),
        center_freq_si=float(cfg["center_freq_si"]),
        bandwidth_si=float(cfg["bandwidth_si"]),
        epsilon_r_background=float(cfg["epsilon_r_background"]),
        system_delay_si=float(cfg["system_delay_si"]),
        diag_load_factor=float(cfg["diag_load_factor"]),
        is_forward_backward=bool(cfg["is_forward_backward"]),
        fov_degrees=float(cfg["fov_degrees"]),
        cir_window=tuple(int(v) for v in cfg["cir_window"]),
        k=int(case["k"]),
        pairs=tuple(tuple(int(v) for v in p) for p in np.asarray(case["pairs"]).tolist()),
        deg_corr=(None if cfg["deg_corr"] is None else tuple(cfg["deg_corr"])),
        ant_calib_phase=(None if cfg["ant_calib_phase"] is None
                         else tuple(cfg["ant_calib_phase"])),
        freq_range_si=tuple(float(v) for v in cfg["freq_range_si"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Q6 硬件侧公共 case 缓存")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--out", default=str(HARDWARE_DIR / "cache"))
    p_build.add_argument("--truth", default=str(HARDWARE_DIR / "truth" / "truth_index.json"))
    p_build.add_argument("--recordings", default=None, help="逗号分隔；缺省全部")
    p_build.add_argument("--force", action="store_true")
    p_build.add_argument("--no-snapshots", action="store_true")
    p_check = sub.add_parser("check")
    p_check.add_argument("--cache", default=str(HARDWARE_DIR / "cache"))
    p_check.add_argument("--sample", type=int, default=3)
    args = parser.parse_args(argv)
    if args.cmd == "build":
        recs = None if not args.recordings else [r for r in args.recordings.split(",") if r]
        index, truth = build_cache(args.out, args.truth, recordings=recs, force=args.force,
                                   with_snapshots=not args.no_snapshots)
        print("[cache] windows=%d recordings=%d -> %s"
              % (index["n_windows"], index["n_recordings"], args.out))
        print("[cache] truth index windows=%d -> %s" % (truth["n_windows"], args.truth))
        return 0
    res = verify_cache(args.cache, sample=args.sample)
    print("[cache] verify pass=%s" % res["pass"])
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
