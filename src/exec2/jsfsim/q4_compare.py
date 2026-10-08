#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q4：仿真–硬件直接跨域比较（规范 §7）。

主比较（至少两项实际交付）
------------------------
1. **目标身份对应的峰位差**：两侧各自用**无门限**最小平方一对一匹配分配到同序真值 identity；
   仅对两侧都有预测的 identity 算 δ_k=‖p̂_sim,k−p̂_hw,k‖，报 √(Σδ_k²/N_joint)、有效 identity 数、
   单侧/双侧缺峰数。另列两侧各自对真值 RMSE 与 DR。
2. **谱相关**：公共有效格点的线性非负伪谱各自除以最大值，逐实测段算 Pearson ρ；
   原始与归一化谱都保存；按环境×配置报告均值/SD/有效数，再按配置等权汇总。

硬约束
------
* 15 配置 + 固定 205 清单（室内 103 / 室外 102）；断言唯一数 = 205。
* 真实 Tx–Rx 几何（Tx ±0.019；Rx2 (-0.019,0,-0.02)、Rx3 (0,0,-0.02)；pairs=(0,2),(0,3),(1,2),(1,3)）；
  2D 仅投影 xy，保留 3D↔2D 投影差异记录。
* 每配置一条 160 状态轨迹（32 状态 EMA 启动 + 保留后 128）；**不是 160 次独立试验**。
* 连续 EMA α=0.25，不每窗重置；**不额外加人工白噪声**。
* 不按相关性挑段、不插值/平移/旋转对齐、不设“相关必须大于某数”。
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from . import data as D
from . import estimate, metrics, time_reference
from . import protocol as P

HW_WINDOWS_CSV = Path("04_Q6/hardware/analysis_nofb_q2q3_20260927/per_window.csv")
HW_CIR_REL = Path("06_仿真实验_20260926/q4/out/hw_cir")


def workspace_root() -> Path:
    """工作区下 `大修执行_20260923/`（内含 04_Q6 与 06_仿真实验_20260926）。"""
    from . import paths
    # paths.RECOVERY_DIR = 大修执行_20260923/06_仿真实验_20260926/recovery_20260929
    override = os.environ.get("JSF_HW_PROJECT_ROOT")
    root = Path(override) if override else paths.RECOVERY_DIR.parents[1]
    if not (root / HW_WINDOWS_CSV).is_file():
        raise RuntimeError("硬件工程根目录不可用；请设置 JSF_HW_PROJECT_ROOT：%s" % root)
    return root


# --------------------------------------------------------------------------- #
# 205 清单
# --------------------------------------------------------------------------- #
def frozen_205(csv_path: Path | None = None) -> dict:
    p = Path(csv_path) if csv_path else workspace_root() / HW_WINDOWS_CSV
    if not p.is_file():
        raise RuntimeError("205 清单来源缺失：%s" % p)
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    sel = [r for r in rows if r.get("method") == "fb_on" and str(r.get("in_205")) == "True"]
    ids = [r["window_id"] for r in sel]
    uniq = sorted(set(ids))
    env = {}
    cfg = {}
    for r in sel:
        env[r["environment"]] = env.get(r["environment"], 0) + 1
    for wid in uniq:
        r = next(x for x in sel if x["window_id"] == wid)
        cfg.setdefault(r["config_id"], []).append(wid)
    if len(uniq) != P.Q4_HW_WINDOWS:
        raise RuntimeError("205 清单断言失败：唯一 window_id = %d != %d" % (len(uniq), P.Q4_HW_WINDOWS))
    if env.get("indoor") != P.Q4_HW_INDOOR or env.get("outdoor") != P.Q4_HW_OUTDOOR:
        raise RuntimeError("环境计数不符：%s（期望 indoor=%d outdoor=%d）"
                           % (env, P.Q4_HW_INDOOR, P.Q4_HW_OUTDOOR))
    return {"source": str(p), "n_windows": len(uniq), "unique_windows": len(uniq),
            "by_environment": env, "window_ids": uniq,
            "by_config": {k: sorted(v) for k, v in sorted(cfg.items())},
            "rule": "method=fb_on 且 in_205=True 的 window_id 冻结；断言唯一数=205",
            "not_214": True, "not_120": True}


# --------------------------------------------------------------------------- #
# 硬件侧输入
# --------------------------------------------------------------------------- #
def hardware_windows(hw_dir: Path | None = None, csv_path: Path | None = None) -> list:
    """返回 205 条硬件窗口记录（含 CIR、真值、几何）。"""
    man = frozen_205(csv_path)
    ids = set(man["window_ids"])
    config_by_window = {w: c for c, windows in man["by_config"].items() for w in windows}
    base = Path(hw_dir) if hw_dir else workspace_root() / HW_CIR_REL
    recs = time_reference.hw_window_manifest(base, ids)
    out = []
    for rid, rec in recs.items():
        truth = json.loads(rec["truth_json"].read_text(encoding="utf-8"))
        with np.load(rec["npz"], allow_pickle=False) as z:
            cir = np.asarray(z["cir_cropped30"])
            ids_all = [str(w) for w in z["window_ids"]]
        for row in rec["rows"]:
            t = (truth.get("windows") or {}).get(row["window_id"]) or {}
            xy = t.get("truth_xy") or t.get("truth_xy_m")
            if not xy:
                continue
            out.append({
                "window_id": row["window_id"], "recording_id": rid,
                "environment": rec["environment"], "config_id": config_by_window[row["window_id"]],
                "k": int(row["k"]), "truth_xy": [[float(a), float(b)] for a, b in xy],
                "cir": cir[row["index_in_npz"]],           # [4,30]
                "pairs": rec["pairs"], "tx_positions": rec["tx_positions"],
                "rx_positions": rec["rx_positions"],
                "cir_offset": rec["cir_offset"], "ft_len": rec["ft_len"],
                "hardware_source": rec["source"], "t_end_s": row["t_end_s"],
            })
    got = {r["window_id"] for r in out}
    missing = sorted(ids - got)
    if missing:
        raise RuntimeError("205 清单中 %d 个 window 在硬件 CIR 包里找不到：%s"
                           % (len(missing), missing[:5]))
    if len(out) != P.Q4_HW_WINDOWS:
        raise RuntimeError("硬件窗口数 %d != 205" % len(out))
    return out


def hw_x(cir_cropped30, *, shift_taps: int = 0) -> np.ndarray:
    """硬件 CFR -> X（可施加已声明时移）。"""
    from . import paths
    sc = paths.sim_common()
    arr = np.asarray(cir_cropped30, dtype=np.complex128)   # [L,4,30]
    X = sc.x_from_cropped(arr)
    if shift_taps:
        # Direct CFR phase preserves all 30 observed taps, including the two that
        # move to physical taps 30/31. An IFFT followed by crop30 silently loses them.
        idx = np.asarray(P.in_band_indices(), dtype=float)
        phase = np.exp(-2j * math.pi * idx * int(shift_taps) / int(P.FFT_LENGTH))
        X = X * np.repeat(phase, P.N_CHANNELS)[None, :]
    return np.ascontiguousarray(X)


def _apply_time_shift(arr: np.ndarray, shift_taps: int) -> np.ndarray:
    """在 CFR 域施加整数 tap 时移（等价于时域循环移位；来源须为已声明参考，非拟合）。"""
    n = int(P.FFT_LENGTH)
    cfr = np.fft.fft(arr, n=n, axis=-1)
    k = np.arange(n)
    cfr = cfr * np.exp(-2j * math.pi * k * int(shift_taps) / n)
    shifted = np.fft.ifft(cfr, n=n, axis=-1)
    return np.ascontiguousarray(shifted[:, :, :P.CROP_LENGTH])


# --------------------------------------------------------------------------- #
# 谱与指标
# --------------------------------------------------------------------------- #
def spectrum_for(rec_geom: dict, k: int, X, grid_x, grid_y, *, tag: str) -> dict:
    from . import paths
    cb = paths.adapter("case_build")
    case = cb.build_case(case_id=tag, clean_id=tag, X=X, k=int(k), grid_x=grid_x, grid_y=grid_y,
                         layer="B", source="fdtd", scene="Q4", tags=["q4"],
                         snr_db=10.0, trial=0, seed=0, L=int(X.shape[0]),
                         resolution_distance_m=None, geometry=rec_geom)
    res = estimate.run_single(case)
    spectrum = res.get("spectrum")
    preds = [[float(p["x"]), float(p["y"])] for p in res["predictions"]]
    return {"status": res["status"], "predictions": preds,
            "spectrum": (None if spectrum is None else np.asarray(spectrum, dtype=float)),
            "case_id": case["id"], "cir_window": list(case["config"]["cir_window"])}


def match_identity(truth_xy, pred_xy) -> dict:
    """无门限最小平方一对一：返回 truth_index -> pred_index。"""
    m = metrics.match_rmse([[float(a), float(b)] for a, b in truth_xy],
                           [[float(a), float(b)] for a, b in pred_xy])
    return {int(t): int(p) for t, p, _d in m["pairs"]}


def pearson(a, b) -> float | None:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        return None
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# 仿真侧
# --------------------------------------------------------------------------- #
def sim_clean_for(cfg_id: str, *, root, workers: int, resolution=None, run_time=None,
                  states: int | None = None, geometry_override=None) -> dict:
    targets = P.Q4_TARGETS[cfg_id]
    cid = P.clean_id_for("fdtd", len(targets), targets)
    rec = {"task_id": "q4_%s" % cfg_id, "clean_id": cid, "family": "Q4", "layer": "B",
           "source": "fdtd", "scene": "Q4", "k": len(targets),
           "targets_xy": [[float(x), float(y)] for x, y in targets],
           "targets_polar": [list(P.xy_to_polar(x, y)) for x, y in targets],
           "snr_db": 10.0, "trials": 0, "L": P.Q4_KEPT_STATES,
           "L_clean": P.Q4_STATES_TOTAL, "clean_seed": P.clean_seed(cid),
           "resolution_distance_m": None, "tags": ["q4", cfg_id]}
    geom = geometry_override or {"tx_positions_m": P.Q4_TX_POSITIONS_M,
                                 "rx_positions_m": P.Q4_RX_POSITIONS_M, "pairs": P.Q4_PAIRS}
    out = D.build_clean_fdtd(rec, root=root, protocol_hash="q4", source_aggregate="q4",
                             resume=True, workers=workers, resolution_px_per_m=resolution,
                             run_time_meep=run_time, background_mode="ema_windowed",
                             geometry=geom, max_states=states,
                             snapshots_override=P.Q4_KEPT_STATES,
                             burnin_states=P.Q4_BURNIN_STATES)
    return {"clean_id": cid, "rec": rec, "geometry": geom, "raw_cir": out["raw_cir"],
            "meta": out["meta"], "reused": out.get("reused", False)}


def sim_x_kept(raw_cir, wids: list) -> np.ndarray:
    """仿真侧：160 状态里保留后 128（EMA 已由 fdtd 层处理），逐窗口取**同一个 128 状态**块。

    说明：仿真只有一条轨迹；硬件每个 window 自有其 128 帧。规范要求“每配置一条真实求解的
    160 状态轨迹，32 状态用于 EMA 启动，保留后 128 状态”——因此这里每个 window 使用同一条
    仿真 128 状态输出（同一几何、同一运动模型），**不把 205 个比较当作 205 次独立求解**。
    """
    from . import paths
    sc = paths.sim_common()
    raw = np.asarray(raw_cir)[:P.Q4_KEPT_STATES]
    return sc.x_from_cropped(raw)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def precompute_sim(cfg_id: str, *, root, workers: int, resolution=None, run_time=None,
                   states: int | None = None, geometry=None) -> dict:
    """用 api 级**状态分片多进程**预填 Q4 仿真解（每进程一个 Meep 实例）。

    与 generate_clean 共享同一 _prepare/cache_key/条目格式与 _solve_one_state 求解路径，
    预填完成后 build_clean_fdtd 全部命中缓存（不重复求解）。
    """
    from fdtd import api as fdtd_api
    targets = P.Q4_TARGETS[cfg_id]
    cid = P.clean_id_for("fdtd", len(targets), targets)
    rec = {"task_id": "q4_%s" % cfg_id, "clean_id": cid, "family": "Q4", "layer": "B",
           "source": "fdtd", "scene": "Q4", "k": len(targets),
           "targets_xy": [[float(x), float(y)] for x, y in targets],
           "targets_polar": [list(P.xy_to_polar(x, y)) for x, y in targets],
           "snr_db": 10.0, "trials": 0, "L": P.Q4_KEPT_STATES,
           "L_clean": P.Q4_STATES_TOTAL, "clean_seed": P.clean_seed(cid),
           "resolution_distance_m": None, "tags": ["q4", cfg_id]}
    geom = geometry or {"tx_positions_m": P.Q4_TX_POSITIONS_M,
                        "rx_positions_m": P.Q4_RX_POSITIONS_M, "pairs": P.Q4_PAIRS}
    scene = D.build_fdtd_scene(rec, resolution_px_per_m=(resolution or P.FDTD_RESOLUTION_PX_PER_M),
                               run_time_meep=(run_time or P.FDTD_RUN_TIME_MEEP),
                               background_mode="ema_windowed", geometry=geom,
                               snapshots=P.Q4_KEPT_STATES,
                               burnin_states=P.Q4_BURNIN_STATES)
    outdir = Path(root) / "cache" / "fdtd" / cid
    cache_root = Path(root) / "cache" / "fdtd" / "_solver_cache"
    res = fdtd_api.precompute_parallel(None, scene, outdir, cache_root=cache_root,
                                       max_states=states, background_mode="ema_windowed",
                                       workers=int(workers))
    return {"config_id": cfg_id, "clean_id": cid, "cache_key": res["cache_key"],
            "stats": res["stats"], "n_total": res["n_total"]}


def precompute_all(root, *, workers: int, configs=None, states=None) -> dict:
    """预填全部（或指定）Q4 配置的仿真状态；已完成的配置命中缓存、只补缺口。"""
    cfgs = configs or list(P.Q4_TARGETS)
    out = []
    for cfg_id in cfgs:
        r = precompute_sim(cfg_id, root=root, workers=workers, states=states)
        out.append(r)
        print("[q4-precompute] %-6s key=%s n_total=%d solved=%s hits=%s"
              % (cfg_id, r["cache_key"][:12], r["n_total"],
                 r["stats"]["solves_actual"], r["stats"]["cache_hits"]), flush=True)
    return {"kind": "q4-precompute", "n_configs": len(cfgs), "configs": out,
            "total_solves_actual": sum(c["stats"]["solves_actual"] for c in out),
            "total_state_hits": sum(c["stats"]["cache_hits"]["state"] for c in out),
            "workers": int(workers)}


def plan(output_root, protocol_path=None, configs=None) -> int:
    from . import paths
    root = Path(output_root)
    man = frozen_205()
    cfgs = configs or list(P.Q4_TARGETS)
    doc = {
        "kind": "q4-plan", "hw_205": {"n_windows": man["n_windows"],
                                      "by_environment": man["by_environment"],
                                      "source": man["source"]},
        "configs": {c: {"targets_xy_m": P.Q4_TARGETS[c],
                        "n_hw_windows": len(man["by_config"].get(c, []))} for c in cfgs},
        "n_configs": len(cfgs), "precheck_configs": list(P.Q4_PRECHECK_CONFIGS),
        "geometry": {"tx_positions_m": P.Q4_TX_POSITIONS_M, "rx_positions_m": P.Q4_RX_POSITIONS_M,
                     "pairs": P.Q4_PAIRS, "projection": "2D xy，保留 3D 坐标与投影差异"},
        "states": {"total": P.Q4_STATES_TOTAL, "burnin": P.Q4_BURNIN_STATES,
                   "kept": P.Q4_KEPT_STATES},
        "background": {"mode": "ema_continuous", "alpha": P.Q4_EMA_ALPHA},
        "no_extra_white_noise": True,
        "cost": {"fdtd_solves_if_miss": len(cfgs) * P.Q4_STATES_TOTAL * len(P.Q4_TX_POSITIONS_M),
                 "note": "15 个仿真输出 + 205 个跨域比较；共用仿真不当作 205 次独立求解"},
        "time_reference_gate": "缺时间参考判定 -> 拒绝给正式“相似度已验证”",
    }
    d = root / "q4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "q4_plan.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"n_windows": man["n_windows"], "by_env": man["by_environment"],
                      "n_configs": len(cfgs), "precheck": doc["precheck_configs"],
                      "cost": doc["cost"]}, ensure_ascii=False, indent=1))
    return 0


def compare_one_config(cfg_id: str, *, root, workers: int, resolution, run_time, states,
                       hw_shift_taps: int, hw: list) -> dict:
    """单个配置的跨域比较（模块级函数，可被多进程调度；不写任何文件）。"""
    targets = P.Q4_TARGETS[cfg_id]
    grid_x, grid_y = P.build_grid()
    out = {"config_id": cfg_id, "targets_xy_m": targets, "per_window": [],
           "missing_sim_side": 0, "missing_hw_side": 0, "errors": []}
    try:
        sim = sim_clean_for(cfg_id, root=root, workers=workers,
                            resolution=(resolution if resolution is not None
                                        else P.FDTD_RESOLUTION_PX_PER_M),
                            run_time=(run_time if run_time is not None
                                      else P.FDTD_RUN_TIME_MEEP),
                            states=states)
        sim_x = sim_x_kept(sim["raw_cir"], [r["window_id"] for r in hw])
        sim_spec = spectrum_for({"tx_positions_m": P.Q4_TX_POSITIONS_M,
                                 "rx_positions_m": P.Q4_RX_POSITIONS_M, "pairs": P.Q4_PAIRS},
                                len(targets), sim_x, grid_x, grid_y, tag="q4_sim_%s" % cfg_id)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "sim_failed"
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    sp = sim_spec["spectrum"]
    if sp is None:
        out["status"] = "no_spectrum"
        out["sim_status"] = sim_spec["status"]
        return out
    out["sim_status"] = sim_spec["status"]
    sim_id = match_identity(targets, sim_spec["predictions"])
    for w in hw:
        if w["config_id"] != cfg_id:
            continue
        try:
            Xh = hw_x(w["cir"], shift_taps=hw_shift_taps)
            spec_h = spectrum_for({"tx_positions_m": w["tx_positions"].tolist(),
                                   "rx_positions_m": w["rx_positions"].tolist(),
                                   "pairs": w["pairs"].tolist()},
                                  int(w["k"]), Xh, grid_x, grid_y,
                                  tag="q4_hw_%s" % w["window_id"])
        except Exception as exc:  # noqa: BLE001
            out["errors"].append({"window_id": w["window_id"],
                                  "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        sph = spec_h["spectrum"]
        hw_id = match_identity(w["truth_xy"], spec_h["predictions"])
        joint, deltas = [], []
        for idx in range(len(w["truth_xy"])):
            hs = idx in hw_id
            ss = idx in sim_id
            if hs and ss:
                hw_p = spec_h["predictions"][hw_id[idx]]
                sim_p = sim_spec["predictions"][sim_id[idx]]
                joint.append(idx)
                deltas.append(math.hypot(hw_p[0] - sim_p[0], hw_p[1] - sim_p[1]))
            elif hs:
                out["missing_sim_side"] += 1
            elif ss:
                out["missing_hw_side"] += 1
        rho = None
        mask_pts = None
        if sph is not None and sph.shape == sp.shape:
            mask = (sp > 0) & (sph > 0)
            mask_pts = int(mask.sum())
            if mask_pts >= 3:
                a, b = sp[mask], sph[mask]
                rho = pearson(a / np.max(np.abs(a)), b / np.max(np.abs(b)))
        out["per_window"].append({
            "window_id": w["window_id"], "recording_id": w["recording_id"],
            "environment": w["environment"], "config_id": cfg_id, "k": int(w["k"]),
            "n_joint_identity": len(joint), "joint_identity_indices": joint,
            "delta_m": deltas,
            "rmse_hw_m": (math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else None),
            "max_delta_m": (max(deltas) if deltas else None),
            "n_hw_predictions": len(spec_h["predictions"]),
            "n_sim_predictions": len(sim_spec["predictions"]),
            "spectral_rho": rho, "common_mask_points": mask_pts,
            "hw_status": spec_h["status"], "sim_status": sim_spec["status"]})
    out["status"] = "ok"
    return out


def run(output_root, protocol_path=None, configs=None, workers: int = 32,
        resolution=None, run_time=None, states=None, hw_shift_taps: int | None = None,
        limit_windows: int | None = None, skip_precompute: bool = False) -> int:
    from . import paths
    root = Path(output_root)
    q4_dir = root / "q4"
    q4_dir.mkdir(parents=True, exist_ok=True)
    man = frozen_205()
    cfgs = configs or list(P.Q4_TARGETS)

    # 时间参考门禁：必须有诊断且非 blocked 才允许给正式跨域结论
    tr_path = q4_dir / "time_reference.json"
    if not tr_path.is_file():
        doc = time_reference.diagnose(workspace_root() / HW_CIR_REL, set(man["window_ids"]))
        time_reference.write(q4_dir, doc)
    tr = json.loads(tr_path.read_text(encoding="utf-8"))
    blocked_tr = bool(tr.get("time_reference", {}).get("blocked", True))
    shift = int(tr.get("time_reference", {}).get("implied_shift_taps") or 0)
    if hw_shift_taps is not None:
        shift = int(hw_shift_taps)
    print("[q4] 时间参考：status=%s shift_taps=%d blocked=%s" % (tr.get("status"), shift, blocked_tr))

    # ---- 阶段 1：并行预填全部仿真状态（4800 次求解；已完成的命中缓存）----
    if not skip_precompute:
        pre = precompute_all(root, workers=workers, configs=cfgs, states=states)
        paths.sim_common().atomic_write_json(q4_dir / "q4_precompute.json", pre)
        print("[q4] 预填完成：实解 %d，复用状态 %d"
              % (pre["total_solves_actual"], pre["total_state_hits"]), flush=True)

    # ---- 阶段 2：跨域比较（按配置并行）----
    hw = hardware_windows()
    if limit_windows:
        hw = hw[: int(limit_windows)]
    per_window, per_cfg = [], {}
    print("[q4] 开始比较：%d 配置 × %d 窗口" % (len(cfgs), len(hw)), flush=True)
    results = []
    if len(cfgs) > 1:
        import concurrent.futures as _cf
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        with _cf.ProcessPoolExecutor(max_workers=min(len(cfgs), 15), mp_context=ctx) as ex:
            futs = [ex.submit(compare_one_config, c, root=str(root), workers=max(1, workers // 8),
                              resolution=resolution, run_time=run_time, states=states,
                              hw_shift_taps=shift, hw=hw) for c in cfgs]
            for f in _cf.as_completed(futs):
                results.append(f.result())
                print("[q4] 配置完成 %d/%d" % (len(results), len(cfgs)), flush=True)
    else:
        results.append(compare_one_config(cfgs[0], root=str(root), workers=workers,
                                          resolution=resolution, run_time=run_time, states=states,
                                          hw_shift_taps=shift, hw=hw))
    for r in sorted(results, key=lambda x: str(x.get("config_id"))):
        cfg_id = r["config_id"]
        per_window.extend(r.get("per_window") or [])
        rows_here = r.get("per_window") or []
        deltas = [d for x in rows_here for d in x["delta_m"]]
        rhos = [x["spectral_rho"] for x in rows_here if x["spectral_rho"] is not None]
        per_cfg[cfg_id] = {
            "config_id": cfg_id, "targets_xy_m": r.get("targets_xy_m"),
            "status": r.get("status"), "error": r.get("error"),
            "n_hw_windows": len(rows_here),
            "n_joint_identities": sum(x["n_joint_identity"] for x in rows_here),
            "peak_rmse_identity_m": (math.sqrt(sum(d * d for d in deltas) / len(deltas))
                                     if deltas else None),
            "peak_rmse_identity_ci95_m": _boot_ci(deltas),
            "missing_sim_side": r.get("missing_sim_side", 0),
            "missing_hw_side": r.get("missing_hw_side", 0),
            "n_spectral_rho": len(rhos),
            "spectral_rho_mean": (float(np.mean(rhos)) if rhos else None),
            "spectral_rho_sd": (float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None),
            "n_errors": len(r.get("errors") or []),
        }
    # 按环境×配置
    by_env_cfg = []
    for env in sorted({r["environment"] for r in per_window}):
        for cfg_id in cfgs:
            sub = [r for r in per_window if r["environment"] == env and r["config_id"] == cfg_id]
            if not sub:
                continue
            deltas = [d for r in sub for d in r["delta_m"]]
            rhos = [r["spectral_rho"] for r in sub if r["spectral_rho"] is not None]
            by_env_cfg.append({
                "environment": env, "config_id": cfg_id, "n_windows": len(sub),
                "n_joint_identities": sum(r["n_joint_identity"] for r in sub),
                "peak_rmse_identity_m": (math.sqrt(sum(d * d for d in deltas) / len(deltas))
                                         if deltas else None),
                "missing_sim_side": sum(r["missing_sim_side"] for r in sub),
                "missing_hw_side": sum(r["missing_hw_side"] for r in sub),
                "spectral_rho_mean": (float(np.mean(rhos)) if rhos else None),
                "spectral_rho_sd": (float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None),
                "n_spectral_rho": len(rhos),
            })
    # 配置等权汇总
    cfg_means_rho = [c["spectral_rho_mean"] for c in per_cfg.values()
                     if isinstance(c, dict) and c.get("spectral_rho_mean") is not None]
    cfg_means_delta = [c["peak_rmse_identity_m"] for c in per_cfg.values()
                       if isinstance(c, dict) and c.get("peak_rmse_identity_m") is not None]
    doc = {
        "kind": "q4-cross-domain", "n_configs": len(cfgs),
        "hw_205": {"n_windows": man["n_windows"], "by_environment": man["by_environment"]},
        "time_reference": tr,
        "hw_shift_taps_applied": shift,
        "asymmetry_note": ("不额外加人工白噪声：比较无人工加噪的简化 FDTD 输出与含真实噪声的实测；"
                           "差异如实解释，不把任意 10 dB 宣称硬件 SNR。"),
        "per_config": per_cfg, "by_environment_config": by_env_cfg,
        "per_window": per_window,
        "summary": {
            "config_equal_weight_spectral_rho_mean": (float(np.mean(cfg_means_rho))
                                                      if cfg_means_rho else None),
            "config_equal_weight_spectral_rho_n": len(cfg_means_rho),
            "config_equal_weight_peak_rmse_m": (float(np.mean(cfg_means_delta))
                                                if cfg_means_delta else None),
            "config_equal_weight_peak_rmse_n": len(cfg_means_delta),
            "note": ("谱相关先按环境×配置给均值/SD 与有效数，再按配置等权汇总，"
                     "不只报被样本数量支配的总体值；不设“相关必须大于某数”筛选。"),
        },
        "blocked_time_reference": blocked_tr,
        "formal_similarity_claim_allowed": (not blocked_tr),
    }
    sc = paths.sim_common()
    sc.atomic_write_json(q4_dir / "q4_cross_domain.json", doc)
    _write_csv(q4_dir / "q4_per_window.csv", per_window)
    _write_csv(q4_dir / "q4_by_environment_config.csv", by_env_cfg)
    print(json.dumps({"n_configs": len(cfgs), "n_windows_compared": len(per_window),
                      "spectral_rho_config_equal_weight":
                          doc["summary"]["config_equal_weight_spectral_rho_mean"],
                      "peak_rmse_config_equal_weight":
                          doc["summary"]["config_equal_weight_peak_rmse_m"],
                      "blocked_time_reference": blocked_tr},
                     ensure_ascii=False, indent=1))
    return 0 if not blocked_tr else 3


def _boot_ci(values, reps=2000, seed=20260929):
    if not values:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = np.arange(arr.size)
    vals = []
    for _ in range(reps):
        pick = rng.choice(idx, size=idx.size, replace=True)
        vals.append(math.sqrt(float(np.mean(arr[pick] ** 2))))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cli(stage: str, output_root, protocol_path=None, configs=None, stage_time_reference=None,
        workers: int = 32, skip_precompute: bool = False) -> int:
    if stage == "plan":
        return plan(output_root, protocol_path, configs)
    if stage == "run":
        return run(output_root, protocol_path, configs, workers=int(workers),
                   skip_precompute=bool(skip_precompute))
    if stage == "precompute":
        pre = precompute_all(Path(output_root), workers=int(workers), configs=configs)
        from . import paths as _p
        _p.sim_common().atomic_write_json(Path(output_root) / "q4" / "q4_precompute.json", pre)
        print(json.dumps({"n_configs": pre["n_configs"],
                          "total_solves_actual": pre["total_solves_actual"],
                          "total_state_hits": pre["total_state_hits"]},
                         ensure_ascii=False, indent=1))
        return 0
    if stage == "time-reference":
        return time_reference_only(output_root)
    raise SystemExit("q4 --stage 只支持 plan|precompute|time-reference|run")


def time_reference_only(output_root) -> int:
    """只跑时间参考诊断并落盘（不生成仿真、不做正式相似度结论）。"""
    from . import paths
    root = Path(output_root)
    man = frozen_205()
    doc = time_reference.diagnose(workspace_root() / HW_CIR_REL, set(man["window_ids"]))
    p = time_reference.write(root / "q4", doc)
    print(json.dumps({"path": str(p), "status": doc["status"],
                      "verdict": doc["time_reference"].get("resolution"),
                      "blocked": doc["time_reference"].get("blocked"),
                      "median_abs_difference_taps":
                          doc["synthetic_test"]["median_abs_difference_taps"]},
                     ensure_ascii=False, indent=1))
    return 0 if doc["status"] == "resolved" else 3


__all__ = ["plan", "run", "cli", "frozen_205", "hardware_windows", "hw_x", "sim_clean_for",
           "time_reference_only"]
