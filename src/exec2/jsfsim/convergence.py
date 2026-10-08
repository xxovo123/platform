#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FDTD 数值前检（规范 §10）。

比较原则
--------
* 先比 **20/40 Meep 时间单位**，后比 **250/350 px/m**；同一比较中**只改被测数值参数**，
  物理域不随分辨率变。
* Courant=0.5；PML 厚度 2c/fc；域覆盖目标圆柱 + 轨迹 + ≥0.75 m 内区裕量（由 fdtd/scenes 派生）。
* 代表场景覆盖：受控 S1/S2/S3/S5、三方向 0.30 m 近间距、Q4 的 G-430/C2-1/C3-1；同配置去重。
* 预检用 **16 个真实状态**；不拿 L=1 代替协方差验证。

固定判据（**不因效果改宽**）
--------------------------
1. 绝对复相关 ≥ 0.90（**一个公共最优复数增益**，四通道 × 全部状态；不逐通道/逐状态对齐）
2. 公共增益对齐后相对 L2 ≤ 0.30（同时保留未经补偿误差与增益值）
3. 峰 tap 差 ≤ 1
4. JSF-MUSIC 两档输出数量相同且最优一对一峰位最大差 ≤ 0.05 m

未通过时先定位采样/相位/数值原因；时长可再比 40/80，网格可再比 350/500；
仍不通过就交付 blocked 清单与证据，**不自动从正式计划删除该场景**。
局部预检不能声称所有 128/160 状态都已收敛。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import data as D
from . import estimate, metrics, paths
from . import protocol as P

CRITERIA = {"min_abs_correlation": 0.90, "max_shape_rel_l2": 0.30, "max_peak_tap_shift": 1,
            "max_peak_position_diff_m": 0.05, "frozen_before_run": True}


def representative_scenes(profile: str = "controlled") -> list:
    """代表场景（同配置去重）。返回 [(tag, rec, geometry)]。"""
    out = []
    if profile == "controlled":
        by_tag = {}
        for task in ("E1_A", "E2_A"):
            for s in P.expand_task(task):
                if abs(float(s["snr_db"]) - 10.0) > 1e-9:
                    continue
                if task == "E1_A" and s.get("layout") in ("S1", "S2", "S3", "S5"):
                    by_tag.setdefault("S%s" % s["layout"], s)
                if task == "E2_A" and abs(float(s.get("d_nominal_m") or 0.0) - 0.30) < 1e-9:
                    by_tag.setdefault("E2_%s" % s["direction"], s)
        for tag, rec in sorted(by_tag.items()):
            out.append((tag, dict(rec, layer="B", source="fdtd"), None))
    else:
        # §10 的数值收敛代表包含深距 G-430；§7 的 G-420 是端到端预跑。
        for cfg in ("G-430", "C2-1", "C3-1"):
            targets = P.Q4_TARGETS[cfg]
            cid = P.clean_id_for("fdtd", len(targets), targets)
            rec = {"task_id": "Q4_%s" % cfg, "clean_id": cid, "family": "Q4", "layer": "B",
                   "source": "fdtd", "scene": "Q4", "k": len(targets),
                   "targets_xy": [[float(x), float(y)] for x, y in targets],
                   "targets_polar": [list(P.xy_to_polar(x, y)) for x, y in targets],
                   "snr_db": 10.0, "trials": 0, "L": P.Q4_KEPT_STATES,
                   "L_clean": P.Q4_STATES_TOTAL, "clean_seed": P.clean_seed(cid),
                   "resolution_distance_m": None, "tags": ["q4", cfg]}
            geom = {"tx_positions_m": P.Q4_TX_POSITIONS_M, "rx_positions_m": P.Q4_RX_POSITIONS_M,
                    "pairs": P.Q4_PAIRS}
            out.append((cfg, rec, geom))
    return out


def _build(root, rec, *, tag, geometry, resolution, run_time, states, background_mode, workers):
    """生成一档参数的解。

    缓存纪律（每档完全隔离，否则会互相判为"签名不匹配"）：
    * **数据目录**：`cache/conv_<res>_<rt>/<clean_id>` —— 每档一套 sidecar/truth/npz。
      若共用 `cache/fdtd/<clean_id>`，低档的签名会被高档覆盖，低档再读即 DataError
      （已本地复现：同场景两档 res 相同但 run_time 不同 -> 两个签名、一个目录）。
    * **求解器状态缓存**：同样按 `conv_<res>_<rt>` 隔离（`JSF_FDTD_CLEAN_SUBDIR` 同时
      作用于数据目录）。
    * 场景快照数用**完整值**（受控 128 / Q4 160）以匹配 `cache_key` 的 `n_snapshots`；
      用 `max_states=states` 只物化前 states 个状态以省算力。
    """
    sub = "conv_%s" % tag
    old = os.environ.get("JSF_FDTD_CLEAN_SUBDIR")
    os.environ["JSF_FDTD_CLEAN_SUBDIR"] = sub
    try:
        out = D.build_clean_fdtd(rec, root=root, protocol_hash="convergence",
                                 source_aggregate="convergence", resume=True, workers=workers,
                                 resolution_px_per_m=resolution, run_time_meep=run_time,
                                 background_mode=background_mode, geometry=geometry,
                                 max_states=int(states))
    finally:
        if old is None:
            os.environ.pop("JSF_FDTD_CLEAN_SUBDIR", None)
        else:
            os.environ["JSF_FDTD_CLEAN_SUBDIR"] = old
    return out


def _common_gain_and_shape(a, b):
    """一个公共最优复数增益 + 形状误差（不逐通道/逐状态对齐）。"""
    import numpy as np
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    num = np.vdot(b, a)          # sum(conj(b)*a)
    den = float(np.vdot(b, b).real)
    g = (num / den) if den > 0 else 0j
    aligned = b * g
    ref = float(np.linalg.norm(a))
    abs_corr = (abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                if np.linalg.norm(a) * np.linalg.norm(b) > 0 else 0.0)
    raw_rel_l2 = (float(np.linalg.norm(b - a) / ref) if ref > 0 else None)
    shape_rel_l2 = (float(np.linalg.norm(aligned - a) / ref) if ref > 0 else None)
    return {"gain_amplitude": float(abs(g)), "gain_phase_rad": float(np.angle(g)),
            "gain_complex": [float(g.real), float(g.imag)],
            "abs_correlation": float(abs_corr), "raw_rel_l2": raw_rel_l2,
            "shape_rel_l2": shape_rel_l2}


def _peak_shift(a, b):
    import numpy as np
    ta = np.abs(np.asarray(a)).argmax(axis=-1).astype(int)
    tb = np.abs(np.asarray(b)).argmax(axis=-1).astype(int)
    d = np.abs(ta - tb)
    return {"max_peak_tap_shift": int(d.max()), "mean_peak_tap_shift": float(d.mean())}


def _music_compare(a, b, rec, bg_mode, clean_id, geometry=None):
    """两档输出跑 JSF-MUSIC，比输出数量与最优一对一峰位最大差（判据 4）。"""
    import numpy as np
    gx, gy = P.build_grid()
    rows = []
    for label, raw in (("low", a), ("high", b)):
        X = D.clean_x(np.asarray(raw), L=int(np.asarray(raw).shape[0]))
        case = _case(rec, X, clean_id=clean_id, grid_x=gx, grid_y=gy, label=label,
                     geometry=geometry)
        res = estimate.run_single(case)
        preds = [[float(p["x"]), float(p["y"])] for p in res["predictions"]]
        rows.append({"label": label, "status": res["status"], "n": len(preds), "pred": preds})
    low, high = rows
    same_count = (low["status"] == high["status"] == "ok" and low["n"] == high["n"])
    max_diff = None
    if same_count and low["n"] > 0:
        import itertools
        best = None
        for perm in itertools.permutations(range(high["n"])):
            d = max(math.hypot(low["pred"][i][0] - high["pred"][perm[i]][0],
                               low["pred"][i][1] - high["pred"][perm[i]][1])
                    for i in range(low["n"]))
            best = d if best is None else min(best, d)
        max_diff = float(best)
    return {"low": low, "high": high, "same_output_count": bool(same_count),
            "max_peak_position_diff_m": max_diff,
            "passed": bool(same_count and max_diff is not None
                           and max_diff <= CRITERIA["max_peak_position_diff_m"] + 1e-12)}


def _case(rec, X, *, clean_id, grid_x, grid_y, label, geometry=None):
    cb = paths.adapter("case_build")
    return cb.build_case(case_id="conv_%s_%s" % (rec["task_id"], label), clean_id=clean_id,
                         X=X, k=int(rec["k"]), grid_x=grid_x, grid_y=grid_y, layer="B",
                         source="fdtd", scene="convergence", tags=["convergence", label],
                         snr_db=10.0, trial=0, seed=0, L=int(X.shape[0]),
                         resolution_distance_m=None, geometry=geometry or {})


def compare_pair(root, rec, *, geometry, tag, low, high, states, background_mode, workers,
                 clean_id, run_music=True):
    """比较两档数值参数。

    缓存纪律：`max_states` **不在**物理签名内（它只是前缀长度），因此每档必须落在
    **不同的缓存子目录**，否则两档会互相拒绝复用。这里给每档独立 tag：
    low 用自己的 `res_rt`，high 用 `res_rt`；两档目录不同、签名各自自洽。
    比较时统一截到相同的状态数（两档都取 `states`），保证逐状态一一对应。
    """
    lo = _build(root, rec, tag="%s_%s" % (tag, low["id"]), geometry=geometry,
                resolution=low["res"], run_time=low["time"], states=states,
                background_mode=background_mode, workers=workers)
    hi = _build(root, rec, tag="%s_%s" % (tag, high["id"]), geometry=geometry,
                resolution=high["res"], run_time=high["time"], states=states,
                background_mode=background_mode, workers=workers)
    a = np.asarray(lo["raw_cir"])[: int(states)]
    b = np.asarray(hi["raw_cir"])[: int(states)]
    if a.shape != b.shape:
        raise RuntimeError("两档状态数不一致：%s vs %s" % (a.shape, b.shape))
    diff = _common_gain_and_shape(a, b)
    diff.update(_peak_shift(a, b))
    c1 = diff["abs_correlation"] >= CRITERIA["min_abs_correlation"]
    c2 = diff["shape_rel_l2"] is not None and diff["shape_rel_l2"] <= CRITERIA["max_shape_rel_l2"]
    c3 = diff["max_peak_tap_shift"] <= CRITERIA["max_peak_tap_shift"]
    music = _music_compare(a, b, rec, background_mode, clean_id, geometry=geometry) if run_music else None
    c4 = bool(music and music["passed"])
    return {
        "tag": tag, "scene_id": clean_id, "low": low, "high": high,
        "n_states_compared": int(a.shape[0]),
        "diff": diff,
        "criteria": dict(CRITERIA),
        "checks": {"abs_correlation_ge_0.90": bool(c1), "shape_rel_l2_le_0.30": bool(c2),
                   "peak_tap_shift_le_1": bool(c3),
                   "music_outputs_match_le_0.05m": bool(c4)},
        "music": music,
        "passed": bool(c1 and c2 and c3 and c4),
        "note": ("公共最优复数增益用于形状误差诊断；同时保留未经补偿的 raw_rel_l2 与增益值。"
                 "预检只用 %d 个真实状态，不代表全部 128/160 状态已收敛。" % int(a.shape[0])),
    }


def run(profile: str, output_root, protocol_path=None, resolution=None, run_time=None,
        states: int = 16, workers: int = 4, escalate: bool = True, only=None) -> int:
    root = Path(output_root)
    out_dir = root / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("== FDTD 数值前检：profile=%s states=%d workers=%d ==" % (profile, states, workers))
    scenes = representative_scenes(profile)
    if only:
        scenes = [s for s in scenes if s[0] in set(only)]
    if not scenes:
        print("没有匹配的代表场景")
        return 1
    bg_mode = "empty_field" if profile == "controlled" else "ema_windowed"
    # 每档用**自己的唯一 id** 作为缓存目录后缀：同一物理参数在不同 stage 中也不会互相误用，
    # 且因为 max_states 不进签名，预检缓存可被正式批次直接复用（16 状态是 128 状态的前缀）。
    n250_20 = {"id": "r250t20", "res": 250.0, "time": 20.0}
    n250_40 = {"id": "r250t40", "res": 250.0, "time": 40.0}
    n350_20 = {"id": "r350t20", "res": 350.0, "time": 20.0}
    n350_40 = {"id": "r350t40", "res": 350.0, "time": 40.0}
    n350_80 = {"id": "r350t80", "res": 350.0, "time": 80.0}
    n500_20 = {"id": "r500t20", "res": 500.0, "time": 20.0}
    stages = [("stage1", n250_20, n250_40), ("stage2", n250_20, n350_20)]
    escalation = [("stage3", n350_40, n350_80), ("stage4", n350_20, n500_20)]
    results, blockers = [], []
    for tag, rec, geometry in scenes:
        cid = rec["clean_id"]
        scene_res = {"tag": tag, "clean_id": cid, "stages": [], "passed": False,
                     "attempted": [s[0] for s in stages + (escalation if escalate else [])]}
        for stage_name, low, high in (stages + (escalation if escalate else [])):
            try:
                r = compare_pair(root, rec, geometry=geometry, tag="%s_%s" % (tag, stage_name),
                                 low=low, high=high, states=states,
                                 background_mode=bg_mode, workers=workers, clean_id=cid)
            except Exception as exc:  # noqa: BLE001
                r = {"tag": "%s_%s" % (tag, stage_name), "scene_id": cid, "error":
                     "%s: %s" % (type(exc).__name__, exc), "passed": False}
            scene_res["stages"].append(r)
            # 时间与空间网格需要分别有通过证据；某一项通过不能跳过另一项。
            passed_names = {s["tag"].rsplit("_", 1)[-1] for s in scene_res["stages"]
                            if s.get("passed")}
            time_ok = bool(passed_names & {"stage1", "stage3"})
            grid_ok = bool(passed_names & {"stage2", "stage4"})
            scene_res["passed"] = time_ok and grid_ok
            scene_res["axes_passed"] = {"time": time_ok, "grid": grid_ok}
            if scene_res["passed"]:
                scene_res["decided_by"] = sorted(passed_names)
                break
        if not scene_res["passed"]:
            scene_res["status"] = "blocked"
            scene_res["reason"] = ("两档数值参数下未通过固定判据；已保留诊断，"
                                   "不自动从正式计划删除该场景")
            blockers.append({"tag": tag, "clean_id": cid})
        results.append(scene_res)
        print("  %-10s %s" % (tag, "PASS" if scene_res["passed"] else "NOT_PASSED"))
    doc = {"kind": "jsf-fdtd-convergence", "profile": profile, "states": int(states),
           "criteria": dict(CRITERIA), "n_scenes": len(results),
           "n_passed": sum(1 for r in results if r["passed"]),
           "blocked": blockers, "results": results,
           "escalation_enabled": bool(escalate),
           "escalation_note": ("按规范 §10 先比 20/40 与 250/350（第一阶段两档）；"
                              "第三/四档（40/80、350/500）仅在本 flag 为真时启用。"),
           "note": ("判据在运行前冻结；只改被测数值参数，物理域不随分辨率变；"
                    "局部预检不代表全部 128/160 状态收敛。")}
    sc = paths.sim_common()
    sc.atomic_write_json(out_dir / ("convergence_%s.json" % profile), doc)
    print(json.dumps({"profile": profile, "n_passed": doc["n_passed"],
                      "n_scenes": doc["n_scenes"], "blocked": blockers}, ensure_ascii=False, indent=1))
    return 0 if doc["n_passed"] == doc["n_scenes"] else 2


__all__ = ["run", "compare_pair", "representative_scenes", "CRITERIA"]
