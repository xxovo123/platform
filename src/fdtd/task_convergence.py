#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务级（QoI）网格收敛检查：res_a vs res_b 五法定位结果一致性（2026-09-27 用户批准）。

动机
----
部分场景的多径干涉图样对网格微扰呈混沌敏感，CIR 波形级收敛（shape_rel_l2，判据 v2）
在该类场景收敛缓慢（350/450 仍 >0.30）；而层 B 的研究结论落在**定位任务**上。
数值实验验证的标准做法是对**感兴趣量（QoI）**做网格收敛：本检查以定位预测为 QoI。

做法（v2）
----------
同一批状态（前 n_states 个，冻结）、同一前端管线（cropped_from_raw → x_from_cropped）、
同一冻结方法参数，在 res_a / res_b 两套网格各生成 clean CIR；**逐分辨率各建一个整 case**
（快照数 = n_states，与正式层 B 的多快照协方差前提一致；v1 的逐状态 L=1 case 会破坏
b1/b2 等方法的协方差前提——本地复现确认 status=failed——故废弃），各跑五法定位，
比较两套预测的一致性。

判据（运行前冻结写入 criteria.json，不得改写；不同判据须换目录）
----------------------------------------------------------------
* require_all_methods_count_equal = true：5/5 方法的预测数一致（某侧 status≠ok 或
  预测数不一致 → 该方法记为不一致，失败不隐藏）
* max_mean_pred_shift_m = 0.05：全部配对预测的平均位移 ≤ 0.05 m（< 半距离格 0.075 m）
* 配对规则：逐方法对 res_a/res_b 预测做贪心最近邻配对
通过 = 全部判据满足。strict_params 与 CIR 收敛同一函数（res/runtime 不进）。
噪声口径：clean CIR（不加噪）——网格敏感性是确定性正向解的属性，与实现一致披露。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import api
from .protocol import atomic_write_json, read_json

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

DEFAULT_TASK_CRITERIA: Dict[str, Any] = {
    "criteria_version": 2,
    "kind": "task-convergence",
    "n_states": 16,
    "require_all_methods_count_equal": True,
    "max_mean_pred_shift_m": 0.05,
    "_note": ("判据运行前冻结（criteria_frozen_before_run=true）；不满足即 not_passed，不得改写。"
              "v2（2026-09-28）：v1 逐状态单快照（L=1）case 破坏 b1/b2 等多快照方法的协方差前提"
              "（本地复现 status=failed），废弃；v2 改为逐分辨率整 case（快照数 = n_states，"
              "与正式层 B 多快照前提一致）跑五法，判据 = 5/5 方法预测数一致 且 配对平均位移 "
              "≤ 0.05 m（< 半距离格 0.075 m）。clean CIR（不加噪），配对 = 逐方法贪心最近邻。"),
}


def freeze_task_criteria(outdir: Path, criteria: Dict[str, Any]) -> None:
    """判据先冻结写入 criteria.json；已存在不同判据 → 拒绝（与 CIR 收敛同一纪律）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "criteria.json"
    if p.is_file():
        old = json.loads(p.read_text(encoding="utf-8"))
        if json.dumps(old, sort_keys=True) != json.dumps(criteria, sort_keys=True):
            raise RuntimeError("已存在不同判据 %s：不得覆盖已冻结判据，须换目录" % p)
        return
    doc = dict(criteria)
    doc["criteria_frozen_before_run"] = True
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _greedy_match(pa: List[List[float]], pb: List[List[float]]):
    """贪心最近邻配对；返回 [(dist, idx_a, idx_b)]。"""
    pairs = []
    used = set()
    for i, p in enumerate(pa):
        best = None
        for j, q in enumerate(pb):
            if j in used:
                continue
            d = math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            used.add(best[1])
            pairs.append((float(best[0]), i, best[1]))
    return pairs


def _preds(res) -> List[List[float]]:
    return [[float(p["x"]), float(p["y"])] for p in (res.get("predictions") or [])]


def run_task_convergence(config: Any, scene_a: Any, scene_b: Any, outdir: str | Path, *,
                         methods: List[str] | None = None, background_mode: str | None = None,
                         workers: int = 1, criteria_in: Any = None,
                         quiet: bool = False) -> tuple:
    """跑 QoI 检查并写 task_convergence.json；返回 (report, exit_code)。"""
    from adapter import case_build, frontend, methods_loader
    import sim_common as sc
    from protocol.sim_protocol import PROTOCOL as SIM_PROTOCOL

    def say(msg):
        if not quiet:
            print("[task-conv] %s" % msg, flush=True)

    outdir = Path(outdir)
    criteria = dict(DEFAULT_TASK_CRITERIA)
    if criteria_in:
        criteria = dict(read_json(criteria_in) if Path(criteria_in).is_file()
                        else json.loads(criteria_in))
    criteria["criteria_frozen_before_run"] = True
    freeze_task_criteria(outdir, criteria)
    n_states = int(criteria["n_states"])

    say("生成 res_a / res_b 各 %d 状态（workers=%d）" % (n_states, workers))
    ra = api.generate_clean(config, scene_a, outdir / "run_resA", resume=True,
                            max_states=n_states, background_mode=background_mode,
                            workers=int(workers), progress=(None if quiet else say))
    rb = api.generate_clean(config, scene_b, outdir / "run_resB", resume=True,
                            max_states=n_states, background_mode=background_mode,
                            workers=int(workers), progress=(None if quiet else say))

    def load_raw(npz_path):
        with np.load(npz_path, allow_pickle=False) as z:
            return np.asarray(z["raw_cir"])

    raw_a = load_raw(ra["clean_npz"])
    raw_b = load_raw(rb["clean_npz"])
    if raw_a.shape != raw_b.shape:
        raise RuntimeError("两分辨率 raw_cir 形状不一致：%s vs %s" % (raw_a.shape, raw_b.shape))
    if int(raw_a.shape[0]) < n_states:
        raise RuntimeError("状态数不足：需要 %d，得到 %d" % (n_states, raw_a.shape[0]))

    methods = list(methods) if methods else ["this_method", "b1_deng", "b2_kakouche",
                                             "c1_jade", "c3_kim"]
    protocol_dict = SIM_PROTOCOL
    params = {m: methods_loader.params_for(m, protocol_dict) for m in methods}
    gx, gy = sc.build_grid()
    scene_obj_a = api.scenes_mod.load_scene(scene_a)
    scene_obj_b = api.scenes_mod.load_scene(scene_b)
    K_a = int(np.asarray(scene_obj_a["targets"]["nominal_centers_m"]).shape[0])
    K_b = int(np.asarray(scene_obj_b["targets"]["nominal_centers_m"]).shape[0])
    if K_a != K_b:
        raise RuntimeError("两分辨率场景目标数不一致：%d vs %d" % (K_a, K_b))

    # 逐分辨率整 case（快照数 = n_states）：与正式层 B 的多快照前提一致
    cases = {}
    preds = {}
    statuses = {}
    for tag, raw, scene_obj in (("a", raw_a, scene_obj_a), ("b", raw_b, scene_obj_b)):
        cc = frontend.cropped_from_raw(np.asarray(raw)[:n_states])
        X = frontend.x_from_cropped(cc)
        case = case_build.build_case(
            case_id="taskconv_%s" % tag, clean_id=str(scene_obj.get("scene_id", "taskconv")),
            X=X, k=K_a, grid_x=gx, grid_y=gy, layer="B", source="fdtd",
            scene="task_conv", tags=[], snr_db=0.0, trial=0, seed=0, L=int(X.shape[0]))
        case_build.validate_case(case)
        cases[tag] = case
        for m in methods:
            res = methods_loader.call(m, case, params[m])
            preds[(tag, m)] = _preds(res)
            statuses[(tag, m)] = str(res.get("status"))
            say("res_%s %-12s status=%s n_pred=%d" % (tag, m, statuses[(tag, m)], len(preds[(tag, m)])))

    per_method = []
    displacements: List[float] = []
    n_equal = 0
    for m in methods:
        pa, pb = preds[("a", m)], preds[("b", m)]
        ok_status = (statuses[("a", m)] == "ok" and statuses[("b", m)] == "ok")
        equal = bool(ok_status and len(pa) == len(pb))
        mean_disp = None
        max_disp = None
        if equal and pa:
            pairs = _greedy_match(pa, pb)
            if len(pairs) == len(pa):
                disp = [d for d, _i, _j in pairs]
                displacements.extend(disp)
                mean_disp = float(np.mean(disp))
                max_disp = float(np.max(disp))
            else:
                equal = False
        if equal:
            n_equal += 1
        per_method.append({"method": m, "n_pred_a": len(pa), "n_pred_b": len(pb),
                           "status_a": statuses[("a", m)], "status_b": statuses[("b", m)],
                           "count_equal": bool(equal), "mean_disp_m": mean_disp,
                           "max_disp_m": max_disp})
        say("方法 %-12s count_equal=%s mean_disp=%s max_disp=%s" % (
            m, equal, ("%.4f" % mean_disp) if mean_disp is not None else "-",
            ("%.4f" % max_disp) if max_disp is not None else "-"))

    agreement = (n_equal / len(methods)) if methods else 0.0
    mean_shift = (float(np.mean(displacements)) if displacements else None)
    require_all = bool(criteria.get("require_all_methods_count_equal", True))
    ok = (agreement >= float(criteria["min_detection_pattern_agreement"])
          if "min_detection_pattern_agreement" in criteria else True)
    if require_all:
        ok = ok and (n_equal == len(methods))
    ok = ok and mean_shift is not None and mean_shift <= float(criteria["max_mean_pred_shift_m"])
    verdict = "passed" if ok else "not_passed"

    cfg_a = api._load_cfg(config, scene_obj_a)
    scene_obj_a["domain"] = api.scenes_mod.resolve_domain(scene_obj_a, fc_hz=float(cfg_a["fc_hz"]))
    from .convergence import strict_params
    report = {
        "kind": "task-convergence", "formal": bool(ok and verdict == "passed"),
        "verdict": verdict, "criteria": criteria,
        "res_a": float(scene_obj_a["meep"]["resolution_px_per_m"]),
        "res_b": float(scene_obj_b["meep"]["resolution_px_per_m"]),
        "n_states": int(n_states), "methods": methods,
        "runs_total": len(methods), "n_equal": n_equal,
        "detection_pattern_agreement": float(agreement),
        "mean_pred_shift_m": mean_shift,
        "per_method": per_method,
        "strict_params": strict_params(cfg_a, scene_obj_a),
        "exit_code": 0 if ok else 2,
        "note": ("verdict=not_passed 必须如实报告，不得改判据；本报告为任务级（QoI）网格收敛证据，"
                 "与 CIR 波形级证据（convergence.json）并列，由 check_evidence 识别"),
        "generated_at": api._now_iso(),
    }
    atomic_write_json(outdir / "task_convergence.json", report)
    say("verdict=%s agreement=%.2f mean_shift=%s" % (
        verdict, agreement, ("%.4f" % mean_shift) if mean_shift is not None else "-"))
    return report, (0 if ok else 2)
