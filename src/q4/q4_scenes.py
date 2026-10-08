# -*- coding: utf-8 -*-
"""Q4 场景导出：inventory 的 15 几何 -> **合法 FDTD scene**（``adapter.scene_export.spec_to_scene``）
+ ``fdtd.api.plan_run`` 验证（**不导入 Meep、不跑 FDTD**）。

* Q4 几何 = 硬件物理 ``tx/rx/pairs``（inventory ``fdtd_geometry``），**不套 main 1Tx 几何**。
* 背景默认 ``ema_windowed``：``plan_run`` 会按真实 burnin 状态计入 n_solves（不得复制快照）。
* 真值为 nominal 静态点；轨迹 provenance 声明与硬件实际不同；本轮 rho 不加噪声、未校准。
* 本轮**仅脚本与 fixture**，不运行 15 FDTD；``generate(run=False)`` 只做 plan_run 校验。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from .q4_batch import DEFAULT_INVENTORY, geometry_index  # noqa: E402
from .q4_common import Q4Error, write_json  # noqa: E402

Q4_BACKGROUND_MODE = "ema_windowed"
Q4_GEOMETRY_OVERRIDE = {
    "use": "manifest_real_txrx_pairs",
    "source": "inventory.geometries[*].fdtd_geometry",
    "must_not_use": "main_1Tx_geometry(sim_common.TX_POSITIONS_M/PAIRS)",
    "note": "经 adapter.scene_export.spec_to_scene(geometry=...) 与 case_build.build_case(geometry=...) 覆盖",
}
Q4_PROVENANCE = {
    "rho_input": "clean_sim_vs_hardware_fixed_rule",
    "added_noise": "none",
    "calibration": "uncalibrated(no_independent_calibration_set)",
    "sim_truth_kind": "nominal_static_points_matched_to_hardware_identity",
    "sim_microtrajectory": "differs_from_hardware_reality(declared)",
    "background_mode": Q4_BACKGROUND_MODE,
}


def q4_geometry(geometry_block: dict) -> dict:
    """manifest ``fdtd_geometry`` -> spec_to_scene/build_case 需要的几何覆盖 dict。"""
    tx = geometry_block.get("tx_positions") or []
    rx = geometry_block.get("rx_positions") or []
    pairs = geometry_block.get("pairs") or []
    if not (tx and rx and pairs):
        raise Q4Error("manifest geometry 缺 tx_positions/rx_positions/pairs")
    return {"tx_positions_m": [[float(v) for v in p] for p in tx],
            "rx_positions_m": [[float(v) for v in p] for p in rx],
            "pairs": [[int(a), int(b)] for a, b in pairs]}


def export_scene_specs(*, inventory_json=DEFAULT_INVENTORY, out_dir,
                       background_mode=Q4_BACKGROUND_MODE) -> dict:
    """15 几何 -> 合法 scene JSON（并逐个 plan_run 验证，无 Meep）。"""
    from adapter import scene_export
    from fdtd import api as fdtd_api
    from protocol.sim_protocol import PROTOCOL

    idx = geometry_index(inventory_json)
    geoms = idx["geometries"]
    if not geoms:
        raise Q4Error("inventory 无几何（先跑 q4_cli.py inventory）：%s" % idx["source"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenes = []
    for gid in sorted(geoms):
        g = geoms[gid]
        geom = q4_geometry(g.get("fdtd_geometry") or {})
        spec = {"clean_id": "q4_%s" % gid, "k": int(g.get("n_targets") or len(g.get("truth_xy") or [])),
                "targets_xy": g.get("truth_xy")}
        scene = scene_export.spec_to_scene(spec, protocol=PROTOCOL, geometry=geom)
        # 背景必须写进 scene（否则后续 convergence 会按默认 empty_field 误检）
        scene["background"] = {"mode": background_mode,
                               "ema": {"bg_m_factor": 4.0, "alpha": 0.25,
                                       "burnin_states": 8, "first_raw": True}}
        plan = fdtd_api.plan_run(scene=scene, background_mode=background_mode)
        write_json(out / (gid + ".scene.json"), scene)
        write_json(out / (gid + ".plan.json"), plan)
        scenes.append({"geometry_id": gid, "scene_id": scene["scene_id"],
                       "n_targets": int(plan["n_targets"]),
                       "snapshots": plan["state_count"]["snapshots"],
                       "burnin_states": plan["state_count"]["burnin_states"],
                       "states_total": plan["state_count"]["states_total"],
                       "geometry": geom})
    doc = {"kind": "q4-scenes-export", "schema_version": 2,
           "inventory": idx["source"], "n_geometries": len(scenes),
           "background_mode": background_mode, "geometry_override": Q4_GEOMETRY_OVERRIDE,
           "provenance": Q4_PROVENANCE, "scenes": scenes,
           "note": "经 adapter.scene_export.spec_to_scene 生成合法 scene，并用 fdtd.api.plan_run 验证；未跑 Meep"}
    write_json(out / "q4_scenes_manifest.json", doc)
    return doc


def generate(*, scene_dir, out_dir, run=False, background_mode=Q4_BACKGROUND_MODE,
             evidence_dir=None, workers=1) -> dict:
    """逐个 scene：``fdtd.api.plan_run`` 校验（无 Meep）。``run=True`` 才真正调用 generate_clean。

    workers > 1 时 generate_clean 内部先 spawn 多进程并行预填状态缓存（数值路径与串行一致）。
    """
    from fdtd import api as fdtd_api
    from fdtd import convergence as conv
    scene_dir = Path(scene_dir)
    scenes = sorted(scene_dir.glob("*.scene.json"))
    if not scenes:
        raise Q4Error("scene_dir 无 *.scene.json：%s（先跑 scenes-export）" % scene_dir)
    steps = []
    for s in scenes:
        scene = json.loads(s.read_text(encoding="utf-8"))
        plan = fdtd_api.plan_run(scene=scene, background_mode=background_mode)
        step = {"scene_id": scene["scene_id"], "scene": str(s), "ran": False,
                "states_total": plan["state_count"]["states_total"],
                "burnin_states": plan["state_count"]["burnin_states"]}
        if run:
            ev = conv.check_evidence(evidence_dir, scene=scene) if evidence_dir else \
                {"passed": False, "reasons": ["未提供 --evidence-dir（正式前检）"]}
            if not ev.get("passed"):
                raise Q4Error("正式前检未通过，拒绝 generate（不绕 convergence.check_evidence）：%s"
                              % ev.get("reasons"))
            outdir = Path(out_dir) / str(scene["scene_id"])
            fdtd_api.generate_clean(None, scene, outdir, background_mode=background_mode,
                                    workers=int(workers))
            step["ran"] = True
        steps.append(step)
    doc = {"kind": "q4-generate", "schema_version": 2,
           "mode": "run" if run else "plan_only", "background_mode": background_mode,
           "workers": int(workers),
           "evidence_dir": str(evidence_dir) if evidence_dir else None,
           "n_scenes": len(steps), "steps": steps, "provenance": Q4_PROVENANCE,
           "note": "plan_only 只做 plan_run（无 Meep）；run 需显式并会真正跑 Meep"}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(out_dir) / "q4_generate.json", doc)
    return doc
