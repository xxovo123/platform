#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1/E2/E4 spec -> FDTD scene JSON（供 fdtd/api.py::generate_clean 直接消费）。

不等待人为映射：本模块把层 A/B 的 spec 直接转成 fdtd/scenes.py 认可的场景字典。
几何与层 A 完全一致（1Tx4Rx 受控主配置，2D xy 投影）；micro-motion seed 由 clean_id 派生，
保证 B 层无噪轨迹可复现且与 clean 身份绑定（不含 SNR/L/family）。
"""

from __future__ import annotations

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np


class SceneExportError(RuntimeError):
    """spec 不可转成 FDTD scene（显式失败）。"""


def _fdtd_defaults(protocol) -> dict:
    d = (protocol or {}).get("fdtd_defaults")
    if not isinstance(d, dict):
        raise SceneExportError("protocol.fdtd_defaults 缺失：FDTD 默认值必须来自协议（不留模块私default）")
    for key in ("scatterer_radius_m", "micro_motion_sigma_m", "micro_motion_radius_m",
                "resolution_px_per_m", "run_time_meep"):
        if key not in d:
            raise SceneExportError("protocol.fdtd_defaults 缺 %s" % key)
    return d


def spec_to_scene(spec: dict, *, protocol: dict | None = None, geometry: dict | None = None,
                  res_override: float | int | None = None) -> dict:
    """spec -> scene dict（fdtd/scenes.py 字段）。

    geometry 选填：Q4 等专用输入几何可覆盖默认 1Tx4Rx（默认仍为主 1Tx4Rx，不改默认行为）。
    覆盖项为 dict{tx_positions_m, rx_positions_m, pairs}（缺项回落到协议主几何）。
    res_override 选填：场景级 Meep 分辨率覆盖（网格收敛逐场景选口径用；数据层覆盖，
    不改协议冻结文件；进 cache key，不进 protocol json）。
    """
    from protocol.sim_protocol import PROTOCOL
    proto = protocol or PROTOCOL
    d = _fdtd_defaults(proto)
    k = int(spec["k"])
    if k < 1:
        raise SceneExportError("k 必须 >= 1")
    clean_id = str(spec["clean_id"])
    centers = [[float(x), float(y), 0.0] for x, y in spec["targets_xy"]]
    r_traj = float(d["micro_motion_radius_m"])
    geom = dict(geometry or {})
    scene = {
        # scene_id = clean_id（几何/源）；case_id 属运行标签，**不进物理/cache key**
        "scene_id": clean_id,
        "geometry": {
            "tx_positions_m": [[float(v) for v in row] for row in
                               (geom.get("tx_positions_m") or sc.TX_POSITIONS_M.tolist())],
            "rx_positions_m": [[float(v) for v in row] for row in
                               (geom.get("rx_positions_m") or sc.RX_POSITIONS_M.tolist())],
            "pairs": [[int(v) for v in row] for row in (geom.get("pairs") or sc.PAIRS.tolist())],
            "projection": {"plane": "xy", "dropped_axis": "z",
                           "note": "2D FDTD 不表达离面 z；只保留 xy 投影，不把 z 当 y"},
        },
        "targets": {
            "nominal_centers_m": centers,
            "radius_m": float(d["scatterer_radius_m"]),
            "overlap_margin_m": 0.005,
            "micro_motion": {
                "seed": int(sc.clean_seed(clean_id)),
                "sigma_m": float(d["micro_motion_sigma_m"]),
                "radius_m": [r_traj] * k,
            },
        },
        "meep": {
            "resolution_px_per_m": float(d["resolution_px_per_m"]),
            "run_time_meep": float(d["run_time_meep"]),
            "pml_thickness_lambda": float(d.get("pml_thickness_lambda", 2.0)),
        },
        "background": {"mode": str(d.get("background_mode", "empty_field"))},
        "snapshots": int(spec.get("L_clean", sc.SNAPSHOTS)),
        "sim_clean_id": clean_id,
    }
    if res_override is not None:
        res_f = float(res_override)
        if res_f <= 0:
            raise SceneExportError("res_override 必须为正数，得到 %r" % (res_override,))
        scene["meep"]["resolution_px_per_m"] = res_f
        scene["meep"]["res_override"] = {
            "resolution_px_per_m": res_f,
            "note": ("场景级分辨率覆盖（网格收敛逐场景选口径）；数据层覆盖，不改协议冻结文件；"
                     "进 cache key 与门禁 want_res，来源与依据记录于运行 provenance/状态账本"),
        }
    return scene


def separation_check(spec: dict, *, protocol: dict | None = None) -> dict:
    """预设计几何合法性检查（多目标间距 vs 不交叠规则）；违规显式失败。"""
    from protocol.sim_protocol import PROTOCOL
    proto = protocol or PROTOCOL
    d = _fdtd_defaults(proto)
    xy = np.asarray(spec["targets_xy"], dtype=float)
    if xy.shape[0] < 2:
        return {"min_separation_m": None, "required_m": None, "ok": True}
    req = 2.0 * float(d["scatterer_radius_m"]) + 2.0 * float(d["micro_motion_radius_m"]) + 0.005
    best = float("inf")
    for i in range(xy.shape[0]):
        for j in range(i + 1, xy.shape[0]):
            best = min(best, float(np.linalg.norm(xy[i] - xy[j])))
    if best < req:
        raise SceneExportError("目标间距 %.4f m < 不交叠要求 %.4f m（%s）"
                               % (best, req, spec.get("case_id")))
    return {"min_separation_m": best, "required_m": req, "ok": True}


def scenes_for_specs(specs, *, protocol: dict | None = None, res_overrides: dict | None = None) -> dict:
    """clean_id -> scene dict（按 clean 去重；逐 spec 做几何安全检查）。

    res_overrides 选填：{clean_id: resolution}，网格收敛逐场景选口径用（见 spec_to_scene）。
    """
    from analytic import scenes as scene_mod
    out = {}
    ro = dict(res_overrides or {})
    for spec in specs:
        separation_check(spec, protocol=protocol)
        cid = spec["clean_id"]
        scene = spec_to_scene(spec, protocol=protocol, res_override=ro.get(cid))
        if cid in out:
            if out[cid]["targets"]["nominal_centers_m"] != scene["targets"]["nominal_centers_m"]:
                raise SceneExportError("clean_id=%s 的目标不一致：FDTD 同几何必须唯一" % cid)
        else:
            out[cid] = scene
    return out


__all__ = ["spec_to_scene", "scenes_for_specs", "separation_check", "SceneExportError"]
