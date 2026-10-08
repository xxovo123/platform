#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Meep 2D 全波求解（**只有本模块在真正求解时惰性导入 meep**）。

物理口径（2026-09-26 冻结，含三处审阅修正）
--------------------------------------------
* 2D（xy）全场 FDTD；长度单位**只允许 1 m**（``length_unit_m == 1.0``），坐标按米直接送
  Meep，``resolution`` 按 pixels/m；非 1 m 单位直接报错，不假装支持多单位。
* **多目标在同一 geometry 内同时存在**（同一次求解），不是“单目标结果相加”。
* **每个 slow-time 状态、每个 Tx 激励各解一次**：``N_solve = (1 背景 + N_state)·N_tx``。
* 目标：金属圆柱，半径 ``1.5·λc``（与旧工程 ``mover_radius`` 同口径）。
* 源：与 MF 参考**同一条** ``pulse.bh_derivative_passband``；``duration = 2/B``。
  **单位与幅度**：Meep 时间单位是 ``1 m / c``，故 Meep 单位下的时间导数比 SI 导数小
  ``sec_per_meep_time`` 倍 —— 这是**全局增益**，不影响波形形状。``source_normalize_to_si=true``
  时显式乘回 ``1/sec_per_meep_time``，使源与 SI 参考同值；该增益与剩余求解 cell transfer
  （源幅度约定、像素内插、数值色散）一并写进 provenance。**绝对幅度未标定**，只用于形状量
  （ρ、峰位），不得用于跨域绝对幅度结论。
* 时间轴：**不假定** ``at_every`` 首调用在 t=0 —— 逐步记录 ``sim.meep_time()`` 的真实时间戳，
  用同一条时间轴生成源 MF 参考；并校验首样本≈0、等间隔（不满足即报错，不外推）。
* ``Courant``：Meep 默认 **0.5**（不是 1/sqrt(2)）；显式从 ``scene.meep.courant`` 传给
  ``mp.Simulation``，预算步数用同一值，避免低估步数。
* 不做任何“按真值调时延”：求解只出原始场记录。
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from . import pulse as pulse_mod


def meep_available() -> bool:
    try:
        import meep  # noqa: F401

        return True
    except Exception:
        return False


def _import_meep():
    try:
        import meep as mp
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "Meep 不可用（%s）。求解必须在远程 mp 环境执行；plan/--help 不需要 Meep。" % exc
        ) from exc
    return mp


def meep_units(scene: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """单位换算与数值参数（单位 / Courant / 源增益 都在此集中裁定）。"""
    c0 = float(cfg["c0_m_per_s"])
    unit = float(scene.get("length_unit_m", 1.0))
    if unit != 1.0:
        raise ValueError(
            "本实现只支持 length_unit_m == 1.0（得到 %r）：坐标按米直接送 Meep，"
            "多单位会同时影响 resolution/PML/几何定位" % unit
        )
    sec_per_meep_time = unit / c0
    duration_s = pulse_mod.duration_from_bandwidth(cfg["bandwidth_hz"])
    meep_cfg = dict(scene.get("meep", {}))
    courant = float(meep_cfg.get("courant", 0.5))
    normalize = bool(cfg.get("source_normalize_to_si", True))
    return {
        "length_unit_m": unit,
        "seconds_per_meep_time": sec_per_meep_time,
        "fc_meep": float(cfg["fc_hz"]) * sec_per_meep_time,
        "duration_s": duration_s,
        "duration_meep": duration_s / sec_per_meep_time,
        "run_time_meep": float(scene["meep"]["run_time_meep"]),
        "resolution_px_per_m": float(scene["meep"]["resolution_px_per_m"]),
        "pml_thickness_meep": float(scene["domain"]["pml_thickness_m"]) / unit,
        "dimensions": int(scene.get("dimensions", 2)),
        "courant": courant,
        "source_normalize_to_si": normalize,
        "source_gain": (1.0 / sec_per_meep_time) if normalize else 1.0,
    }


def estimate_cost(
    scene: Mapping[str, Any],
    cfg: Mapping[str, Any],
    n_target_states: int,
    n_background: int,
    measured_cell_updates_per_s: float | None = None,
) -> Dict[str, Any]:
    """求解预算（**估计**，不冒充实测）：格子数 × 步数 × 求解次数。

    ``dt_est = Courant/resolution``，Courant 取 ``scene.meep.courant``（默认 0.5，与 Meep 一致）。
    """
    units = meep_units(scene, cfg)
    res = units["resolution_px_per_m"]
    size = [float(v) for v in scene["domain"]["size_m"]]
    n_cells = int(math.ceil(size[0] * res)) * int(math.ceil(size[1] * res))
    dt_est = units["courant"] / res
    n_steps = int(units["run_time_meep"] / dt_est) + 1
    n_tx = int(np.asarray(scene["geometry"]["tx_positions_m"], dtype=float).shape[0])
    n_solves = (int(n_background) + int(n_target_states)) * n_tx
    updates = float(n_cells) * float(n_steps) * float(n_solves)
    out: Dict[str, Any] = {
        "kind": "estimate_only",
        "basis": "n_solves = (n_background + n_states) * n_tx；dt_est = Courant/resolution（显式 Courant，Meep 默认 0.5）",
        "n_cells": n_cells,
        "courant": units["courant"],
        "dt_meep_est": dt_est,
        "n_steps_est": n_steps,
        "n_tx": n_tx,
        "n_background_solves_per_tx": int(n_background),
        "n_target_states": int(n_target_states),
        "n_solves": n_solves,
        "cell_updates_est": updates,
        "note": "真实 dt/步数以 sim.fields.dt 为准；本表仅为预算估计，不是实测性能",
    }
    if measured_cell_updates_per_s:
        rate = float(measured_cell_updates_per_s)
        out["measured_cell_updates_per_s"] = rate
        out["est_core_hours_from_measured_rate"] = updates / rate / 3600.0
    return out


def _to_meep_point(mp, p2_m: Sequence[float], center_m: Sequence[float]):
    """xy（米）→ Meep 坐标：**不除 length_unit_m**（单位已强制 1 m），只做域中心平移。"""
    return mp.Vector3(float(p2_m[0]) - float(center_m[0]), float(p2_m[1]) - float(center_m[1]), 0.0)


def run_single_solve(
    scene: Mapping[str, Any],
    cfg: Mapping[str, Any],
    centers_2d_m: np.ndarray | None,
    tx_index: int,
    *,
    record: bool = True,
) -> Dict[str, Any]:
    """一次求解：``centers_2d_m``（``None`` = 空场背景）在**同一 geometry 内同时存在**，
    单个 Tx 激励。返回原始通带场记录（**未做任何背景/时延处理**）。
    """
    mp = _import_meep()
    units = meep_units(scene, cfg)
    dom = scene["domain"]
    ctr = [float(v) for v in dom["center_m"]]
    geom_cfg = scene["geometry"]
    tx_positions = np.asarray(geom_cfg["tx_positions_m"], dtype=float)[:, :2]
    rx_positions = np.asarray(geom_cfg["rx_positions_m"], dtype=float)[:, :2]
    radius_m = float(scene["targets"].get("radius_m", 0.0))
    material_name = str(scene["targets"].get("material", "metal"))
    if material_name != "metal":
        raise ValueError("当前只实现金属目标（mp.metal）；得到 %s" % material_name)

    geometry: List[Any] = []
    n_in_geometry = 0
    if centers_2d_m is not None:
        centers_arr = np.asarray(centers_2d_m, dtype=float)
        if centers_arr.ndim != 2 or centers_arr.shape[1] != 2:
            raise ValueError("centers_2d_m 形状应为 [K, 2]，得到 %s" % (centers_arr.shape,))
        n_in_geometry = int(centers_arr.shape[0])
        for p in centers_arr:
            geometry.append(
                mp.Cylinder(
                    center=_to_meep_point(mp, p, ctr),
                    radius=radius_m,
                    height=mp.inf,
                    material=mp.metal,
                )
            )

    duration_meep = units["duration_meep"]
    fc_meep = units["fc_meep"]
    gain = float(units["source_gain"])

    def src_func(t):
        return gain * pulse_mod.bh_derivative_passband(t, fc_meep, duration_meep)

    try:
        src_time = mp.CustomSource(src_func=src_func, end_time=duration_meep)
    except TypeError:  # 兼容不同 Meep 版本的 CustomSource 签名
        src_time = mp.CustomSource(src_func=src_func)
    tx_pt = _to_meep_point(mp, tx_positions[int(tx_index)], ctr)
    source = mp.Source(src=src_time, component=mp.Ez, center=tx_pt)

    size = [float(v) for v in dom["size_m"]]
    sim = mp.Simulation(
        cell_size=mp.Vector3(size[0], size[1], 0.0),
        resolution=units["resolution_px_per_m"],
        Courant=units["courant"],
        dimensions=units["dimensions"],
        boundary_layers=[mp.PML(units["pml_thickness_meep"])],
        geometry=geometry,
        sources=[source],
        default_material=mp.Medium(epsilon=1.0),
    )
    sim.init_sim()
    dt_meep = float(sim.fields.dt)
    run_time_meep = units["run_time_meep"]
    rx_pts = [_to_meep_point(mp, p, ctr) for p in rx_positions]
    times: List[float] = []
    samples: List[List[float]] = []

    def step(sim_now):
        if record:
            times.append(float(sim_now.meep_time()))
            samples.append([float(np.real(sim_now.get_field_point(mp.Ez, pt))) for pt in rx_pts])

    t_start = time.time()
    sim.run(mp.at_every(dt_meep, step), until=run_time_meep)
    wall = time.time() - t_start
    del sim
    gc.collect()

    sec_per_meep_time = units["seconds_per_meep_time"]
    if record:
        t_meep = np.asarray(times, dtype=float)
        if t_meep.size < 2:
            raise RuntimeError("记录样本过少（%d）：at_every 未按预期触发" % t_meep.size)
        t_first_meep = float(t_meep[0])
        if t_first_meep < 0.0 or t_first_meep > 0.5 * run_time_meep:
            raise RuntimeError("首个记录样本时间戳异常：t=%r（run_time=%r）" % (t_first_meep, run_time_meep))
        d = np.diff(t_meep)
        if not np.allclose(d, d[0], rtol=1e-9, atol=0.0):
            raise RuntimeError("记录时间戳非等间隔（max|Δ-Δ0|=%.3e）" % float(np.max(np.abs(d - d[0]))))
        records = np.asarray(samples, dtype=float).T  # [n_rx, n_time]
        t_s = t_meep * sec_per_meep_time
    else:
        records = np.zeros((len(rx_pts), 0), dtype=float)
        t_s = np.zeros((0,), dtype=float)
    return {
        "records": records,
        "t_s": t_s,
        "dt_meep": dt_meep,
        "dt_s": dt_meep * sec_per_meep_time,
        "n_steps": int(records.shape[-1]),
        "wall_time_s": wall,
        "tx_index": int(tx_index),
        "n_targets_in_geometry": n_in_geometry,
        "units": dict(units),
        "sim_meta": {
            "cell_size_m": size,
            "resolution_px_per_m": units["resolution_px_per_m"],
            "courant": units["courant"],
            "pml_thickness_m": float(units["pml_thickness_meep"]),
            "dimensions": units["dimensions"],
            "source_component": "Ez",
            "source_amplitude": 1.0,
            "source_gain_applied": gain,
            "source_normalize_to_si": units["source_normalize_to_si"],
            "seconds_per_meep_time": sec_per_meep_time,
            "source_end_time_s": units["duration_s"],
            "run_time_meep": run_time_meep,
            "time_origin_definition": "t=0 为 Tx 源起始时刻（Meep 绝对时间）；MF 对齐用物理 filter lag",
            "t_first_sample_meep": float(t_meep[0]),
            "t_first_sample_s": float(t_s[0]),
            "t_first_sample_note": "at_every(dt) 首个回调在 t=dt（实测），不是 0；参考与数据共用该绝对时间轴，不外推",
            "absolute_amplitude_calibrated": False,
            "amplitude_note": "含求解 cell transfer（源幅度约定/像素内插/数值色散）；只用于形状量比较",
        },
    }


def measured_updates_per_second(result: Mapping[str, Any]) -> float | None:
    """用一次真实求解的实测耗时反推 cell-updates/s（**实测**，供后续预算换算）。"""
    wall = float(result.get("wall_time_s") or 0.0)
    meta = result.get("sim_meta", {})
    if wall <= 0 or not meta:
        return None
    size = meta.get("cell_size_m", [0.0, 0.0])
    res = float(meta.get("resolution_px_per_m") or 0.0)
    n_cells = math.ceil(float(size[0]) * res) * math.ceil(float(size[1]) * res)
    n_steps = float(result.get("n_steps") or 0.0)
    if n_cells <= 0 or n_steps <= 0:
        return None
    return n_cells * n_steps / wall
