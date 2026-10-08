#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""场景（几何 + 目标 + 数值参数）的装载、校验与派生量计算。

几何纪律（2026-09-26）
---------------------
* FDTD 是 **2D（xy 平面）**：所有 z 被丢弃，只保留 xy 投影；2D 不能表达离面高度。
* 主档 1Tx4Rx：Tx``[0,0,0]``、Rx x=``[-0.038,-0.019,0,0.019]``、y=0（**新几何**，
  **不声称历史复原**；历史 2Tx×4Rx 口径另见 Q4 档）。
* Q4 档保留硬件原始 3D 坐标与通道对，2D 只做投影，并把 3D↔2D 双程时延偏差
  写进 metadata（``delay_deviation_metadata``），**不把 z 当 y**、不据此调时延。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .protocol import read_json

SCENE_DIR = Path(__file__).resolve().parent / "protocol" / "scenes"


def builtin_scene(scene_id: str) -> Dict[str, Any]:
    path = SCENE_DIR / ("%s.json" % scene_id)
    if not path.is_file():
        avail = sorted(p.stem for p in SCENE_DIR.glob("*.json"))
        raise FileNotFoundError("未找到场景 %s；可用：%s" % (scene_id, avail))
    return read_json(path)


def load_scene(scene: Any) -> Dict[str, Any]:
    """接受 场景字典 / JSON 路径 / 内建 scene_id。返回校验后并已解析 domain 的字典。"""
    if isinstance(scene, Mapping):
        obj = dict(scene)
    elif isinstance(scene, (str, Path)):
        p = Path(scene)
        if p.is_file():
            obj = read_json(p)
        else:
            obj = builtin_scene(str(scene))
    else:
        raise TypeError("scene 需为 dict / 路径 / scene_id")
    problems = validate_scene(obj)
    if problems:
        raise ValueError("场景不合法：%s" % problems)
    obj["domain"] = resolve_domain(obj)
    return obj


def validate_scene(scene: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in ("scene_id", "geometry", "targets", "meep"):
        if key not in scene:
            problems.append("缺少字段 %s" % key)
    if problems:
        return problems
    geom = scene["geometry"]
    txs = np.asarray(geom.get("tx_positions_m", []), dtype=float)
    rxs = np.asarray(geom.get("rx_positions_m", []), dtype=float)
    pairs = np.asarray(geom.get("pairs", []), dtype=int)
    if txs.ndim != 2 or txs.shape[1] != 3:
        problems.append("tx_positions_m 形状应为 [n_tx, 3]")
    if rxs.ndim != 2 or rxs.shape[1] != 3:
        problems.append("rx_positions_m 形状应为 [n_rx, 3]")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] != 4:
        problems.append("pairs 形状应为 [4, 2]（本协议为四通道）")
    else:
        for tx_i, rx_i in pairs:
            if not (0 <= tx_i < txs.shape[0]):
                problems.append("pairs 含越界 tx 下标 %d" % tx_i)
            if not (0 <= rx_i < rxs.shape[0]):
                problems.append("pairs 含越界 rx 下标 %d" % rx_i)
    targets = scene["targets"]
    centers = np.asarray(targets.get("nominal_centers_m", []), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        problems.append("targets.nominal_centers_m 形状应为 [K, 3]")
    mm = targets.get("micro_motion", {})
    for key in ("seed", "sigma_m", "radius_m"):
        if key not in mm:
            problems.append("targets.micro_motion 缺 %s（seed 决定可复现性）" % key)
    meep = scene["meep"]
    for key in ("resolution_px_per_m", "run_time_meep"):
        if key not in meep:
            problems.append("meep 缺 %s" % key)
    return problems


def target_centers_2d(scene: Mapping[str, Any]) -> np.ndarray:
    """目标标称中心的 xy 投影 ``[K, 2]``（丢弃 z；2D 无法表达离面）。"""
    centers = np.asarray(scene["targets"]["nominal_centers_m"], dtype=float)
    return centers[:, :2].copy()


def antenna_positions_2d(scene: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    geom = scene["geometry"]
    return {
        "tx": np.asarray(geom["tx_positions_m"], dtype=float)[:, :2].copy(),
        "rx": np.asarray(geom["rx_positions_m"], dtype=float)[:, :2].copy(),
    }


def resolve_domain(scene: Mapping[str, Any], fc_hz: float = 7.9872e9) -> Dict[str, Any]:
    """派生计算域：覆盖全目标（含目标半径 + **轨迹半径**）+ 裕量 + PML。

    ``inner_bounds_m`` 一律定义为**实际内区** = ``center ± size/2 ∓ PML``（与是否显式给
    ``size_m`` 无关）；显式给 ``size_m`` 时**不再**用派生目标包围盒冒充内区。最后校验
    所有目标（含半径与轨迹半径）确实落在实际内区内，越界即报错（几何拒非法，不事后筛）。
    """
    dom_cfg = dict(scene.get("domain", {}))
    margin = float(dom_cfg.get("margin_m", 0.75))
    lam = 299792458.0 / float(fc_hz)
    meep_cfg = scene["meep"]
    pml_m = float(meep_cfg.get("pml_thickness_lambda", 2.0)) * lam
    radius = float(scene["targets"].get("radius_m", 0.0))
    traj_r_max = max(_traj_radius_list(scene))
    reach = radius + traj_r_max
    pos = antenna_positions_2d(scene)
    centers = target_centers_2d(scene)
    xs = list(pos["tx"][:, 0]) + list(pos["rx"][:, 0]) + list(centers[:, 0] - reach) + list(centers[:, 0] + reach)
    ys = list(pos["tx"][:, 1]) + list(pos["rx"][:, 1]) + list(centers[:, 1] - reach) + list(centers[:, 1] + reach)
    inner_x = [min(xs) - margin, max(xs) + margin]
    inner_y = [min(ys) - margin, max(ys) + margin]
    # 已解析过的 domain（derived=True）不当作显式覆盖：目标/几何变了必须重新派生，
    # 否则会把旧域当“显式 size_m”沿用，导致 slack<0 的假错误。
    explicit = ("size_m" in dom_cfg) and not bool(dom_cfg.get("_resolved") or dom_cfg.get("derived"))
    if explicit:
        size = [float(v) for v in dom_cfg["size_m"]]
        center = [float(v) for v in dom_cfg.get("center_m", [(inner_x[0] + inner_x[1]) / 2.0, (inner_y[0] + inner_y[1]) / 2.0])]
    else:
        size = [inner_x[1] - inner_x[0] + 2 * pml_m, inner_y[1] - inner_y[0] + 2 * pml_m]
        center = [(inner_x[0] + inner_x[1]) / 2.0, (inner_y[0] + inner_y[1]) / 2.0]
    # 实际内区（与是否显式 size_m 无关）：域边界内缩一个 PML 厚度
    actual_inner_x = [center[0] - size[0] / 2.0 + pml_m, center[0] + size[0] / 2.0 - pml_m]
    actual_inner_y = [center[1] - size[1] / 2.0 + pml_m, center[1] + size[1] / 2.0 - pml_m]
    out = {
        "size_m": size,
        "center_m": center,
        "inner_bounds_m": {"x": actual_inner_x, "y": actual_inner_y},
        "required_bounds_m": {"x": inner_x, "y": inner_y},
        "margin_m": margin,
        "pml_thickness_m": pml_m,
        "target_reach_m": reach,
        "derived": "size_m" not in dom_cfg,
        "_resolved": True,
    }
    # slack = 物体外缘到实际内区边界的最小余量（正数 = 内部有余量）
    slack_x = min(min(xs) - actual_inner_x[0], actual_inner_x[1] - max(xs))
    slack_y = min(min(ys) - actual_inner_y[0], actual_inner_y[1] - max(ys))
    out["inner_slack_m"] = {"x": float(slack_x), "y": float(slack_y), "min": float(min(slack_x, slack_y))}
    if min(slack_x, slack_y) < 0.0:
        raise ValueError(
            "目标/天线（含半径 %.4f + 轨迹半径 %.4f）越出实际内区：slack_x=%.4f slack_y=%.4f；"
            "请扩大 size_m 或减小裕量/PML/轨迹半径" % (radius, traj_r_max, slack_x, slack_y)
        )
    return out


def _traj_radius_list(scene: Mapping[str, Any]) -> List[float]:
    """逐目标轨迹半径（``micro_motion.radius_m`` 可为标量或列表）。"""
    mm = dict(scene["targets"].get("micro_motion", {}))
    n_k = int(np.asarray(scene["targets"]["nominal_centers_m"], dtype=float).shape[0])
    radius = mm.get("radius_m", 0.0)
    vals = [float(x) for x in (radius if isinstance(radius, (list, tuple)) else [radius] * n_k)]
    if len(vals) != n_k:
        raise ValueError("micro_motion.radius_m 长度必须等于 K=%d" % n_k)
    return vals


def channel_names(scene: Mapping[str, Any]) -> List[str]:
    pairs = np.asarray(scene["geometry"]["pairs"], dtype=int)
    return ["tx%d-rx%d" % (int(t), int(r)) for t, r in pairs]


def delay_deviation_metadata(scene: Mapping[str, Any], c0_m_per_s: float = 2.99792458e8) -> Dict[str, Any]:
    """Q4 用：真实 3D 几何与 2D 共面投影的**双程时延偏差**（metadata，不是补偿值）。

    ``Δτ = (L_3d - L_2d)/c0``，``L = |tx-p| + |rx-p|``；3D 用硬件原始坐标（含 z=-0.02），
    2D 用 xy 投影。2D FDTD 无法表达 z，故该偏差**只报告、不补偿**。
    """
    geom = scene["geometry"]
    tx3 = np.asarray(geom["tx_positions_m"], dtype=float)
    rx3 = np.asarray(geom["rx_positions_m"], dtype=float)
    pairs = np.asarray(geom["pairs"], dtype=int)
    centers3 = np.asarray(scene["targets"]["nominal_centers_m"], dtype=float)
    rows: List[Dict[str, Any]] = []
    for tx_i, rx_i in pairs:
        for k, p3 in enumerate(centers3):
            p2 = np.array([p3[0], p3[1]], dtype=float)
            l3 = float(np.linalg.norm(tx3[tx_i] - p3) + np.linalg.norm(rx3[rx_i] - p3))
            l2 = float(np.linalg.norm(tx3[tx_i, :2] - p2) + np.linalg.norm(rx3[rx_i, :2] - p2))
            rows.append(
                {
                    "tx_index": int(tx_i),
                    "rx_index": int(rx_i),
                    "target_index": int(k),
                    "path_3d_m": l3,
                    "path_2d_m": l2,
                    "delay_3d_s": l3 / c0_m_per_s,
                    "delay_2d_s": l2 / c0_m_per_s,
                    "delay_deviation_s": (l3 - l2) / c0_m_per_s,
                }
            )
    dev = np.array([r["delay_deviation_s"] for r in rows], dtype=float)
    return {
        "definition": "tau_3d_minus_2d = (|tx-p|+|rx-p|)/c0 在 3D 与 xy 投影下的差；只报告，不补偿",
        "rows": rows,
        "delay_deviation_s": {"min": float(dev.min()), "max": float(dev.max()), "mean": float(dev.mean())},
    }


def meep_target_support_m(scene: Mapping[str, Any]) -> float:
    """目标几何支撑（用于 cost 估计与域覆盖检查）：最大双程距离。"""
    pos = antenna_positions_2d(scene)
    centers = target_centers_2d(scene)
    radius = float(scene["targets"].get("radius_m", 0.0)) + max(_traj_radius_list(scene))
    best = 0.0
    for p in centers:
        for t in pos["tx"]:
            for r in pos["rx"]:
                best = max(best, float(np.linalg.norm(t - p) + np.linalg.norm(r - p)) + 2.0 * radius)
    return best


def covered_range_m(scene: Mapping[str, Any]) -> float:
    """域（扣除 PML）沿**视轴**能覆盖的单程距离。

    视轴方向由目标相对阵列的一侧决定：目标在 +y 侧 → ``inner_y_max - tx_y``；
    目标在 -y 侧 → ``tx_y - inner_y_min``（2026-09-26 修正：原来无条件用
    ``inner_y_max - tx_y``，在 -y 侧会给出错误的正值）。
    """
    dom = scene["domain"]
    pos = antenna_positions_2d(scene)
    centers = target_centers_2d(scene)
    y_inner = dom["inner_bounds_m"]["y"]
    tx_y = float(np.mean(pos["tx"][:, 1]))
    target_y = float(np.mean(centers[:, 1]))
    if target_y >= tx_y:
        return float(y_inner[1]) - tx_y
    return tx_y - float(y_inner[0])


def separation_rule(scene: Mapping[str, Any]) -> Dict[str, Any]:
    """冻结的“目标不交叠”规则：``d_min >= 2·r_target + 2·r_traj + overlap_margin``。

    保守性：轨迹被约束在每个目标半径 ``r_traj`` 的球内，所以任意状态下两目标中心距离
    ≥ 标称距离 − 2·r_traj；因此该不等式成立即保证**任何状态**下两个半径 ``r_target`` 的
    金属圆都不相交（``overlap_margin_m`` 为额外余量）。
    """
    r_target = float(scene["targets"].get("radius_m", 0.0))
    r_traj = max(_traj_radius_list(scene))
    margin = float(scene["targets"].get("overlap_margin_m", 0.005))
    return {
        "target_radius_m": r_target,
        "trajectory_radius_max_m": r_traj,
        "overlap_margin_m": margin,
        "required_min_separation_m": 2.0 * r_target + 2.0 * r_traj + margin,
        "rule": "min_nominal_separation >= 2*r_target + 2*r_traj + overlap_margin（逐状态成立）",
    }


def assert_geometry_rules(scene: Mapping[str, Any], traj: np.ndarray | None = None) -> Dict[str, Any]:
    """几何合法性（**预设计拒绝**，不是事后筛结果）：目标不交叠 + 目标留在内区内。

    含轨迹时逐状态检查实际最小间距；不含轨迹时用标称距离 − 2·r_traj 的保守界。
    违规直接抛 ``ValueError``（几何拒非法），不得“跑完再看哪个状态好看”。
    """
    rule = separation_rule(scene)
    centers = target_centers_2d(scene)
    n_k = centers.shape[0]
    problems: List[str] = []
    nom_min = float("inf")
    for i in range(n_k):
        for j in range(i + 1, n_k):
            nom_min = min(nom_min, float(np.linalg.norm(centers[i] - centers[j])))
    worst = nom_min - 2.0 * rule["trajectory_radius_max_m"] if n_k > 1 else float("inf")
    if n_k > 1 and worst < rule["required_min_separation_m"]:
        problems.append(
            "目标可能交叠：标称最小间距 %.4f m − 2·r_traj %.4f m = %.4f m < 要求 %.4f m"
            % (nom_min, rule["trajectory_radius_max_m"], worst, rule["required_min_separation_m"])
        )
    state_min = float("inf")
    if traj is not None:
        traj = np.asarray(traj, dtype=float)
        for l in range(traj.shape[0]):
            for i in range(n_k):
                for j in range(i + 1, n_k):
                    state_min = min(state_min, float(np.linalg.norm(traj[l, i] - traj[l, j])))
        if n_k > 1 and state_min < rule["required_min_separation_m"]:
            problems.append("实际轨迹出现交叠：最小间距 %.4f m" % state_min)
    reach = float(scene["targets"].get("radius_m", 0.0)) + rule["trajectory_radius_max_m"]
    inner = scene["domain"]["inner_bounds_m"]
    slack = float("inf")
    for p in centers:
        slack = min(slack, float(p[0] - reach - inner["x"][0]), float(inner["x"][1] - (p[0] + reach)),
                    float(p[1] - reach - inner["y"][0]), float(inner["y"][1] - (p[1] + reach)))
    if slack < 0.0:
        problems.append("目标（含半径+轨迹半径）越出内区：最小 slack %.4f m" % slack)
    if problems:
        raise ValueError("几何拒非法：%s" % problems)
    return {
        "rule": rule,
        "min_nominal_separation_m": None if nom_min == float("inf") else nom_min,
        "worst_case_separation_m": None if worst == float("inf") else worst,
        "min_state_separation_m": None if state_min == float("inf") else state_min,
        "min_inner_slack_m": slack,
    }
