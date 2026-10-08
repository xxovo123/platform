#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""慢时间微动序列：seed 可复现、逐状态各不相同（禁止“复制 10 条冒充 128”）。

生成算法（**冻结文本，可被独立实现复算**）
------------------------------------------
模型 ``bounded_random_walk``：

    rng = np.random.default_rng(seed)
    traj[0] = centers                     # 第 0 个状态 = 标称位置
    for l in 1..L-1:                      # 外层：状态
        for k in 0..K-1:                  # 内层：目标（顺序固定，影响 rng 消费顺序）
            step = rng.normal(0, sigma_k, size=2)
            p = traj[l-1, k] + step
            if |p - centers[k]| > radius_k:   p = centers[k] + (p-centers[k]) * radius_k/|p-centers[k]|
            traj[l, k] = p

* 每个目标独立抽步（同一 rng、固定消费顺序）；步长按 **米** 记录，不按“格”。
* ``radius_m`` 是有界约束（防止漂出域），不是平滑核。
* 状态唯一性：调用方必须校验 ``state_hashes`` 无重复（``assert_unique_states``），
  这是“128 必须真 128”的机器可检条件。
* rng 只用在轨迹上；Meep 求解本身确定，无随机项。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .protocol import sha256_hex

MODEL_ID = "bounded_random_walk_v1"


def build_trajectory(
    spec: Mapping[str, Any],
    centers_m: Sequence[Sequence[float]],
) -> np.ndarray:
    """按冻结算法生成 ``[L, K, 2]`` 的 2D（xy）目标轨迹。"""
    model = str(spec.get("model", MODEL_ID))
    if model not in (MODEL_ID, "bounded_random_walk"):
        raise ValueError("未知微动模型：%s" % model)
    centers = np.asarray(centers_m, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_m 形状应为 [K, 2]（2D 平面投影），得到 %s" % (centers.shape,))
    n_states = int(spec["L"])
    k_targets = centers.shape[0]
    sigma = spec.get("sigma_m", 0.01)
    radius = spec.get("radius_m", 0.10)
    sigma_list = [float(x) for x in (sigma if isinstance(sigma, (list, tuple)) else [sigma] * k_targets)]
    radius_list = [float(x) for x in (radius if isinstance(radius, (list, tuple)) else [radius] * k_targets)]
    if not (len(sigma_list) == len(radius_list) == k_targets):
        raise ValueError("sigma_m / radius_m 长度必须等于目标数 K=%d" % k_targets)
    if n_states < 1:
        raise ValueError("L 必须 >= 1")
    rng = np.random.default_rng(int(spec["seed"]))
    traj = np.zeros((n_states, k_targets, 2), dtype=float)
    traj[0] = centers
    for l in range(1, n_states):
        for k in range(k_targets):
            step = rng.normal(0.0, sigma_list[k], size=2)
            cand = traj[l - 1, k] + step
            delta = cand - centers[k]
            dist = float(np.linalg.norm(delta))
            if dist > radius_list[k]:
                cand = centers[k] + delta * (radius_list[k] / dist)
            traj[l, k] = cand
    return traj


def state_hashes(traj: np.ndarray) -> list[str]:
    """逐状态的精确哈希（float64 位型参与；不做小数位 round）。"""
    return [sha256_hex({"positions_m": np.asarray(traj[l], dtype=float).tolist()}) for l in range(traj.shape[0])]


def assert_unique_states(traj: np.ndarray) -> list[str]:
    """断言所有状态互不相同；返回状态哈希列表。"""
    hashes = state_hashes(traj)
    if len(set(hashes)) != len(hashes):
        dup = {h for h in hashes if hashes.count(h) > 1}
        raise ValueError("存在重复状态（禁止复制快照冒充 L 个真实状态）：%s" % sorted(dup))
    return hashes


def trajectory_source_hash(spec: Mapping[str, Any], centers_m: Sequence[Sequence[float]]) -> str:
    """轨迹源 hash：模型/种子/步长/半径/L + 标称中心，进入 cache key。"""
    return sha256_hex(
        {
            "model": str(spec.get("model", MODEL_ID)),
            "seed": int(spec["seed"]),
            "sigma_m": spec.get("sigma_m", 0.01),
            "radius_m": spec.get("radius_m", 0.10),
            "L": int(spec["L"]),
            "centers_m": np.asarray(centers_m, dtype=float).tolist(),
        }
    )


def describe(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """写进 provenance 的可读说明（供独立复算）。"""
    return {
        "model": str(spec.get("model", MODEL_ID)),
        "algorithm_text": (
            "rng=default_rng(seed); traj[0]=centers; "
            "for l in 1..L-1: for k in 0..K-1: step=rng.normal(0,sigma_k,2); "
            "cand=traj[l-1,k]+step; if |cand-centers[k]|>radius_k: project onto ball(centers[k],radius_k)"
        ),
        "spec": dict(spec),
    }
