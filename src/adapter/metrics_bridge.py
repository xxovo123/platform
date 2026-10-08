#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""metrics 桥：解析并调用独立的 metrics 包（接口由 metrics 代理冻结）。

接口（metrics 代理 Aristotle 冻结）::
    from metrics import evaluate_predictions, aggregate
    evaluate_predictions(truth_xy, pred_xy, resolution_distance_m=None, status='ok') -> dict
    aggregate(rows) -> dict

本桥只做解析 + 调用 + 显式失败；**不改** metrics 实现，也不在本包内重写指标。
解析顺序：环境变量 SIM_METRICS_PATH（metrics 包的父目录）> 本包目录（含 metrics/）。
"""

from __future__ import annotations

import importlib
import os
import sys

import sim_common as sc


class MetricsBridgeError(RuntimeError):
    """metrics 包不可用（显式失败，不静默降级为自研指标）。"""


_MOD = None


def _resolve_paths() -> list:
    paths = []
    env = os.environ.get("SIM_METRICS_PATH")
    if env:
        paths.append(env)
    paths.append(str(sc.PKG_DIR))
    return paths


def load_metrics():
    """导入 metrics 包；不可用则显式失败（附解析路径与原因）。"""
    global _MOD
    if _MOD is not None:
        return _MOD
    tried = []
    for p in _resolve_paths():
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            mod = importlib.import_module("metrics")
        except Exception as exc:  # noqa: BLE001
            tried.append((p, "%s: %s" % (type(exc).__name__, exc)))
            continue
        for need in ("evaluate_predictions", "aggregate"):
            if not hasattr(mod, need):
                tried.append((p, "metrics 缺接口 %s" % need))
                mod = None
                break
        if mod is not None:
            _MOD = mod
            return mod
    raise MetricsBridgeError("metrics 包不可用；解析路径与原因：%s（可设 SIM_METRICS_PATH 指向其父目录）"
                             % tried)


def evaluate_predictions(truth_xy, pred_xy, *, resolution_distance_m=None, status="ok") -> dict:
    mod = load_metrics()
    return mod.evaluate_predictions(list(truth_xy), list(pred_xy or []),
                                    resolution_distance_m=resolution_distance_m, status=str(status))


def evaluate_failed(truth_xy, *, resolution_distance_m=None, status="failed") -> dict:
    """失败/NA 行：已知真值 + 空预测 + 非 ok 状态（统一计分母）。"""
    return evaluate_predictions(truth_xy, [], resolution_distance_m=resolution_distance_m,
                                status=status)


def aggregate(rows) -> dict:
    return load_metrics().aggregate(list(rows))


__all__ = ["load_metrics", "evaluate_predictions", "evaluate_failed", "aggregate",
           "MetricsBridgeError"]
