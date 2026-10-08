#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""估计器装配：只调用冻结 JSF-MUSIC（`this_method`），不改其数学。

硬规则
------
* `methods` 只允许 `["this_method"]`；任何其他取值**直接拒绝**（外部仿真基线不接入、不运行）。
* 不重写导向矢量/协方差/FB/对角加载/取峰；只装配 case（X、几何、K、网格）与调用。
* 方法异常/非法输出 -> `status='failed'` 且保留在分母（不静默丢弃）。
"""

from __future__ import annotations

from . import paths

ONLY_METHOD = "this_method"


class MethodScopeError(RuntimeError):
    """越界方法（显式拒绝）。"""


def assert_methods(methods) -> list:
    if methods is None:
        return [ONLY_METHOD]
    seq = [str(m) for m in methods]
    if seq != [ONLY_METHOD]:
        raise MethodScopeError(
            "本轮只允许 methods=['this_method']（JSF-MUSIC）；收到 %r。"
            "外部仿真基线不接入、不运行。" % (seq,))
    return seq


def load_estimator():
    """冻结方法装载器（vendor，byte-identical）。"""
    return paths.adapter("methods_loader")


def run_single(case: dict, params: dict | None = None) -> dict:
    """调用冻结 JSF-MUSIC。返回归一化结果（status/predictions/spectrum/diagnostics）。"""
    ml = load_estimator()
    return ml.call(ONLY_METHOD, case, params or ml.params_for(ONLY_METHOD, _protocol_dict()))


def _protocol_dict() -> dict:
    from . import protocol as P
    return {"methods": {"params_by_method": {ONLY_METHOD: {}}}, **P.build_protocol()}


def module_hash() -> str | None:
    return load_estimator().module_hash(ONLY_METHOD)


__all__ = ["ONLY_METHOD", "MethodScopeError", "assert_methods", "run_single",
           "module_hash", "load_estimator"]
