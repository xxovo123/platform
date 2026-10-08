# -*- coding: utf-8 -*-
"""仿真侧通用指标包（known-K 匹配 + TP/FN/FP + 条件误差 + 标准 OSPA）。

公共接口::

    from metrics import evaluate_predictions, aggregate

口径冻结来源见 ``sim_metrics`` 模块 docstring；本包独立于 ``knownk_eval`` / ``ospa_postprocess``。
"""

from .sim_metrics import (
    SCHEMA_VERSION,
    DEFAULT_GATE_M,
    GATE_TOLERANCE_M,
    OSPA_C_M,
    OSPA_P,
    RESOLUTION_ABS_GATE_M,
    RESOLUTION_REL_DIVISOR,
    MetricError,
    aggregate,
    evaluate_predictions,
    match_by_gate,
    ospa,
    ospa_squared,
    polar_of,
    wrap_deg,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_GATE_M",
    "GATE_TOLERANCE_M",
    "OSPA_C_M",
    "OSPA_P",
    "RESOLUTION_ABS_GATE_M",
    "RESOLUTION_REL_DIVISOR",
    "MetricError",
    "aggregate",
    "evaluate_predictions",
    "match_by_gate",
    "ospa",
    "ospa_squared",
    "polar_of",
    "wrap_deg",
]
