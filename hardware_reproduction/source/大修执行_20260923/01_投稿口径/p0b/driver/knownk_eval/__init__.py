# -*- coding: utf-8 -*-
"""P0b 统一 known-K 评估 driver（独立实现，可供硬件与后续仿真/不同算法复用）。

公开 API::

    from knownk_eval import (load_protocol, evaluate_spectrum, evaluate_predictions,
                             evaluate_windows, summarise)

两条入口：
  * 谱方法路径：evaluate_spectrum(image, grid_x, grid_y, k, truth_xy, protocol)
    先按统一取峰规则（3x3 非严格极大 + 8 连通平台去重 + 线性 0.3 阈值 + 0.10 m NMS + cap K）
    得到 (x, y) 峰，再按 0.30 m 一对一最优匹配与 known-K 指标口径评估。
  * 连续域方法路径：evaluate_predictions(truth_xy, predictions_xy, protocol, k=...)
    直接吃方法原生 (x, y) 预测，不强制栅格化（路线 8.4：保留原生求解结构），
    K 上限由方法自身原生结构给出（或经 cap_by_score 显式声明截断并留痕）。

失败显式记录、不静默丢弃；参数全部由协议驱动并严格校验。
本包不读源工程、不写历史产物；历史 frozen_protocol.json 保持只读。
"""

from .protocol import (  # noqa: F401
    HASH_FIELD,
    MatchingRule,
    PeakRule,
    Protocol,
    ProtocolError,
    canonical_sha256,
    canonical_text,
    check_override,
    load_protocol,
    sha256_file,
)
from .evaluate import (  # noqa: F401
    EvaluationError,
    WindowInput,
    WindowResult,
    evaluate_predictions,
    evaluate_spectrum,
    evaluate_window,
    evaluate_windows,
    summarise,
)
from .matching import MatchingError, match_targets  # noqa: F401
from .metrics import (  # noqa: F401
    DuplicateWindowError, aggregate, check_unique_windows, error_stats, polar_of, wrap_deg,
)
from .selfcheck import behavioral_self_check, validate_semantic_fields  # noqa: F401

__all__ = [
    "HASH_FIELD", "MatchingRule", "PeakRule", "Protocol", "ProtocolError",
    "canonical_sha256", "canonical_text", "check_override", "load_protocol", "sha256_file",
    "EvaluationError", "WindowInput", "WindowResult", "evaluate_predictions",
    "evaluate_spectrum", "evaluate_window", "evaluate_windows", "summarise",
    "MatchingError", "match_targets", "aggregate", "error_stats", "polar_of", "wrap_deg",
    "DuplicateWindowError", "check_unique_windows", "behavioral_self_check",
    "validate_semantic_fields",
]
