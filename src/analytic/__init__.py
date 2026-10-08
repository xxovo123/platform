# -*- coding: utf-8 -*-
"""层 A（解析受控多目标 CIR/CFR）正向生成与场景矩阵。

反循环约束：本包的正向生成是**独立实现**，不得调用被测代码（JSF-MUSIC / 各基线）的内部
流形或导向来造输入。真值只写独立 JSON，不进入 case。
"""

from .forward import (  # noqa: F401
    MethodFreeForward, coefficients_from_spec, delays_for_targets, invariant_check,
    raw_cir_from_coeffs,
)
from . import scenes  # noqa: F401

__all__ = ["MethodFreeForward", "coefficients_from_spec", "delays_for_targets",
           "raw_cir_from_coeffs", "invariant_check", "scenes"]
