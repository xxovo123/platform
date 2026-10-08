#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 B（FDTD 全波）生成包：import-safe。

纪律（2026-09-26 冻结，详见 README.md / API_CONTRACT.md）
----------------------------------------------------------
* 本包 ``import`` **不得**导入 meep；Meep 只在真正求解时惰性导入（solver.py）。
* 源脉冲与匹配滤波参考共用同一公式实现（pulse.bh_derivative_passband），由结构保证。
* 输出 ``clean`` case 只含算法可见字段：``raw_cir[L,4,32]`` + 时间原点 + 协议常量；
  真值/全轨迹/诊断一律进独立 provenance，**不进 case**。
* 本轮为“代码 + tiny smoke”；正式 128 states / 250 / 350 矩阵须另行授权。
"""

from __future__ import annotations

__all__ = ["generate_clean", "plan_run", "PROTOCOL_VERSION"]

from .api import generate_clean, plan_run  # noqa: E402  (api 内部同样不导入 meep)
from .protocol import PROTOCOL_VERSION  # noqa: E402

__version__ = "20260926.1"
