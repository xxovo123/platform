# -*- coding: utf-8 -*-
"""仿真实验协议包：sim_protocol 是唯一事实来源，build_protocol 生成可冻结 JSON。"""

from .sim_protocol import PROTOCOL, build_protocol, protocol_hash  # noqa: F401

__all__ = ["PROTOCOL", "build_protocol", "protocol_hash"]
