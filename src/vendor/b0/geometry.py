# -*- coding: utf-8 -*-
"""虚拟通道映射（独立实现）：按物理位置去重并按 x 排序，与工程口径一致。

规则（记录在源码与冻结协议里）：虚拟位置 = tx_pos + rx_pos（use_phase_center=False 时
不取中点），按 ndigits 四舍五入去重（保留首个出现的 (tx_idx, rx_idx)），再按虚拟位置 x
升序排序。本模块独立实现该规则，并在运行日志里与源工程 util.create_virtual_array_mapping
的结果逐对比对（一致性记录为 provenance，不作为算法结论）。
"""

from __future__ import annotations

import numpy as np


def virtual_pairs(tx_positions, rx_positions, tx_idx_list, rx_idx_list, *,
                  use_phase_center_convention: bool = False, ndigits: int = 4):
    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)
    factor = 0.5 if use_phase_center_convention else 1.0
    seen = {}
    for tx_idx in tx_idx_list:
        for rx_idx in rx_idx_list:
            if int(tx_idx) >= tx_positions.shape[0] or int(rx_idx) >= rx_positions.shape[0]:
                continue
            v_pos = (tx_positions[int(tx_idx)] + rx_positions[int(rx_idx)]) * factor
            key = tuple(np.round(v_pos, ndigits))
            if key not in seen:
                seen[key] = (int(tx_idx), int(rx_idx))
    ordered = sorted(seen.items(), key=lambda item: item[0][0])
    return [pair for _key, pair in ordered], np.array([list(key) for key, _pair in ordered])

