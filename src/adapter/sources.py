#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据源登记：层 A（解析，本包）与层 B（FDTD，fdtd 代理的 fdtd/api.py）。

两层契约一致：clean raw CIR = raw_cir[L_clean, 4, 32]，tap0 = 物理零延迟；
不做 crop2（由 adapter 前端统一处理）；无噪轨迹按 clean_id 复用；噪声由 runner 在下游统一施加。
"""

from __future__ import annotations

import os

from analytic import cache as analytic_cache


class SourceError(RuntimeError):
    """数据源不可用（显式失败）。"""


class AnalyticSource:
    name = "analytic"
    layer = "A"

    def available(self) -> tuple:
        return True, "analytic forward（本包独立实现）"

    def build_clean(self, clean_spec, *, cache_dir, protocol_hash, source_aggregate,
                    resume: bool = True):
        return analytic_cache.build_clean(clean_spec, cache_dir=cache_dir,
                                          protocol_hash=protocol_hash,
                                          source_aggregate=source_aggregate)

    def load_clean(self, cache_dir, clean_id, *, expect_signature=None):
        return analytic_cache.load_clean(cache_dir, clean_id, expect_signature=expect_signature)

    def load_truth(self, cache_dir, clean_id):
        return analytic_cache.load_truth(cache_dir, clean_id)


def get_source(source: str):
    """'analytic' -> AnalyticSource；'fdtd'/'B' -> FdtdSource（fdtd 代理的 fdtd/api.py）。"""
    if source == "analytic":
        return AnalyticSource()
    if source in ("fdtd", "B", "b"):
        from .fdtd_source import FdtdSource
        return FdtdSource(os.environ.get("SIM_FDTD_PROVIDER") or "fdtd.api")
    raise SourceError("未知 source=%r" % source)


__all__ = ["AnalyticSource", "SourceError", "get_source"]
