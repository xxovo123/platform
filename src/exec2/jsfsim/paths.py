#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路径引导：把新版包与冻结数值包接起来。

新版代码放在 `execution2/`，**冻结数值代码只读复用**（不复制、不修改）：
  * 冻结包 = `../execution/bundle/`（含带限修复与层 B 物理范围缓存签名）；
  * 本模块把冻结包目录加入 sys.path，再由新版模块 import `sim_common` / `adapter.*` /
    `fdtd.*` / `analytic.*` / `vendor`。

新版**刻意不 import** 冻结包的：
  * `analytic/scenes.py`（旧 S1–S5 带 (0.017,-0.013) 平移；新版用 jsfsim.protocol 展开）；
  * `protocol/sim_protocol.py` 的场景/预算常量（新版以机器协议 JSON 为准）；
  * `sim_common.clean_seed` / `trial_seed`（旧种子口径；新版用规范 §6 canonical-JSON 口径）。
冻结包的 `sim_common` 常量（fc/B/Ts/taps/网格/取峰门限/几何）与规范一致，直接复用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
EXEC2_DIR = PKG_DIR.parent
RECOVERY_DIR = EXEC2_DIR.parent


def frozen_bundle_dir() -> Path:
    """冻结数值包：默认 `../execution/bundle`（sibling），只读复用，不复制不改。"""
    env = os.environ.get("JSF_FROZEN_BUNDLE")
    if env:
        return Path(env).resolve()
    return (RECOVERY_DIR / "execution" / "bundle").resolve()


def ensure_paths() -> dict:
    """把冻结包与新版包加入 sys.path（幂等）。返回解析结果。"""
    bundle = frozen_bundle_dir()
    if not (bundle / "sim_common.py").is_file():
        raise RuntimeError("冻结包不可用（缺 sim_common.py）：%s" % bundle)
    added = []
    for p in (str(bundle), str(EXEC2_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    return {"frozen_bundle": str(bundle), "exec2": str(EXEC2_DIR), "added": added}


def sim_common():
    """冻结包的协议常量/前端/噪声工具（只读复用）。"""
    ensure_paths()
    import sim_common as sc  # noqa: PLC0415
    return sc


def adapter(name: str):
    """冻结包的 adapter 子模块。"""
    ensure_paths()
    import importlib
    return importlib.import_module("adapter.%s" % name)


__all__ = ["PKG_DIR", "EXEC2_DIR", "RECOVERY_DIR", "frozen_bundle_dir", "ensure_paths",
           "sim_common", "adapter"]
