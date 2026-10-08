#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冻结方法装载（vendor，byte-identical）+ 统一 estimate 调用 + 搜索剖面 params。

路径覆盖（不修改 vendored 源）：vendor 模块内部用 __file__ 推 WS/PROTOCOL 等常量，vendored
副本的 parents[] 与本地工程不同，故 bootstrap 先把 vendor/{,common,src} 加入 sys.path 使 import
成功，再在 import 后**覆盖**模块级 PROTOCOL/WS 等常量指向 vendor 内字节一致副本。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys

import sim_common as sc

MAIN_METHODS = ["this_method", "b1_deng", "b2_kakouche", "c1_jade", "c3_kim"]
REFERENCE_METHODS = ["b0_standard_music"]
METHOD_ORDER = MAIN_METHODS + REFERENCE_METHODS

MODULE_FILES = {
    "this_method": "common/q6_methods_this_method.py",
    "b0_standard_music": "common/q6_methods_b0_standard_music.py",
    # 深度须使 parents[2]/common/q6_contract.py 命中 vendor/common（冻结模块的定位规则）
    "b1_deng": "hardware/methods/b1_deng.py",
    "b2_kakouche": "hardware/methods/b2_kakouche.py",
    "c1_jade": "hardware/methods/c1_jade.py",
    "c3_kim": "hardware/methods/c3_kim.py",
}

_LOADED = {}


class MethodLoadError(RuntimeError):
    """方法模块不可装载（显式失败；调用方记为 failed 并保留在分母）。"""


def bootstrap() -> dict:
    """把 vendor 路径加入 sys.path（幂等）。返回 sys.path 前缀记录。"""
    added = []
    for p in (sc.VENDOR_DIR, sc.VENDOR_DIR / "common", sc.VENDOR_DIR / "src"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
            added.append(sp)
    # 预导入，确保 b0/knownk_eval/radar_algorithm 来自 vendor（不被本地旧路径抢先）
    pre = {}
    for name in ("b0", "knownk_eval", "radar_algorithm"):
        try:
            mod = importlib.import_module(name)
            pre[name] = getattr(mod, "__file__", None)
        except Exception as exc:  # noqa: BLE001
            pre[name] = "IMPORT_FAILED: %s: %s" % (type(exc).__name__, exc)
    return {"added": added, "vendor": str(sc.VENDOR_DIR), "preimported": pre}


def _fix_module_paths(mod) -> None:
    """覆盖 vendored 模块里由 __file__ 推出的路径常量，指向 vendor 内副本。"""
    proto = sc.VENDOR_DIR / "protocol" / "knownk_protocol.json"
    for attr, value in (("WS", sc.WS), ("PROTOCOL", proto),
                        ("SRC_DIR", sc.VENDOR_DIR / "src"),
                        ("P0B_DRIVER", sc.VENDOR_DIR),
                        ("P0B_DIR", sc.VENDOR_DIR),
                        ("B0_DIR", sc.VENDOR_DIR),
                        ("COMMON_DIR", sc.VENDOR_DIR / "common")):
        if hasattr(mod, attr):
            setattr(mod, attr, value)


def load(method: str):
    bootstrap()
    if method in _LOADED:
        return _LOADED[method]
    if method not in MODULE_FILES:
        raise MethodLoadError("未注册方法 %r（可选：%s）" % (method, ", ".join(METHOD_ORDER)))
    path = sc.VENDOR_DIR / MODULE_FILES[method]
    if not path.is_file():
        raise MethodLoadError("vendor 模块缺失：%s（先运行 scripts/make_vendor_snapshot.py）" % path)
    name = "simv_" + method
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(name, None)
        raise MethodLoadError("导入 %s 失败：%s: %s" % (method, type(exc).__name__, exc))
    _fix_module_paths(mod)
    _LOADED[method] = mod
    return mod


def params_for(method: str, protocol: dict) -> dict:
    return dict((protocol.get("methods", {}).get("params_by_method", {}) or {}).get(method) or {})


def _entry(mod):
    return getattr(mod, "estimate", None) or getattr(mod, "make_estimate", None)


def _accepts_params(entry) -> bool:
    try:
        return len(inspect.signature(entry).parameters) >= 2
    except (TypeError, ValueError):
        return False


def normalize(result) -> dict:
    """规范化方法输出，并用 vendor q6_contract.validate_estimate_output 硬校验
    （finite / status / pred 数量 / 真值泄漏）；不合规抛 ValueError 由 call() 记为 failed。
    """
    if not isinstance(result, dict):
        raise ValueError("estimate 返回非 dict：%r" % type(result).__name__)
    status = str(result.get("status", "failed"))
    preds = list(result.get("predictions") or [])
    if status == "ok" and result.get("spectrum") is None:
        for item in preds:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                raise ValueError("prediction 非法（缺 x/y）：%r" % (item,))
    diag = result.get("diagnostics") or {}
    return {"status": status, "predictions": preds,
            "spectrum": result.get("spectrum"), "diagnostics": diag}


def _validate_output(normalized: dict, case: dict) -> None:
    """调用 vendor q6_contract.validate_estimate_output（finite/status/predcount + 真值检查）。"""
    if str(sc.VENDOR_DIR / "common") not in sys.path:
        sys.path.insert(0, str(sc.VENDOR_DIR / "common"))
    import q6_contract as contract
    contract.validate_estimate_output(normalized, case)


def call(method: str, case: dict, params: dict | None = None) -> dict:
    """统一调用：异常 -> status='failed'（保留在分母，不静默丢弃）。"""
    try:
        mod = load(method)
    except MethodLoadError as exc:
        return {"status": "failed", "predictions": [], "spectrum": None,
                "diagnostics": {"reason": "module_load_failed", "detail": str(exc)}}
    entry = _entry(mod)
    if entry is None:
        return {"status": "failed", "predictions": [], "spectrum": None,
                "diagnostics": {"reason": "interface_missing"}}
    try:
        if params is not None and _accepts_params(entry):
            result = entry(case, params)
        else:
            result = entry(case)
    except Exception as exc:  # noqa: BLE001 - 显式失败，计分母
        return {"status": "failed", "predictions": [], "spectrum": None,
                "diagnostics": {"reason": "estimate_exception",
                                "detail": "%s: %s" % (type(exc).__name__, exc)}}
    try:
        normalized = normalize(result)
        _validate_output(normalized, case)
        return normalized
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "predictions": [], "spectrum": None,
                "diagnostics": {"reason": "output_invalid",
                                "detail": "%s: %s" % (type(exc).__name__, exc)}}


def module_hash(method: str) -> str | None:
    rel = MODULE_FILES.get(method)
    if not rel:
        return None
    p = sc.VENDOR_DIR / rel
    return sc.sha256_file(p) if p.is_file() else None


__all__ = ["MAIN_METHODS", "REFERENCE_METHODS", "METHOD_ORDER", "MODULE_FILES", "bootstrap",
           "load", "params_for", "call", "normalize", "_validate_output", "module_hash",
           "MethodLoadError"]
