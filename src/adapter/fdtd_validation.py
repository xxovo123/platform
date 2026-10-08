#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 B 门禁（**纯薄包装**）：只调用 fdtd 代理已就绪的严格 helper，不自造/不弱化判定。

对接（deployment/validation_schema.md + fdtd/convergence.py::check_evidence）
------------------------------------------------------------------------
* 唯一判定入口：fdtd.convergence.check_evidence(evidence_dir, scene=..., protocol_path=...)
  —— 其内部已严格实现：formal=true（tiny/validation-only 直接拒绝）、criteria 冻结、
  comparisons 全通过且非空、strict_params 严格相等、目标 (res, runtime) 必须是 tested_combos
  的 strict member、verdict=passed。本模块**不重复实现、也不放宽**。
* 证据可为：目录（含 convergence.json 或 convergence_manifest.json）、单报告文件（取其目录）、
  或 manifest（{"dirs":[...]} 或 {"reports":[...]} 列目录/文件）。
* helper 未就绪（不可导入/缺 check_evidence）-> 明确 blocked（正式 B 暂不可跑；交付需 wired）。
* 只读证据；不写/不改 fdtd 任何文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class ValidationGateError(RuntimeError):
    """门禁不可用/输入不合规（显式失败）。"""


def _fdtd_conv():
    pkg = str(Path(__file__).resolve().parents[1])
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from fdtd import convergence as conv
    return conv


def _load_json(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ValidationGateError("证据缺失：%s" % p)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValidationGateError("不可解析：%s: %s" % (type(exc).__name__, exc))


def evidence_dirs(path) -> list:
    """用户输入 -> 证据目录列表（目录 / 单报告文件 / manifest）。"""
    p = Path(path)
    if p.is_dir():
        return [p]
    obj = _load_json(p)
    if isinstance(obj, dict):
        items = obj.get("dirs") or obj.get("reports")
        if isinstance(items, list) and items:
            base = p.resolve().parent
            out = []
            for item in items:
                if not isinstance(item, str):
                    raise ValidationGateError("manifest 条目必须是路径字符串：%r" % (item,))
                q = Path(item)
                q = q if q.is_absolute() else base / q
                out.append(q if q.is_dir() else q.parent)
            return out
    return [p.parent]


def gate_scenes(scenes_map: dict, validation_path, *, protocol=None, protocol_path=None) -> dict:
    """逐 scene 调用 fdtd.convergence.check_evidence；任一未 applicable+passed 即 overall 拒绝。"""
    try:
        conv = _fdtd_conv()
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "blocked": True, "route": "unavailable",
                "reason": "fdtd.convergence helper 未就绪（正式 B 暂不可跑；交付需 wired）：%s: %s"
                          % (type(exc).__name__, exc),
                "report_path": str(validation_path), "by_scene": {}, "reasons": [],
                "note": "不弱化为本地内置判定；缺 helper 即 blocked。"}
    if not hasattr(conv, "check_evidence"):
        return {"passed": False, "blocked": True, "route": "unavailable",
                "reason": "fdtd.convergence 缺 check_evidence（交付需 wired）",
                "report_path": str(validation_path), "by_scene": {}, "reasons": [], "note": "不弱化。"}
    dirs = evidence_dirs(validation_path)
    by_scene, reasons, uncovered = {}, [], []
    for cid, scene in scenes_map.items():
        entries, covered, matched = [], False, None
        for d in dirs:
            try:
                chk = conv.check_evidence(d, scene=scene, protocol_path=protocol_path)
                entries.append({"evidence_dir": str(d), "applicable": chk.get("applicable"),
                                "passed": chk.get("passed"), "matched_entry": chk.get("matched_entry"),
                                "matched_combo": chk.get("matched_combo"),
                                "want": chk.get("want"), "reasons": chk.get("reasons"),
                                "semantics": chk.get("semantics")})
                if chk.get("applicable") and chk.get("passed"):
                    covered, matched = True, str(d)
                    break
            except Exception as exc:  # noqa: BLE001
                entries.append({"evidence_dir": str(d),
                                "error": "%s: %s" % (type(exc).__name__, exc)})
        by_scene[cid] = {"entries": entries, "covered": covered, "matched_evidence_dir": matched}
        if not covered:
            uncovered.append(cid)
            reasons.append("%s 无 applicable+passed 的 check_evidence 证据" % cid)
    return {"passed": not uncovered, "blocked": False, "route": "fdtd.convergence.check_evidence",
            "evidence_dirs": [str(d) for d in dirs], "by_scene": by_scene,
            "n_scenes": len(scenes_map), "n_uncovered": len(uncovered),
            "uncovered_scenes": uncovered, "reasons": reasons,
            "report_path": str(validation_path),
            "note": ("严格调用 fdtd.convergence.check_evidence；formal=false/未冻结/comparisons 空/"
                     "strict_params 不等/(res,runtime) 非 tested 组合 均拒绝；只读证据。")}


__all__ = ["ValidationGateError", "evidence_dirs", "gate_scenes"]
