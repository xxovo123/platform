# -*- coding: utf-8 -*-
"""结果目录哈希清单（ISSUE-3）：让结果与驱动代码事后可校验。

生成 results/manifest.sha256.json，内含：
  * 本目录每个产物的相对路径 / 字节数 / SHA256；
  * driver 每个 .py 文件的 SHA256；
  * 协议文件路径 + canonical SHA256 + file SHA256；
  * 生成时间与命令行。
replay_log.json 会引用该清单（字段 manifest）。
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .protocol import canonical_sha256, sha256_file

MANIFEST_NAME = "manifest.sha256.json"
DRIVER_DIR = Path(__file__).resolve().parent


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_manifest(results_dir, protocol, *, command=None, extra_files=()) -> dict:
    results_dir = Path(results_dir)
    artifacts = {}
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        artifacts[_rel(path, results_dir)] = {"bytes": path.stat().st_size,
                                              "sha256": sha256_file(path)}
    driver_files = {}
    for path in sorted(DRIVER_DIR.glob("*.py")):
        driver_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    extras = {}
    for path in extra_files:
        path = Path(path)
        if path.is_file():
            extras[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    payload = {
        "schema": "p0b-results-manifest/1",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "command": command,
        "protocol": {
            "path": str(protocol.path),
            "protocol_id": protocol.protocol_id,
            "protocol_revision": protocol.protocol_revision,
            "canonical_sha256": protocol.canonical_sha256_value,
            "file_sha256": protocol.file_sha256,
        },
        "n_artifacts": len(artifacts),
        "artifacts": artifacts,
        "driver_dir": str(DRIVER_DIR),
        "driver_files": driver_files,
        "n_driver_files": len(driver_files),
        "extra_files": extras,
    }
    target = results_dir / MANIFEST_NAME
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
                      encoding="utf-8")
    return payload


def verify_manifest(results_dir, *, protocol_path=None) -> dict:
    """按清单复算：返回 {path: bool} 与总体 pass（供独立验收复用）。"""
    results_dir = Path(results_dir)
    manifest_path = results_dir / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, info in (payload.get("artifacts") or {}).items():
        path = results_dir / rel
        if not path.is_file():
            mismatches.append({"path": rel, "issue": "missing"})
            continue
        if sha256_file(path) != info["sha256"]:
            mismatches.append({"path": rel, "issue": "sha256_mismatch"})
    for name, info in (payload.get("driver_files") or {}).items():
        path = DRIVER_DIR / name
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            mismatches.append({"path": str(path), "issue": "driver_sha256_mismatch"})
    result = {
        "manifest": str(manifest_path),
        "n_artifacts": len(payload.get("artifacts") or {}),
        "n_driver_files": len(payload.get("driver_files") or {}),
        "mismatches": mismatches,
        "pass": not mismatches,
        "canonical_sha256_of_driver_dir_recorded": (payload.get("protocol") or {}).get(
            "canonical_sha256"),
    }
    if protocol_path is not None:
        payload_protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        result["protocol_canonical_matches_file"] = (
            canonical_sha256(payload_protocol)
            == (payload.get("protocol") or {}).get("canonical_sha256"))
    return result

