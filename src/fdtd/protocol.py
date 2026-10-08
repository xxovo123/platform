#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""协议常量、规范化 JSON、哈希与原子写入工具（**不导入 meep**）。

版本纪律
--------
* ``PROTOCOL_VERSION``：公共协议字段/语义变化时递增。
* ``PULSE_FORMULA_ID``：源脉冲公式变化时递增（源与 MF 参考同时受影响）。
* ``CHAIN_VERSION``：RF→IQ→抗混叠→匹配滤波链路实现变化时递增。
三者都进入 cache key 与 provenance，避免“同名不同物”的缓存串用。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROTOCOL_VERSION = "fdtd-protocol-20260926.2"  # .2：crop_start 2->0（2026-09-26）
PULSE_FORMULA_ID = "bh_derivative_v1"
CHAIN_VERSION = "rfchain-20260926.1"

#: 物理与链路默认值（可被 protocol JSON / 调用方覆盖）。数值来源：
#: 硬件 meta 与旧仿真字面量（见 大修执行_20260923/仿真实验完整方案_20260926.md §4.1）。
PHYSICS_DEFAULTS: Dict[str, Any] = {
    "fc_hz": 7.9872e9,
    "bandwidth_hz": 5.0e8,
    "sample_interval_s": 1.0e-9,
    "raw_taps": 32,
    "crop_start": 0,  # 2026-09-26 v2：仿真不裁前 2 物理 tap（v1 曾为 2）
    "fft_length": 32,
    "snapshots": 128,
    # 抗混叠：复数基带低通截止，Nyquist = 1/(2*Ts) = 500 MHz（复数采样 → ±500 MHz）
    "aa_cutoff_hz": 4.0e8,
    "aa_pad_min_factor": 2.0,
    "match_filter_mode": "full",
    "c0_m_per_s": 2.99792458e8,
}

#: clean case（算法可见）允许出现的键；真值/轨迹/诊断键一律不允许。
CLEAN_CASE_ALLOWED_KEYS = frozenset(
    {
        "raw_cir",
        "pair_index",
        "tx_index",
        "rx_index",
        "state_index",
        "t_origin_s",
        "t_origin_definition",
        "background_mode",
        "sample_interval_s",
        "fc_hz",
        "bandwidth_hz",
        "raw_taps",
        "crop_start",
        "fft_length",
        "snapshots",
        "scene_id",
        "channel_names",
        "protocol_version",
        "chain_version",
        "cache_key",
    }
)

#: 明显属于真值/诊断的键名片段（大小写不敏感），用于 case 清洗断言。
TRUTH_KEY_MARKERS = (
    "truth",
    "target",
    "nominal",
    "trajectory",
    "label",
    "diagnostic",
    "expected_",
)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并：``override`` 覆盖 ``base``，不修改入参。"""
    out: Dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def canonical_json_bytes(obj: Any) -> bytes:
    """规范化 JSON 字节串：sorted keys、紧凑分隔符、浮点按 repr 往返。

精度纪律（2026-09-26 修正）
--------------------------
**不得**对浮点做任何统一小数位的 round 后再哈希。物理量单位不同（Hz / s / m / 无量纲），
按固定小数位 round 会：
  * 把绝对值很小的量（如 ps 级时间 1e-13 s → ``round(x, 12) == 0.0``）归零，与真 0 碰撞；
  * 对绝对值很大的量（如 fc ≈ 8e9）丢掉小数位以下的相对信息。
因此这里只用 ``json.dumps`` 的 float ``repr``（Python 保证最短往返表示）：同一个 float64
位型 → 同一字节串；不同位型 → 不同字节串。
"""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    """对象 → 规范化 JSON → sha256 hex。"""
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(bytes(obj)).hexdigest()
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def code_fingerprint(paths: Iterable[Path | str], root: Path | str | None = None) -> str:
    """源码指纹：**相对路径** + 文件 sha256 的排序拼接哈希（用于 cache key）。

用相对路径而不是 ``p.name``，避免不同目录下同名文件（如两个 ``__init__.py``、
或 ``pulse.py`` 与别的 ``pulse.py``）互相碰撞、也避免两条不同文件集合同哈希。
未给 ``root`` 时退化为绝对路径（仍然唯一，只是不便于跨机器复现）。
    """
    pairs: list[tuple[str, str | None]] = []
    for raw in paths:
        p = Path(raw).resolve()
        if root is not None:
            try:
                label = str(p.relative_to(Path(root).resolve()))
            except ValueError:
                label = str(p)
        else:
            label = str(p)
        pairs.append((label, file_sha256(p) if p.is_file() else None))
    pairs.sort(key=lambda kv: kv[0])
    return sha256_hex([list(item) for item in pairs])[:32]


def package_code_fingerprint() -> str:
    """本包内**可能影响数值结果**的全部源码指纹（含调用链，勿漏）。

    取 ``fdtd/*.py`` 全部（不含 ``tests/``）：protocol / pulse / rf_chain / trajectory /
    scenes / cache / solver / api / cli / __init__ / __main__。任何一处改动（包括驱动层
    参数组装）都会换 cache key，宁可多失效一次，也不要读到“同名不同物”的缓存。
    """
    here = Path(__file__).resolve().parent
    files = sorted(p for p in here.glob("*.py") if p.is_file())
    return code_fingerprint(files, root=here)


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """原子写：同目录临时文件 + ``os.replace``（避免半截文件被当成有效缓存）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def atomic_write_json(path: Path | str, obj: Any) -> Path:
    return atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


def atomic_write_npz(path: Path | str, **arrays: Any) -> Path:
    """原子写 npz（先写同目录临时文件，再 ``os.replace``）。"""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-", suffix=".npz.part")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def assert_clean_case(keys: Iterable[str]) -> None:
    """断言 clean case 不含真值/诊断字段（大小写不敏感）。"""
    bad = []
    for key in keys:
        low = str(key).lower()
        if key not in CLEAN_CASE_ALLOWED_KEYS:
            bad.append(key)
            continue
        if any(marker in low for marker in TRUTH_KEY_MARKERS):
            bad.append(key)
    if bad:
        raise ValueError("clean case 含非算法可见键：%s" % sorted(bad))
