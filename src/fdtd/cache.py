#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""缓存键与原子续跑（键基于**完整物理参数 + 轨迹源 hash**，不基于文件名约定）。

纪律
----
* cache key = sha256(协议版本 + 公式/链路版本 + 本包源码指纹 + 物理参数 + Meep 数值参数
  + **解析后的 domain（size/center/PML/margin）** + 场景几何/目标 + 轨迹源 hash + 背景模式)；
  **任一物理/数值参数变化必然换键**。
* 每个 (状态, 或 background) 单独一个 .npz，**原子写**（临时文件 + ``os.replace``）；
  manifest 记录**文件 sha256 + 字节数**；manifest 的读-改-写由 ``manifest.lock`` 跨平台
  互斥锁串行化（并行分片求解时多进程同写 manifest 不丢条目）。
* 续跑判定 ``has()`` **不只看 key**：还要复算文件 sha256、完整读回并校验
  （必需键、shape、finite、时间轴单调且等间隔、n_time 与时间轴一致）。
* 浮点哈希不做小数位 round（见 protocol.canonical_json_bytes 的精度纪律）。
* 本模块**不提供删除入口**（无 rmtree）：缓存清理由人手动决定，避免误删大目录。
"""

from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

import numpy as np

from .protocol import (
    CHAIN_VERSION,
    PROTOCOL_VERSION,
    PULSE_FORMULA_ID,
    atomic_write_json,
    atomic_write_npz,
    file_sha256,
    package_code_fingerprint,
    read_json,
    sha256_hex,
)

#: cache key 必须是 64 位小写十六进制（sha256）；不满足直接拒绝，避免路径拼接污染。
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


@contextlib.contextmanager
def _exclusive_lock(path: Path, timeout_s: float = 300.0) -> Iterator[None]:
    """跨平台互斥锁（fcntl / msvcrt）：保护 manifest.json 的读-改-写。

    并行分片求解时多个进程会同时 ``save()`` 不同状态条目；npz 本身按条目独立原子写
    互不冲突，但 manifest 是单文件读-改-写，必须串行化，否则会丢条目（丢条目的
    后果是 ``has()`` 判否 -> 该状态被重复求解，浪费但不致脏数据）。
    锁文件与被保护文件同目录（``<dir>/manifest.lock``），只锁不写内容。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        try:
            import fcntl  # POSIX

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except ImportError:
            import msvcrt  # Windows

            deadline = time.monotonic() + float(timeout_s)
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
        yield
    finally:
        try:
            try:
                import fcntl  # POSIX

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except ImportError:
                import msvcrt  # Windows

                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            fh.close()

#: 进入 cache key 的物理/数值参数字段（缺一不可，缺字段直接报错）。
KEY_PHYSICS_FIELDS = (
    "fc_hz",
    "bandwidth_hz",
    "sample_interval_s",
    "raw_taps",
    "crop_start",
    "fft_length",
    "snapshots",
    "aa_cutoff_hz",
    "aa_pad_min_factor",
    "match_filter_mode",
    "c0_m_per_s",
)


def cache_key(
    cfg: Mapping[str, Any],
    scene: Mapping[str, Any],
    trajectory_source_hash: str,
    background_mode: str,
    ema: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    physics = {}
    missing = [k for k in KEY_PHYSICS_FIELDS if k not in cfg]
    if missing:
        raise KeyError("cache key 缺物理字段：%s" % missing)
    for k in KEY_PHYSICS_FIELDS:
        physics[k] = cfg[k]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "pulse_formula_id": PULSE_FORMULA_ID,
        "chain_version": CHAIN_VERSION,
        "code_fingerprint": package_code_fingerprint(),
        "physics": physics,
        "meep": dict(scene.get("meep", {})),
        "domain": dict(scene.get("domain", {})),
        "geometry": dict(scene.get("geometry", {})),
        "targets": dict(scene.get("targets", {})),
        "dimensions": scene.get("dimensions", 2),
        "length_unit_m": scene.get("length_unit_m", 1.0),
        "trajectory_source_hash": trajectory_source_hash,
        "background_mode": background_mode,
        "ema": dict(ema or {}),
        "extra": dict(extra or {}),
    }
    return sha256_hex(payload)


class RunCache:
    """``<root>/<cache_key>/`` 下的原子缓存：background.npz 与 state_XXXX.npz。"""

    def __init__(self, root: Path | str, key: str):
        self.key = str(key)
        if not _KEY_RE.match(self.key):
            raise ValueError("cache key 必须是 64 位小写 hex（sha256），得到 %r" % self.key)
        self.root = Path(root).resolve()
        child = (self.root / self.key).resolve()
        if child == self.root or not child.is_relative_to(self.root):
            raise ValueError("cache 目录越出根目录：%s" % child)
        self.dir = child
        self.dir.mkdir(parents=True, exist_ok=True)

    def entry_name(self, kind: str, index: int | None = None) -> str:
        return kind if kind == "background" else "%s_%05d" % (kind, int(index))

    def path_for(self, kind: str, index: int | None = None) -> Path:
        if kind == "background":
            return self.dir / "background.npz"
        if kind == "state":
            if index is None:
                raise ValueError("state 条目必须给 index")
            return self.dir / ("state_%05d.npz" % int(index))
        raise ValueError("未知 kind：%s" % kind)

    def has(self, kind: str, index: int | None = None) -> bool:
        """条目可用性：文件存在 + 字节数一致 + 文件 sha256 一致 + 读回结构/数值校验通过。"""
        path = self.path_for(kind, index)
        if not path.is_file():
            return False
        entry = self.manifest_entries().get(self.entry_name(kind, index))
        if not entry or entry.get("sha256") is None:
            return False
        if int(entry.get("bytes", -1)) != int(path.stat().st_size):
            return False
        if str(entry.get("sha256")) != file_sha256(path):
            return False
        try:
            self.load(kind, index)
        except Exception:
            return False
        return True

    def save(self, kind: str, index: int | None, arrays: Mapping[str, Any], meta: Mapping[str, Any]) -> Path:
        path = self.path_for(kind, index)
        payload: Dict[str, Any] = {
            "cache_key": np.array(self.key),
            "kind": np.array(kind),
            "index": np.array(-1 if index is None else int(index)),
            "meta_json": np.array(__import__("json").dumps(dict(meta), ensure_ascii=False, sort_keys=True)),
        }
        for name, value in arrays.items():
            payload[str(name)] = np.asarray(value)
        written = atomic_write_npz(path, **payload)
        entry = {
            "kind": kind,
            "index": (-1 if index is None else int(index)),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
            "n_records_dim0": int(np.asarray(payload["records"]).shape[0]) if "records" in payload else None,
            "n_time": int(np.asarray(payload["records"]).shape[-1]) if "records" in payload else None,
        }
        self.update_manifest({"entries": {self.entry_name(kind, index): entry}})
        return written

    def load(self, kind: str, index: int | None = None) -> Dict[str, Any]:
        import json

        path = self.path_for(kind, index)
        with np.load(path, allow_pickle=False) as npz:
            out = {name: npz[name] for name in npz.files}
        if str(out["cache_key"].item()) != self.key:
            raise ValueError("缓存键不一致（不应发生，需清理重跑）：%s" % path)
        out["meta"] = json.loads(str(out.pop("meta_json").item()))
        out.pop("cache_key", None)
        out.pop("kind", None)
        out.pop("index", None)
        self._validate_entry(kind, out, path)
        return out

    def _validate_entry(self, kind: str, data: Mapping[str, Any], path: Path) -> None:
        """结构 + 数值校验：缺键、非有限值、shape 不匹配、时间轴非等间隔一律拒绝。"""
        for key in ("records", "t_s"):
            if key not in data:
                raise ValueError("缓存条目缺 %s：%s" % (key, path))
        rec = np.asarray(data["records"])
        t_s = np.asarray(data["t_s"], dtype=float)
        if rec.ndim < 2:
            raise ValueError("records 维度应 >= 2，得到 %s" % (rec.shape,))
        if t_s.ndim != 1 or t_s.shape[0] != rec.shape[-1]:
            raise ValueError("t_s 形状 %s 与 records 末轴 %d 不一致" % (t_s.shape, rec.shape[-1]))
        if rec.shape[-1] < 2:
            raise ValueError("records 时间轴过短：%d" % rec.shape[-1])
        if not np.all(np.isfinite(rec)):
            raise ValueError("records 含非有限值（NaN/Inf）：%s" % path)
        d = np.diff(t_s)
        if not np.all(d > 0):
            raise ValueError("t_s 非严格单调递增：%s" % path)
        if not np.allclose(d, d[0], rtol=1e-9, atol=0.0):
            raise ValueError("t_s 非等间隔（at_every 未按预期触发？）：%s" % path)

    def manifest_entries(self) -> Dict[str, Any]:
        path = self.dir / "manifest.json"
        if not path.is_file():
            return {}
        try:
            return dict(read_json(path).get("entries", {}))
        except Exception:
            return {}

    def update_manifest(self, patch: Mapping[str, Any]) -> Path:
        path = self.dir / "manifest.json"
        with _exclusive_lock(self.dir / "manifest.lock"):
            cur: Dict[str, Any] = read_json(path) if path.is_file() else {"cache_key": self.key, "entries": {}}
            cur.setdefault("entries", {})
            for k, v in patch.items():
                if k == "entries" and isinstance(v, Mapping):
                    cur["entries"].update(v)
                else:
                    cur[k] = v
            return atomic_write_json(path, cur)

    def disk_usage_bytes(self) -> int:
        total = 0
        for p in self.dir.glob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

def free_bytes(path: Path | str) -> int:
    """磁盘余量（只读查询，供部署前容量检查）。"""
    import shutil as _shutil

    return int(_shutil.disk_usage(str(path)).free)
