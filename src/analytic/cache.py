#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 A clean 数据缓存：解析正向 -> clean raw CIR npz（key=raw_cir, complex [L_clean,4,32]）。

硬性约束
--------
* 一个 clean_id 只生成一份 clean 序列（E4 的多个 L 档共享同一 clean_id -> 同一 128 序列，
  运行期取前 L 快照切片）。
* clean npz **只含 raw 数据与必要非真值 metadata**：不含目标坐标 / k / 任何真值字段
  （真值只写独立 truth JSON，供 evaluation 使用）。
* 读写都做完整性校验：签名（协议哈希 + 源指纹 + clean 规格）+ npz sha256 + 形状 + 有限性；
  truth JSON 亦有 sha256 校验（防篡改影响 metrics）。
* 原子写入；签名/哈希不严格匹配即拒绝复用（不默用旧 cache）。
"""

from __future__ import annotations

import json
from pathlib import Path

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np

from . import forward

INDEX_NAME = "analytic_index.json"
TRUTH_DIR = "truth"
NPZ_NON_TRUTH_KEYS = ("clean_id", "L_clean", "sample_interval_s", "fc_hz", "bandwidth_hz",
                      "raw_taps", "crop_start", "fft_length", "snapshots", "protocol_version",
                      "t_origin_definition")


class CacheError(RuntimeError):
    """缓存签名/完整性不匹配（显式失败，不静默降级）。"""


def _index_path(cache_dir) -> Path:
    return Path(cache_dir) / INDEX_NAME


def _load_index(cache_dir) -> dict:
    p = _index_path(cache_dir)
    if not p.is_file():
        return {"kind": "sim-analytic-index", "entries": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_index(cache_dir, index: dict) -> None:
    sc.atomic_write_json(_index_path(cache_dir), index)


def clean_signature(clean_spec: dict, *, protocol_hash: str, source_aggregate: str) -> str:
    return sc.param_hash({"clean_spec": clean_spec, "protocol_hash": protocol_hash,
                          "source_aggregate": source_aggregate})


def _npz_path(cache_dir, clean_id: str) -> Path:
    return Path(cache_dir) / "clean" / ("%s.npz" % clean_id)


def _truth_path(cache_dir, clean_id: str) -> Path:
    return Path(cache_dir) / TRUTH_DIR / ("%s.json" % clean_id)


def build_clean(clean_spec: dict, *, cache_dir, protocol_hash: str, source_aggregate: str,
                force: bool = False) -> dict:
    """生成/校验 clean 序列（幂等）。返回 meta（**索引/provenance**，非 case 可见）。"""
    clean_id = str(clean_spec["clean_id"])
    sig = clean_signature(clean_spec, protocol_hash=protocol_hash,
                          source_aggregate=source_aggregate)
    index = _load_index(cache_dir)
    npz_path = _npz_path(cache_dir, clean_id)
    entry = index["entries"].get(clean_id)
    if entry is not None and not force:
        if str(entry.get("signature")) != sig:
            raise CacheError("clean_id=%s 缓存签名不匹配（已有 %s != 本次 %s）：拒绝复用旧 cache"
                             % (clean_id, str(entry.get("signature"))[:12], sig[:12]))
        if not npz_path.is_file():
            raise CacheError("索引有 %s 但 npz 缺失：%s" % (clean_id, npz_path))
        if sc.sha256_file(npz_path) != entry.get("npz_sha256"):
            raise CacheError("clean_id=%s npz 哈希与索引不符（文件被改动）" % clean_id)
        return dict(entry, reused=True)

    L_clean = int(clean_spec.get("L_clean", sc.SNAPSHOTS))
    k = int(clean_spec["k"])
    targets_xy = clean_spec["targets_xy"]
    amps = None
    if clean_spec.get("power_diff_db"):
        from . import scenes as scene_mod
        amps = scene_mod.amplitude_vector(k, float(clean_spec["power_diff_db"]))
    fwd = forward.MethodFreeForward(
        targets_xy, k=k, L=L_clean, rng=sc.rng_from(sc.clean_seed(clean_id)), amps=amps,
        coherence=float(clean_spec.get("coherence", 0.0)),
        compensate_crop=bool(clean_spec.get("compensate_crop", False)))
    raw = np.ascontiguousarray(np.asarray(fwd.raw, dtype=np.complex128))
    if raw.shape != (L_clean, sc.N_CHANNELS, sc.RAW_TAPS):
        raise CacheError("clean raw 形状异常：%s" % (raw.shape,))
    if not np.all(np.isfinite(raw)):
        raise CacheError("clean raw 含 NaN/Inf")
    sc.atomic_write_npz(
        npz_path, raw_cir=raw, clean_id=np.array(clean_id),
        L_clean=np.int64(L_clean), sample_interval_s=np.array(float(sc.SAMPLE_INTERVAL_S)),
        fc_hz=np.array(float(sc.FC_HZ)), bandwidth_hz=np.array(float(sc.BANDWIDTH_HZ)),
        raw_taps=np.int64(sc.RAW_TAPS), crop_start=np.int64(sc.CROP_START),
        fft_length=np.int64(sc.FFT_LENGTH), snapshots=np.int64(L_clean),
        protocol_version=np.array(str(_protocol_version())),
        t_origin_definition=np.array("raw tap0 = physical zero delay; crop offset retained"))
    truth = {
        "clean_id": clean_id, "k": k,
        "targets_xy_m": [[float(x), float(y)] for x, y in targets_xy],
        "targets_polar": [[float(r), float(t)] for r, t in clean_spec["targets_polar"]],
        "scene": clean_spec.get("scene"), "family": clean_spec.get("family"),
        "note": "真值独立 JSON，不进入 case（case 由 q6_contract 硬门控真值字段）。",
    }
    truth_path = _truth_path(cache_dir, clean_id)
    sc.atomic_write_json(truth_path, truth)
    meta = {
        "clean_id": clean_id, "k": k, "L_clean": int(L_clean),
        "scene": clean_spec.get("scene"), "family": clean_spec.get("family"), "source": "analytic",
        "targets_xy_m": [[float(x), float(y)] for x, y in targets_xy],
        "targets_polar": clean_spec["targets_polar"],
        "clean_seed": int(sc.clean_seed(clean_id)),
        "protocol_hash": protocol_hash, "source_aggregate": source_aggregate, "signature": sig,
        "time_origin": "physical_zero(raw tap0 = tau 0); crop offset retained & disclosed",
        "crop_offset_range_m": float(sc.crop_offset_range_m()),
        "compensate_crop_diagnostic_only": bool(clean_spec.get("compensate_crop", False)),
        "coherence": float(clean_spec.get("coherence", 0.0)),
        "power_diff_db": float(clean_spec.get("power_diff_db", 0.0)),
        "max_delay_tap": float(fwd.meta()["max_delay_tap"]),
        "noise_prefix_policy": "noise drawn for L_clean then sliced to L (deterministic prefix)",
        "npz_path": str(npz_path), "npz_sha256": sc.sha256_file(npz_path),
        "truth_path": str(truth_path), "truth_sha256": sc.sha256_file(truth_path),
        "npz_keys": list(NPZ_NON_TRUTH_KEYS), "raw_shape": list(raw.shape),
        "built_at": sc.now_iso(),
    }
    index["entries"][clean_id] = meta
    index["n_entries"] = len(index["entries"])
    _save_index(cache_dir, index)
    return dict(meta, reused=False)


def load_clean(cache_dir, clean_id: str, *, expect_signature: str | None = None) -> dict:
    """读取 clean 序列；校验签名 + npz sha256 + 形状 + 有限性。"""
    index = _load_index(cache_dir)
    entry = index["entries"].get(str(clean_id))
    npz_path = _npz_path(cache_dir, clean_id)
    if entry is None or not npz_path.is_file():
        raise CacheError("clean_id=%s 缓存缺失：%s" % (clean_id, npz_path))
    if expect_signature is not None and str(entry.get("signature")) != str(expect_signature):
        raise CacheError("clean_id=%s 签名不匹配：拒绝复用旧 cache" % clean_id)
    if sc.sha256_file(npz_path) != entry.get("npz_sha256"):
        raise CacheError("clean_id=%s npz 哈希与索引不符（文件被改动）" % clean_id)
    with np.load(npz_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_cir"])
    expect_shape = (int(entry["L_clean"]), sc.N_CHANNELS, sc.RAW_TAPS)
    if raw.shape != expect_shape:
        raise CacheError("clean_id=%s raw 形状 %s != %s" % (clean_id, raw.shape, expect_shape))
    if not np.all(np.isfinite(raw)):
        raise CacheError("clean_id=%s raw 含 NaN/Inf" % clean_id)
    return {"raw_cir": raw, "meta": entry, "path": str(npz_path)}


def load_truth(cache_dir, clean_id: str) -> dict:
    """读取真值 JSON；校验 sha256（防篡改影响 metrics）。"""
    index = _load_index(cache_dir)
    entry = index["entries"].get(str(clean_id))
    if entry is None:
        raise CacheError("clean_id=%s 索引缺失" % clean_id)
    p = _truth_path(cache_dir, clean_id)
    if not p.is_file():
        raise CacheError("clean_id=%s 真值缺失：%s" % (clean_id, p))
    have = sc.sha256_file(p)
    if entry.get("truth_sha256") and have != entry["truth_sha256"]:
        raise CacheError("clean_id=%s 真值哈希与索引不符（被篡改）" % clean_id)
    return json.loads(p.read_text(encoding="utf-8"))


def _protocol_version() -> str:
    try:
        from protocol.sim_protocol import PROTOCOL_VERSION
        return PROTOCOL_VERSION
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = ["build_clean", "load_clean", "load_truth", "clean_signature", "CacheError",
           "INDEX_NAME", "TRUTH_DIR", "NPZ_NON_TRUTH_KEYS"]
