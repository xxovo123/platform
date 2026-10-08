#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 B 数据源：薄适配 fdtd 代理的 fdtd/api.py::generate_clean（不改其实现）。

契约对齐（与层 A 一致）
----------------------
* clean raw CIR：raw_cir [L_clean, 4, 32]，**tap0 = 物理零延迟**；不做 crop2（由 adapter 前端处理）。
* 无噪轨迹只生成一次并按 clean_id 复用（跨 SNR/L）；噪声由本 runner 在下游统一施加。
* sidecar 只存**相对路径**（相对 sidecar 所在目录）+ npz/truth sha256 + 形状；
  load 时校验 sha256 + 形状 + 有限性，签名漂移即拒绝（不默用旧 cache，也不依赖绝对路径）。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import sim_common as sc

import numpy as np

from .scene_export import spec_to_scene

SIDECAR = "sim_source.json"

# FDTD-CACHE-PHYSICS-SCOPE-20260929: 层 B clean 的缓存身份只含**物理范围**，
# 不含算法数值源码 aggregate / 协议版本号。理由：层 A 正向或算法源码变化不应
# 强迫重解 FDTD；而目标几何、背景、网格、收敛档、协议物理常量变化必须失配。
FDTD_SIGNATURE_SCOPE = "physics_only_v1"


def fdtd_physics_scope(clean_spec, *, geometry=None, max_states=None, formal=True,
                       res_override=None) -> dict:
    """层 B clean 签名用的物理范围字典（确定性；不含算法源码哈希）。"""
    return {
        "scope": FDTD_SIGNATURE_SCOPE,
        "clean_spec": clean_spec,
        "geometry": geometry or "protocol_main",
        "max_states": max_states,
        "formal": bool(formal),
        "res_override": res_override,
        "protocol_physics": {
            "fc_hz": float(sc.FC_HZ),
            "bandwidth_hz": float(sc.BANDWIDTH_HZ),
            "sample_interval_s": float(sc.SAMPLE_INTERVAL_S),
            "raw_taps": int(sc.RAW_TAPS),
            "crop_start": int(sc.CROP_START),
            "fft_length": int(sc.FFT_LENGTH),
            "snapshots": int(sc.SNAPSHOTS),
            "n_channels": int(sc.N_CHANNELS),
            "epsilon_r_background": float(sc.EPSILON_R_BACKGROUND),
        },
    }


def fdtd_signature(clean_spec, *, geometry=None, max_states=None, formal=True,
                   res_override=None) -> str:
    """层 B clean 的物理范围签名（替代含 source_aggregate 的旧口径）。"""
    return sc.param_hash(fdtd_physics_scope(clean_spec, geometry=geometry,
                                            max_states=max_states, formal=bool(formal),
                                            res_override=res_override))



class FdtdSourceError(RuntimeError):
    """层 B 数据源不可用（显式失败）。"""


class FdtdSource:
    name = "fdtd"
    layer = "B"

    def __init__(self, module_name: str = "fdtd.api"):
        self.module_name = module_name
        self._mod = None
        self._reason = ""

    def _load(self):
        if self._mod is None:
            base = str(sc.PKG_DIR)
            import sys
            if base not in sys.path:
                sys.path.insert(0, base)
            try:
                self._mod = importlib.import_module(self.module_name)
            except Exception as exc:  # noqa: BLE001
                self._reason = ("fdtd provider %s 不可导入：%s: %s"
                                % (self.module_name, type(exc).__name__, exc))
        return self._mod

    def available(self) -> tuple:
        mod = self._load()
        if mod is None or not hasattr(mod, "generate_clean"):
            return False, (self._reason or ("provider %s 缺 generate_clean" % self.module_name))
        return True, "provider %s" % self.module_name

    def _dir(self, cache_dir, clean_id: str) -> Path:
        return Path(cache_dir) / "fdtd" / str(clean_id)

    def build_clean(self, clean_spec, *, cache_dir, protocol_hash, source_aggregate,
                    resume: bool = True, max_states=None, dry_run=False,
                    geometry=None, formal: bool = True, workers: int = 1,
                    res_overrides: dict | None = None):
        ok, reason = self.available()
        if not ok:
            raise FdtdSourceError(reason)
        clean_id = str(clean_spec["clean_id"])
        res_ov = (res_overrides or {}).get(clean_id)
        # FDTD-CACHE-PHYSICS-SCOPE-20260929：签名只看物理范围（不含 source_aggregate / 协议版本号）
        sig = fdtd_signature(clean_spec, geometry=geometry, max_states=max_states,
                             formal=bool(formal), res_override=res_ov)
        outdir = self._dir(cache_dir, clean_id)
        sidecar = outdir / SIDECAR
        if resume and sidecar.is_file():
            have = json.loads(sidecar.read_text(encoding="utf-8"))
            if str(have.get("signature")) != sig:
                raise FdtdSourceError("clean_id=%s 层 B 签名不匹配：拒绝复用旧 cache" % clean_id)
            return dict(have, reused=True)
        scene = (clean_spec.get("fdtd_scene")
                 or spec_to_scene(clean_spec, geometry=geometry,
                                  res_override=(res_overrides or {}).get(clean_id)))
        res = self._mod.generate_clean(None, scene, outdir,
                                       cache_root=Path(cache_dir) / "fdtd" / "_solver_cache",
                                       resume=resume, max_states=max_states, workers=int(workers),
                                       background_mode=clean_spec.get("background_mode"),
                                       dry_run=dry_run)
        if dry_run:
            return {"dry_run": True, "plan": res.get("plan"), "clean_id": clean_id, "scene": scene}
        npz_path = Path(res["clean_npz"])
        truth_path = Path(res["truth_json"])
        with np.load(npz_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_cir"])
        expect = (int(raw.shape[0]), sc.N_CHANNELS, sc.RAW_TAPS)
        if tuple(raw.shape) != expect:
            raise FdtdSourceError("层 B raw_cir 形状 %s != %s" % (tuple(raw.shape), expect))
        if not np.all(np.isfinite(raw)):
            raise FdtdSourceError("层 B raw_cir 含 NaN/Inf")
        meta = {
            "clean_id": clean_id, "k": int(clean_spec["k"]), "L_clean": int(raw.shape[0]),
            "scene": clean_spec.get("scene"), "source": "fdtd",
            "targets_xy_m": clean_spec["targets_xy"], "targets_polar": clean_spec["targets_polar"],
            "signature": sig, "signature_scope": FDTD_SIGNATURE_SCOPE,
            "protocol_hash": protocol_hash, "source_aggregate": source_aggregate,
            "time_origin": "physical_zero(raw tap0 = tau 0); crop offset retained & disclosed",
            "crop_offset_range_m": float(sc.crop_offset_range_m()),
            "clean_npz_rel": str(npz_path.name), "truth_json_rel": str(truth_path.name),
            "npz_sha256": sc.sha256_file(npz_path), "truth_sha256": sc.sha256_file(truth_path),
            "cache_key": res.get("cache_key"), "background_mode": res.get("background_mode"),
            "raw_shape": list(raw.shape), "formal": bool(formal), "max_states": max_states,
            "noise_prefix_policy": "noise drawn for L_clean then sliced to L (deterministic prefix)",
            "built_at": sc.now_iso(),
        }
        sc.atomic_write_json(sidecar, meta)
        return dict(meta, reused=False)

    def load_clean(self, cache_dir, clean_id, *, expect_signature=None):
        outdir = self._dir(cache_dir, clean_id)
        sidecar = outdir / SIDECAR
        if not sidecar.is_file():
            raise FdtdSourceError("层 B 缓存缺失：%s" % sidecar)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if expect_signature is not None and str(meta.get("signature")) != str(expect_signature):
            raise FdtdSourceError("clean_id=%s 层 B 签名不匹配：拒绝复用旧 cache" % clean_id)
        npz_path = outdir / str(meta["clean_npz_rel"])
        if not npz_path.is_file():
            raise FdtdSourceError("层 B clean npz 缺失：%s" % npz_path)
        if sc.sha256_file(npz_path) != meta.get("npz_sha256"):
            raise FdtdSourceError("clean_id=%s 层 B npz 哈希与索引不符（被改动）" % clean_id)
        with np.load(npz_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_cir"])
        expect = (int(meta["L_clean"]), sc.N_CHANNELS, sc.RAW_TAPS)
        if tuple(raw.shape) != expect:
            raise FdtdSourceError("层 B raw_cir 形状 %s != %s" % (tuple(raw.shape), expect))
        if not np.all(np.isfinite(raw)):
            raise FdtdSourceError("层 B raw_cir 含 NaN/Inf")
        return {"raw_cir": raw, "meta": meta, "path": str(npz_path)}

    def load_truth(self, cache_dir, clean_id):
        outdir = self._dir(cache_dir, clean_id)
        meta = json.loads((outdir / SIDECAR).read_text(encoding="utf-8"))
        p = outdir / str(meta["truth_json_rel"])
        if not p.is_file():
            raise FdtdSourceError("层 B 真值缺失：%s" % p)
        if sc.sha256_file(p) != meta.get("truth_sha256"):
            raise FdtdSourceError("clean_id=%s 层 B 真值哈希与索引不符（被篡改）" % clean_id)
        return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["FdtdSource", "FdtdSourceError", "SIDECAR"]
