# -*- coding: utf-8 -*-
"""Q4 可移植硬件 CIR 包导出：``hardware cropped30`` 四通道复数 CIR（+ 可选 raw32）。

两条来源（**同一套时间中点规则，不按相关性挑快照**）：

1. ``--source cache``（默认）：读既有 ``04_Q6/hardware/cache/cases/*.npz``。
   该缓存由 ``b0.replay_windows`` **每录制一次扫描**生成，背景 EMA 跨窗连续，
   且已按 ``cropped = arr[:, :, cir_offset:ft_len]`` 截成 30 tap。
2. ``--source bin``：用 ``b0.cfr_replay.replay_windows`` 对 ``.bin`` 做**单次扫描**重放，
   背景 EMA 跨窗保持（**不每窗重置**），同时导出 ``cir_cropped30`` 与 ``cir_raw32``。
   需要 ``--b0-dir`` 指向含 ``b0`` 包的目录；缺依赖时显式报错。

输出（每录制一个 npz + 包级 manifest）::

    q4/out/hw_cir/<recording_id>.npz
    q4/out/hw_cir/manifest.json

通道选择复用硬件 4 虚拟通道对 ``(0,2)(0,3)(1,2)(1,3)``（``select_virtual_channels``）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .q4_common import (CROPPED_TAPS, DEFAULT_CACHE_DIR, DEFAULT_CIR_OFFSET,
                        DEFAULT_FT_LEN, DEFAULT_TRUTH_INDEX, Q4Error, environment_of,
                        geometry_of, load_json, mid_index, resolve_path,
                        select_virtual_channels, sha256_file, workspace_root, write_json)

DEFAULT_B0_DIR = "大修执行_20260923/04_Q6/B0_同模型校验"


def _cache_npz(cache_dir, recording_id) -> Path:
    p = Path(cache_dir) / (recording_id + ".npz")
    if not p.is_file():
        raise Q4Error("缓存缺失：%s" % p)
    return p


def list_cache_recordings(cache_dir, recordings=None) -> list:
    d = Path(cache_dir)
    if not d.is_dir():
        raise Q4Error("cache_dir 不存在：%s" % d)
    ids = sorted(p.name[:-len(".npz")] for p in d.glob("*.npz"))
    if recordings:
        want = set(recordings)
        ids = [r for r in ids if r in want]
        missing = want - set(ids)
        if missing:
            raise Q4Error("请求的录制不在缓存中：%s" % sorted(missing))
    return ids


def _pack_windows(window_records: list, meta_common: dict, source: str) -> dict:
    """把逐窗记录整理成 npz payload（不产生 object dtype；可 allow_pickle=False 读取）。

    L 不一致时退化为逐窗键。``window_ids`` / ``truth_xy_json`` 用 **unicode dtype**（安全）。
    """
    payload = dict(meta_common)
    payload["source"] = source
    payload["window_ids"] = np.array([w["window_id"] for w in window_records], dtype="U")
    payload["t_end_s"] = np.array([w["t_end_s"] for w in window_records], dtype=float)
    payload["k"] = np.array([w["k"] for w in window_records], dtype=np.int64)
    payload["truth_xy_json"] = np.array([json.dumps(w["truth_xy"]) for w in window_records], dtype="U")
    payload["mid_index"] = np.array([w["mid_index"] for w in window_records], dtype=np.int64)
    payload["L_list"] = np.array([w["cir30"].shape[0] for w in window_records], dtype=np.int64)
    uniform = len(set(int(v) for v in payload["L_list"])) == 1
    payload["stacked_uniform_L"] = bool(uniform)
    if uniform:
        payload["cir_cropped30"] = np.stack([w["cir30"].astype(np.complex64) for w in window_records], axis=0)
        if all(w.get("cir32") is not None for w in window_records):
            payload["cir_raw32"] = np.stack([w["cir32"].astype(np.complex64) for w in window_records], axis=0)
    else:
        for w in window_records:
            payload["cir_cropped30__%s" % w["window_id"]] = w["cir30"].astype(np.complex64)
            if w.get("cir32") is not None:
                payload["cir_raw32__%s" % w["window_id"]] = w["cir32"].astype(np.complex64)
    return payload


def _write_truth_json(out_dir, recording_id, window_records) -> Path:
    """真值单列 json（避免把 truth 塞进 npz；新包可纯 allow_pickle=False 读取）。"""
    doc = {
        "recording_id": recording_id,
        "note": "真值独立存放；window_id -> {k, truth_xy, t_end_s, mid_index}",
        "windows": {w["window_id"]: {"k": int(w["k"]), "truth_xy": w["truth_xy"],
                                     "t_end_s": float(w["t_end_s"]),
                                     "mid_index": int(w["mid_index"])}
                    for w in window_records},
    }
    path = Path(out_dir) / (recording_id + ".truth.json")
    write_json(path, doc)
    return path


def _guard_overwrite(out_dir, source, recording_ids, force):
    """避免不同 source 静默覆盖（同 source 覆盖仍允许）。"""
    man = Path(out_dir) / "manifest.json"
    if man.is_file():
        try:
            prev = load_json(man)
        except Exception:
            prev = {}
        if prev.get("source") and prev["source"] != source and not force:
            raise Q4Error("输出目录已有不同 source 的包（existing=%s, new=%s）；如确要覆盖请加 --force"
                          % (prev["source"], source))
    for rid in recording_ids:
        p = Path(out_dir) / (rid + ".npz")
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            prev = str(z["source"]) if "source" in z.files else ""
        if prev and prev != source and not force:
            raise Q4Error("录制包 %s 由不同 source 生成（%s != %s）；如确要覆盖请加 --force"
                          % (p.name, prev, source))


def _truth_from_npz(z, window_id) -> list:
    key = "truth_xy__%s" % window_id
    if key in z.files:
        return [[float(v) for v in p] for p in z[key]]
    return None


def _common_meta(z) -> dict:
    return {
        "pairs": np.asarray(z["pairs"], dtype=np.int64),
        "tx_positions": np.asarray(z["tx_positions"], dtype=float),
        "rx_positions": np.asarray(z["rx_positions"], dtype=float),
        "frequencies_hz": np.asarray(z["frequencies_hz"], dtype=float),
        "cir_offset": np.int64(DEFAULT_CIR_OFFSET),
        "ft_len": np.int64(DEFAULT_FT_LEN),
        "cropped_taps": np.int64(CROPPED_TAPS),
    }


def export_from_cache(cache_dir, out_dir, *, truth_index=DEFAULT_TRUTH_INDEX, recordings=None,
                      force=False) -> dict:
    cache_dir = resolve_path(cache_dir, workspace_root())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_index = resolve_path(truth_index, workspace_root()) if truth_index else None
    truth_windows = None
    if truth_index and Path(truth_index).is_file():
        truth_windows = (load_json(truth_index).get("windows") or {})
    ids = list_cache_recordings(cache_dir, recordings)
    source = "cache:%s" % cache_dir
    _guard_overwrite(out_dir, source, ids, force)
    manifest = []
    for rid in ids:
        # 注意：冻结 Q6 缓存的 window_ids 是 object dtype，**读取源**必须 allow_pickle=True；
        # 本函数写出的新包本身是 pickle-free（window_ids/truth 用 unicode + 独立 truth.json）。
        with np.load(_cache_npz(cache_dir, rid), allow_pickle=True) as z:
            common = _common_meta(z)
            wids = [str(w) for w in z["window_ids"]]
            recs = []
            for wid in wids:
                snap = z["snapshots__%s" % wid]            # (n_rx, n_tx, 30, L)
                pairs = [(int(p[0]), int(p[1])) for p in np.asarray(z["pairs"])]
                cir30 = select_virtual_channels(snap, pairs)  # (L, 4, 30)
                truth = _truth_from_npz(z, wid)
                if truth is None and truth_windows and wid in truth_windows:
                    truth = [[float(v) for v in p] for p in truth_windows[wid]["truth_xy"]]
                if truth is None:
                    raise Q4Error("%s/%s 缺 truth；请提供 --truth-index" % (rid, wid))
                recs.append({
                    "window_id": wid,
                    "t_end_s": float(z["t_end_s"][wids.index(wid)]),
                    "k": int(z["k"][wids.index(wid)]),
                    "truth_xy": truth,
                    "cir30": cir30,
                    "cir32": None,
                    "mid_index": mid_index(cir30.shape[0]),
                })
            common["environment"] = environment_of(rid)
            common["geometry_id"] = geometry_of(rid)
            common["recording_id"] = rid
            payload = _pack_windows(recs, common, source)
        np.savez_compressed(out_dir / (rid + ".npz"), **payload)
        truth_path = _write_truth_json(out_dir, rid, recs)
        manifest.append({"recording_id": rid, "environment": payload["environment"],
                         "geometry_id": payload["geometry_id"], "n_windows": len(recs),
                         "n_channels": int(payload["pairs"].shape[0]), "taps": CROPPED_TAPS,
                         "L_uniform": bool(payload["stacked_uniform_L"]),
                         "truth_json": truth_path.name,
                         "sha256": sha256_file(out_dir / (rid + ".npz"))})
    doc = {
        "kind": "q4-hw-cir-package",
        "schema_version": 1,
        "source": source,
        "cir_convention": "cropped30 = arr[:, :, cir_offset:ft_len]; background EMA kept across windows by single-scan replay",
        "channel_rule": "4 virtual pairs (0,2)(0,3)(1,2)(1,3)",
        "mid_rule": "fixed L//2 snapshot (no oracle shift, no correlation-based snapshot pick)",
        "n_recordings": len(manifest),
        "recordings": manifest,
        "pickle_free": True,
        "no_credentials": True,
    }
    write_json(out_dir / "manifest.json", doc)
    return doc


def _load_b0(b0_dir):
    b0_dir = resolve_path(b0_dir, workspace_root())
    if not (Path(b0_dir) / "b0" / "cfr_replay.py").is_file():
        raise Q4Error("--source bin 需要 b0 包（cfr_replay.py）；未在 %s 找到" % b0_dir)
    if str(b0_dir) not in sys.path:
        sys.path.insert(0, str(b0_dir))
    import b0  # noqa: WPS433
    return b0


def export_from_bin(data_dir, out_dir, *, b0_dir=DEFAULT_B0_DIR, cache_dir=DEFAULT_CACHE_DIR,
                    truth_index=DEFAULT_TRUTH_INDEX, recordings=None, ring=128, force=False) -> dict:
    """单次扫描重放 .bin -> raw32 + cropped30；背景 EMA 跨窗保持。"""
    b0 = _load_b0(b0_dir)
    data_dir = resolve_path(data_dir, workspace_root())
    cache_dir = resolve_path(cache_dir, workspace_root())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_index = resolve_path(truth_index, workspace_root()) if truth_index else None
    truth_windows = load_json(truth_index).get("windows") if (truth_index and Path(truth_index).is_file()) else {}
    ids = list_cache_recordings(cache_dir, recordings)
    source = "bin_replay:%s" % data_dir
    _guard_overwrite(out_dir, source, ids, force)
    manifest = []
    for rid in ids:
        # 窗口清单取自缓存（tap/通道口径一致），t 值用于重放
        with np.load(_cache_npz(cache_dir, rid), allow_pickle=True) as z:
            wids = [str(w) for w in z["window_ids"]]
            t_list = [float(v) for v in z["t_end_s"]]
            pairs = [(int(p[0]), int(p[1])) for p in np.asarray(z["pairs"])]
            common = _common_meta(z)
        bin_path = data_dir / (rid + ".bin")
        if not bin_path.is_file():
            raise Q4Error("bin 缺失：%s" % bin_path)
        meta = b0.load_meta(bin_path)
        captures = b0.replay_windows(bin_path, meta, t_list, ring=ring)  # 单次扫描，EMA 跨窗连续
        recs = []
        for wid, t in zip(wids, t_list):
            cap = captures.get(float(t))
            if cap is None or cap.get("unavailable"):
                raise Q4Error("%s/%s 重放不可用" % (rid, wid))
            arr = cap["arr"]                                        # (n_rx, n_tx, ft_len, ring)
            cir32 = select_virtual_channels(arr[:, :, :DEFAULT_FT_LEN, :], pairs)  # (L,4,32)
            cir30 = cir32[:, :, DEFAULT_CIR_OFFSET:DEFAULT_CIR_OFFSET + CROPPED_TAPS]
            truth = None
            if wids and truth_windows and wid in truth_windows:
                truth = [[float(v) for v in p] for p in truth_windows[wid]["truth_xy"]]
            if truth is None:
                raise Q4Error("%s/%s 缺 truth；请提供 --truth-index" % (rid, wid))
            recs.append({"window_id": wid, "t_end_s": float(t), "k": len(truth), "truth_xy": truth,
                         "cir30": cir30, "cir32": cir32, "mid_index": mid_index(cir30.shape[0])})
        common["environment"] = environment_of(rid)
        common["geometry_id"] = geometry_of(rid)
        common["recording_id"] = rid
        payload = _pack_windows(recs, common, source)
        np.savez_compressed(out_dir / (rid + ".npz"), **payload)
        truth_path = _write_truth_json(out_dir, rid, recs)
        manifest.append({"recording_id": rid, "n_windows": len(recs),
                         "truth_json": truth_path.name,
                         "sha256": sha256_file(out_dir / (rid + ".npz"))})
    doc = {
        "kind": "q4-hw-cir-package",
        "schema_version": 1,
        "source": source,
        "cir_convention": "raw32 = arr[:,:,:32]; cropped30 = raw32[..., 2:32]; background EMA (alpha=1/bg_m_factor) kept across windows in one scan per recording",
        "channel_rule": "4 virtual pairs (0,2)(0,3)(1,2)(1,3)",
        "mid_rule": "fixed L//2 snapshot",
        "n_recordings": len(manifest),
        "recordings": manifest,
        "pickle_free": True,
        "no_credentials": True,
    }
    write_json(out_dir / "manifest.json", doc)
    return doc
