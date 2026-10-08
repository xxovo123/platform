# -*- coding: utf-8 -*-
"""Q4 清单：从本地硬件 meta 整理几何 / 有效录制 manifest（不含任何凭据）。

有效性口径（**不按文件大小发明阈值**）
--------------------------------------
1. **冻结有效 32 录制索引**：以既有冻结产物为准 —— ``04_Q6/hardware/truth/truth_index.json``
   的录制集合（与 ``04_Q6/hardware/cache/cases/*.npz`` 必须一致）。
2. **已知无效**：不在冻结索引里，但录制自身 metadata 给出**事实**（每通道实际帧数 <
   冻结 ring/``max_snapshots``=128），记 ``valid=false`` + 事实 ``invalid_reason``。
3. **未知**：既不在索引、metadata 又不能证实不足 -> ``valid=null`` + ``status='unverified'``，
   **明确标未核**，不伪推“明显截断”。

几何合并纪律
------------
按 ``geometry_id`` 合并时，逐录制核对 **truth 坐标签名** 与 **FDTD 几何/时间参数签名**；
同一 ``geometry_id`` 出现不同签名即 ``Q4Error``（**不静默合并同名坐标变更**）。

清单同时给出 **FDTD 生成所需几何**：``tx_positions`` / ``rx_positions`` / 虚拟 ``pairs`` /
``center_freq`` / ``bandwidth`` / ``tap_interval_si`` / ``cir_offset`` / ``ft_len`` /
``max_snapshots`` / ``snapshot_rate`` / ``fov_degrees`` / ``epsilon_r`` / ``forward_backward`` / ``diag_load`` /
``ant_calib_phase`` / ``imaging_grid_spec``。

用法::

    python -B q4/q4_cli.py inventory --out q4/manifest
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .q4_common import (DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, DEFAULT_TRUTH_INDEX, Q4Error,
                        environment_of, geometry_of, resolve_path, sha256_file, workspace_root,
                        write_json)

DEFAULT_MANIFEST_OUT = "大修执行_20260923/06_仿真实验_20260926/q4/manifest"


def _frozen_valid_index(truth_index, cache_dir) -> tuple:
    """冻结有效录制集合：truth_index 录制 ∩（若在）cache/cases 录制。两者不一致则报错。"""
    truth_index = resolve_path(truth_index, workspace_root())
    if not Path(truth_index).is_file():
        raise Q4Error("冻结 truth_index 缺失：%s" % truth_index)
    doc = json.loads(Path(truth_index).read_text(encoding="utf-8-sig"))
    from_truth = {str(w["recording_id"]) for w in (doc.get("windows") or {}).values()}
    cache_dir = resolve_path(cache_dir, workspace_root())
    from_cache = {p.name[:-len(".npz")] for p in Path(cache_dir).glob("*.npz")} if Path(cache_dir).is_dir() else set()
    if from_cache and from_cache != from_truth:
        raise Q4Error("冻结索引不一致：truth_index=%d 录制 vs cache=%d 录制；差集 truth-only=%s cache-only=%s"
                      % (len(from_truth), len(from_cache), sorted(from_truth - from_cache),
                         sorted(from_cache - from_truth)))
    return from_truth, str(truth_index), sha256_file(truth_index)


def _virtual_pairs(mp: dict) -> list:
    txs = list(mp.get("virt_tx_idx") or [])
    rxs = list(mp.get("virt_rx_idx") or [])
    return [[int(t), int(r)] for t in txs for r in rxs]


def _fdtd_geometry(meta: dict) -> dict:
    mp = ((meta.get("algo_params") or {}).get("2D-MUSIC") or {})
    return {
        "tx_positions": [[float(v) for v in p] for p in (meta.get("tx_positions") or [])],
        "rx_positions": [[float(v) for v in p] for p in (meta.get("rx_positions") or [])],
        "virt_tx_idx": [int(v) for v in (mp.get("virt_tx_idx") or [])],
        "virt_rx_idx": [int(v) for v in (mp.get("virt_rx_idx") or [])],
        "pairs": _virtual_pairs(mp),
        "center_freq_si": float(mp.get("center_freq")) if mp.get("center_freq") is not None else None,
        "bandwidth_si": float(mp.get("bandwidth")) if mp.get("bandwidth") is not None else None,
        "epsilon_r": float(mp.get("epsilon_r")) if mp.get("epsilon_r") is not None else None,
        "tap_interval_si": float(meta.get("tap_interval_si")) if meta.get("tap_interval_si") is not None else None,
        "cir_offset": int(mp.get("cir_offset")) if mp.get("cir_offset") is not None else None,
        "ft_len": int(meta.get("ft_len")) if meta.get("ft_len") is not None else None,
        "max_snapshots": int(meta.get("max_snapshots")) if meta.get("max_snapshots") is not None else None,
        "snapshot_rate_hz": float(meta.get("snapshot_rate")) if meta.get("snapshot_rate") is not None else None,
        "music_snapshots": int(mp.get("music_snapshots")) if mp.get("music_snapshots") is not None else None,
        "forward_backward": bool(mp.get("forward_backward")) if mp.get("forward_backward") is not None else None,
        "diag_load": float(mp.get("diag_load")) if mp.get("diag_load") is not None else None,
        "fov_degrees": float(mp.get("fov_degrees")) if mp.get("fov_degrees") is not None else None,
        "ant_calib_en": bool(mp.get("ant_calib_en")) if mp.get("ant_calib_en") is not None else None,
        "ant_calib_phase": [float(v) for v in (mp.get("ant_calib_phase") or [])],
        "imaging_grid_spec": [float(v) for v in (meta.get("imaging_grid_spec") or [])],
    }


def _signature(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _min_frames(meta: dict):
    counts = ((meta.get("recording") or {}).get("channel_frame_counts") or {})
    vals = [int(v) for v in counts.values()] if counts else []
    return (min(vals) if vals else None), counts


def scan_recordings(data_dir=None, *, truth_index=DEFAULT_TRUTH_INDEX, cache_dir=DEFAULT_CACHE_DIR) -> dict:
    """扫描 meta，按冻结索引判定有效性；返回 (rows, frozen_info)。"""
    root = resolve_path(data_dir or DEFAULT_DATA_DIR, workspace_root())
    if not root.is_dir():
        raise Q4Error("data_dir 不存在：%s" % root)
    frozen, truth_src, truth_sha = _frozen_valid_index(truth_index, cache_dir)
    rows = []
    for meta_path in sorted(root.glob("*.bin.meta")):
        recording_id = meta_path.name[:-len(".bin.meta")]
        bin_path = meta_path.with_name(recording_id + ".bin")
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        collection = meta.get("collection") or {}
        truth = [[float(p[0]), float(p[1])] for p in (collection.get("truth_points") or [])]
        n_targets = int(collection.get("n_targets") or len(truth))
        geom = _fdtd_geometry(meta)
        geom["truth_xy"] = truth
        min_frames, counts = _min_frames(meta)
        ring = geom.get("max_snapshots")

        if recording_id in frozen:
            valid, status, reason = True, "valid", None
        elif min_frames is not None and ring is not None and min_frames < int(ring):
            valid, status = False, "invalid"
            reason = ("insufficient_frames_min%d_lt_ring%d(recording.duration_s=%s)"
                      % (min_frames, int(ring), (meta.get("recording") or {}).get("duration_s")))
        else:
            valid, status = None, "unverified"
            reason = ("not_in_frozen_valid_index_and_metadata_does_not_prove_insufficient_frames"
                      if min_frames is not None else "not_in_frozen_valid_index_and_no_frame_counts")

        rows.append({
            "recording_id": recording_id,
            "environment": collection.get("environment") or environment_of(recording_id),
            "geometry_id": geometry_of(recording_id),
            "config_id": collection.get("config_id"),
            "n_targets": n_targets,
            "truth_xy": truth,
            "valid": valid,
            "status": status,
            "invalid_reason": reason,
            "min_channel_frames": min_frames,
            "channel_frame_counts": counts,
            "bin_bytes": int(bin_path.stat().st_size) if bin_path.is_file() else None,
            "truth_signature": _signature(truth),
            "geometry_signature": _signature(geom),
            "fdtd_geometry": geom,
        })
    if not rows:
        raise Q4Error("未在 %s 找到任何 *.bin.meta" % root)
    return {"rows": rows, "frozen": frozen, "truth_index": truth_src, "truth_index_sha256": truth_sha}


def build_manifest(data_dir=None, *, truth_index=DEFAULT_TRUTH_INDEX, cache_dir=DEFAULT_CACHE_DIR) -> dict:
    scanned = scan_recordings(data_dir, truth_index=truth_index, cache_dir=cache_dir)
    rows = scanned["rows"]
    valid = [r for r in rows if r["valid"] is True]
    invalid = [r for r in rows if r["valid"] is False]
    unverified = [r for r in rows if r["valid"] is None]
    if {r["recording_id"] for r in valid} != scanned["frozen"]:
        raise Q4Error("有效录制集合与冻结索引不一致：有效=%d 冻结=%d"
                      % (len(valid), len(scanned["frozen"])))

    # 几何合并：核对 truth 签名 + FDTD 几何签名；不一致显式报错
    geometries = {}
    for r in rows:
        g = geometries.setdefault(r["geometry_id"], {
            "geometry_id": r["geometry_id"], "n_targets": r["n_targets"],
            "truth_xy": r["truth_xy"], "fdtd_geometry": r["fdtd_geometry"],
            "truth_signatures": set(), "geometry_signatures": set(),
            "recordings": [], "n_valid_recordings": 0, "environments": [],
        })
        g["truth_signatures"].add(r["truth_signature"])
        g["geometry_signatures"].add(r["geometry_signature"])
        g["recordings"].append({"recording_id": r["recording_id"], "environment": r["environment"],
                                "valid": r["valid"], "status": r["status"],
                                "invalid_reason": r["invalid_reason"]})
        if r["valid"] is True:
            g["n_valid_recordings"] += 1
        if r["environment"] not in g["environments"]:
            g["environments"].append(r["environment"])
    conflicts = []
    for gid, g in geometries.items():
        if len(g["truth_signatures"]) > 1 or len(g["geometry_signatures"]) > 1:
            conflicts.append({"geometry_id": gid, "n_truth_signatures": len(g["truth_signatures"]),
                              "n_geometry_signatures": len(g["geometry_signatures"]),
                              "recordings": [r["recording_id"] for r in g["recordings"]]})
        g["truth_signatures"] = sorted(g["truth_signatures"])
        g["geometry_signatures"] = sorted(g["geometry_signatures"])
    if conflicts:
        raise Q4Error("同名 geometry_id 出现不一致的 truth/几何签名，拒绝静默合并：%s"
                      % json.dumps(conflicts, ensure_ascii=False))

    by_k = {}
    for r in valid:
        by_k[str(r["n_targets"])] = by_k.get(str(r["n_targets"]), 0) + 1
    return {
        "kind": "q4-inventory",
        "schema_version": 2,
        "source": "<workspace>/%s" % (data_dir or DEFAULT_DATA_DIR),
        "validity_basis": {
            "frozen_valid_index": "%s (sha256=%s)" % (scanned["truth_index"], scanned["truth_index_sha256"]),
            "rule": "valid = in frozen valid index; invalid = metadata proves insufficient frames (< ring); else unverified",
            "size_threshold_used": False,
        },
        "n_recordings_total": len(rows),
        "n_recordings_valid": len(valid),
        "n_recordings_invalid": len(invalid),
        "n_recordings_unverified": len(unverified),
        "n_geometries": len(geometries),
        "valid_recordings_by_k": by_k,
        "recordings": [r for r in rows],
        "geometries": [geometries[g] for g in sorted(geometries)],
        "excluded_invalid": [{"recording_id": r["recording_id"], "reason": r["invalid_reason"]} for r in invalid],
        "unverified": [{"recording_id": r["recording_id"], "reason": r["invalid_reason"]} for r in unverified],
        "notes": [
            "双环境 / 重复录制保留（indoor/outdoor 成对；G-410 与 C3-1 各有 _002）",
            "有效性以冻结索引为准，不按文件大小推测；未知录制显式 unverified",
            "几何块含 FDTD 生成所需 tx/rx/pairs/时间参数，不只 truth",
            "本清单不含账号/口令/认证信息，可由脚本重新生成",
        ],
    }


def write_manifest(manifest: dict, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "q4_inventory.json", manifest)
    with (out / "q4_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "environment", "geometry_id", "config_id", "n_targets",
                    "truth_xy", "valid", "status", "invalid_reason", "min_channel_frames",
                    "bin_bytes", "center_freq_si", "bandwidth_si", "cir_offset", "ft_len",
                    "snapshot_rate_hz", "pairs"])
        for r in manifest["recordings"]:
            g = r["fdtd_geometry"]
            w.writerow([r["recording_id"], r["environment"], r["geometry_id"], r["config_id"],
                        r["n_targets"], r["truth_xy"], r["valid"], r["status"],
                        r["invalid_reason"] or "", r["min_channel_frames"], r["bin_bytes"],
                        g.get("center_freq_si"), g.get("bandwidth_si"), g.get("cir_offset"),
                        g.get("ft_len"), g.get("snapshot_rate_hz"), g.get("pairs")])
    # 单独一份 FDTD 几何清单（仅有效几何）
    fdtd = {g["geometry_id"]: g["fdtd_geometry"] for g in manifest["geometries"] if g["n_valid_recordings"] > 0}
    write_json(out / "q4_fdtd_geometries.json", {
        "kind": "q4-fdtd-geometry-index", "schema_version": 1,
        "note": "仅含有效几何；供 FDTD 生成使用（tx/rx/pairs/时间参数/truth）",
        "geometries": fdtd})
    return {"json": str(out / "q4_inventory.json"), "csv": str(out / "q4_inventory.csv"),
            "fdtd_geometries": str(out / "q4_fdtd_geometries.json")}
