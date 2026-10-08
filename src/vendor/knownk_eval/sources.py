# -*- coding: utf-8 -*-
"""输入装载（只读）：硬件缓存谱 npz、P0a 底层真值表、历史逐窗预测。

本模块只读 Q2/Q3 历史评估产物，不写回、不改动。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .evaluate import WindowInput

BRANCH_MODE = "oracle"
BRANCH_POSTPROCESS = "nms"


def read_csv_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def branch(rows, mode=BRANCH_MODE, postprocess=BRANCH_POSTPROCESS) -> list[dict]:
    return [r for r in rows if r.get("mode") == mode and r.get("postprocess") == postprocess]


def load_spectra(path):
    """返回 (grid_x, grid_y, {array_key: image})。"""
    with np.load(Path(path), allow_pickle=True) as z:
        gx = np.asarray(z["gx"], dtype=float)
        gy = np.asarray(z["gy"], dtype=float)
        images = {k: np.asarray(z[k], dtype=float) for k in z.files if k.endswith("__image")}
    return gx, gy, images


def load_per_window_branch(data_dir) -> dict:
    """per_window.csv 的 oracle+nms 分支，按 window_id 索引。"""
    rows = branch(read_csv_rows(Path(data_dir) / "per_window.csv"))
    out = {}
    for row in rows:
        out[row["window_id"]] = {
            "recording_id": row["recording_id"],
            "window_id": row["window_id"],
            "t_end_s": float(row["t_end_s"]),
            "environment": row["environment"],
            "config_id": row["config_id"],
            "n_truth": int(row["n_truth"]),
            "k_hat": int(row["k_hat"]),
            "tp": int(row["tp"]),
            "fn": int(row["fn"]),
            "fp": int(row["fp"]),
            "n_predictions": int(row["n_predictions"]),
            "truth_points": json.loads(row["truth_points"]),
            "predictions": json.loads(row["predictions"]),
            "spectrum_valid": row["spectrum_valid"],
            "vmax": float(row["vmax"]) if row["vmax"] not in ("", None) else None,
        }
    return out


def load_recorded_predictions(data_dir) -> dict:
    """outcome_predictions.csv 的 oracle+nms 分支，按 window_id 索引（用于取峰逐项核对）。"""
    rows = branch(read_csv_rows(Path(data_dir) / "outcome_predictions.csv"))
    out = {}
    for row in rows:
        out.setdefault(row["window_id"], []).append({
            "pred_index": int(row["pred_index"]),
            "outcome": row["outcome"],
            "x": float(row["pred_x"]),
            "y": float(row["pred_y"]),
            "value": float(row["pred_value"]) if row["pred_value"] not in ("", None) else None,
        })
    for key in out:
        out[key].sort(key=lambda item: item["pred_index"])
    return out


def load_outcome_targets(data_dir) -> dict:
    """outcome_targets.csv 的 oracle+nms 分支（每窗每个真值的逐项结果），按 window_id 索引。"""
    rows = branch(read_csv_rows(Path(data_dir) / "outcome_targets.csv"))
    out = {}
    for row in rows:
        out.setdefault(row["window_id"], []).append({
            "truth_index": int(row["truth_index"]),
            "outcome": row["outcome"],
            "pred_index": int(row["pred_index"]) if row["pred_index"] not in ("", None) else None,
            "truth_x": float(row["truth_x"]),
            "truth_y": float(row["truth_y"]),
            "e_r_m": float(row["e_r_m"]) if row["e_r_m"] not in ("", None) else None,
            "e_theta_deg": float(row["e_theta_deg"]) if row["e_theta_deg"] not in ("", None) else None,
            "e_xy_m": float(row["e_xy_m"]) if row["e_xy_m"] not in ("", None) else None,
        })
    for key in out:
        out[key].sort(key=lambda item: item["truth_index"])
    return out


def cached_window_keys(spectra_dir) -> dict:
    """扫描 spectra/*.npz，返回 {recording_id: [array_key, ...]}（仅 oracle 谱）。"""
    out = {}
    for path in sorted(Path(spectra_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            out[path.stem] = [k for k in z.files if k.endswith("__oracle__image")]
    return out


def iter_cached_windows(spectra_dir, data_dir, *, windows=None, recording_ids=None):
    """构造 WindowInput 列表：缓存谱 + 网格 + 给定 K + 真值（全部来自历史只读产物）。"""
    truth_index = load_per_window_branch(data_dir)
    want = None if windows is None else set(windows)
    rec_want = None if recording_ids is None else set(recording_ids)
    produced = 0
    for path in sorted(Path(spectra_dir).glob("*.npz")):
        recording_id = path.stem
        if rec_want is not None and recording_id not in rec_want:
            continue
        gx, gy, images = load_spectra(path)
        for key in sorted(images):
            if not key.endswith("__oracle__image") or "__mdl__" in key:
                continue
            window_id = key.replace("__oracle__image", "")
            if want is not None and window_id not in want:
                continue
            info = truth_index.get(window_id)
            if info is None:
                continue
            yield WindowInput(
                window_id=window_id,
                recording_id=recording_id,
                environment=info["environment"],
                config_id=info["config_id"],
                n_truth=int(info["n_truth"]),
                k=int(info["k_hat"]),
                truth_xy=[tuple(p) for p in info["truth_points"]],
                input_kind="spectrum",
                image=images[key],
                grid_x=gx,
                grid_y=gy,
                source="cached_spectra_oracle",
                notes=["谱来自历史缓存 npz（oracle 分支），网格为同一文件内 gx/gy"],
            )
            produced += 1
            if want is not None and produced >= len(want):
                return
