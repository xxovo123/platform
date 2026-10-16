# -*- coding: utf-8 -*-
"""坐标约定、误差与分组聚合（独立实现）。

坐标：雷达相位中心为原点，视轴 = -Y，theta > 0 为 +X 侧；
      r = hypot(x, y), theta_deg = degrees(atan2(x, -y))；角度差 wrap 到 (-180, 180]。
聚合：分子分母分别累加；误差只条件于匹配 TP；描述性统计，不作 CI / 显著性推断。
"""

from __future__ import annotations

import math

import numpy as np


class DuplicateWindowError(ValueError):
    """同一批结果里出现重复 window_id：必须显式报错，禁止静默去重/重复计数。"""


def wrap_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if value <= -180.0 + 1e-12:
        value = 180.0
    return value


def polar_of(x: float, y: float):
    """论文坐标系：视轴 = -Y，theta > 0 为 +X 侧。"""
    return (math.hypot(float(x), float(y)), math.degrees(math.atan2(float(x), -float(y))))


def error_stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "median": None, "p95": None, "std": None}
    return {
        "n": int(arr.size),
        "mae": float(np.mean(arr)),
        "rmse": float(math.sqrt(float(np.mean(arr ** 2)))),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


GROUP_DIMS = ("overall", "environment", "n_truth", "environment_n", "config")


def as_record_dict(record) -> dict:
    """接受 dict 或带 to_dict() 的结果对象（WindowResult）。"""
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return record


def group_label(record: dict, dim: str) -> str:
    record = as_record_dict(record)
    env = record.get("environment") or ""
    n = int(record.get("n_truth") or 0)
    if dim == "overall":
        return "overall"
    if dim == "environment":
        return env
    if dim == "n_truth":
        return "N=%d" % n
    if dim == "environment_n":
        return "%s N=%d" % (env, n)
    if dim == "config":
        return "%s|%s|N=%d" % (env, record.get("config_id") or "", n)
    raise ValueError("未知分组维度：%s" % dim)


def sort_group_labels(labels, dim: str):
    if dim == "overall":
        return ["overall"]
    if dim == "environment":
        return sorted(labels)
    if dim == "n_truth":
        return sorted(labels, key=lambda s: int(s.split("=")[1]))
    order_env = {"indoor": 0, "outdoor": 1}
    if dim == "environment_n":
        return sorted(labels, key=lambda s: (order_env.get(s.split(" ")[0], 9),
                                             int(s.split("=")[1])))
    return sorted(labels, key=lambda s: (order_env.get(s.split("|")[0], 9),
                                         s.split("|")[1], int(s.split("=")[-1])))


def group_stats(records) -> dict:
    """按“分子分母分别累加”的口径汇总一组窗记录。"""
    records = [as_record_dict(r) for r in records]
    if not records:
        stats = {
            "n_windows": 0, "n_recordings": 0, "n_windows_failed": 0,
            "n_truth_total": 0, "n_predictions_total": 0, "tp": 0, "fn": 0, "fp": 0,
            "dr": None, "miss_rate": None, "fp_per_estimate": None,
            "n_windows_with_fp": 0, "p_window_with_fp": None,
        }
        for name in ("e_r_m", "e_theta_deg", "e_xy_m"):
            stats[name] = error_stats([])
        return stats
    windows = {r["window_id"] for r in records}
    if len(windows) != len(records):
        seen = {}
        for r in records:
            seen.setdefault(r["window_id"], 0)
            seen[r["window_id"]] += 1
        dup = sorted(k for k, v in seen.items() if v > 1)
        raise DuplicateWindowError(
            "同一聚合组内出现重复 window_id（会同时污染分子与窗数分母）：%s"
            % ", ".join(dup[:10]))
    recordings = {r["recording_id"] for r in records}
    tp = sum(int(r["tp"]) for r in records)
    fn = sum(int(r["fn"]) for r in records)
    fp = sum(int(r["fp"]) for r in records)
    n_pred = sum(int(r["n_predictions"]) for r in records)
    n_windows_with_fp = sum(1 for r in records if int(r["fp"]) >= 1)
    n_failed = sum(1 for r in records if r.get("status") != "ok")
    n_truth_total = tp + fn
    n_win = len(windows)
    stats = {
        "n_windows": n_win,
        "n_recordings": len(recordings),
        "n_windows_failed": n_failed,
        "n_truth_total": n_truth_total,
        "n_predictions_total": n_pred,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "dr": (tp / n_truth_total) if n_truth_total else None,
        "miss_rate": (fn / n_truth_total) if n_truth_total else None,
        "fp_per_estimate": (fp / n_win) if n_win else None,
        "n_windows_with_fp": n_windows_with_fp,
        "p_window_with_fp": (n_windows_with_fp / n_win) if n_win else None,
    }
    tp_errors = [row for r in records for row in r.get("errors", [])]
    for name, key in (("e_r_m", "e_r"), ("e_theta_deg", "e_theta"), ("e_xy_m", "e_xy")):
        stats[name] = error_stats([row.get(key) for row in tp_errors])
    return stats


STATS_COLUMNS = (
    ["group_dim", "group", "n_windows", "n_recordings", "n_windows_failed", "n_truth_total",
     "n_predictions_total", "tp", "fn", "fp", "dr", "miss_rate", "fp_per_estimate",
     "n_windows_with_fp", "p_window_with_fp"]
    + ["%s_%s" % (name, stat)
       for name in ("e_r_m", "e_theta_deg", "e_xy_m")
       for stat in ("n", "mae", "rmse", "median", "p95", "std")]
)


def flatten_stats(label: str, dim: str, stats: dict) -> dict:
    row = {"group_dim": dim, "group": label}
    for key in ("n_windows", "n_recordings", "n_windows_failed", "n_truth_total",
                "n_predictions_total", "tp", "fn", "fp", "dr", "miss_rate",
                "fp_per_estimate", "n_windows_with_fp", "p_window_with_fp"):
        row[key] = stats[key]
    for name in ("e_r_m", "e_theta_deg", "e_xy_m"):
        for stat in ("n", "mae", "rmse", "median", "p95", "std"):
            row["%s_%s" % (name, stat)] = stats[name][stat]
    return row


def check_unique_windows(records) -> None:
    """整批唯一性检查：重复 window_id 直接报错（不静默去重）。"""
    records = [as_record_dict(r) for r in records]
    seen = {}
    for record in records:
        wid = record.get("window_id")
        seen[wid] = seen.get(wid, 0) + 1
    dup = sorted(str(k) for k, v in seen.items() if v > 1)
    if dup:
        raise DuplicateWindowError(
            "评估结果含重复 window_id（同一窗被算了两次，p_window_with_fp 可能 > 1）：%s"
            % ", ".join(dup[:10]))


def aggregate(records, *, require_unique_windows: bool = True) -> list[dict]:
    """返回所有分组维度的统计行（overall / environment / n_truth / environment_n / config）。

    空输入时仍返回一行 overall（全 0，比率为 None），避免调用方 KeyError。
    """
    records = [as_record_dict(r) for r in records]
    if require_unique_windows:
        check_unique_windows(records)
    rows = []
    for dim in GROUP_DIMS:
        buckets: dict[str, list] = {}
        for record in records:
            buckets.setdefault(group_label(record, dim), []).append(record)
        if not buckets:
            if dim != "overall":
                continue
            buckets = {"overall": []}
        for label in sort_group_labels(list(buckets), dim):
            rows.append(flatten_stats(label, dim, group_stats(buckets[label])))
    return rows
