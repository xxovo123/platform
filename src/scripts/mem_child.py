#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E6 峰值内存子进程：**隔离进程**单次跑一个方法，报告 RSS 峰值（统一口径）。

用法（由 run_experiments 调用）::
    python -B scripts/mem_child.py <case_npz> <case_meta_json> <method> <params_json>
输出：单行 JSON {method, status, peak_rss_bytes, peak_source, seconds}。
peak_source: "resource.ru_maxrss"(Linux/macOS) / "psutil" / "unavailable_on_platform"。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import sim_common as sc  # noqa: E402  先于 numpy：线程 env 先固定

import numpy as np  # noqa: E402


def _peak_rss_bytes():
    from adapter import warmcold
    return warmcold.rss_measurement()


def main(argv) -> int:
    case_npz, meta_json, method = argv[0], argv[1], argv[2]
    mode = str(argv[3]) if len(argv) > 3 and argv[3] else "cold"
    params = json.loads(argv[4]) if len(argv) > 4 and argv[4] else None
    from adapter import case_build, methods_loader
    with np.load(case_npz, allow_pickle=False) as z:
        X = np.asarray(z["X"])
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    gx = np.asarray(meta["grid_x"], dtype=float)
    gy = np.asarray(meta["grid_y"], dtype=float)
    case = case_build.build_case(case_id=meta["case_id"], clean_id=meta["clean_id"], X=X,
                                 k=int(meta["k"]), grid_x=gx, grid_y=gy, layer="A",
                                 source="analytic", scene=meta.get("scene", ""),
                                 tags=[], snr_db=meta.get("snr_db"), trial=0, seed=0,
                                 L=int(meta["L"]))
    t0 = time.perf_counter()
    if mode == "warm":
        from adapter import warmcold
        support = warmcold.warm_support(method)
        if not support["supported"]:
            seconds = time.perf_counter() - t0
            print(json.dumps({"method": method, "mode": mode, "status": "not_applicable",
                              "reason": support["reason"], "peak_rss_bytes": None,
                              "peak_source": "not_applicable", "seconds": seconds},
                             ensure_ascii=False))
            return 0
        manifold = warmcold.prepare(method, case)
        res = warmcold.solve_warm(method, case, manifold)
    else:
        res = methods_loader.call(method, case, params)
    seconds = time.perf_counter() - t0
    rss = _peak_rss_bytes()
    record = {"method": method, "mode": mode, "status": res["status"],
              "rss_bytes": rss["rss_bytes"], "rss_kind": rss["rss_kind"],
              "rss_source": rss["rss_source"], "seconds": seconds,
              "n_predictions": len(res["predictions"]), "platform": sys.platform,
              "python": sys.version.split()[0],
              "measurement_scope": ("whole_child_warm_including_prepare_and_one_solve"
                                    if mode == "warm" else
                                    "whole_child_cold_including_import_and_one_call"),
              "note": ("rss_kind=peak 才是真峰值；current_not_peak 只是当前 RSS，不得冒称 peak。")}
    if rss["rss_kind"] == "peak":
        record["peak_rss_bytes"] = rss["rss_bytes"]
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
