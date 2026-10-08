# -*- coding: utf-8 -*-
"""Q4 评估：clean_cir[L,4,32] -> adapter.frontend -> adapter.case_build(k=manifest, geometry override)
-> 五法 call（``adapter.methods_loader`` + ``params_for(PROTOCOL)``）-> ``geom/<method>.pred.json`` + nominal truth。

**显式依赖**（不猜 module）：``from adapter import case_build, frontend, methods_loader``；
``from protocol.sim_protocol import PROTOCOL``。

几何：Q4 用硬件物理 tx/rx/pairs（inventory ``fdtd_geometry``），**不套 main 1Tx**；
``build_case(..., geometry=...)`` 的新接口做覆盖，config（含 FB）保持不动。

本轮定位：脚本 + fixture；**不跑 15 FDTD**。可用 tiny 合成 raw 送**实际五法**验证一次。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import sim_common as sc  # noqa: E402  （统一网格/协议唯一事实来源）

from .q4_batch import DEFAULT_INVENTORY, geometry_index  # noqa: E402
from .q4_common import Q4Error, geometry_of, resolve_path, workspace_root, write_json  # noqa: E402
from .q4_scenes import q4_geometry  # noqa: E402

MAIN_METHODS = ("this_method", "b1_deng", "b2_kakouche", "c1_jade", "c3_kim")


def _deps():
    """显式导入真实依赖（缺即报错，不做 module 泛化猜测）。"""
    from adapter import case_build, frontend, methods_loader
    from protocol.sim_protocol import PROTOCOL
    return case_build, frontend, methods_loader, PROTOCOL


def deps_ready() -> dict:
    try:
        case_build, frontend, methods_loader, protocol = _deps()
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"ready": True, "case_build": case_build.__name__, "frontend": frontend.__name__,
            "methods_loader": methods_loader.__name__, "protocol_version": protocol.get("version")}


def select_channels(raw: np.ndarray, pairs) -> np.ndarray:
    """``[L, n_tx, n_rx, T]`` -> ``[L, 4, T]``（按 Q4 物理 pairs 取）；已是 [L,4,T] 则原样返回。"""
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[1] == 4:
        return raw
    if raw.ndim != 4:
        raise Q4Error("clean raw 需 [L,n_tx,n_rx,T] 或 [L,4,T]，收到 shape=%s" % (raw.shape,))
    out = np.empty((raw.shape[0], len(pairs), raw.shape[-1]), dtype=raw.dtype)
    for i, (tx, rx) in enumerate(pairs):
        out[:, i, :] = raw[:, int(tx), int(rx), :]
    return out


def assert_pairs_match(tx_index, rx_index, pairs) -> list:
    """clean 的 tx_index/rx_index 必须与 manifest pairs **一致**（不只是记录）。"""
    if tx_index is None or rx_index is None:
        raise Q4Error("clean_cir.npz 缺 tx_index/rx_index，无法与 manifest pairs 校验")
    got = [[int(a), int(b)] for a, b in zip(np.asarray(tx_index).ravel(), np.asarray(rx_index).ravel())]
    want = [[int(a), int(b)] for a, b in pairs]
    if got != want:
        raise Q4Error("clean pairs %s != manifest pairs %s（几何口径不一致）" % (got, want))
    return got


def truth_nominal(clean_dir, geometry_block) -> list:
    """nominal 静态真值：优先 FDTD ``truth.json`` 的 nominal 3D（取 xy），否则 geometry_block.truth_xy；**不可空**。"""
    tpath = Path(clean_dir) / "truth.json"
    if tpath.is_file():
        doc = json.loads(tpath.read_text(encoding="utf-8-sig"))
        c3 = doc.get("targets_nominal_3d_m")
        if c3:
            return [[float(p[0]), float(p[1])] for p in c3]
    if geometry_block.get("truth_xy"):
        return [[float(p[0]), float(p[1])] for p in geometry_block["truth_xy"]]
    raise Q4Error("缺 nominal 真值：需 FDTD truth.json(targets_nominal_3d_m) 或 inventory geometry.truth_xy")


def build_q4_case(raw4, *, geometry_block, k, clean_id, grid_x, grid_y, trial=0, snr_db=None, scene=None):
    """frontend -> X；case_build（geometry=Q4 物理几何覆盖，config/FB 不动）；validate。"""
    case_build, frontend, _loader, _proto = _deps()
    raw4 = np.asarray(raw4)
    if raw4.ndim != 3 or raw4.shape[1] != 4 or raw4.shape[-1] != 32:
        raise Q4Error("Q4 前端要求 raw [L,4,32]，收到 shape=%s" % (raw4.shape,))
    X = frontend.x_from_raw(raw4)
    geom = q4_geometry(geometry_block)
    L = int(raw4.shape[0])
    case = case_build.build_case(
        case_id="q4_%s" % clean_id, clean_id="q4_%s" % clean_id, X=X, k=int(k),
        grid_x=grid_x, grid_y=grid_y, layer="B", source="fdtd", scene=scene, tags={},
        snr_db=snr_db, trial=int(trial), seed=0, L=L, geometry=geom)
    case_build.validate_case(case)
    return case


def run_methods(case: dict, methods=MAIN_METHODS, protocol=None) -> dict:
    """五法统一调用：``methods_loader.call(method, case, params_for(method, PROTOCOL))``。"""
    _cb, _fe, methods_loader, proto = _deps()
    proto = protocol or proto
    out = {}
    for method in methods:
        params = methods_loader.params_for(method, proto)
        res = methods_loader.call(method, case, params)
        out[method] = {"status": res.get("status"),
                       "predictions": [[float(p["x"]), float(p["y"])]
                                       for p in (res.get("predictions") or [])],
                       "diagnostics": res.get("diagnostics")}
    return out


def eval_clean(*, clean_dir, out_dir, geometry_block=None, geometry_id=None, k=None,
               methods=MAIN_METHODS) -> dict:
    """读 ``clean_cir.npz`` -> 校验 pairs -> 统一 ``sc.build_grid()`` -> frontend -> case -> 五法
    -> ``<geom>/<method>.pred.json``（``{__single__: {status, predictions}}``）+ ``truth.json``。

    网格固定用协议 **sc.build_grid()**（不裁域）；pairs 必须与 manifest 一致；真值不可空。
    """
    clean_dir = Path(clean_dir)
    npz_path = clean_dir / "clean_cir.npz"
    if not npz_path.is_file():
        raise Q4Error("缺 clean_cir.npz：%s" % npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        raw = np.asarray(z["raw_cir"])
        tx_index = np.asarray(z["tx_index"], int) if "tx_index" in z.files else None
        rx_index = np.asarray(z["rx_index"], int) if "rx_index" in z.files else None
        scene_id = str(z["scene_id"]) if "scene_id" in z.files else clean_dir.name
    if geometry_block is None:
        raise Q4Error("eval_clean 需要 geometry_block（inventory.fdtd_geometry，Q4 物理几何）")
    pairs = q4_geometry(geometry_block)["pairs"]
    got_pairs = assert_pairs_match(tx_index, rx_index, pairs)          # 与 manifest 校验
    raw4 = select_channels(raw, pairs)
    truth = truth_nominal(clean_dir, geometry_block)                   # 不可空
    gx, gy = sc.build_grid()                                          # 统一主域网格
    case = build_q4_case(raw4, geometry_block=geometry_block,
                         k=int(k if k is not None else len(truth)), clean_id=scene_id,
                         grid_x=gx, grid_y=gy)
    preds = run_methods(case, methods=methods)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 供 batch 消费的 4 通道 clean CIR（[L,4,32]，单场景共享 -> __single__）
    np.savez_compressed(out_dir / "clean_cir.npz", raw_cir=raw4.astype(np.complex64),
                        scene_id=np.array(scene_id), snapshots=np.array(int(raw4.shape[0])))
    write_json(out_dir / "truth.json", {"__single__": truth})
    written = {}
    for method, res in preds.items():
        # schema 兼容 load_predictions：值是 dict{status,predictions}，不是把 status 混进映射
        write_json(out_dir / (method + ".pred.json"),
                   {"__single__": {"status": res["status"], "predictions": res["predictions"]}})
        written[method] = res["status"]
    doc = {"kind": "q4-eval-clean", "schema_version": 1, "scene_id": scene_id,
           "geometry_id": geometry_id, "n_methods": len(preds), "status": written,
           "pairs": pairs, "pairs_validated": (got_pairs == pairs),
           "grid": {"nx": int(len(gx)), "ny": int(len(gy)),
                    "x": [float(gx[0]), float(gx[-1])], "y": [float(gy[0]), float(gy[-1])]},
           "truth_nominal_xy": truth,
           "provenance": {"added_noise": "none", "calibration": "uncalibrated",
                          "sim_truth_kind": "nominal_static",
                          "sim_microtrajectory": "differs_from_hardware(declared)"},
           "note": "写入 <geom>/<method>.pred.json + truth.json；batch 按这些 exact files 读取"}
    write_json(out_dir / "q4_eval.json", doc)
    return doc


def eval_plan(*, inventory_json=DEFAULT_INVENTORY) -> dict:
    """就绪检查（显式依赖）；不产出 prediction。"""
    idx = geometry_index(inventory_json)
    return {"kind": "q4-eval-plan", "schema_version": 2, "inventory": idx["source"],
            "n_geometries": len(idx["geometries"]), "geometries": sorted(idx["geometries"]),
            "deps": deps_ready(), "methods": list(MAIN_METHODS),
            "geometry_override": "manifest_real_txrx_pairs",
            "provenance": {"added_noise": "none", "calibration": "uncalibrated"},
            "note": "显式 from adapter import case_build/frontend/methods_loader + params_for(PROTOCOL)；不猜 module"}
