#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对外接口：plan_run（预算/清单，不跑 Meep）与 generate_clean（生成算法可见 case）。

产出契约（与主实现代理的 adapter 对齐）
---------------------------------------
outdir/clean_cir.npz（算法可见）:
    raw_cir            complex128 [L, 4, 32]   # 未 crop；adapter 自行 crop_start（v2=0，不裁）
    pair_index / tx_index / rx_index           [4]
    state_index                                 [L]
    t_origin_s                                  0.0（Tx 源起始时刻）
    sample_interval_s / fc_hz / bandwidth_hz / raw_taps / crop_start / fft_length / snapshots
    scene_id / channel_names / protocol_version / chain_version / cache_key / background_mode

outdir/truth.json：逐状态目标 xy 真值 + K + 投影说明（独立文件，不进 case）。
outdir/provenance/*：完整轨迹、状态哈希、轨迹源 hash、延迟偏差 metadata、实测诊断、
实测频谱、MF 参考、运行清单与文件哈希。

工程纪律（2026-09-26）
---------------------
* 内存：按状态分块走 RF 链，不把 128 状态原始记录与 Hilbert 中间量全堆在内存里。
* JSON 安全：complex ndarray 不直接进 JSON（否则算完才崩）；MF 参考单独存 npz。
* resume 记账：分别报 solves_planned / solves_actual / cache_hits，不把计划数当实测。
* 并行预填（2026-09-26）：``generate_clean(workers=N>1)`` 先经 ``precompute_parallel``
  spawn 多进程分片求解状态（每任务 1 状态，分片互斥，worker 内 has() 双保险），
  再串行组装；数值路径与串行共用 ``_solve_one_state``，缓存条目逐位同构。
  manifest 读-改-写由 cache.manifest.lock 互斥锁保护。
* outdir 保护：目标目录已有不同 cache key 的产物时，开头即拒，不在结尾覆盖 clean。
* 不做 crop2（留给 adapter）；不按真值调时延；empty-field 与硬件 EMA 分别声明。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np

from . import cache as cache_mod
from . import protocol, rf_chain, scenes as scenes_mod, trajectory as trajectory_mod
from .protocol import (
    CHAIN_VERSION,
    PROTOCOL_VERSION,
    atomic_write_json,
    atomic_write_npz,
    deep_merge,
    file_sha256,
    read_json,
)

BACKGROUND_MODES = ("empty_field", "ema_windowed")
EMA_FREEZE_NOTE = (
    "ema_windowed 冻结硬件 b0.cfr_replay 语义：每通道 alpha=1/bg_m_factor（默认 bg_m_factor=4 -> 0.25）；"
    "首帧 out=cur 且 ref=cur；其后 out=cur-ref 再 ref=(1-alpha)*ref+alpha*cur；跨窗连续不重置；"
    "丢弃 startup（burnin）后使用真实状态。与 empty_field 不是同一条链。"
)
EMPTY_FIELD_NOTE = "empty_field：无目标空场差分（线性、精确散射场）；与硬件 EMA 背景处理不等同。"


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    """numpy / complex -> JSON 可序列化（complex 转 {real, imag}）。"""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": np.asarray(obj.real).tolist(), "imag": np.asarray(obj.imag).tolist()}
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


def _load_cfg(config: Any, scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    if config is None:
        base: Dict[str, Any] = {}
    elif isinstance(config, Mapping):
        base = dict(config)
    else:
        base = dict(read_json(config))
    cfg = deep_merge(protocol.PHYSICS_DEFAULTS, base)
    return deep_merge(cfg, dict(scene_obj.get("protocol_overrides", {})))


def _resolve_background(scene_obj: Mapping[str, Any], cfg: Mapping[str, Any], background_mode: str | None):
    bg_cfg = dict(scene_obj.get("background", {}))
    mode = str(background_mode or bg_cfg.get("mode") or cfg.get("background_mode") or "empty_field")
    if mode not in BACKGROUND_MODES:
        raise ValueError("background_mode 必须是 %s，得到 %r" % (list(BACKGROUND_MODES), mode))
    ema: Dict[str, Any] = {}
    if mode == "ema_windowed":
        ema = dict(bg_cfg.get("ema", {}))
        ema.setdefault("bg_m_factor", 4.0)
        ema.setdefault("alpha", 1.0 / float(ema["bg_m_factor"]))
        ema.setdefault("burnin_states", int(bg_cfg.get("burnin_states", 8)))
        ema.setdefault("first_raw", True)
    return mode, ema

def _prepare(config: Any, scene: Any, *, max_states: int | None, background_mode: str | None):
    """装载配置/场景、派生域、建轨迹、校验几何规则、算 cache key。"""
    scene_obj = scenes_mod.load_scene(scene)
    cfg = _load_cfg(config, scene_obj)
    scene_obj["domain"] = scenes_mod.resolve_domain(scene_obj, fc_hz=float(cfg["fc_hz"]))
    mode, ema = _resolve_background(scene_obj, cfg, background_mode)
    n_snap = min(int(scene_obj.get("snapshots", cfg["snapshots"])), int(cfg["snapshots"]))
    if max_states is not None:
        n_snap = min(n_snap, int(max_states))
    if n_snap < 1:
        raise ValueError("快照数必须 >= 1，得到 %d" % n_snap)
    burnin = int(ema.get("burnin_states", 0)) if mode == "ema_windowed" else 0
    n_total = n_snap + burnin
    spec = dict(scene_obj["targets"]["micro_motion"])
    spec["L"] = n_total
    centers = scenes_mod.target_centers_2d(scene_obj)
    traj = trajectory_mod.build_trajectory(spec, centers)
    state_hashes = trajectory_mod.assert_unique_states(traj)
    geometry_rules = scenes_mod.assert_geometry_rules(scene_obj, traj)
    src_hash = trajectory_mod.trajectory_source_hash(spec, centers)
    key = cache_mod.cache_key(cfg, scene_obj, src_hash, mode, ema=ema,
                              extra={"n_snapshots": n_snap, "burnin_states": burnin, "n_states_total": n_total})
    return {"cfg": cfg, "scene": scene_obj, "bg_mode": mode, "ema": ema, "n_snapshots": n_snap,
            "burnin": burnin, "n_total": n_total, "traj": traj, "centers": centers,
            "state_hashes": state_hashes, "trajectory_spec": spec,
            "trajectory_source_hash": src_hash, "cache_key": key, "geometry_rules": geometry_rules}


def _public_protocol(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any], prep: Mapping[str, Any]) -> Dict[str, Any]:
    geom = scene_obj["geometry"]
    return {
        "protocol_version": PROTOCOL_VERSION, "chain_version": CHAIN_VERSION,
        "scene_id": scene_obj["scene_id"],
        "fc_hz": float(cfg["fc_hz"]), "bandwidth_hz": float(cfg["bandwidth_hz"]),
        "sample_interval_s": float(cfg["sample_interval_s"]),
        "raw_taps": int(cfg["raw_taps"]), "crop_start": int(cfg["crop_start"]),
        "fft_length": int(cfg["fft_length"]), "snapshots": int(prep["n_snapshots"]),
        "aa_cutoff_hz": float(cfg["aa_cutoff_hz"]),
        "background_mode": prep["bg_mode"], "ema": dict(prep["ema"]),
        "geometry": {"tx_positions_m": geom["tx_positions_m"], "rx_positions_m": geom["rx_positions_m"],
                     "pairs": geom["pairs"], "projection": geom.get("projection", {})},
        "domain": scene_obj["domain"], "meep": scene_obj["meep"],
        "_note": "adapter 侧自行做 crop_start（v2=0，不裁窗）；本文件不含真值",
    }


def plan_run(config: Any = None, scene: Any = None, *, max_states: int | None = None,
             background_mode: str | None = None, measured_cell_updates_per_s: float | None = None) -> Dict[str, Any]:
    """只算清单与预算（不导入 Meep、不写数据）。"""
    from . import solver as solver_mod

    prep = _prepare(config, scene, max_states=max_states, background_mode=background_mode)
    scene_obj = prep["scene"]
    n_bg = 1 if prep["bg_mode"] == "empty_field" else 0
    cost = solver_mod.estimate_cost(scene_obj, prep["cfg"], prep["n_total"], n_bg,
                                    measured_cell_updates_per_s=measured_cell_updates_per_s)
    return {
        "scene_id": scene_obj["scene_id"], "prepared_at": _now_iso(),
        "protocol": _public_protocol(prep["cfg"], scene_obj, prep),
        "channel_names": scenes_mod.channel_names(scene_obj),
        "n_targets": int(np.asarray(scene_obj["targets"]["nominal_centers_m"]).shape[0]),
        "target_radius_m": float(scene_obj["targets"]["radius_m"]),
        "covered_range_m": scenes_mod.covered_range_m(scene_obj),
        "max_two_way_path_m": scenes_mod.meep_target_support_m(scene_obj),
        "geometry_rules": prep["geometry_rules"],
        "trajectory": trajectory_mod.describe(prep["trajectory_spec"]),
        "state_count": {"snapshots": prep["n_snapshots"], "burnin_states": prep["burnin"],
                        "states_total": prep["n_total"]},
        "cache_key": prep["cache_key"], "cost": cost,
        "notes": ["cost 为估计值（n_cells x n_steps x n_solves），不是实测性能",
                  "ema_windowed 的 burnin 状态必须真实求解（计入 n_solves），不得复制快照",
                  EMA_FREEZE_NOTE if prep["bg_mode"] == "ema_windowed" else EMPTY_FIELD_NOTE],
    }


def _select_channels(cir: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """[L, n_tx, n_rx, T] -> [L, 4, T]（按 pairs 的 (tx, rx) 取）。"""
    out = np.empty((cir.shape[0], pairs.shape[0], cir.shape[-1]), dtype=cir.dtype)
    for i, (tx_i, rx_i) in enumerate(np.asarray(pairs, dtype=int)):
        out[:, i, :] = cir[:, int(tx_i), int(rx_i), :]
    return out


def _diagnostics(traj: np.ndarray, scene_obj: Mapping[str, Any], cir: np.ndarray, Ts: float) -> Dict[str, Any]:
    """provenance 专用诊断（不进算法 case）：几何期望时延 vs 实测峰位。

    expected 取该状态下最远目标的双程时延（多目标同场时的上界），只作对照。
    """
    geom = scene_obj["geometry"]
    tx = np.asarray(geom["tx_positions_m"], dtype=float)[:, :2]
    rx = np.asarray(geom["rx_positions_m"], dtype=float)[:, :2]
    pairs = np.asarray(geom["pairs"], dtype=int)
    c0 = 2.99792458e8
    expected = np.zeros((traj.shape[0], pairs.shape[0]), dtype=float)
    for l in range(traj.shape[0]):
        for i, (tx_i, rx_i) in enumerate(pairs):
            d = [float(np.linalg.norm(tx[tx_i] - traj[l, k]) + np.linalg.norm(rx[rx_i] - traj[l, k]))
                 for k in range(traj.shape[1])]
            expected[l, i] = float(np.max(d)) / c0
    return {"expected_delay_s_farthest_target": expected, "expected_tap_farthest_target": expected / float(Ts),
            "measured_peak_tap": np.abs(cir).argmax(axis=-1).astype(int),
            "measured_peak_abs": np.abs(cir).max(axis=-1),
            "note": "峰位对照几何期望时延；未用于时延补偿，也未用于挑结果"}

def _outdir_previous_key(outdir: Path) -> Tuple[str | None, str]:
    keyfile = outdir / ".cache_key"
    if keyfile.is_file():
        return keyfile.read_text(encoding="utf-8").strip(), ".cache_key"
    clean = outdir / "clean_cir.npz"
    if clean.is_file():
        with np.load(clean, allow_pickle=False) as npz:
            if "cache_key" in npz.files:
                return str(npz["cache_key"].item()), "clean_cir.npz"
    return None, ""


def assert_outdir_available(outdir: Path, key: str) -> None:
    """开头即拒：目标目录已有不同 cache key 的产物时不得继续（不在结尾覆盖 clean）。"""
    prev, where = _outdir_previous_key(Path(outdir))
    if prev and prev != key:
        raise RuntimeError("outdir %s 已有不同 cache key 的产物（%s: %s... != %s...）：请换目录或显式清理，不在结尾覆盖 clean"
                           % (outdir, where, prev[:16], key[:16]))


def _solve_one_state(run_cache: "cache_mod.RunCache", scene_obj: Mapping[str, Any], cfg: Mapping[str, Any],
                     traj: np.ndarray, l: int, n_tx: int, solver_mod: Any,
                     *, say: Callable[[str], None] | None = None, n_total: int | None = None) -> np.ndarray:
    """求解单个状态（geometry 内全部目标同时存在，逐 Tx 激励）并原子落盘；返回该状态时间轴。

    串行 generate_clean 与并行 worker **共用本路径**（同 save 条目格式/meta），保证
    并行预填与串行求解产生的缓存逐位一致。
    """
    per_tx = []
    t_s = None
    for tx in range(n_tx):
        if say is not None and n_total is not None:
            say("状态 %d/%d，Tx %d/%d 求解（同一 geometry 内 %d 目标同时存在）"
                % (l + 1, n_total, tx + 1, n_tx, traj.shape[1]))
        res = solver_mod.run_single_solve(scene_obj, cfg, traj[l], tx)
        per_tx.append(res["records"])
        if t_s is None:
            t_s = np.asarray(res["t_s"], dtype=float)
    if len({r.shape[-1] for r in per_tx}) != 1:
        raise RuntimeError("不同 Tx 的记录长度不一致：%s" % [r.shape[-1] for r in per_tx])
    run_cache.save("state", l, {"records": np.stack(per_tx, axis=0), "t_s": t_s},
                   meta={"state_index": int(l), "targets_2d_m": traj[l].tolist(),
                         "n_targets_in_geometry": int(traj.shape[1]),
                         "same_geometry_multi_target": True, "role": "target_state"})
    return t_s


def _solve_states_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """spawn 子进程入口（模块级函数保证可 pickle）：求解分到的状态分片。

    worker 内每个状态再有 ``has()`` 双保险（分片互斥下正常不会触发）；子进程独立
    导入 meep，不继承父进程的求解器状态。返回本 worker 实解的状态清单与计数。
    """
    import os as _os

    from . import solver as solver_mod

    run_cache = cache_mod.RunCache(payload["cache_root"], str(payload["cache_key"]))
    scene_obj = payload["scene_obj"]
    cfg = payload["cfg"]
    traj = np.asarray(payload["traj"])
    n_tx = int(payload["n_tx"])
    solved = []
    for l in payload["states"]:
        l = int(l)
        if run_cache.has("state", l):
            continue
        _solve_one_state(run_cache, scene_obj, cfg, traj, l, n_tx, solver_mod)
        solved.append(l)
    return {"worker_pid": int(_os.getpid()), "solved": solved, "n_solves": len(solved) * n_tx}


def precompute_parallel(config: Any, scene: Any, outdir: str | Path, *, cache_root: str | Path | None = None,
                        max_states: int | None = None, background_mode: str | None = None,
                        workers: int = 2, progress: Callable[[str], None] | None = None) -> Dict[str, Any]:
    """并行预填 RunCache：background 串行 + 状态按 ``spawn`` 多进程分片求解。

    与 generate_clean(resume=True) 共享同一 _prepare/cache_key/条目格式与
    _solve_one_state 求解路径：预填完成后 generate_clean 全部 cache 命中，只做
    RF 链组装，不重复求解。纪律：
    * 状态分片互斥（每个状态只分给一个 worker）；worker 内 has() 双保险；
    * spawn（非 fork）：子进程独立解释器 + 独立导入 meep；
    * 每进程一个 Meep 实例 —— workers 受 CPU 核数与（单解峰值内存 × workers）约束；
    * 任一 worker 失败即抛出；resume 语义保证已完成状态不丢，重跑只补缺口；
    * 本函数不写 clean/truth/provenance（那是 generate_clean 的职责），只填缓存。
    """
    from . import solver as solver_mod

    say = progress or (lambda msg: None)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers 必须 >= 1，得到 %d" % workers)
    prep = _prepare(config, scene, max_states=max_states, background_mode=background_mode)
    cfg, scene_obj = prep["cfg"], prep["scene"]
    outdir = Path(outdir)
    assert_outdir_available(outdir, prep["cache_key"])
    n_tx = int(np.asarray(scene_obj["geometry"]["tx_positions_m"], dtype=float).shape[0])
    n_bg = 1 if prep["bg_mode"] == "empty_field" else 0
    cache_root = Path(cache_root) if cache_root else (outdir / "cache")
    run_cache = cache_mod.RunCache(cache_root, prep["cache_key"])
    stats = {"solves_planned": int(prep["n_total"]) * n_tx + (n_tx if n_bg else 0),
             "solves_actual": 0, "cache_hits": {"background": 0, "state": 0},
             "workers": workers, "mode": "precompute_parallel"}
    if n_bg:
        for tx in range(n_tx):
            if run_cache.has("background", tx):
                stats["cache_hits"]["background"] += 1
                say("背景复用 tx=%d" % tx)
                continue
            say("背景求解 tx=%d（预填阶段背景串行，不做进程扇出）" % tx)
            res = solver_mod.run_single_solve(scene_obj, cfg, None, tx)
            stats["solves_actual"] += 1
            run_cache.save("background", tx, {"records": res["records"], "t_s": res["t_s"]},
                           meta={"tx_index": tx, "role": "background_empty_field",
                                 "wall_time_s": res["wall_time_s"], "n_steps": res["n_steps"], **res["sim_meta"]})
    missing = [l for l in range(prep["n_total"]) if not run_cache.has("state", l)]
    stats["cache_hits"]["state"] = int(prep["n_total"]) - len(missing)
    if missing:
        if workers == 1 or len(missing) == 1:
            done_serial = stats["cache_hits"]["state"]
            for l in missing:
                _solve_one_state(run_cache, scene_obj, cfg, prep["traj"], int(l), n_tx, solver_mod)
                stats["solves_actual"] += n_tx
                done_serial += 1
                say("状态进度 %d/%d（串行）" % (done_serial, prep["n_total"]))
        else:
            import concurrent.futures as _cf
            import multiprocessing as _mp

            ctx = _mp.get_context("spawn")
            # 每任务 1 个状态：单状态求解远重于任务调度开销，粒度最细即负载最均衡。
            chunks = [[int(l)] for l in missing]
            payloads = [{"cache_root": str(cache_root), "cache_key": prep["cache_key"],
                         "scene_obj": scene_obj, "cfg": cfg, "traj": prep["traj"],
                         "n_tx": n_tx, "states": ch} for ch in chunks]
            say("并行预填：%d 个待解状态 / %d workers（spawn，每任务 1 状态）"
                % (len(missing), workers))
            done_states = stats["cache_hits"]["state"]
            with _cf.ProcessPoolExecutor(max_workers=min(workers, len(chunks)),
                                         mp_context=ctx) as ex:
                for out in ex.map(_solve_states_worker, payloads):
                    done_states += len(out["solved"])
                    stats["solves_actual"] += int(out["n_solves"])
                    say("状态进度 %d/%d（worker pid=%d 本批 %d 个）"
                        % (done_states, prep["n_total"], out["worker_pid"], len(out["solved"])))
    # 收尾自检：全部状态必须 has() 通过，否则视为预填失败（不允许静默缺口）。
    still_missing = [l for l in range(prep["n_total"]) if not run_cache.has("state", l)]
    if still_missing:
        raise RuntimeError("并行预填后仍有 %d 个状态缺失（首几个 %s）：缓存可能被并发破坏，请检查"
                           % (len(still_missing), still_missing[:5]))
    say("预填完成：cache_key=%s（实解 %d，复用 bg=%d/state=%d）"
        % (prep["cache_key"][:16], stats["solves_actual"],
           stats["cache_hits"]["background"], stats["cache_hits"]["state"]))
    return {"cache_key": prep["cache_key"], "cache_root": str(cache_root),
            "n_total": int(prep["n_total"]), "n_tx": n_tx, "stats": stats,
            "bg_mode": prep["bg_mode"], "outdir": str(outdir)}


def generate_clean(config: Any, scene: Any, outdir: str | Path, *, cache_root: str | Path | None = None,
                   resume: bool = True, max_states: int | None = None, background_mode: str | None = None,
                   dry_run: bool = False, diagnostics_taps: int = 64, block_states: int = 8,
                   workers: int = 1,
                   progress: Callable[[str], None] | None = None) -> Dict[str, Any]:
    """生成算法可见 clean case（raw_cir[L,4,32]）与独立 truth/provenance。

    workers > 1 时先经 ``precompute_parallel`` 多进程预填状态缓存（数值路径与串行
    完全一致，仅调度并行），随后组装全部命中缓存；workers=1 时行为与本表历史版本
    逐位一致（先背景后状态串行）。
    """
    from . import solver as solver_mod

    say = progress or (lambda msg: None)
    prep = _prepare(config, scene, max_states=max_states, background_mode=background_mode)
    cfg, scene_obj = prep["cfg"], prep["scene"]
    outdir = Path(outdir)
    assert_outdir_available(outdir, prep["cache_key"])
    n_tx = int(np.asarray(scene_obj["geometry"]["tx_positions_m"], dtype=float).shape[0])
    n_bg = 1 if prep["bg_mode"] == "empty_field" else 0
    stats = {"solves_planned": int(prep["n_total"]) * n_tx + (n_tx if n_bg else 0),
             "solves_actual": 0, "cache_hits": {"background": 0, "state": 0}}
    plan = plan_run(config, scene, max_states=max_states, background_mode=background_mode)
    if dry_run:
        say("dry_run：只输出计划，不求解")
        outdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(outdir / "plan.json", _jsonable(plan))
        return {"dry_run": True, "plan": plan, "outdir": str(outdir)}

    cache_root = Path(cache_root) if cache_root else (outdir / "cache")
    run_cache = cache_mod.RunCache(cache_root, prep["cache_key"])
    say("cache key = %s（root=%s）" % (prep["cache_key"], cache_root))
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers 必须 >= 1，得到 %d" % workers)
    if workers > 1:
        pre = precompute_parallel(config, scene, outdir, cache_root=cache_root,
                                  max_states=max_states, background_mode=background_mode,
                                  workers=workers, progress=say)
        say("并行预填完成：实解 %d，复用 bg=%d/state=%d（后续组装全部命中缓存）"
            % (pre["stats"]["solves_actual"], pre["stats"]["cache_hits"]["background"],
               pre["stats"]["cache_hits"]["state"]))

    bg_records = None
    t_s = None
    if n_bg:
        bg_list = []
        for tx in range(n_tx):
            if resume and run_cache.has("background", tx):
                ent = run_cache.load("background", tx)
                stats["cache_hits"]["background"] += 1
                say("背景复用 tx=%d（已复算 hash/shape/finite/时间轴）" % tx)
            else:
                say("背景求解 tx=%d" % tx)
                res = solver_mod.run_single_solve(scene_obj, cfg, None, tx)
                stats["solves_actual"] += 1
                run_cache.save("background", tx, {"records": res["records"], "t_s": res["t_s"]},
                               meta={"tx_index": tx, "role": "background_empty_field",
                                     "wall_time_s": res["wall_time_s"], "n_steps": res["n_steps"], **res["sim_meta"]})
                ent = run_cache.load("background", tx)
            bg_list.append(ent["records"])
            if t_s is None:
                t_s = np.asarray(ent["t_s"], dtype=float)
        bg_records = np.stack(bg_list, axis=0)

    for l in range(prep["n_total"]):
        if resume and run_cache.has("state", l):
            stats["cache_hits"]["state"] += 1
            if t_s is None:
                t_s = np.asarray(run_cache.load("state", l)["t_s"], dtype=float)
            say("状态 %d/%d 复用" % (l + 1, prep["n_total"]))
            continue
        ts_l = _solve_one_state(run_cache, scene_obj, cfg, prep["traj"], l, n_tx, solver_mod,
                                say=say, n_total=prep["n_total"])
        stats["solves_actual"] += n_tx
        if t_s is None:
            t_s = ts_l
    if t_s is None:
        raise RuntimeError("没有可用时间轴（无背景且无状态记录）")

    time_out = rf_chain.output_time_axis(t_s, cfg["sample_interval_s"])
    n_out = int(time_out.shape[0])
    if int(cfg["raw_taps"]) > n_out:
        raise ValueError("raw_taps=%d 超过可用 tap 数 %d：请增大 run_time_meep" % (int(cfg["raw_taps"]), n_out))
    keep_taps = int(min(max(int(cfg["raw_taps"]), int(diagnostics_taps)), n_out))
    ref_ds = rf_chain.prepare_reference_ds(t_s, cfg, time_out)
    parts = []
    chain_info: Dict[str, Any] = {}
    step = max(1, int(block_states))
    for start in range(0, prep["n_total"], step):
        idx = list(range(start, min(start + step, prep["n_total"])))
        block = np.stack([run_cache.load("state", l)["records"] for l in idx], axis=0)
        cir_b, chain_info = rf_chain.records_to_aligned_cir(
            block, bg_records, t_s, cfg, keep_taps=keep_taps,
            background_mode=prep["bg_mode"], ema_alpha=prep["ema"].get("alpha"), ref_ds=ref_ds)
        parts.append(cir_b)
        say("RF 链处理状态 %d-%d/%d" % (idx[0] + 1, idx[-1] + 1, prep["n_total"]))
    cir_full = np.concatenate(parts, axis=0)
    cir_ch = _select_channels(cir_full, np.asarray(scene_obj["geometry"]["pairs"], dtype=int))
    if prep["bg_mode"] == "ema_windowed":
        cir_ch = rf_chain.apply_state_ema(cir_ch, float(prep["ema"]["alpha"]),
                                          first_raw=bool(prep["ema"].get("first_raw", True)),
                                          burnin=prep["burnin"])
    raw_cir = np.ascontiguousarray(cir_ch[..., : int(cfg["raw_taps"])])
    if raw_cir.shape != (prep["n_snapshots"], 4, int(cfg["raw_taps"])):
        raise RuntimeError("raw_cir 形状异常：%s" % (raw_cir.shape,))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ".cache_key").write_text(prep["cache_key"] + "\n", encoding="utf-8")
    pairs = np.asarray(scene_obj["geometry"]["pairs"], dtype=int)
    atomic_write_npz(outdir / "clean_cir.npz",
                     raw_cir=raw_cir.astype(np.complex128),
                     pair_index=np.arange(4, dtype=int),
                     tx_index=pairs[:, 0].astype(int), rx_index=pairs[:, 1].astype(int),
                     state_index=np.arange(prep["n_snapshots"], dtype=int),
                     t_origin_s=np.array(0.0),
                     sample_interval_s=np.array(float(cfg["sample_interval_s"])),
                     fc_hz=np.array(float(cfg["fc_hz"])),
                     bandwidth_hz=np.array(float(cfg["bandwidth_hz"])),
                     raw_taps=np.array(int(cfg["raw_taps"])),
                     crop_start=np.array(int(cfg["crop_start"])),
                     fft_length=np.array(int(cfg["fft_length"])),
                     snapshots=np.array(int(prep["n_snapshots"])),
                     scene_id=np.array(str(scene_obj["scene_id"])),
                     channel_names=np.array(scenes_mod.channel_names(scene_obj)),
                     t_origin_definition=np.array("t=0 为 Tx 源起始时刻；MF 按物理 filter lag 对齐；未按真值补偿"),
                     background_mode=np.array(str(prep["bg_mode"])),
                     protocol_version=np.array(PROTOCOL_VERSION),
                     chain_version=np.array(CHAIN_VERSION),
                     cache_key=np.array(str(prep["cache_key"])))
    with np.load(outdir / "clean_cir.npz", allow_pickle=False) as npz:
        protocol.assert_clean_case(npz.files)

    atomic_write_json(outdir / "truth.json", {
        "scene_id": scene_obj["scene_id"], "K": int(prep["centers"].shape[0]),
        "projection": scene_obj["geometry"].get("projection", {}), "snapshots": int(prep["n_snapshots"]),
        "targets_xy_m": prep["traj"][prep["burnin"]:].tolist(),
        "targets_nominal_3d_m": scene_obj["targets"]["nominal_centers_m"],
        "note": "真值独立文件，不进算法 case；2D 只取 xy 投影，离面 z 不可表达"})

    prov = outdir / "provenance"
    atomic_write_npz(prov / "trajectory.npz", trajectory_xy_m=prep["traj"],
                     state_hashes=np.array(prep["state_hashes"]),
                     used_snapshots_xy_m=prep["traj"][prep["burnin"]:])
    diag = _diagnostics(prep["traj"][prep["burnin"]:], scene_obj, raw_cir, float(cfg["sample_interval_s"]))
    spec = rf_chain.mean_spectrum(raw_cir, float(cfg["sample_interval_s"]))
    atomic_write_npz(prov / "diagnostics.npz",
                     expected_delay_s=diag["expected_delay_s_farthest_target"],
                     expected_tap=diag["expected_tap_farthest_target"],
                     measured_peak_tap=diag["measured_peak_tap"],
                     measured_peak_abs=diag["measured_peak_abs"],
                     spectrum_freq_hz=spec["freq_hz"], spectrum_mean_magnitude=spec["mean_magnitude"])
    if "reference" in chain_info:
        atomic_write_npz(prov / "mf_reference.npz", reference=np.asarray(chain_info["reference"]))
    atomic_write_json(prov / "chain_info.json",
                      _jsonable({k: v for k, v in chain_info.items() if not isinstance(v, np.ndarray)}))
    atomic_write_json(prov / "trajectory_spec.json",
                      _jsonable({"spec": prep["trajectory_spec"],
                                 "describe": trajectory_mod.describe(prep["trajectory_spec"]),
                                 "trajectory_source_hash": prep["trajectory_source_hash"],
                                 "state_hashes_unique": len(set(prep["state_hashes"])) == len(prep["state_hashes"])}))
    atomic_write_json(prov / "delay_deviation.json", _jsonable(scenes_mod.delay_deviation_metadata(scene_obj)))
    atomic_write_json(prov / "geometry_rules.json", _jsonable(prep["geometry_rules"]))
    atomic_write_json(outdir / "protocol.json", _jsonable(_public_protocol(cfg, scene_obj, prep)))

    outputs = ["clean_cir.npz", "truth.json", "protocol.json", "provenance/trajectory.npz",
               "provenance/diagnostics.npz", "provenance/chain_info.json", "provenance/mf_reference.npz",
               "provenance/trajectory_spec.json", "provenance/delay_deviation.json",
               "provenance/geometry_rules.json"]
    outputs = [rel for rel in outputs if (outdir / rel).is_file()]
    manifest = {
        "generated_at": _now_iso(), "cache_key": prep["cache_key"], "cache_dir": str(run_cache.dir),
        "cache_bytes": run_cache.disk_usage_bytes(),
        "solves_planned": stats["solves_planned"], "solves_actual_this_run": stats["solves_actual"],
        "cache_hits": stats["cache_hits"], "workers": int(workers),
        "state_count": {"snapshots": prep["n_snapshots"], "burnin_states": prep["burnin"],
                        "states_total": prep["n_total"]},
        "same_geometry_multi_target": True, "background_mode": prep["bg_mode"],
        "background_note": EMA_FREEZE_NOTE if prep["bg_mode"] == "ema_windowed" else EMPTY_FIELD_NOTE,
        "chain_info_summary": _jsonable({k: v for k, v in chain_info.items() if not isinstance(v, np.ndarray)}),
        "geometry_rules": _jsonable(prep["geometry_rules"]),
        "files": {rel: {"sha256": file_sha256(outdir / rel), "bytes": int((outdir / rel).stat().st_size)}
                  for rel in outputs},
        "cost_plan": plan["cost"]}
    atomic_write_json(prov / "run_summary.json", manifest)
    atomic_write_json(outdir / "plan.json", _jsonable(plan))
    say("完成：%s（raw_cir %s；本次实解 %d，复用 %s）" % (outdir, raw_cir.shape, stats["solves_actual"], stats["cache_hits"]))
    return {"outdir": str(outdir), "clean_npz": str(outdir / "clean_cir.npz"),
            "truth_json": str(outdir / "truth.json"),
            "provenance_json": str(prov / "run_summary.json"),
            "raw_cir_shape": list(raw_cir.shape), "cache_key": prep["cache_key"],
            "cache_dir": str(run_cache.dir), "background_mode": prep["bg_mode"],
            "stats": stats, "plan": plan}


