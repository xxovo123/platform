#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据生成：层 A（解析，带限修复后）与层 B（FDTD 多目标同场），噪声按规范 §6。

层 A
----
复用冻结包的 `analytic/forward.py`（已带 `BANDLIMIT-FIX-20260929`：IFFT 前显式乘 `band_mask`）。
**不复用** `analytic/cache.py`，因为它的 clean 种子走冻结口径（`sim_common.clean_seed`）；
新版按规范 §6 的 canonical-JSON seed 自行生成与落盘，缓存身份含协议版本与代码指纹。

层 B
----
复用冻结包 `fdtd/api.py::generate_clean`（Meep 2D；同一 geometry 内全部目标同时存在；
每目标有界随机游走）。几何由本模块按规范构造（场景 seed 用新版 clean seed），
**不复用** 冻结 `adapter/scene_export.py` 的 seed 口径。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from . import paths
from . import protocol as P

SIDECAR = "clean_meta.json"
TRUTH_NAME = "truth.json"


class DataError(RuntimeError):
    """数据生成/加载不合规（显式失败）。"""


def _sc():
    return paths.sim_common()


def _root(root) -> Path:
    return Path(root)


def _clean_dir(root, layer: str, clean_id: str) -> Path:
    # 允许把 fdtd 数据目录整体重定向（收敛前检用它把自己的解与正式批次隔离开；
    # 求解器状态缓存由 fdtd 的 cache_root 单独管理，不受本函数影响）。
    if layer == "B":
        sub = os.environ.get("JSF_FDTD_CLEAN_SUBDIR")
        if sub:
            return _root(root) / "cache" / sub / clean_id
    return _root(root) / "cache" / ("analytic" if layer == "A" else "fdtd") / clean_id


# --------------------------------------------------------------------------- #
# 层 A
# --------------------------------------------------------------------------- #
def build_clean_analytic(rec: dict, *, root, protocol_hash: str, source_aggregate: str,
                         force: bool = False) -> dict:
    sc = _sc()
    import numpy as np
    from analytic import forward as fwd_mod

    clean_id = str(rec["clean_id"])
    outdir = _clean_dir(root, "A", clean_id)
    npz_path = outdir / "clean_cir.npz"
    truth_path = outdir / TRUTH_NAME
    sidecar = outdir / SIDECAR
    sig = _signature(rec, role="analytic", protocol_hash=protocol_hash,
                     source_aggregate=source_aggregate)
    if sidecar.is_file() and not force:
        have = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(have.get("signature")) != sig:
            raise DataError("层 A clean_id=%s 签名不匹配（已有 %s != 本次 %s）：拒绝复用"
                            % (clean_id, str(have.get("signature"))[:12], sig[:12]))
        if not npz_path.is_file() or sc.sha256_file(npz_path) != have.get("npz_sha256"):
            raise DataError("层 A clean_id=%s npz 缺失或哈希不符" % clean_id)
        with np.load(npz_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_cir"])
        return {"clean_id": clean_id, "raw_cir": raw, "meta": have, "reused": True}

    L_clean = int(rec.get("L_clean", P.SNAPSHOTS))
    rng = np.random.default_rng(int(rec["clean_seed"]))
    f = fwd_mod.MethodFreeForward(rec["targets_xy"], k=int(rec["k"]), L=L_clean, rng=rng,
                                  coherence=float(rec.get("coherence", 0.0)),
                                  band_mode="bandlimited_500mhz")
    raw = np.ascontiguousarray(np.asarray(f.raw, dtype=np.complex128))
    if raw.shape != (L_clean, P.N_CHANNELS, P.RAW_TAPS):
        raise DataError("层 A raw 形状异常：%s" % (raw.shape,))
    if not np.all(np.isfinite(raw)):
        raise DataError("层 A raw 含 NaN/Inf")

    # 生成后自检：截窗前带外能量比例（规范 §6 硬门槛）
    band_check = out_of_band_fraction(f, freq_axis_hz=P.fft_freq_axis_hz())
    if band_check["oob_fraction_before_crop"] > 1e-12:
        raise DataError("层 A 带外能量 %.3e 超门槛 1e-12：带限掩膜未生效？"
                        % band_check["oob_fraction_before_crop"])

    outdir.mkdir(parents=True, exist_ok=True)
    sc.atomic_write_npz(npz_path, raw_cir=raw)
    truth = {"clean_id": clean_id, "k": int(rec["k"]),
             "targets_xy_m": [[float(x), float(y)] for x, y in rec["targets_xy"]],
             "targets_polar": [[float(r), float(t)] for r, t in rec["targets_polar"]],
             "scene": rec.get("scene"), "layer": "A",
             "note": "真值独立 JSON；不进算法 case。"}
    sc.atomic_write_json(truth_path, truth)
    meta = {
        "clean_id": clean_id, "layer": "A", "source": "analytic", "k": int(rec["k"]),
        "L_clean": int(L_clean), "scene": rec.get("scene"),
        "targets_xy_m": rec["targets_xy"], "targets_polar": rec["targets_polar"],
        "clean_seed": int(rec["clean_seed"]),
        "seed_scheme": "jsf_sim_20260929_v1 canonical-json sha256",
        "signature": sig, "protocol_hash": protocol_hash, "source_aggregate": source_aggregate,
        "protocol_version": P.PROTOCOL_VERSION,
        "band_mode": "bandlimited_500mhz",
        "band_check": band_check,
        "time_origin": "raw tap0 = physical zero delay (tau=0); crop_start=0",
        "crop_start": P.CROP_START, "raw_taps": P.RAW_TAPS, "fft_length": P.FFT_LENGTH,
        "npz_sha256": sc.sha256_file(npz_path), "truth_sha256": sc.sha256_file(truth_path),
        "raw_shape": list(raw.shape),
        "noise_prefix_policy": "每 clean/trial 先抽 [128,4,30] 标准噪声再取前缀",
        "built_at": sc.now_iso(),
    }
    sc.atomic_write_json(sidecar, meta)
    return {"clean_id": clean_id, "raw_cir": raw, "meta": meta, "reused": False}


def out_of_band_fraction(fwd, *, freq_axis_hz) -> dict:
    """截窗前（生成端 H 域）带外能量比例与截窗后泄漏（分别记录，不混判）。"""
    import numpy as np
    axis = np.asarray(freq_axis_hz, dtype=float)
    mask = np.abs(axis - P.FC_HZ) <= P.BAND_HALF_HZ + 1e-6
    H = np.fft.fft(np.asarray(fwd.raw, dtype=np.complex128), n=P.FFT_LENGTH, axis=-1)
    p_in = float(np.sum(np.abs(H[..., mask]) ** 2))
    p_out = float(np.sum(np.abs(H[..., ~mask]) ** 2))
    total = p_in + p_out
    # 截窗后（[0:30] + fft32 补零）的泄漏，单独披露
    cropped = np.asarray(fwd.raw, dtype=np.complex128)[:, :, :P.CROP_LENGTH]
    Hc = np.fft.fft(cropped, n=P.FFT_LENGTH, axis=-1)
    pc_out = float(np.sum(np.abs(Hc[..., ~mask]) ** 2))
    pc_tot = float(np.sum(np.abs(Hc) ** 2))
    return {"oob_fraction_before_crop": (p_out / total) if total > 0 else None,
            "oob_fraction_after_crop": (pc_out / pc_tot) if pc_tot > 0 else None,
            "note": ("before_crop 是规范 §6 的硬门槛（<1e-12）；after_crop 是 [0:30] 截窗"
                     "引入的谱泄漏，单独记录，不用它误判生成失败。")}


# --------------------------------------------------------------------------- #
# 层 B
# --------------------------------------------------------------------------- #
def build_fdtd_scene(rec: dict, *, resolution_px_per_m=None, run_time_meep=None,
                     background_mode=None, geometry=None, snapshots=None,
                     burnin_states=None) -> dict:
    """按规范构造 fdtd 场景字典（几何种子用新版 clean seed）。

    EMA 背景（Q4）必须显式给 `burnin_states`：规范 §7 要求"32 状态用于 EMA 启动，
    保留后 128 状态"。`snapshots` 是**保留（kept）数**，不是总状态数；
    总求解数 = snapshots(kept) + burnin_states（预热）。
    """
    k = int(rec["k"])
    r_traj = float(P.MICRO_MOTION_RADIUS_M)
    geom = dict(geometry or {})
    mode = str(background_mode or P.BACKGROUND_MODE_B)
    n_keep = int(snapshots if snapshots is not None else rec.get("L_clean", P.SNAPSHOTS))
    bg = {"mode": mode}
    if mode == "ema_windowed":
        bn = int(burnin_states if burnin_states is not None else P.Q4_BURNIN_STATES)
        bg["burnin_states"] = bn
        bg["ema"] = {"alpha": float(P.Q4_EMA_ALPHA), "burnin_states": bn,
                     "first_raw": True, "bg_m_factor": 4.0}
    return {
        "scene_id": str(rec["clean_id"]),
        "geometry": {
            "tx_positions_m": [[float(v) for v in row] for row in
                               (geom.get("tx_positions_m") or P.TX_POSITIONS_M)],
            "rx_positions_m": [[float(v) for v in row] for row in
                               (geom.get("rx_positions_m") or P.RX_POSITIONS_M)],
            "pairs": [[int(v) for v in row] for row in (geom.get("pairs") or P.PAIRS)],
            "projection": {"plane": "xy", "dropped_axis": "z",
                           "note": "2D FDTD 不表达离面 z；只保留 xy 投影，不把 z 当 y"},
        },
        "targets": {
            "nominal_centers_m": [[float(x), float(y), 0.0] for x, y in rec["targets_xy"]],
            "radius_m": float(P.SCATTERER_RADIUS_M),
            "overlap_margin_m": 0.005,
            "micro_motion": {"seed": int(rec["clean_seed"]),
                             "sigma_m": float(P.MICRO_MOTION_SIGMA_M),
                             "radius_m": [r_traj] * k},
        },
        "meep": {
            "resolution_px_per_m": float(resolution_px_per_m or P.FDTD_RESOLUTION_PX_PER_M),
            "run_time_meep": float(run_time_meep or P.FDTD_RUN_TIME_MEEP),
            "pml_thickness_lambda": float(P.PML_THICKNESS_LAMBDA),
        },
        "background": bg,
        "snapshots": n_keep,
        "sim_clean_id": str(rec["clean_id"]),
    }


def build_clean_fdtd(rec: dict, *, root, protocol_hash: str, source_aggregate: str,
                     resume: bool = True, workers: int = 1, resolution_px_per_m=None,
                     run_time_meep=None, background_mode=None, geometry=None,
                     max_states=None, dry_run=False, burnin_states=None,
                     snapshots_override=None) -> dict:
    sc = _sc()
    import numpy as np
    from fdtd import api as fdtd_api

    clean_id = str(rec["clean_id"])
    outdir = _clean_dir(root, "B", clean_id)
    npz_path = outdir / "clean_cir.npz"
    truth_path = outdir / TRUTH_NAME
    sidecar = outdir / SIDECAR
    res = resolution_px_per_m or P.FDTD_RESOLUTION_PX_PER_M
    rt = run_time_meep or P.FDTD_RUN_TIME_MEEP
    bg = background_mode or P.BACKGROUND_MODE_B
    sig = _signature(rec, role="clean", protocol_hash=protocol_hash,
                     source_aggregate=source_aggregate, resolution_px_per_m=res,
                     run_time_meep=rt, background_mode=bg, geometry=geometry,
                     burnin_states=burnin_states, snapshots_override=snapshots_override)
    if resume and sidecar.is_file() and not dry_run:
        have = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(have.get("signature")) != sig:
            raise DataError("层 B clean_id=%s 签名不匹配（已有 %s != 本次 %s）：拒绝复用旧结果"
                            % (clean_id, str(have.get("signature"))[:12], sig[:12]))
        if not npz_path.is_file() or sc.sha256_file(npz_path) != have.get("npz_sha256"):
            raise DataError("层 B clean_id=%s npz 缺失或哈希不符" % clean_id)
        with np.load(npz_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_cir"])
        # max_states 不在签名内（只是前缀长度），所以缓存可能**浅于**本次需求：
        # 预检用 16 状态建的缓存不能冒充正式 128 状态。深度不足即按本次的完整需求重解
        # （需要的状态是同一确定性的前缀，重解得到相同的物理状态）。
        # raw_cir 已经去掉 EMA 预热；Q4 的 L_clean=160 不能要求 raw 再有160帧。
        need = min(int(snapshots_override if snapshots_override is not None
                       else rec.get("L_clean", P.SNAPSHOTS)), P.SNAPSHOTS)
        if max_states is not None:
            need = min(need, int(max_states))
        if int(raw.shape[0]) < need:
            print("[data] 层 B %s 缓存深度 %d < 需要 %d：按完整需求重解（前缀确定性一致）"
                  % (clean_id, int(raw.shape[0]), need), flush=True)
            sidecar.unlink(missing_ok=True)
            npz_path.unlink(missing_ok=True)
            return build_clean_fdtd(rec, root=root, protocol_hash=protocol_hash,
                                    source_aggregate=source_aggregate, resume=resume,
                                    workers=workers, resolution_px_per_m=res,
                                    run_time_meep=rt, background_mode=bg, geometry=geometry,
                                    max_states=max_states, dry_run=dry_run,
                                    burnin_states=burnin_states,
                                    snapshots_override=snapshots_override)
        return {"clean_id": clean_id, "raw_cir": raw, "meta": have, "reused": True}

    scene = build_fdtd_scene(rec, resolution_px_per_m=res, run_time_meep=rt,
                            background_mode=bg, geometry=geometry,
                            snapshots=(snapshots_override
                                       if snapshots_override is not None
                                       else rec.get("L_clean", P.SNAPSHOTS)),
                            burnin_states=burnin_states)
    # 求解器状态缓存：与数据目录同一隔离口径（前检每档独立，避免跨档串用状态）
    _cache_sub = os.environ.get("JSF_FDTD_CLEAN_SUBDIR")
    cache_root = (_root(root) / "cache" / (_cache_sub or "fdtd") / "_solver_cache")
    res_out = fdtd_api.generate_clean(None, scene, outdir, cache_root=cache_root, resume=resume,
                                      max_states=max_states, background_mode=bg,
                                      dry_run=dry_run, workers=int(workers))
    if dry_run:
        return {"clean_id": clean_id, "dry_run": True, "plan": res_out.get("plan"),
                "scene": scene, "reused": False}
    n_path, t_path = Path(res_out["clean_npz"]), Path(res_out["truth_json"])
    with np.load(n_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_cir"])
    expect = (int(raw.shape[0]), P.N_CHANNELS, P.RAW_TAPS)
    if tuple(raw.shape) != expect:
        raise DataError("层 B raw 形状 %s != %s" % (tuple(raw.shape), expect))
    if not np.all(np.isfinite(raw)):
        raise DataError("层 B raw 含 NaN/Inf")
    if n_path != npz_path:
        import shutil
        shutil.copy2(n_path, npz_path)
        shutil.copy2(t_path if Path(t_path) == truth_path else t_path, truth_path)
    truth = json.loads(Path(t_path).read_text(encoding="utf-8"))
    meta = {
        "clean_id": clean_id, "layer": "B", "source": "fdtd", "k": int(rec["k"]),
        "L_clean": int(raw.shape[0]), "scene": rec.get("scene"),
        "targets_xy_m": rec["targets_xy"], "targets_polar": rec["targets_polar"],
        "clean_seed": int(rec["clean_seed"]),
        "seed_scheme": "jsf_sim_20260929_v1 canonical-json sha256",
        "signature": sig, "signature_scope": "jsf_physics_scope_v1",
        "protocol_hash": protocol_hash, "source_aggregate": source_aggregate,
        "protocol_version": P.PROTOCOL_VERSION,
        "geometry": {"tx_positions_m": scene["geometry"]["tx_positions_m"],
                     "rx_positions_m": scene["geometry"]["rx_positions_m"],
                     "pairs": scene["geometry"]["pairs"]},
        "meep": scene["meep"], "background_mode": bg,
        "scatterer_radius_m": float(P.SCATTERER_RADIUS_M),
        "micro_motion": scene["targets"]["micro_motion"],
        "time_origin": "raw tap0 = physical zero delay (tau=0)；与层 A 同一延迟参考",
        "cache_key": res_out.get("cache_key"), "solves": res_out.get("stats"),
        "npz_sha256": sc.sha256_file(npz_path), "truth_sha256": sc.sha256_file(truth_path),
        "raw_shape": list(raw.shape), "truth_K": truth.get("K"),
        "built_at": sc.now_iso(),
    }
    sc.atomic_write_json(sidecar, meta)
    return {"clean_id": clean_id, "raw_cir": raw, "meta": meta, "reused": False}


def load_clean(root, layer: str, clean_id: str, *, expect_signature=None) -> dict:
    sc = _sc()
    import numpy as np
    outdir = _clean_dir(root, layer, clean_id)
    sidecar = outdir / SIDECAR
    if not sidecar.is_file():
        raise DataError("clean 缓存缺失：%s" % sidecar)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if expect_signature is not None and str(meta.get("signature")) != str(expect_signature):
        raise DataError("clean_id=%s 签名不匹配：拒绝复用" % clean_id)
    npz_path = outdir / "clean_cir.npz"
    if sc.sha256_file(npz_path) != meta.get("npz_sha256"):
        raise DataError("clean_id=%s npz 哈希不符" % clean_id)
    with np.load(npz_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_cir"])
    if tuple(raw.shape) != (int(meta["L_clean"]), P.N_CHANNELS, P.RAW_TAPS):
        raise DataError("clean_id=%s raw 形状 %s 与 sidecar 不符" % (clean_id, raw.shape))
    if not np.all(np.isfinite(raw)):
        raise DataError("clean_id=%s raw 含 NaN/Inf" % clean_id)
    return {"raw_cir": raw, "meta": meta}


def _signature(rec: dict, *, role: str, protocol_hash: str, source_aggregate: str,
               **extra) -> str:
    """缓存身份：实际阵列/精确真值/运动源/带宽版本/时间原点/背景/网格时长/代码指纹。"""
    sc = _sc()
    payload = {
        "role": role, "clean_id": str(rec["clean_id"]),
        "k": int(rec["k"]), "L_clean": int(rec.get("L_clean", P.SNAPSHOTS)),
        "targets_xy": [[float(x), float(y)] for x, y in rec["targets_xy"]],
        "clean_seed": int(rec["clean_seed"]),
        "protocol_version": P.PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "source_aggregate": source_aggregate,
        "band_impl": "bandlimit-fix-20260929:mask-applied-in-forward",
        "time_origin": "raw_tap0_physical_zero;crop_start=%d" % P.CROP_START,
        "background_mode": P.BACKGROUND_MODE_B,
        "fc_hz": P.FC_HZ, "bandwidth_hz": P.BANDWIDTH_HZ, "ts": P.SAMPLE_INTERVAL_S,
        "raw_taps": P.RAW_TAPS, "fft_length": P.FFT_LENGTH,
        "geometry": {"tx_positions_m": P.TX_POSITIONS_M, "rx_positions_m": P.RX_POSITIONS_M,
                     "pairs": P.PAIRS},
        "array": "1Tx4Rx_controlled",
        "motion": {"sigma_m": P.MICRO_MOTION_SIGMA_M, "radius_m": P.MICRO_MOTION_RADIUS_M},
        "scatterer_radius_m": P.SCATTERER_RADIUS_M,
    }
    # 注意：**max_states 刻意不进签名**。它只表示"本次解算前多少个状态"，是数据的前缀长度，
    # 不改变任何状态的物理内容与数值路径；若放进签名，同一场景在 16 状态（预检）与 128 状态
    # （正式）之间会互相拒绝复用，导致收敛前检与正式批次无法共享同一个物理缓存。
    # 真正影响数值身份的量（几何/真值/种子/带宽实现/时间原点/背景/网格/时长/代码指纹）全部在内。
    payload.update({k: (json.loads(json.dumps(v)) if isinstance(v, dict) else v)
                    for k, v in extra.items() if k != "max_states"})
    return sc.param_hash(payload)


# --------------------------------------------------------------------------- #
# 噪声（规范 §6）
# --------------------------------------------------------------------------- #
def noise_units(physical_id: str, trial: int, *, l_available: int = P.SNAPSHOTS):
    """每 clean_id/trial 先抽 [128,4,30] 标准复白噪；改变 L 取前缀。"""
    import numpy as np
    seed = P.noise_seed(physical_id, trial)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((l_available, P.N_CHANNELS, P.CROP_LENGTH))
            + 1j * rng.standard_normal((l_available, P.N_CHANNELS, P.CROP_LENGTH))) / math.sqrt(2.0)


def build_x(raw_cir, *, L_requested: int, snr_db: float, physical_id: str, trial: int) -> dict:
    """无噪 raw -> 截窗 30 tap -> 加噪（处理后 SNR）-> X。

    严格按规范：噪声在**背景处理后、FFT 前的 30-tap 复数 CIR** 上加；
    Ps = 全 L×4×30 均方；Pn = Ps·10^(-SNR/10)；加噪后 FFT 取 17 带内频点。
    """
    import numpy as np
    sc = _sc()
    raw = np.asarray(raw_cir)
    if raw.ndim != 3 or raw.shape[-1] != P.RAW_TAPS:
        raise DataError("raw_cir 必须是 [L,C,32]，收到 %s" % (raw.shape,))
    l_avail = int(raw.shape[0])
    l_req = int(L_requested)
    if l_avail < l_req:
        raise DataError("clean 快照不足：可用 %d < 请求 %d（拒绝复制快照凑数）" % (l_avail, l_req))
    l_actual = l_req
    cropped = np.ascontiguousarray(raw[:l_actual, :, P.CROP_START:P.CROP_START + P.CROP_LENGTH])
    ps = float(np.mean(np.abs(cropped) ** 2))
    if not math.isfinite(ps) or ps <= 0.0:
        raise DataError("Ps=%r 非有限或非正：按规范判 failed，不静默加噪声底" % ps)
    pn = ps * 10.0 ** (-float(snr_db) / 10.0)
    sigma = math.sqrt(pn / 2.0)
    noise_length = max(l_avail, P.SNAPSHOTS)
    unit = noise_units(physical_id, trial, l_available=noise_length)[:l_actual]
    if unit.shape != cropped.shape:
        raise DataError("数据与噪声形状不一致：%s vs %s" % (cropped.shape, unit.shape))
    noisy = cropped + sigma * unit
    X = sc.x_from_cropped(noisy)
    # 实际实现 SNR：只用**真正的噪声分量** sigma*unit 的实测功率，不从 noisy-cropped 反推
    p_noise_realised = float(np.mean(np.abs(sigma * unit) ** 2))
    realised = (10.0 * math.log10(ps / p_noise_realised)
                if p_noise_realised > 0 else None)
    return {
        "X": np.ascontiguousarray(X),
        "L_actual": int(l_actual),
        "diag": {"Ps": ps, "Pn_target": float(pn), "sigma": float(sigma),
                 "noise_power_realised": p_noise_realised,
                 "snr_target_db": float(snr_db), "snr_realised_db": realised,
                 "L_requested": l_req, "L_actual": int(l_actual), "L_available": l_avail,
                 "noise_seed": int(P.noise_seed(physical_id, trial)),
                 "noise_shape_drawn": [noise_length, P.N_CHANNELS, P.CROP_LENGTH],
                 "background_processed_before_noise": True,
                 "note": ("处理后 SNR（非 RF 前端 SNR）；同 trial 跨参数为配对噪声试验，"
                          "不声称跨条件独立。有限样本实际 SNR 允许波动，不做噪声再归一化。")},
    }


def clean_x(raw_cir, *, L: int):
    """无噪 X（诊断/样例用）：截窗 30 tap -> fft32 -> 17 带内频点。"""
    import numpy as np
    sc = _sc()
    return sc.x_from_cropped(sc.crop_raw(np.asarray(raw_cir)[:int(L)]))


__all__ = ["DataError", "build_clean_analytic", "build_clean_fdtd", "build_fdtd_scene",
           "load_clean", "build_x", "noise_units", "out_of_band_fraction", "SIDECAR", "TRUTH_NAME"]
