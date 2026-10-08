#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E6 cold/warm 计时支持：真正预构字典（不是把同一调用跑两遍）。

定义（协议 §E6）
---------------
* cold：从共有 X 起算，并计入本次所需的字典/流形构建 -> pred。
* warm：从已构建字典/流形 + 共有 X 起算 -> pred。
* 通用前端 CIR->X 另列，不并入上述两段。

支持矩阵（诚实申报，不隐瞒）
---------------------------
* this_method / b0_standard_music：字典 = 网格流形 A（数据无关）。warm 复用 b0.standard_music
  的 steering_matrix 预构 A，跳过流形构建；与各自 estimate() 逐格核对（verify）。
* b1_deng / b2_kakouche / c3_kim：冻结模块在 estimate() 内部构造角/时延流形，**无公开预构 API**；
  在不改冻结算法的前提下无法给出真正 warm -> warm **不适用**（附原因）。
* c1_jade：闭式平移不变，无字典/网格 -> warm 与 cold 同构（非预构差异），warm 不适用。
"""

from __future__ import annotations

import sys
import json
import subprocess
import time

import sim_common as sc   # 先于 numpy：线程 env 必须在 import numpy 前固定

import numpy as np

WARM_NOT_APPLICABLE = {
    "b1_deng": "冻结模块 estimate() 内部构造角流形 C(theta)/Q(theta)，无公开预构 API（不改算法即无法真正 warm）",
    "b2_kakouche": "冻结模块 estimate() 内部构造 IMUSIC/LCMV/距离 MUSIC 流形，无公开预构 API",
    "c3_kim": "冻结模块 estimate() 内部构造 DFT-ESPRIT 时延字典，无公开预构 API",
    "c1_jade": "闭式平移不变（JADE），无字典/网格，warm 与 cold 同构（不存在预构差异）",
}
WARM_SUPPORTED = ("this_method", "b0_standard_music")


def warm_support(method: str) -> dict:
    if method in WARM_SUPPORTED:
        return {"supported": True, "dictionary": "grid_manifold_A（b0.standard_music.steering_matrix）",
                "reason": "数据无关流形可预构；warm 跳过流形构建"}
    return {"supported": False, "dictionary": None,
            "reason": WARM_NOT_APPLICABLE.get(method, "未注册方法")}


def _b0():
    if str(sc.VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(sc.VENDOR_DIR))
    from b0 import standard_music as sm
    return sm


def _cfg_from_case(case: dict, sm):
    cfg = case["config"]
    return sm.MusicConfig(
        fft_length=int(cfg["fft_length"]), tap_interval_si=float(cfg["tap_interval_si"]),
        center_freq_si=float(cfg["center_freq_si"]), bandwidth_si=float(cfg["bandwidth_si"]),
        epsilon_r_background=float(cfg["epsilon_r_background"]),
        system_delay_si=float(cfg["system_delay_si"]),
        diag_load_factor=float(cfg["diag_load_factor"]),
        is_forward_backward=bool(cfg["is_forward_backward"]),
        fov_degrees=float(cfg["fov_degrees"]),
        cir_window=tuple(int(v) for v in cfg["cir_window"]), k=int(case["k"]),
        pairs=tuple(tuple(int(v) for v in p) for p in np.asarray(case["pairs"]).tolist()),
        deg_corr=(None if cfg["deg_corr"] is None else tuple(cfg["deg_corr"])),
        ant_calib_phase=(None if cfg["ant_calib_phase"] is None else tuple(cfg["ant_calib_phase"])),
        freq_range_si=tuple(float(v) for v in cfg["freq_range_si"]))


def prepare(method: str, case: dict) -> dict | None:
    """预构字典（数据无关）。仅对支持的网格 MUSIC 方法；其余返回 None。"""
    if method not in WARM_SUPPORTED:
        return None
    sm = _b0()
    cfg = _cfg_from_case(case, sm)
    gx = np.asarray(case["grid_x"], dtype=float)
    gy = np.asarray(case["grid_y"], dtype=float)
    A, mask, _vx, _vy = sm.steering_matrix(cfg, case["tx_positions"], case["rx_positions"], gx, gy)
    A.flags.writeable = False
    return {"kind": "grid_manifold_A", "cfg": cfg, "A": A, "mask": mask,
            "shape": (int(gy.size), int(gx.size))}


def _peaks(case: dict, image: np.ndarray):
    from knownk_eval import load_protocol, peaks as kp
    proto = load_protocol(str(sc.VENDOR_DIR / "protocol" / "knownk_protocol.json"))
    peaks, _status = kp.nms_peaks(np.asarray(image, dtype=float),
                                  np.asarray(case["grid_x"], dtype=float),
                                  np.asarray(case["grid_y"], dtype=float),
                                  proto.peaks, k=int(case["k"]))
    return [{"x": float(p["x"]), "y": float(p["y"]), "score": float(p["value"])} for p in peaks]


def spectrum_from_manifold(method: str, case: dict, manifold: dict, X: np.ndarray) -> np.ndarray:
    """warm 求解：字典（A）已给，只做协方差 + EVD + 谱。

    **同实现原则**：this_method 走 **row 约定**（R = X^H X / L、denom = Re(sum((A@P)*conj(A)))），
    与 q6_methods_this_method.music_image_from_X 内部实现一致；b0_standard_music 走 **column 约定**
    （复用 b0.sample_covariance）。这样 warm 与 cold 的差异**只来自省掉流形 A 构建**，
    不把协方差实现差异算进 cold-warm 差（避免"用 B0 列协方差重写主方法"的口径混用）。
    """
    sm = _b0()
    cfg, A, mask = manifold["cfg"], np.asarray(manifold["A"]), manifold["mask"]
    X = np.asarray(X, dtype=np.complex128)
    M = int(X.shape[1])
    k = int(case["k"])
    if k >= M:
        raise ValueError("K=%d 必须小于 M=%d" % (k, M))
    L = int(X.shape[0])
    if method == "b0_standard_music":
        R, info = sm.sample_covariance(X, cfg)
        if info.get("zero_energy"):
            raise ValueError("协方差零能量（退化输入）")
        _eigvals, eigvecs = np.linalg.eigh(R)
        En = eigvecs[:, : M - k]
        projector = En @ En.conj().T
        denom = np.real(np.sum((np.conj(A) @ projector) * A, axis=1))
    elif method == "this_method":
        R = (X.T.conj() @ X) / float(L)
        if bool(case["config"]["is_forward_backward"]):
            J = np.flip(np.eye(M), axis=0)
            R = 0.5 * (R + J @ R.conj() @ J)
        trace = complex(np.trace(R))
        if (not np.isfinite(trace.real)) or trace.real <= 0.0:
            raise ValueError("协方差无能量或非有限：trace=%s" % (trace,))
        R = R + np.eye(M) * float(case["config"]["diag_load_factor"]) * trace
        _eigvals, eigvecs = np.linalg.eigh(R)
        En = eigvecs[:, : M - k]
        projector = En @ En.conj().T
        denom = np.real(np.sum((A @ projector) * np.conj(A), axis=1))
    else:
        raise ValueError("warm 未实现方法 %r" % method)
    denom = np.where(np.isnan(denom), np.inf, denom)
    denom = np.maximum(denom, 0.0)
    spectrum = 1.0 / (denom + 1e-12)
    spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
    image = np.zeros(manifold["shape"], dtype=float)
    image[mask] = np.maximum(spectrum, 0.0)
    return image


def solve_warm(method: str, case: dict, manifold: dict) -> dict:
    t0 = time.perf_counter()
    image = spectrum_from_manifold(method, case, manifold, np.asarray(case["X"]))
    preds = _peaks(case, image)
    return {"status": "ok", "predictions": preds, "spectrum": image,
            "diagnostics": {"eval_path": "warm_prebuilt_grid_manifold",
                            "dictionary": manifold["kind"],
                            "wall_seconds_warm_solve": time.perf_counter() - t0}}


def verify(method: str, case: dict, params: dict | None = None) -> dict:
    """逐格核对：warm（预构 A + 同 X）与 cold（estimate）是否一致。

    判据用**坐标比较 + 谱相对容差**（不要求 score 浮点逐字相等）。
    """
    from . import methods_loader as ml
    support = warm_support(method)
    if not support["supported"]:
        return {"method": method, "supported": False, "reason": support["reason"]}
    # 预热：先丢弃一次 cold+warm，排除首次 import / knownk 协议 JSON 读入等 IO 差异，
    # 使"warm 与 cold 同实现"的等价结论不受协议 load/IO 干扰（协议对象在模块内缓存）。
    try:
        ml.load(method)
        ml.call(method, case, params)
        solve_warm(method, case, prepare(method, case))
    except Exception:  # noqa: BLE001
        pass
    manifold = prepare(method, case)
    cold = ml.call(method, case, params)
    try:
        warm = solve_warm(method, case, manifold)
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "supported": True, "equivalent": False,
                "reason": "warm 求解失败：%s: %s" % (type(exc).__name__, exc)}
    cs, ws = cold.get("spectrum"), warm.get("spectrum")
    def _coords(res):
        return sorted((round(float(p["x"]), 9), round(float(p["y"]), 9))
                      for p in (res.get("predictions") or []))
    out = {"method": method, "supported": True, "prewarm_discarded": True,
           "note": "prewarm 丢弃一次，排除首次 import/协议 JSON load-IO；等价性判据=坐标+谱相对容差",
           "cold_status": cold.get("status"), "warm_status": warm.get("status"),
           "n_pred_cold": len(cold.get("predictions") or []),
           "n_pred_warm": len(warm.get("predictions") or []),
           "coords_identical": (_coords(cold) == _coords(warm))}
    if cs is not None and ws is not None:
        ca, wa = np.asarray(cs, dtype=float), np.asarray(ws, dtype=float)
        denom = float(np.linalg.norm(ca.ravel()) * np.linalg.norm(wa.ravel()))
        scale = max(float(np.max(np.abs(ca))), 1e-300)
        out.update({"max_abs_diff": float(np.max(np.abs(ca - wa))),
                    "max_rel_diff": float(np.max(np.abs(ca - wa)) / scale),
                    "spectrum_cosine": (float(ca.ravel() @ wa.ravel() / denom) if denom > 0 else 1.0)})
        out["equivalent"] = bool(out["coords_identical"] and out["max_rel_diff"] <= 1e-9
                                 and out["spectrum_cosine"] >= 1.0 - 1e-9)
    else:
        out["equivalent"] = bool(out["coords_identical"])
    return out


def timed_cold_warm(method: str, case: dict, params: dict | None, n_reps: int = 5) -> dict:
    """计时：cold（estimate 全流程）与 warm（预构 A + solve）。

    纪律：
    * 计时前先 ml.load(method) **预热导入**，使 t_cold 不含首次 import。
    * 逐次记录 status 与秒数，**不静默过滤失败**；失败次数单列，且失败不给耗时优势
      （失败项不计入 median/p95，但 n_failed 与逐次 status 一并落盘）。
    """
    from . import methods_loader as ml
    support = warm_support(method)
    try:
        ml.load(method)                      # 预热导入/装载，不计入 t_cold
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "load_failed": True, "reason": str(exc)}
    # 丢弃式 warm-up：首次 cold 调用含 import/编译/缓存效应，正式计时前先跑一次并丢弃结果
    warmup_support = support["supported"]
    try:
        ml.call(method, case, params)
        if warmup_support:
            solve_warm(method, case, prepare(method, case))
    except Exception:  # noqa: BLE001
        pass
    reps = []
    for _ in range(max(1, int(n_reps))):
        t0 = time.perf_counter()
        res = ml.call(method, case, params)
        cold_status = str(res["status"])   # 显式记录 cold status：failed 单列、不计入 median/P95
        reps.append({"status": cold_status, "seconds": time.perf_counter() - t0, "phase": "cold"})
    if support["supported"]:
        manifold = prepare(method, case)
        for _ in range(max(1, int(n_reps))):
            t0 = time.perf_counter()
            try:
                solve_warm(method, case, manifold)
                reps.append({"status": "ok", "seconds": time.perf_counter() - t0, "phase": "warm"})
            except Exception as exc:  # noqa: BLE001
                reps.append({"status": "failed", "seconds": time.perf_counter() - t0,
                             "phase": "warm", "reason": "%s: %s" % (type(exc).__name__, exc)})
                continue
    def _stats(phase):
        arr = np.asarray([r["seconds"] for r in reps if r["phase"] == phase and r["status"] == "ok"],
                         dtype=float)
        if arr.size == 0:
            return {"median": None, "p95": None, "n": 0}
        return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
                "n": int(arr.size)}
    n_failed = int(sum(1 for r in reps if r["status"] == "failed"))
    cold_statuses = sorted({r["status"] for r in reps if r["phase"] == "cold"})
    warm_ok = [r for r in reps if r["phase"] == "warm"]
    warm_status = ("not_applicable" if not support["supported"]
                   else ("ok" if warm_ok and all(r["status"] == "ok" for r in warm_ok)
                         else "failed"))
    return {"method": method, "warm_supported": support["supported"],
            "warm_reason": None if support["supported"] else support["reason"],
            "cold_s": _stats("cold"), "warm_s": _stats("warm"),
            "n_failed": n_failed, "n_repetitions": len(reps), "repetitions": reps,
            "cold_status": ("/".join(cold_statuses) if cold_statuses else "none"),
            "warm_status": warm_status,
            "warm_note": ("warm 不适用（无预构 API/闭式）" if not support["supported"]
                          else "warm = 预构流形 A + 复用共有 X")}


def rss_measurement() -> dict:
    """统一 RSS 口径：**区分真峰值与当前值**，不把 current 冒充 peak。

    * Linux/macOS：resource.getrusage(RUSAGE_SELF).ru_maxrss（真峰值；Linux kB、macOS bytes）
      -> rss_kind="peak"。
    * Windows：psutil memory_info().peak_wset（真峰值）-> rss_kind="peak"；
      无 peak_wset 时退回 memory_info().rss -> rss_kind="current_not_peak"（字段名/语义如实标注）。
    * 都不可用 -> rss_kind="unavailable"。
    """
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        factor = 1 if sys.platform == "darwin" else 1024
        return {"rss_bytes": value * factor, "rss_kind": "peak",
                "rss_source": "resource.ru_maxrss", "platform": sys.platform}
    except Exception:
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if sys.platform == "win32" and hasattr(mi, "peak_wset"):
            return {"rss_bytes": int(mi.peak_wset), "rss_kind": "peak",
                    "rss_source": "psutil.peak_wset", "platform": sys.platform}
        return {"rss_bytes": int(mi.rss), "rss_kind": "current_not_peak",
                "rss_source": "psutil.current_rss", "platform": sys.platform}
    except Exception:
        return {"rss_bytes": None, "rss_kind": "unavailable",
                "rss_source": "unavailable_on_platform", "platform": sys.platform}


def peak_rss_bytes_self():
    """兼容旧名：返回 (rss_bytes, rss_source)。语义见 rss_measurement（可能是 current 而非 peak）。"""
    m = rss_measurement()
    return m["rss_bytes"], ("%s[%s]" % (m["rss_source"], m["rss_kind"]))


def measure_peak_memory(method: str, mode: str, case_npz, meta_json, params=None, *,
                        timeout: int = 900, python: str | None = None, child: str | None = None) -> dict:
    """E6 峰值内存：**隔离子进程**单次运行一个方法（cold 或 warm），报告 RSS 峰值（统一口径）。

    子进程 scripts/mem_child.py 内用 resource.getrusage(...).ru_maxrss / psutil 取 RSS 峰值
    （maxrss），保证不同方法在同一口径下比较；warm 不支持则直接记 not_applicable。
    """
    mode = str(mode)
    if mode == "warm" and not warm_support(method)["supported"]:
        return {"method": method, "mode": mode, "status": "not_applicable",
                "reason": warm_support(method)["reason"], "peak_rss_bytes": None,
                "peak_source": "not_applicable"}
    child = child or str(sc.PKG_DIR / "scripts" / "mem_child.py")
    cmd = [python or sys.executable, "-B", child, str(case_npz), str(meta_json), method, mode,
           json.dumps(params or {})]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
    except Exception as exc:
        return {"method": method, "mode": mode, "error": "%s: %s" % (type(exc).__name__, exc)}
    if not out.stdout.strip():
        return {"method": method, "mode": mode, "error": (out.stderr or "")[-300:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


__all__ = ["warm_support", "prepare", "solve_warm", "spectrum_from_manifold", "verify",
           "timed_cold_warm", "measure_peak_memory", "peak_rss_bytes_self", "rss_measurement",
           "WARM_NOT_APPLICABLE", "WARM_SUPPORTED"]
