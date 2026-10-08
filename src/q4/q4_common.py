# -*- coding: utf-8 -*-
"""Q4 公共工具：路径、几何命名、通道选择、时间中点、CIR 相似度（ρ 复数 / ρ_env 包络）。

口径要点（对齐 ``仿真实验完整方案_20260926.md`` §E5 与用户 2026-09-26 口径）：

* **硬件 cropped30**：硬件链路 ``cropped = arr[:, :, cir_offset:ft_len]``（原始 32 -> 30 tap），
  再做背景处理；本模块直接复用既有缓存中已按此口径处理好的 4 通道复数 CIR。
* **时间中点快照固定**：每窗取 ``L // 2`` 号慢时间快照，**不按相关性挑快照**（禁止 oracle shift）。
* **四通道联合向量**上算复数 ``rho = |sᴴh| / (||s||·||h||)``；包络用**窗内时间 RMS 包络**。
* 范数 < ``EPS`` 记 ``undefined``（返回 ``None``），不参与均值。
* 跨域 peak delta：**先各自按 truth 匹配到同身份**，再对同一身份算 ``||p_sim - p_hw||``；

  **不得**直接用 sim-hw 之间的最佳配对冒充身份对应。

本模块及 q4 子包**不 import** ``q6_case_cache``（其硬编码 Windows 根），只读取其落盘的 npz。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

# 复用 metrics 作为匹配/误差/OSPA 的单一来源（q4 与 metrics 同属 06_仿真实验_20260926）
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
from metrics import evaluate_predictions, match_by_gate, polar_of, wrap_deg  # noqa: E402

EPS = 1e-12
DEFAULT_CIR_OFFSET = 2      # 硬件 meta.algo_params['2D-MUSIC'].cir_offset
DEFAULT_FT_LEN = 32
CROPPED_TAPS = 30
DEFAULT_CACHE_DIR = "大修执行_20260923/04_Q6/hardware/cache/cases"
DEFAULT_TRUTH_INDEX = "大修执行_20260923/04_Q6/hardware/truth/truth_index.json"
DEFAULT_DATA_DIR = "Q2 数据采集/data_in_adult"

ENV_PREFIXES = ("indoor_", "outdoor_")


class Q4Error(ValueError):
    """Q4 输入/结构不合规（显式失败，不静默降级）。"""


# --------------------------------------------------------------------------- #
# 路径与 IO
# --------------------------------------------------------------------------- #
def workspace_root() -> Path:
    """工作区根（D:\\论文\\小论文修改）：本文件位于 06_仿真实验_20260926/q4/ 下。"""
    return Path(__file__).resolve().parents[3]


def resolve_path(path, base=None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base) if base else workspace_root()) / p


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 录制 / 几何命名
# --------------------------------------------------------------------------- #
def environment_of(recording_id: str) -> str:
    for prefix in ENV_PREFIXES:
        if recording_id.startswith(prefix):
            return prefix[:-1]
    return "unknown"


def geometry_of(recording_id: str) -> str:
    """去环境前缀、去尾部 _001/_002 录制序号 -> 几何标识（15 个几何）。"""
    name = recording_id
    for prefix in ENV_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    parts = name.split("_")
    if parts and len(parts[-1]) == 3 and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def base_truth_index():
    return DEFAULT_TRUTH_INDEX


# --------------------------------------------------------------------------- #
# 通道选择 / tap 口径
# --------------------------------------------------------------------------- #
def select_virtual_channels(arr, pairs):
    """arr: (n_rx, n_tx, taps, L) -> (L, n_ch, taps) 复数 CIR。

    ``pairs[c] = (tx_idx, rx_idx)``（与缓存内 ``pairs`` 同序，(0,2),(0,3),(1,2),(1,3)）。
    """
    a = np.asarray(arr)
    if a.ndim != 4:
        raise Q4Error("snapshots 必须是 (n_rx, n_tx, taps, L)，收到 shape=%s" % (a.shape,))
    n_rx, n_tx, taps, L = a.shape
    out = np.empty((L, len(pairs), taps), dtype=np.complex64)
    for c, pair in enumerate(pairs):
        tx, rx = int(pair[0]), int(pair[1])
        if not (0 <= tx < n_tx and 0 <= rx < n_rx):
            raise Q4Error("pair %r 越界（n_rx=%d n_tx=%d）" % (tuple(pair), n_rx, n_tx))
        out[:, c, :] = a[rx, tx, :, :].T
    return out


def mid_index(L: int) -> int:
    """固定时间中点索引 = L // 2（默认 L=128 -> 64），不用相关性挑快照。"""
    if int(L) <= 0:
        raise Q4Error("L 必须是正整数，收到 %r" % (L,))
    return int(L) // 2


def crop_fixed(raw, cir_offset: int = DEFAULT_CIR_OFFSET, taps: int = CROPPED_TAPS):
    """固定截窗（不搜索时移，禁止 oracle shift）：raw[..., off:off+taps]"""
    a = np.asarray(raw)
    off = int(cir_offset)
    if a.shape[-1] < off + int(taps):
        raise Q4Error("raw tap=%d 不足以按 offset=%d 截 %d tap" % (a.shape[-1], off, taps))
    return a[..., off:off + int(taps)]


def raw32_to_cropped30(raw32, cir_offset: int = DEFAULT_CIR_OFFSET):
    """仿真侧 raw ``(L, C, 32)`` -> 硬件 30 口径 ``(L, C, 30)``（固定 offset，无时移搜索）。"""
    return crop_fixed(raw32, cir_offset=cir_offset, taps=CROPPED_TAPS)


# --------------------------------------------------------------------------- #
# 相似度（复数 ρ / 包络 ρ_env）
# --------------------------------------------------------------------------- #
def _norm(vec) -> float:
    return float(np.linalg.norm(np.asarray(vec).ravel()))


def complex_rho(s, h):
    """四通道联合复数向量相似度 ``|sᴴh| / (||s||·||h||)``。

    零范数或含 NaN/Inf -> ``None``（undefined，不写 NaN 进 JSON）；结果 clamp 到 ``[0, 1]``。
    """
    sv = np.asarray(s).ravel().astype(np.complex128)
    hv = np.asarray(h).ravel().astype(np.complex128)
    if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(hv))):
        return None
    ns, nh = _norm(sv), _norm(hv)
    if ns < EPS or nh < EPS:
        return None
    return float(min(1.0, max(0.0, float(abs(np.vdot(sv, hv))) / (ns * nh))))


def rms_envelope(cir):
    """窗内时间 RMS 包络：cir ``(L, C, T)`` -> 实数 ``(C, T)`` = sqrt(mean_L |cir|^2)。"""
    a = np.asarray(cir)
    if a.ndim != 3:
        raise Q4Error("rms_envelope 需要 (L, C, T)，收到 shape=%s" % (a.shape,))
    return np.sqrt(np.mean(np.abs(a) ** 2, axis=0))


def envelope_rho(env_s, env_h):
    """包络相似度：实数非负向量的余弦；零范数 / NaN / Inf -> None；clamp 到 [0, 1]。"""
    a = np.asarray(env_s, dtype=float).ravel()
    b = np.asarray(env_h, dtype=float).ravel()
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return None
    return float(max(0.0, min(1.0, float(np.dot(a, b)) / (na * nb))))


def window_similarity(sim_cir, hw_cir):
    """同一窗的 sim/hw 相似度：mid 快照复数 ρ + 时间 RMS 包络 ρ_env。

    ``sim_cir``/``hw_cir`` 均为 ``(L, C, T)``，且 T 必须一致（调用方先统一到 30 口径）。
    含 NaN/Inf 的输入 -> 对应 ρ 记 ``None``（undefined）并置 ``inputs_non_finite=True``，
    **不把 NaN 写进 JSON**。
    """
    s = np.asarray(sim_cir)
    h = np.asarray(hw_cir)
    if s.shape != h.shape:
        raise Q4Error("sim/hw CIR shape 不一致：%s vs %s" % (s.shape, h.shape))
    if s.ndim != 3:
        raise Q4Error("CIR 必须是 (L, C, T)，收到 shape=%s" % (s.shape,))
    L = s.shape[0]
    mi = mid_index(L)
    finite = bool(np.all(np.isfinite(s)) and np.all(np.isfinite(h)))
    return {
        "mid_index": int(mi),
        "rho_complex": complex_rho(s[mi], h[mi]) if finite else None,
        "rho_env": envelope_rho(rms_envelope(s), rms_envelope(h)) if finite else None,
        "inputs_non_finite": (not finite),
        "T_taps": int(s.shape[-1]),
        "L": int(L),
    }


# --------------------------------------------------------------------------- #
# 跨域身份匹配
# --------------------------------------------------------------------------- #
def identity_matches(truth_xy, pred_xy, gate_m: float = 0.30) -> dict:
    """把预测按 truth 身份匹配（复用 metrics 的 known-K 匹配口径）。"""
    m = match_by_gate(truth_xy, pred_xy, gate_m=gate_m)
    by_identity = {}
    for ti, pi, dist in m["pairs"]:
        by_identity[int(ti)] = {"truth_index": int(ti), "pred_index": int(pi),
                                "distance_m": float(dist),
                                "truth_xy": [float(truth_xy[ti][0]), float(truth_xy[ti][1])],
                                "pred_xy": [float(pred_xy[pi][0]), float(pred_xy[pi][1])]}
    return {"by_identity": by_identity, "tp": int(m["tp"]),
            "unmatched_truth": list(m["fn"]), "unmatched_pred": list(m["fp"]),
            "n_truth": len(truth_xy), "n_pred": len(pred_xy)}


def own_errors(truth_xy, pred_xy) -> dict:
    """单个域内：TP 条件 e_xy/e_r/e_theta（复用 metrics）。"""
    r = evaluate_predictions(truth_xy, pred_xy)
    return {"errors": r["errors"], "tp": r["tp"], "fn": r["fn"], "fp": r["fp"],
            "n_truth": r["n_truth"], "n_predictions": r["n_predictions"],
            "ospa": r["ospa"], "ospa2": r["ospa2"]}


def cross_domain_peak_delta(sim_truth, sim_pred, hw_truth, hw_pred, identity_map=None,
                            truth_tol_m: float = 1e-9) -> dict:
    """同一真值身份的跨域峰位差 + 各自相对 **本域 truth** 误差 + 未配对计数。

    * 先在 **sim 域**按 sim truth 匹配 sim pred -> identity -> sim 估计；
    * 再在 **hw 域**按 hw truth 匹配 hw pred -> identity -> hw 估计；
    * 对两域都匹配到同一 identity 的点算 ``delta = ||p_sim - p_hw||``；
      每个域各自的 ``e_xy/e_r/e_theta`` **用各自 truth 坐标**（sim 用 sim truth，hw 用 hw truth）。
    * **identity 成立前提**：两域 truth 为**同序同点**（容差内）或显式给 ``identity_map``；
      否则直接 ``Q4Error``，**不得**按索引硬套。

    ``identity_map``：``{sim_identity_index: hw_identity_index}``；默认要求两域 truth 逐点一致。
    """

    def _same_ordered(a, b, tol=truth_tol_m):
        if len(a) != len(b):
            return False
        for (ax, ay), (bx, by) in zip(a, b):
            if math.hypot(ax - bx, ay - by) > tol:
                return False
        return True

    if identity_map is None:
        if not _same_ordered(sim_truth, hw_truth):
            raise Q4Error("sim/hw truth 顺序或坐标不一致，identity 索引不成立；"
                          "请确保同几何同 ordered points，或显式提供 identity_map")
        identity_map = {i: i for i in range(len(sim_truth))}
    identity_map = {int(k): int(v) for k, v in identity_map.items()}

    sim_m = identity_matches(sim_truth, sim_pred)
    hw_m = identity_matches(hw_truth, hw_pred)
    items = []
    for sim_id in sorted(sim_m["by_identity"]):
        hw_id = identity_map.get(sim_id)
        if hw_id is None or hw_id not in hw_m["by_identity"]:
            continue
        s = sim_m["by_identity"][sim_id]
        h = hw_m["by_identity"][hw_id]
        sx, sy = s["pred_xy"]
        hx, hy = h["pred_xy"]
        s_tx, s_ty = sim_truth[sim_id]
        h_tx, h_ty = hw_truth[hw_id]
        sr_t, sth_t = polar_of(s_tx, s_ty)
        hr_t, hth_t = polar_of(h_tx, h_ty)
        sim_r, sim_th = polar_of(sx, sy)
        hw_r, hw_th = polar_of(hx, hy)
        items.append({
            "identity": int(sim_id),
            "hw_identity": int(hw_id),
            "sim_truth_xy": [float(s_tx), float(s_ty)],
            "hw_truth_xy": [float(h_tx), float(h_ty)],
            "sim_pred_xy": [float(sx), float(sy)],
            "hw_pred_xy": [float(hx), float(hy)],
            "delta_sim_hw_peak_m": float(math.hypot(sx - hx, sy - hy)),
            "sim_e_xy_m": float(math.hypot(sx - s_tx, sy - s_ty)),
            "hw_e_xy_m": float(math.hypot(hx - h_tx, hy - h_ty)),
            "sim_e_r_m": float(abs(sim_r - sr_t)),
            "hw_e_r_m": float(abs(hw_r - hr_t)),
            "sim_e_theta_deg": float(abs(wrap_deg(sim_th - sth_t))),
            "hw_e_theta_deg": float(abs(wrap_deg(hw_th - hth_t))),
        })
    return {
        "n_common_identities": len(items),
        "items": items,
        "sim_unmatched_truth": sim_m["unmatched_truth"],
        "sim_unmatched_pred": sim_m["unmatched_pred"],
        "hw_unmatched_truth": hw_m["unmatched_truth"],
        "hw_unmatched_pred": hw_m["unmatched_pred"],
        "identity_check": ("same_ordered_truth" if identity_map == {i: i for i in range(len(sim_truth))}
                           else "explicit_identity_map"),
        "truth_source": "per_domain_own_truth",
        "note": "peak delta 只在两域都按各自 truth 匹配到同一 identity 后计算；各自误差用各自 truth；不得用 sim-hw 最佳配对冒充身份对应",
    }
