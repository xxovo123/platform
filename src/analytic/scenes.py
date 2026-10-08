#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""场景矩阵（E1/E2/E3/E4）构造：把协议参数展开成可执行的 case 规格。

关键约束（2026-09-26 澄清）
--------------------------
* clean_id 只由 geometry + source + coherence + power_diff（物理 seed 由此派生）决定，
  不含 SNR / L / family。因此：
    - E1_B 的 9 个 SNR 共享同一条无噪轨迹（无噪 cache 跨 SNR 复用）；
    - E4 的多个 L 档共享同一 128 clean 序列（运行期取前 L 切片）；
    - E4 的 S2 与 E1 的 S2（同几何/同 source/同相干）自动去重为同一 clean 序列。
* 噪声 trial seed 用 (clean_id, trial)：同一 clean_id 跨 SNR/跨 L 共用同一批标准复高斯抽样
  （跨 SNR 只缩放 sigma；跨 L 取前缀），该策略显式披露（protocol.noise.noise_prefix_policy）。
* 离网格真值：只对 E1（及共享其 S2 几何的 E4）施加统一笛卡尔微平移；E2 不施加笛卡尔平移
  （否则破坏“横向同 r / 径向同角”），E2 用 theta=12 度等非网格角度已离网。
* E2 斜向按严格欧氏间距 d 构造：p0 +/- (d/(2*sqrt2))*(u_r + u_t)，u_r/u_t 为正交单位向量，
  两目标距离精确等于 d（名义即实际）。

坐标 p = (r*sin(theta), -r*cos(theta))。每个 spec 字段见 README。
"""

from __future__ import annotations

import math

import sim_common as sc

OFFGRID_SHIFT_M = [0.017, -0.013]


def _polar_list(pairs_rt):
    return [[float(sc.polar_to_xy(r, th)[0]), float(sc.polar_to_xy(r, th)[1])] for r, th in pairs_rt]


def _shift(xy):
    dx, dy = float(OFFGRID_SHIFT_M[0]), float(OFFGRID_SHIFT_M[1])
    return [[float(x) + dx, float(y) + dy] for x, y in xy]


def _snr_tag(snr_db: float) -> str:
    return "snr%+03d" % int(round(float(snr_db)))


def clean_id_for(source: str, k: int, targets_xy, *, coherence: float = 0.0,
                 power_diff_db: float = 0.0) -> str:
    """clean_id = f(source, k, geometry, coherence, power_diff)；不含 SNR/L/family。"""
    key = {
        "source": str(source), "k": int(k),
        "targets_xy": [[round(float(x), 6), round(float(y), 6)] for x, y in targets_xy],
        "coherence": round(float(coherence), 6),
        "power_diff_db": round(float(power_diff_db), 6),
    }
    return "cln_" + sc.sha256_text(sc.canonical_json(key))[:12]


def _mk(case_id, family, layer, scene, tags, targets_xy, targets_polar, *, L, snr_db, trials,
        L_clean=None, resolution_distance_m=None, coherence: float = 0.0, power_diff_db: float = 0.0,
        optional: bool = False, extra=None):
    src = "fdtd" if layer == "B" else "analytic"
    cid = clean_id_for(src, len(targets_xy), targets_xy, coherence=coherence,
                       power_diff_db=power_diff_db)
    spec = {
        "case_id": case_id, "clean_id": cid, "family": family, "layer": layer, "source": src,
        "scene": scene, "tags": list(tags), "k": len(targets_xy),
        "targets_polar": [[float(r), float(t)] for r, t in targets_polar],
        "targets_xy": [[float(x), float(y)] for x, y in targets_xy],
        "L": int(L), "L_clean": int(L_clean if L_clean is not None else max(int(L), sc.SNAPSHOTS)),
        "snr_db": float(snr_db), "trials": int(trials),
        "resolution_distance_m": (None if resolution_distance_m is None else float(resolution_distance_m)),
        "coherence": float(coherence), "power_diff_db": float(power_diff_db),
    }
    if optional:
        spec["optional"] = True
    if extra:
        spec.update(extra)
    return spec


# --------------------------------------------------------------------------- #
# E1：受控布局 + SNR 扫描（统一离网格微平移；clean 跨 SNR 复用）
# --------------------------------------------------------------------------- #
def e1_specs(layer: str = "A", *, trials: int | None = None) -> list:
    from protocol.sim_protocol import E1_LAYOUTS, E1_SNR_DB, E1_TRIALS, E1_B_TRIALS
    fam = "E1_A" if layer == "A" else "E1_B"
    n_tr = int(trials if trials is not None else (E1_TRIALS if layer == "A" else E1_B_TRIALS))
    specs = []
    for layout, rt in E1_LAYOUTS.items():
        xy = _shift(_polar_list(rt))
        for snr in E1_SNR_DB:
            cid = "%s_%s_%s" % (fam, layout, _snr_tag(snr))
            specs.append(_mk(cid, fam, layer, "E1", ["layout=%s" % layout], xy, rt,
                             L=sc.SNAPSHOTS, snr_db=snr, trials=n_tr))
    return specs


# --------------------------------------------------------------------------- #
# E2：间距 / 可分性（严格欧氏 d；不施加笛卡尔平移）
# --------------------------------------------------------------------------- #
def e2_pair(direction: str, d_m: float, *, r0: float, theta0_deg: float):
    """返回 (targets_xy, targets_polar, actual_euclidean_distance_m)。三方向严格 d。"""
    d = float(d_m)
    th0 = math.radians(float(theta0_deg))
    u_r = (math.sin(th0), -math.cos(th0))          # 径向单位向量
    u_t = (math.cos(th0), math.sin(th0))           # 切向单位向量
    if direction == "transverse":
        half = math.degrees(math.asin(min(1.0, (d / 2.0) / float(r0))))
        rt = [[float(r0), float(theta0_deg) + half], [float(r0), float(theta0_deg) - half]]
        xy = _polar_list(rt)
    elif direction == "radial":
        rt = [[float(r0) - d / 2.0, float(theta0_deg)], [float(r0) + d / 2.0, float(theta0_deg)]]
        xy = _polar_list(rt)
    elif direction == "oblique":
        p0 = sc.polar_to_xy(r0, theta0_deg)
        step = d / (2.0 * math.sqrt(2.0))
        off = tuple(step * (u_r[i] + u_t[i]) for i in range(2))
        xy = [[p0[0] + off[0], p0[1] + off[1]], [p0[0] - off[0], p0[1] - off[1]]]
        rt = [list(sc.xy_to_polar(x, y)) for x, y in xy]
    else:
        raise ValueError("未知方向 %r（只支持 transverse/radial/oblique）" % direction)
    actual = math.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1])
    return xy, rt, float(actual)


def e2_specs(layer: str = "A", *, trials: int | None = None) -> list:
    from protocol.sim_protocol import (E2_BASE_R0_M, E2_BASE_THETA0_DEG, E2_SPACINGS_M,
                                       E2_SNR_DB, E2_TRIALS, E2_B_SPACINGS_M, E2_B_SNR_DB,
                                       E2_B_TRIALS)
    fam = "E2_A" if layer == "A" else "E2_B"
    spacings = E2_SPACINGS_M if layer == "A" else E2_B_SPACINGS_M
    snrs = E2_SNR_DB if layer == "A" else E2_B_SNR_DB
    n_tr = int(trials if trials is not None else (E2_TRIALS if layer == "A" else E2_B_TRIALS))
    specs = []
    for direction in ("transverse", "radial", "oblique"):
        for d in spacings:
            xy, rt, actual = e2_pair(direction, d, r0=E2_BASE_R0_M, theta0_deg=E2_BASE_THETA0_DEG)
            if abs(actual - float(d)) > 1e-9:
                raise ValueError("E2 %s d=%.3f 实际间距 %.9f != 名义：构造错误"
                                 % (direction, d, actual))
            for snr in snrs:
                cid = "%s_%s_d%03d_%s" % (fam, direction, int(round(d * 1000)), _snr_tag(snr))
                specs.append(_mk(cid, fam, layer, "E2",
                                 ["direction=%s" % direction, "d_nominal_m=%.3f" % d,
                                  "d_actual_m=%.6f" % actual],
                                 xy, rt, L=sc.SNAPSHOTS, snr_db=snr, trials=n_tr,
                                 resolution_distance_m=actual,
                                 extra={"d_nominal_m": float(d), "d_actual_m": float(actual)}))
    return specs


# --------------------------------------------------------------------------- #
# E3：可选挑战（默认关闭）
# --------------------------------------------------------------------------- #
def e3_optional_specs(*, trials: int | None = None) -> list:
    from protocol.sim_protocol import (E3_OPT_CORR, E3_OPT_POWER_DIFF_DB, E3_OPT_TRIALS,
                                       E4_BASE_LAYOUT, E1_LAYOUTS)
    rt = E1_LAYOUTS[E4_BASE_LAYOUT]
    xy = _shift(_polar_list(rt))
    n_tr = int(trials if trials is not None else E3_OPT_TRIALS)
    specs = []
    for pd in E3_OPT_POWER_DIFF_DB:
        for corr in E3_OPT_CORR:
            cid = "E3_A_OPT_%s_pd%02d_corr%03d_snr%+03d" % (E4_BASE_LAYOUT, int(pd),
                                                            int(round(corr * 100)), 10)
            specs.append(_mk(cid, "E3_A_OPT", "A", "E3",
                             ["optional=true", "power_diff_db=%.1f" % pd, "corr=%.2f" % corr],
                             xy, rt, L=sc.SNAPSHOTS, snr_db=10.0, trials=n_tr,
                             coherence=float(corr), power_diff_db=float(pd), optional=True))
    return specs


# --------------------------------------------------------------------------- #
# E4：快照敏感性（同一 128 clean 序列的前 L 切片；与 E1 同几何时 clean_id 去重）
# --------------------------------------------------------------------------- #
def e4_specs(layer: str = "A", *, trials: int | None = None) -> list:
    from protocol.sim_protocol import (E1_LAYOUTS, E1_SNR_DB, E4_BASE_LAYOUT, E4_L_LIST,
                                       E4_TRIALS, E4_B_TRIALS)
    fam = "E4_A" if layer == "A" else "E4_B"
    rt = E1_LAYOUTS[E4_BASE_LAYOUT]
    xy = _shift(_polar_list(rt))
    n_tr = int(trials if trials is not None else (E4_TRIALS if layer == "A" else E4_B_TRIALS))
    specs = []
    for snr in E1_SNR_DB:
        base = "%s_%s_%s" % (fam, E4_BASE_LAYOUT, _snr_tag(snr))
        for L in E4_L_LIST:
            specs.append(_mk("%s_L%03d" % (base, L), fam, layer, "E4",
                             ["layout=%s" % E4_BASE_LAYOUT, "L=%d" % L], xy, rt,
                             L=int(L), L_clean=sc.SNAPSHOTS, snr_db=snr, trials=n_tr))
    return specs


# --------------------------------------------------------------------------- #
# HW：硬件实测点位族（2026-09-28 用户指令）。点位 = 硬件 214 窗真值的 7 个实测位置
# （q4/out/hw_cir/*.truth.json 聚合），坐标系与硬件一致（目标在 -y 半平面）。
# 参数不进 protocol（SNR/L/trials 在本文件内定义）——protocol json 与既有 cache 不受影响。
# --------------------------------------------------------------------------- #
HW_POINTS_XY = (
    (0.0, -1.0),            # 1 m 正前
    (0.643, -0.766),        # 1 m 右 40°
    (-0.643, -0.766),       # 1 m 左 40°
    (0.0, -2.0),            # 2 m 正前
    (1.286, -1.532),        # 2 m 右 40°
    (-1.286, -1.532),       # 2 m 左 40°
    (0.0, -3.0),            # 3 m 正前
)
HW_SNR_DB = 10.0
HW_TRIALS = 5


def hw_specs(layer: str = "A", *, trials: int | None = None) -> list:
    fam = "HW_A" if layer == "A" else "HW_B"
    n_tr = int(trials if trials is not None else HW_TRIALS)
    specs = []
    for i, (x, y) in enumerate(HW_POINTS_XY, 1):
        r, th = sc.xy_to_polar(float(x), float(y))
        specs.append(_mk("%s_P%d" % (fam, i), fam, layer, "HW",
                         ["hw_point=%d" % i, "src=hardware_truth_214win"],
                         [[float(x), float(y)]], [[float(r), float(th)]],
                         L=sc.SNAPSHOTS, snr_db=HW_SNR_DB, trials=n_tr))
    return specs


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
LAYER_A_FAMILIES = ("E1_A", "E2_A", "E4_A", "HW_A")
LAYER_B_FAMILIES = ("E1_B", "E2_B", "E4_B", "HW_B")


def specs_for(family: str, *, trials: int | None = None, include_e3_optional: bool = False) -> list:
    fam = str(family).upper()
    if fam in ("E1_A", "E1_B"):
        return e1_specs(fam[-1], trials=trials)
    if fam in ("E2_A", "E2_B"):
        return e2_specs(fam[-1], trials=trials)
    if fam in ("E4_A", "E4_B"):
        return e4_specs(fam[-1], trials=trials)
    if fam in ("HW_A", "HW_B"):
        return hw_specs(fam[-1], trials=trials)
    if fam == "E3":
        return e3_optional_specs(trials=trials) if include_e3_optional else []
    if fam == "ALL_A":
        out = []
        for f in LAYER_A_FAMILIES:
            out.extend(specs_for(f, trials=trials))
        if include_e3_optional:
            out.extend(e3_optional_specs(trials=trials))
        return out
    if fam == "ALL_B":
        out = []
        for f in LAYER_B_FAMILIES:
            out.extend(specs_for(f, trials=trials))
        return out
    raise ValueError("未知 family %r" % family)


def resolve_clean_specs(specs) -> dict:
    """按 clean_id 去重（同几何/同 source/同相干即同一 clean 序列，跨 SNR/L 复用）。"""
    out = {}
    for s in specs:
        cid = s["clean_id"]
        rec = {"clean_id": cid, "k": s["k"], "targets_xy": s["targets_xy"],
               "targets_polar": s["targets_polar"], "L_clean": s["L_clean"],
               "scene": s["scene"], "source": s["source"],
               "coherence": float(s.get("coherence", 0.0)),
               "power_diff_db": float(s.get("power_diff_db", 0.0))}
        if cid in out:
            prev = out[cid]
            if (prev["targets_xy"] != rec["targets_xy"] or prev["k"] != rec["k"]
                    or prev["L_clean"] != rec["L_clean"]):
                raise ValueError("clean_id=%s 参数不一致（同几何必须共享同一 clean 序列）" % cid)
        else:
            out[cid] = rec
    return out


def amplitude_vector(k: int, power_diff_db: float):
    """多目标功率差（dB）：第 i 个目标幅度 = 10^(-i*pd/20)（相对首目标）。"""
    pd = float(power_diff_db)
    if pd == 0.0:
        return None
    return [10.0 ** (-i * pd / 20.0) for i in range(int(k))]


__all__ = ["e1_specs", "e2_specs", "e2_pair", "e3_optional_specs", "e4_specs", "specs_for",
           "resolve_clean_specs", "amplitude_vector", "clean_id_for", "OFFGRID_SHIFT_M",
           "LAYER_A_FAMILIES", "LAYER_B_FAMILIES"]
