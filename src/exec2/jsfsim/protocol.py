#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机器协议构建（规范 2026-09-29）。

唯一数值/场景依据：`大修执行_20260923/仿真执行与论文叙事交接规范_20260929.md`。
本模块只产出**可 JSON 序列化的协议字典**并做自校验；不建 cache、不跑方法、不跑 Meep。

与冻结包的差异（本模块刻意覆盖）：
  * S1–S5 使用规范精确极坐标，**不施加旧平移 `(0.017,-0.013)`**（offgrid_shift_m = [0,0]）；
  * clean key 只由 source/K/精确几何/相干/power_diff 决定 —— 新版站位天然给出不同 clean_id，
    因此不复用旧（带平移）解析与 FDTD 缓存；
  * seed 规范改为规范 §6 的 canonical-JSON + SHA256（salt `jsf_sim_20260929_v1`），
    不使用冻结包的 `seed_int("sim20260926:clean", ...)` 口径。
"""

from __future__ import annotations

import hashlib
import json
import math

PROTOCOL_ID = "jsf-20260929"
PROTOCOL_VERSION = "jsf-20260929-noshift-bandfix-1"
SEED_SALT = "jsf_sim_20260929_v1"

# ---- §2 固定公共设置 ------------------------------------------------------- #
FC_HZ = 7.9872e9
BANDWIDTH_HZ = 5.0e8
BAND_HALF_HZ = 0.5 * BANDWIDTH_HZ
SAMPLE_INTERVAL_S = 1.0e-9
RAW_TAPS = 32
CROP_START = 0
CROP_LENGTH = 30
FFT_LENGTH = 32
SNAPSHOTS = 128
N_CHANNELS = 4
C0 = 299792458.0
SYSTEM_DELAY_SI = 0.0
EPSILON_R_BACKGROUND = 1.0
DIAG_LOAD_FACTOR = 1.0e-5
IS_FORWARD_BACKWARD = True
FOV_DEGREES = 180.0
GRID_STEP_M = 0.05
NMS_MIN_DIST_M = 0.10
PEAK_THRESHOLD_REL = 0.30
GATE_M = 0.30
RESOLUTION_ABS_GATE_M = 0.10
RESOLUTION_REL_DIVISOR = 3.0

GRID_X_MIN, GRID_X_MAX = -2.2, 2.2
GRID_Y_MIN, GRID_Y_MAX = -3.2, -0.55

#: 1Tx4Rx 受控几何（§2）。Q4 另用硬件真实 Tx–Rx，见 q4 协议段。
TX_POSITIONS_M = [[0.0, 0.0, 0.0]]
RX_POSITIONS_M = [[-0.038, 0.0, 0.0], [-0.019, 0.0, 0.0], [0.0, 0.0, 0.0], [0.019, 0.0, 0.0]]
PAIRS = [[0, 0], [0, 1], [0, 2], [0, 3]]
SPACING_M = 0.019

#: **新版关键**：不施加任何统一平移（规范 §3）。
OFFGRID_SHIFT_M = [0.0, 0.0]

# ---- §3 站位 S1–S5（精确极坐标 (r_m, theta_deg)） -------------------------- #
S1_S5_LAYOUTS = {
    "S1": [[1.8, 12.0]],
    "S2": [[1.8, 12.0], [1.8, -12.0]],
    "S3": [[1.5, 12.0], [2.1, 12.0]],
    "S4": [[1.5, -12.0], [2.1, 12.0]],
    "S5": [[1.5, -20.0], [1.8, 0.0], [2.1, 20.0]],
}

# ---- §3 SNR / trials ------------------------------------------------------- #
SNR_SCAN_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
E1_A_TRIALS = 200
E1_B_TRIALS = 50

# ---- §4 间距实验 ----------------------------------------------------------- #
E2_BASE_R0_M = 1.8
E2_BASE_THETA0_DEG = 12.0
E2_DIRECTIONS = ["transverse", "radial", "oblique"]
E2_A_SPACINGS_M = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
E2_A_SNR_DB = [0.0, 10.0, 20.0]
E2_A_TRIALS = 100
E2_B_SPACINGS_M = [0.30, 0.50, 1.00]
E2_B_SNR_DB = [10.0]
E2_B_TRIALS = 50

# ---- §5 快照实验（固定 S2） ------------------------------------------------ #
E4_LAYOUT = "S2"
E4_L_LIST = [10, 16, 32, 64, 128]
E4_A_SNR_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
E4_A_TRIALS = 100
E4_B_SNR_DB = [0.0, 10.0, 20.0]
E4_B_TRIALS = 50

# ---- §2 FDTD 层 B 物理参数 ------------------------------------------------ #
SCATTERER_RADIUS_M = 1.5 * C0 / FC_HZ          # ≈ 0.056327 m
MICRO_MOTION_SIGMA_M = 0.002
MICRO_MOTION_RADIUS_M = 0.01
BACKGROUND_MODE_B = "empty_field"              # 受控层用空场相减（§2）
FDTD_RESOLUTION_PX_PER_M = 350.0               # 由 §10 收敛比较选定，见 convergence 段
FDTD_RUN_TIME_MEEP = 20.0
PML_THICKNESS_LAMBDA = 2.0
FDTD_COURANT = 0.5

# ---- §7 Q4 ---------------------------------------------------------------- #
Q4_TARGETS = {
    "G-370": [[-0.643, -0.766]],
    "G-380": [[-1.286, -1.532]],
    "G-410": [[0.0, -1.0]],
    "G-420": [[0.0, -2.0]],
    "G-430": [[0.0, -3.0]],
    "G-450": [[0.643, -0.766]],
    "G-460": [[1.286, -1.532]],
    "C2-1": [[-1.286, -1.532], [0.0, -2.0]],
    "C2-3": [[-0.643, -0.766], [0.643, -0.766]],
    "C2-4": [[-0.643, -0.766], [1.286, -1.532]],
    "C2-5": [[-1.286, -1.532], [0.0, -1.0]],
    "C2-6": [[-0.643, -0.766], [0.0, -1.0]],
    "C3-1": [[-1.286, -1.532], [0.0, -2.0], [1.286, -1.532]],
    "C3-3": [[-0.643, -0.766], [0.0, -2.0], [0.643, -0.766]],
    "C3-4": [[-0.643, -0.766], [0.0, -1.0], [1.286, -1.532]],
}
Q4_PRECHECK_CONFIGS = ["G-420", "C2-1", "C3-1"]
Q4_TX_POSITIONS_M = [[-0.019, 0.0, 0.0], [0.019, 0.0, 0.0]]
Q4_RX_POSITIONS_M = [[-0.019, 0.0, -0.02], [0.0, 0.0, -0.02]]
Q4_PAIRS = [[0, 0], [0, 1], [1, 0], [1, 1]]
Q4_STATES_TOTAL = 160
Q4_BURNIN_STATES = 32
Q4_KEPT_STATES = 128
Q4_EMA_ALPHA = 0.25
Q4_HW_WINDOWS = 205
Q4_HW_INDOOR = 103
Q4_HW_OUTDOOR = 102

# ---- §10 计算预算上界 ------------------------------------------------------ #
# 规范 §5 明确：E4_A 的 SNR 档 = −10、−5、0、5、10、15、20、25、30（9 档）、每条件 100 trial；
# E4_B = 0、10、20（3 档）、每条件 50 trial；L 档 5 个（L=10,16,32,64,128）。
# 据此 E4_A = 5×9×100 = 4500，E4_B = 5×3×50 = 750。
# **规范 §10 表内 E4_A 写作 2250、合计 21900，与该任务自身定义不自洽**（2250 只对应 5 档 SNR，
# 与 §5 的 9 档冲突；且 E4_B 按 3 档已得 750，其 5L×3SNR×50 算式是对的）。
# 本实现按 §5 的显式定义取 9 档，并把该算术差异原样记录，不自行改条件：
bATCH_E4_A_DEFINITION_CONFLICT = {
    "spec_table_value": 2250, "spec_total_table_value": 21900,
    "derived_from_section5": 4500, "derived_total": 23400,
    "resolution": ("按 §5 显式定义（9 档 SNR）实现；差异上报，未擅自改条件。"
                   "若主代理裁定应取 §10 表的 2250，需先修改 §5 的 SNR 列表并重生成协议。"),
}
BATCH_UPPER_BOUND = {
    "E1_A": 5 * 9 * 200,
    "E1_B": 5 * 9 * 50,
    "E2_A": 3 * 8 * 3 * 100,
    "E2_B": 3 * 3 * 1 * 50,
    "E4_A": 1 * 5 * len(E4_A_SNR_DB) * 100,
    "E4_B": 1 * 5 * len(E4_B_SNR_DB) * 50,
}
BATCH_TOTAL_UPPER_BOUND = sum(BATCH_UPPER_BOUND.values())


class ProtocolError(RuntimeError):
    """协议自校验失败（显式失败，不静默放行）。"""


# --------------------------------------------------------------------------- #
# 坐标 / 频点
# --------------------------------------------------------------------------- #
def polar_to_xy(r_m: float, theta_deg: float):
    th = math.radians(float(theta_deg))
    return (float(r_m) * math.sin(th), -float(r_m) * math.cos(th))


def xy_to_polar(x: float, y: float):
    return (math.hypot(float(x), float(y)), math.degrees(math.atan2(float(x), -float(y))))


def fft_freq_axis_hz():
    """np.fft.fftfreq(fft_length, Ts) + fc，与冻结后端同一构造。"""
    return [k / (FFT_LENGTH * SAMPLE_INTERVAL_S) + FC_HZ for k in _fftfreq_indices()]


def _fftfreq_indices():
    return [k if k <= FFT_LENGTH // 2 else k - FFT_LENGTH for k in range(FFT_LENGTH)]


def in_band_indices():
    """带内频点索引（原始 fftfreq 索引升序，非单调）。"""
    out = []
    for i, k in enumerate(_fftfreq_indices()):
        f = k / (FFT_LENGTH * SAMPLE_INTERVAL_S) + FC_HZ
        if FC_HZ - BAND_HALF_HZ - 1e-6 <= f <= FC_HZ + BAND_HALF_HZ + 1e-6:
            out.append(i)
    return out


def frequencies_hz():
    axis = fft_freq_axis_hz()
    return [axis[i] for i in in_band_indices()]


# --------------------------------------------------------------------------- #
# 网格
# --------------------------------------------------------------------------- #
def grid_axis(lo, hi, step):
    n = int(round((float(hi) - float(lo)) / float(step)))
    return [round(float(lo) + float(step) * i, 9) for i in range(n + 1)]


def build_grid():
    return grid_axis(GRID_X_MIN, GRID_X_MAX, GRID_STEP_M), grid_axis(GRID_Y_MIN, GRID_Y_MAX, GRID_STEP_M)


# --------------------------------------------------------------------------- #
# seed 规范（§6）
# --------------------------------------------------------------------------- #
def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def seed_from(payload: dict) -> int:
    """SHA256(canonical_json(payload)) 前 8 字节 -> 无符号大端整数（不使用内置 hash）。"""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def clean_seed(physical_id: str) -> int:
    """clean seed 不含 SNR/L/trial。"""
    return seed_from({"salt": SEED_SALT, "kind": "clean", "physical_id": str(physical_id)})


def noise_seed(physical_id: str, trial: int) -> int:
    """noise seed 含 trial、不含 SNR/L/method。"""
    return seed_from({"salt": SEED_SALT, "kind": "noise", "physical_id": str(physical_id),
                      "trial": int(trial)})


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #
def _shift(xy, shift=OFFGRID_SHIFT_M):
    dx, dy = float(shift[0]), float(shift[1])
    return [[float(x) + dx, float(y) + dy] for x, y in xy]


def e2_pair(direction: str, d_m: float, *, r0=E2_BASE_R0_M, theta0_deg=E2_BASE_THETA0_DEG):
    """§4 三方向构造；返回 (xy, polar, 实际欧氏间距)。"""
    d = float(d_m)
    th0 = math.radians(float(theta0_deg))
    u_r = (math.sin(th0), -math.cos(th0))
    u_t = (math.cos(th0), math.sin(th0))
    if direction == "transverse":
        half = math.degrees(math.asin(min(1.0, (d / 2.0) / float(r0))))
        polar = [[float(r0), float(theta0_deg) + half], [float(r0), float(theta0_deg) - half]]
        xy = [list(polar_to_xy(r, t)) for r, t in polar]
    elif direction == "radial":
        polar = [[float(r0) - d / 2.0, float(theta0_deg)], [float(r0) + d / 2.0, float(theta0_deg)]]
        xy = [list(polar_to_xy(r, t)) for r, t in polar]
    elif direction == "oblique":
        p0 = polar_to_xy(r0, theta0_deg)
        step = d / (2.0 * math.sqrt(2.0))
        off = tuple(step * (u_r[i] + u_t[i]) for i in range(2))
        xy = [[p0[0] + off[0], p0[1] + off[1]], [p0[0] - off[0], p0[1] - off[1]]]
        polar = [list(xy_to_polar(x, y)) for x, y in xy]
    else:
        raise ProtocolError("未知方向 %r" % direction)
    actual = math.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1])
    return xy, polar, float(actual)


def clean_id_for(source: str, k: int, targets_xy, *, protocol_version: str = PROTOCOL_VERSION) -> str:
    """clean_id = f(source, K, 精确几何, 协议版本)。几何为精确 float64 展开后的坐标。"""
    key = {"source": str(source), "k": int(k),
           "targets_xy": [[float(x), float(y)] for x, y in targets_xy],
           "protocol_version": str(protocol_version)}
    return "cln_" + hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()[:12]


def _scene_record(*, task_id: str, family: str, layer: str, scene: str, k: int,
                  targets_xy, targets_polar, snr_db, trials, L, L_clean,
                  tags, resolution_distance_m=None, extra=None) -> dict:
    source = "fdtd" if layer == "B" else "analytic"
    cid = clean_id_for(source, k, targets_xy)
    rec = {
        "task_id": str(task_id), "clean_id": cid, "family": family, "layer": layer,
        "source": source, "scene": scene, "k": int(k),
        "targets_xy": [[float(x), float(y)] for x, y in targets_xy],
        "targets_polar": [[float(r), float(t)] for r, t in targets_polar],
        "snr_db": float(snr_db), "trials": int(trials), "L": int(L), "L_clean": int(L_clean),
        "resolution_distance_m": (None if resolution_distance_m is None else float(resolution_distance_m)),
        "tags": list(tags),
        "clean_seed": clean_seed(cid),
    }
    if extra:
        rec.update(extra)
    return rec


def snr_tag(snr_db) -> str:
    return "snr%+03d" % int(round(float(snr_db)))


def expand_e1(layer: str, trials: int):
    fam = "E1_A" if layer == "A" else "E1_B"
    out = []
    for layout in sorted(S1_S5_LAYOUTS):
        polar = S1_S5_LAYOUTS[layout]
        xy = _shift([list(polar_to_xy(r, t)) for r, t in polar])
        for snr in SNR_SCAN_DB:
            task_id = "%s_%s_%s" % (fam, layout, snr_tag(snr))
            out.append(_scene_record(task_id=task_id, family=fam, layer=layer, scene="E1",
                                     k=len(polar), targets_xy=xy, targets_polar=polar,
                                     snr_db=snr, trials=trials, L=SNAPSHOTS, L_clean=SNAPSHOTS,
                                     tags=["layout=%s" % layout],
                                     extra={"layout": layout}))
    return out


def expand_e2(layer: str, trials: int, spacings, snrs):
    fam = "E2_A" if layer == "A" else "E2_B"
    out = []
    for direction in E2_DIRECTIONS:
        for d in spacings:
            xy, polar, actual = e2_pair(direction, d)
            if abs(actual - float(d)) > 1e-9:
                raise ProtocolError("E2 %s d=%.3f 实际 %.12f 超差" % (direction, d, actual))
            for snr in snrs:
                task_id = "%s_%s_d%03d_%s" % (fam, direction, int(round(float(d) * 1000)), snr_tag(snr))
                out.append(_scene_record(task_id=task_id, family=fam, layer=layer, scene="E2",
                                         k=2, targets_xy=xy, targets_polar=polar, snr_db=snr,
                                         trials=trials, L=SNAPSHOTS, L_clean=SNAPSHOTS,
                                         tags=["direction=%s" % direction,
                                               "d_nominal_m=%.3f" % d, "d_actual_m=%.6f" % actual],
                                         resolution_distance_m=actual,
                                         extra={"direction": direction, "d_nominal_m": float(d),
                                                "d_actual_m": float(actual)}))
    return out


def expand_e4(layer: str, trials: int, snrs):
    """§5：固定 S2，同一 128 clean 序列取前 L 个（clean 去重为同一 physical_id）。"""
    fam = "E4_A" if layer == "A" else "E4_B"
    polar = S1_S5_LAYOUTS[E4_LAYOUT]
    xy = _shift([list(polar_to_xy(r, t)) for r, t in polar])
    out = []
    for snr in snrs:
        for L in E4_L_LIST:
            task_id = "%s_%s_%s_L%03d" % (fam, E4_LAYOUT, snr_tag(snr), L)
            out.append(_scene_record(task_id=task_id, family=fam, layer=layer, scene="E4",
                                     k=len(polar), targets_xy=xy, targets_polar=polar, snr_db=snr,
                                     trials=trials, L=L, L_clean=SNAPSHOTS,
                                     tags=["layout=%s" % E4_LAYOUT, "L=%d" % L],
                                     extra={"layout": E4_LAYOUT}))
    return out


TASKS = ("E1_A", "E1_B", "E2_A", "E2_B", "E4_A", "E4_B")


def expand_task(task: str, *, trials_override=None):
    if task not in TASKS:
        raise ProtocolError("未知任务 %r（可选 %s）" % (task, list(TASKS)))
    layer = task[-1]
    if task == "E1_A":
        return expand_e1("A", E1_A_TRIALS if trials_override is None else trials_override)
    if task == "E1_B":
        return expand_e1("B", E1_B_TRIALS if trials_override is None else trials_override)
    if task == "E2_A":
        return expand_e2("A", E2_A_TRIALS if trials_override is None else trials_override,
                         E2_A_SPACINGS_M, E2_A_SNR_DB)
    if task == "E2_B":
        return expand_e2("B", E2_B_TRIALS if trials_override is None else trials_override,
                         E2_B_SPACINGS_M, E2_B_SNR_DB)
    if task == "E4_A":
        return expand_e4("A", E4_A_TRIALS if trials_override is None else trials_override, E4_A_SNR_DB)
    return expand_e4("B", E4_B_TRIALS if trials_override is None else trials_override, E4_B_SNR_DB)


# --------------------------------------------------------------------------- #
# 协议字典
# --------------------------------------------------------------------------- #
def build_protocol() -> dict:
    gx, gy = build_grid()
    indices = in_band_indices()
    freqs = frequencies_hz()
    return {
        "kind": "jsf-sim-protocol",
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_spec": "大修执行_20260923/仿真执行与论文叙事交接规范_20260929.md",
        "method_scope": {"methods": ["this_method"],
                         "note": "只运行 JSF-MUSIC；外部仿真基线不接入、不运行。"},
        "fc_hz": FC_HZ, "bandwidth_hz": BANDWIDTH_HZ, "band_half_hz": BAND_HALF_HZ,
        "sample_interval_s": SAMPLE_INTERVAL_S, "raw_taps": RAW_TAPS,
        "crop_start": CROP_START, "crop_length": CROP_LENGTH, "fft_length": FFT_LENGTH,
        "snapshots": SNAPSHOTS, "n_channels": N_CHANNELS,
        "speed_of_light_m_s": C0, "system_delay_si": SYSTEM_DELAY_SI,
        "epsilon_r_background": EPSILON_R_BACKGROUND,
        "diag_load_factor": DIAG_LOAD_FACTOR, "is_forward_backward": IS_FORWARD_BACKWARD,
        "frequency_plan": {
            "axis_rule": "np.fft.fftfreq(fft_length=32, d=Ts) + fc_hz",
            "in_band_indices": [int(i) for i in indices],
            "n_freq_bins": len(indices),
            "frequencies_hz": [float(f) for f in freqs],
            "order": "fft_index_ascending_non_monotonic（原始频序；禁止自行排序）",
            "feature_order": "frequency-major then channel (index = f*C + ch)",
        },
        "band_limiting": {
            "marker": "BANDLIMIT-FIX-20260929",
            "primary_band_mode": "bandlimited_500mhz",
            "diagnostic_band_mode": "full_band_ideal",
            "mask": "S(f)=1 iff |f-fc| <= 250 MHz, else 0（IFFT 前显式作用于 H）",
            "call_site_required": "analytic/forward.py::raw_cir_from_coeffs 内 `H = H * s_mask`",
            "oob_energy_gate": 1e-12,
            "oob_gate_note": "截窗前带外能量比例 < 1e-12（双精度）；截窗后谱泄漏单独记录。",
        },
        "geometry_controlled": {
            "kind": "1Tx4Rx_controlled",
            "tx_positions_m": TX_POSITIONS_M, "rx_positions_m": RX_POSITIONS_M, "pairs": PAIRS,
            "spacing_m": SPACING_M,
            "coordinate": {"polar_to_xy": "p = (r*sin(theta), -r*cos(theta))",
                           "half_plane": "y < 0", "theta_sign": "theta>0 = +X 侧"},
            "not_hardware_geometry": True,
            "hardware_geometry_used_in": "q4（见 q4 段）",
        },
        "offgrid_shift_m": list(OFFGRID_SHIFT_M),
        "offgrid_shift_policy": ("新版取消旧 (0.017,-0.013) 统一平移；站位由精确极坐标 float64 展开。"
                                 "该改变已进入 clean key（协议版本入键）。"),
        "layouts_S1_S5": {k: [[float(r), float(t)] for r, t in v] for k, v in S1_S5_LAYOUTS.items()},
        "snr_scan_db": list(SNR_SCAN_DB),
        "trials": {"E1_A": E1_A_TRIALS, "E1_B": E1_B_TRIALS, "E2_A": E2_A_TRIALS,
                   "E2_B": E2_B_TRIALS, "E4_A": E4_A_TRIALS, "E4_B": E4_B_TRIALS},
        "e2": {"base_r0_m": E2_BASE_R0_M, "base_theta0_deg": E2_BASE_THETA0_DEG,
               "directions": list(E2_DIRECTIONS),
               "A": {"spacings_m": list(E2_A_SPACINGS_M), "snr_db": list(E2_A_SNR_DB),
                     "trials": E2_A_TRIALS},
               "B": {"spacings_m": list(E2_B_SPACINGS_M), "snr_db": list(E2_B_SNR_DB),
                     "trials": E2_B_TRIALS},
               "euclidean_tolerance_m": 1e-9},
        "e4": {"layout": E4_LAYOUT, "L_list": list(E4_L_LIST),
               "A": {"snr_db": list(E4_A_SNR_DB), "trials": E4_A_TRIALS},
               "B": {"snr_db": list(E4_B_SNR_DB), "trials": E4_B_TRIALS},
               "prefix_rule": "同一 128 clean 序列取前 L 个；不得复制快照凑数"},
        "search_grid": {
            "kind": "cartesian_2d",
            "grid_x": [float(gx[0]), float(gx[-1]), GRID_STEP_M],
            "grid_y": [float(gy[0]), float(gy[-1]), GRID_STEP_M],
            "n_points": [len(gx), len(gy)],
            "fov_degrees": FOV_DEGREES,
            "unified_static_domain": True,
        },
        "peak_rule": {
            "window": 3, "strict": False, "include_center": True, "boundary": "edge",
            "plateau_connectivity": 8, "plateau_representative": "plateau_max",
            "plateau_tie_break": "min_row_then_col",
            "threshold_mode": "linear_relative_to_global_finite_max",
            "threshold_rel": PEAK_THRESHOLD_REL,
            "nms_min_dist_m": NMS_MIN_DIST_M,
            "nms_accept_rule": "dist >= min_dist_m",
            "count_cap": "K", "allow_fewer_than_K": True,
            "frozen_implementation": "vendor/knownk_eval/peaks.py::nms_peaks",
            "note": "沿用冻结规则；不得为不同条件调阈值。",
        },
        "metrics": {
            "rmse_matching": "无门限最大基数最小平方一对一（位置/距离/方位共用配对）",
            "rmse_pooling": "跨 trial 汇 SSE 与配对数后开方（不平均各次 RMSE）",
            "dr_gate_m": GATE_M, "dr_rule": "先最大 TP 再最小总距离；FN=真值数-TP，FP=预测数-TP",
            "pairs_naming": "pairs/n_truth 只能称配对覆盖率，不叫 DR",
            "success_rule": "预测数=K 且所有目标在 0.30 m 内一对一命中",
            "e2_resolution_gate_m": "min(0.10, d/3)",
            "e2_success_rule": "输出恰好 2 点且存在一对一匹配使两点误差均 <= min(0.10,d/3)",
            "ci": {"success": "Wilson 95%", "rmse": "固定 seed trial 整组 bootstrap 2000 次"},
            "sd": "每条件 trial 级 RMSE 样本 SD（ddof=1）与有效 trial 数",
        },
        "noise": {
            "model": "complex_gaussian_same_variance_half_variance_each_part",
            "reference": "背景处理后、FFT 前的 30-tap 复数 CIR",
            "formula": "Ps = mean(|s|^2) over L x 4 x 30; Pn = Ps*10^(-SNR/10); n = sqrt(Pn/2)*(u+jv)",
            "per_L_scaling": "不同 L 以各自实际 clean 前缀功率定标",
            "not_rf_snr": True,
            "zero_signal_policy": "Ps = 0 或非有限 -> 标 failed，不静默加噪声底",
            "noise_draw": "每 clean_id/trial 先抽 [128,4,30] 标准噪声，改变 L 取前缀，改变 SNR 只缩放",
            "independence_note": "同 trial 跨参数是配对噪声试验，不能声称跨条件独立",
        },
        "seed": {
            "scheme": "SHA256(canonical_json(payload)) 前 8 字节无符号大端整数",
            "salt": SEED_SALT,
            "canonical_json": "sort_keys=True, separators=(',',':'), UTF-8",
            "clean_payload": "{salt, kind:'clean', physical_id}（不含 SNR/L/trial）",
            "noise_payload": "{salt, kind:'noise', physical_id, trial}（不含 SNR/L/method）",
            "no_builtin_hash": True, "order_independent": True,
        },
        "fdtd_layer_b": {
            "geometry": "2D xy；金属圆柱；全部目标同时置于同一次求解几何",
            "scatterer_radius_m": SCATTERER_RADIUS_M,
            "scatterer_radius_rule": "1.5*c/fc",
            "micro_motion": {"sigma_m": MICRO_MOTION_SIGMA_M, "radius_m": MICRO_MOTION_RADIUS_M,
                             "model": "有界随机游走，首状态为标称中心，逐状态逐坐标 SD=0.002 m，半径限制 0.01 m"},
            "background_mode": BACKGROUND_MODE_B,
            "meep": {"resolution_px_per_m": FDTD_RESOLUTION_PX_PER_M,
                     "run_time_meep": FDTD_RUN_TIME_MEEP,
                     "pml_thickness_lambda": PML_THICKNESS_LAMBDA, "courant": FDTD_COURANT},
            "no_independent_sum": "不得把独立目标回波叠加代替多目标 FDTD",
        },
        "q4": {
            "targets_xy_m": Q4_TARGETS,
            "precheck_configs": list(Q4_PRECHECK_CONFIGS),
            "n_configs": len(Q4_TARGETS),
            "tx_positions_m": Q4_TX_POSITIONS_M, "rx_positions_m": Q4_RX_POSITIONS_M,
            "pairs": Q4_PAIRS,
            "z_projection": "二维求解仅投影 xy；保留原 3D 坐标与投影差异记录；不得把 z=-0.02 当 y",
            "states": {"total": Q4_STATES_TOTAL, "burnin": Q4_BURNIN_STATES, "kept": Q4_KEPT_STATES},
            "background": {"mode": "ema_continuous", "alpha": Q4_EMA_ALPHA,
                           "rule": "首状态参考=该状态；后续先 output=cur-ref，再 ref=.75*ref+.25*cur；"
                                   "32 状态预热后取 128 输出；不每窗重置"},
            "no_extra_white_noise": True,
            "hw_windows": {"total": Q4_HW_WINDOWS, "indoor": Q4_HW_INDOOR, "outdoor": Q4_HW_OUTDOOR,
                           "source": "04_Q6/hardware/analysis_nofb_q2q3_20260927/per_window.csv 中 "
                                     "method=fb_on 且 in_205=True 的 window_id 冻结",
                           "not_214": True, "not_120": True},
            "time_reference": {
                "requirement": "显式保存 raw tap0 物理参考、crop 索引、CIR/CFR 剩余时移、导向时延参考、"
                               "已应用系统延迟及其来源",
                "alignment_test": "以无噪已知时延合成测试确认同一物理目标的处理链对齐",
                "forbidden": "不允许根据测试目标误差反向拟合 2 ns/相位补偿",
                "blocked_marker": "blocked_time_reference",
            },
        },
        "cache_policy": {
            "analytic": "带限修复 + 无平移站位 -> 旧 analytic clean 签名失配，必须重新生成",
            "fdtd": "clean key 含精确真值/运动源/带宽版本/时间原点/背景/网格/时长/代码指纹；"
                    "参数不同不得冒用旧结果",
            "noise_clean_separate": True,
        },
        "budget_upper_bound": {"by_task": dict(BATCH_UPPER_BOUND),
                               "total": BATCH_TOTAL_UPPER_BOUND,
                               "spec_table_total": 21900,
                               "e4_a_definition_conflict": dict(bATCH_E4_A_DEFINITION_CONFLICT),
                               "note": "去重前、仅本文方法的算法评估次数上界；E4_A 按 §5 显式 9 档 SNR。"},
        "failure_policy": {
            "keep_all_trials": True, "algorithm_failure_kept": True,
            "solver_failure_marked_blocked": True,
            "no_threshold_tuning": True, "no_scene_deletion": True,
            "pause_conditions": ["物理时间参考不清", "数值不收敛", "输入 hash 不符",
                                 "指标复算不一致", "缺必要硬件预测/谱", "运行资源不足"],
        },
    }


def self_check(proto: dict) -> dict:
    """协议自校验：无平移、带宽调用点标记、方法唯一、Q4 清单齐备、预算上界正确。"""
    checks = {}
    checks["no_offgrid_shift"] = list(proto["offgrid_shift_m"]) == [0.0, 0.0]
    checks["methods_only_this_method"] = list(proto["method_scope"]["methods"]) == ["this_method"]
    checks["n_freq_bins_17"] = int(proto["frequency_plan"]["n_freq_bins"]) == 17
    checks["band_marker"] = proto["band_limiting"]["marker"] == "BANDLIMIT-FIX-20260929"
    checks["q4_15_configs"] = len(Q4_TARGETS) == 15
    checks["q4_precheck_3"] = set(Q4_PRECHECK_CONFIGS) == {"G-420", "C2-1", "C3-1"}
    checks["q4_205"] = (Q4_HW_WINDOWS == 205 and Q4_HW_INDOOR + Q4_HW_OUTDOOR == 205)
    checks["budget_matches_section5_definition"] = (
        BATCH_UPPER_BOUND["E4_A"] == 1 * 5 * len(E4_A_SNR_DB) * 100
        and BATCH_UPPER_BOUND["E4_B"] == 1 * 5 * len(E4_B_SNR_DB) * 50
        and BATCH_TOTAL_UPPER_BOUND == sum(BATCH_UPPER_BOUND.values()))
    checks["e4_a_conflict_recorded"] = bool(bATCH_E4_A_DEFINITION_CONFLICT["spec_table_value"] == 2250)
    checks["s1_s5_no_shift_matches_spec"] = all(
        abs(proto["layouts_S1_S5"]["S1"][0][0] - 1.8) < 1e-15
        and abs(proto["layouts_S1_S5"]["S1"][0][1] - 12.0) < 1e-15 for _ in [0])
    checks["grid_89x54"] = tuple(proto["search_grid"]["n_points"]) == (89, 54)
    return {"passed": bool(all(checks.values())), "checks": checks,
            "failed": sorted(k for k, v in checks.items() if not v)}


def main(argv=None) -> int:
    import argparse
    import pathlib
    ap = argparse.ArgumentParser(prog="protocol_build")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    proto = build_protocol()
    checks = self_check(proto)
    payload = {"protocol": proto, "self_check": checks,
               "protocol_hash": hashlib.sha256(canonical_json(proto).encode("utf-8")).hexdigest()}
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(proto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (p.parent / "protocol_selfcheck.json").write_text(
            json.dumps({"self_check": checks,
                        "protocol_hash": payload["protocol_hash"],
                        "protocol_version": proto["protocol_version"]},
                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_hash": payload["protocol_hash"],
                      "self_check": checks}, ensure_ascii=False, indent=1))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
