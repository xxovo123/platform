#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仿真实验 2026-09-26 公共工具：协议常量、几何/网格、前端（crop0->fft32->17 频点）、
噪声、确定性种子、参数哈希、原子写入与 run 目录管理。

本模块是协议层唯一事实来源：协议 JSON、adapter、analytic、runner 都从这里取常量与派生量，
避免各处硬编码不一致。

关键口径（与方案 仿真实验完整方案_20260926.md 对齐）
--------------------------------------------------
* 中心频率/带宽/采样：fc = 7.9872 GHz，B = 500 MHz，Ts = 1 ns（fs = 1 GSa/s）。
* raw 32 / 算法窗 30（2026-09-26 v2 口径）：clean 数据是 raw 32 tap（npz key=raw_cir，复数 [L,4,32]）；
  后端入口 crop_start=0（不裁前 2 个物理抽头）-> 窗 [0:30] 共 30 tap -> fft(n=32) 补零 -> 带内 17 频点。
  两者分别命名、分别记录（raw_cir vs X），不得混用。
* 频序：17 个带内频点按 np.fft.fftfreq 的原始索引序取出（非单调），特征序为
  频率优先、通道其次（index = f*C + ch），与硬件后端 / B0 包逐字一致。
* 几何（1Tx4Rx 受控主配置）：Tx = [0,0,0]；Rx x = [-0.038,-0.019,0,0.019]，y = z = 0；
  pairs = (0,i), i=0..3；间距 19 mm。与硬件虚拟通道/坐标同预算，但不是严格相同物理几何，
  也不是历史复原（硬件物理 Rx 含 z=-0.02 上下排，本协议 2D 不表达离面 z）。
* 坐标：目标位置 p = (r*sin(theta), -r*cos(theta))（视轴 -Y，theta>0 = +X 侧）。
* 统一静态域：grid_x = [-2.2,2.2] 步长 0.05；grid_y = [-3.2,-0.55] 步长 0.05；
  fov_degrees = 180（与硬件同）；距离环由 tap [crop_start, crop_start+窗长] 派生
  （v2 crop_start=0: [0, 4.49688687] m）。统一静态域，不为任何 scene 裁剪。
* rmax 与角步：B1/B2 角步按共同实际最大 r（不是名义 3 m）复算，见 domain_rmax_m / angle_step_deg。
* 噪声口径（本方案主张的新统一协议，非旧代码事实）：复高斯、同方差；参考信号 = 四通道无噪、
  经背景处理后的实际输入 CIR（30 tap 窗，起点 0）；功率按 全 L x 4ch x 30tap 全局均方。
* 种子：clean 信号序列只按 case_id 生成；噪声 trial 只按 (case_id, trial) 生成；与方法无关。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# BLAS 线程固定：必须在 import numpy 之前设置（与硬件侧同一 CPU/线程口径）
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
#: 预先存在的值（若不同则被强制覆盖为 "1"；覆盖记录进 provenance，不静默）
PRE_EXISTING_THREAD_ENV = {v: os.environ.get(v) for v in BLAS_THREAD_VARS}
for _var in BLAS_THREAD_VARS:
    os.environ[_var] = "1"
BLAS_THREAD_OVERRIDES = {v: PRE_EXISTING_THREAD_ENV[v] for v in BLAS_THREAD_VARS
                         if PRE_EXISTING_THREAD_ENV[v] not in (None, "1")}

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
WS = PKG_DIR.parents[1]
PROTOCOL_DIR = PKG_DIR / "protocol"
ANALYTIC_DIR = PKG_DIR / "analytic"
ADAPTER_DIR = PKG_DIR / "adapter"
VENDOR_DIR = PKG_DIR / "vendor"
SCRIPTS_DIR = PKG_DIR / "scripts"
METRICS_DIR = PKG_DIR / "metrics"

DEFAULT_DATA_ROOT = Path("/data1/zsy/paper_revision_sim_20260926")


def default_data_root() -> Path:
    """数据根：环境变量 SIM_DATA_ROOT 优先，其次 /data1/zsy/...，本地不存在时退回包内 .simdata。"""
    env = os.environ.get("SIM_DATA_ROOT")
    if env:
        return Path(env)
    if DEFAULT_DATA_ROOT.parent.exists():
        return DEFAULT_DATA_ROOT
    return PKG_DIR / ".simdata"


# --------------------------------------------------------------------------- #
# 协议常量（参数主档）
# --------------------------------------------------------------------------- #
C0 = 299792458.0
FC_HZ = 7.9872e9
BANDWIDTH_HZ = 5.0e8
SAMPLE_INTERVAL_S = 1.0e-9
RAW_TAPS = 32
CROP_START = 0  # 2026-09-26 v2：仿真不裁前 2 个物理抽头（时间原点 = 物理零延迟；v1 曾为 2）
N_TAPS_WINDOW = 30  # 算法窗长保持 30 tap（起点 crop_start；受冻结 c3 ref-delay 搜索窗 < 无模糊周期 32 ns 约束，且与 v1 派生量一致）
FFT_LENGTH = 32
SNAPSHOTS = 128
N_CHANNELS = 4
EPSILON_R_BACKGROUND = 1.0
SYSTEM_DELAY_SI = 0.0
DIAG_LOAD_FACTOR = 1.0e-5
IS_FORWARD_BACKWARD = True
FOV_DEGREES = 180.0
GRID_STEP_M = 0.05
NMS_MIN_DIST_M = 0.10
GATE_M = 0.30
OSPA_C_M = 0.30

GRID_X_MIN, GRID_X_MAX = -2.2, 2.2
GRID_Y_MIN, GRID_Y_MAX = -3.2, -0.55

# 1Tx4Rx 受控主几何（明确新配置；非历史复原，非严格等同硬件物理几何）
TX_POSITIONS_M = np.array([[0.0, 0.0, 0.0]], dtype=float)
RX_POSITIONS_M = np.array([[-0.038, 0.0, 0.0],
                           [-0.019, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.019, 0.0, 0.0]], dtype=float)
PAIRS = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
VIRTUAL_X_M = np.array([-0.038, -0.019, 0.0, 0.019], dtype=float)
SPACING_M = 0.019

GEOMETRY_NOTE = (
    "1Tx4Rx 受控主配置（2026-09-26 明确新配置）：Tx=[0,0,0]，Rx x=[-0.038,-0.019,0,0.019]，"
    "y=z=0，pairs=(0,i) i=0..3，保持 19 mm 间距。与硬件虚拟通道/坐标同预算（x 坐标与 19 mm 间距一致），"
    "但不是严格相同物理几何（硬件物理 Rx 含 z=-0.02 上下排，本协议 2D 不表达离面 z），"
    "也不是历史复原。"
)


# --------------------------------------------------------------------------- #
# 哈希 / 序列化
# --------------------------------------------------------------------------- #
def canonical_json(obj) -> str:
    """稳定 JSON（键排序、无多余空白、ensure_ascii=False）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_hash(*parts) -> str:
    """参数哈希：任意 JSON 可序列化对象 -> 稳定 sha256（配置严格匹配用）。"""
    return sha256_text("|".join(canonical_json(p) for p in parts))


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 原子写入 / run 目录（已有不同 run 拒覆）
# --------------------------------------------------------------------------- #
class SimError(RuntimeError):
    """显式失败（不静默降级）。"""


def atomic_write_bytes(path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (".tmp_%s_%d_%s" % (path.name, os.getpid(), time.time_ns()))
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path, obj) -> Path:
    return atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=1,
                                                sort_keys=False) + "\n").encode("utf-8"))


def atomic_write_npz(path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (".tmp_%s_%d_%s" % (path.name, os.getpid(), time.time_ns()))
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


RUN_MARKER = "run.json"
RUNID_ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"


def validate_runid(runid) -> str:
    """runid 必须是单一名字：不含路径分隔符、不是 '.'/'..'、不是绝对路径；否则显式失败。"""
    name = str(runid)
    if not name:
        raise SimError("runid 不能为空")
    if name in (".", ".."):
        raise SimError("runid 不能是 %r" % name)
    if name != Path(name).name:
        raise SimError("runid=%r 含路径分隔符：必须是单一名字" % name)
    if Path(name).is_absolute() or (len(name) > 1 and name[1] == ":"):
        raise SimError("runid=%r 是绝对路径：必须相对 runs/ 的单一名字" % name)
    bad = sorted(set(ch for ch in name if ch not in RUNID_ALLOWED))
    if bad:
        raise SimError("runid=%r 含非法字符 %s（允许 %s）" % (name, bad, RUNID_ALLOWED))
    return name


#: 数值 source 允许清单（**只**这些进入 config_hash；新增 test/审计/生成物不改变 run 身份）
NUMERICAL_SOURCE_DIRS = ("adapter", "analytic", "protocol", "metrics", "fdtd", "vendor")
NUMERICAL_SOURCE_FILES = ("sim_common.py", "run_experiments.py", "scripts/mem_child.py")
#: 运行数值的非 .py 资产（如冻结协议 JSON）也必须进 hash
NUMERICAL_SOURCE_NONPY = ("protocol/sim_protocol.json", "vendor/protocol/knownk_protocol.json")
#: 任何命中这些路径片段的文件一律排除（tests / 审计 / 生成 / 运行根）
NUMERICAL_EXCLUDE_PARTS = ("__pycache__", ".simdata", "tests", "verification", "deployment",
                           "_tmp", "figuresout", "q4out")


def numerical_source_fingerprint(root=None) -> dict:
    """**数值 source 允许清单**指纹：只含会影响数值结果的文件。

    排除 tests / verification / deployment / q4|figures 输出 / 运行根（runs、cache、results…）
    等生成或非数值路径，避免"新增测试/审计/输出"破坏 run 身份与续跑。
    """
    root = Path(root) if root else PKG_DIR
    manifest = {}
    for d in NUMERICAL_SOURCE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if any(part in rel.split("/") for part in NUMERICAL_EXCLUDE_PARTS):
                continue
            manifest[rel] = sha256_file(p)
    for rel in tuple(NUMERICAL_SOURCE_FILES) + tuple(NUMERICAL_SOURCE_NONPY):
        p = root / rel
        if p.is_file():
            manifest[rel] = sha256_file(p)
    agg = sha256_text(canonical_json(manifest))
    return {"root": str(root), "kind": "numerical_source_allowlist", "n_files": len(manifest),
            "aggregate": agg, "files": manifest,
            "allow_dirs": list(NUMERICAL_SOURCE_DIRS), "allow_files": list(NUMERICAL_SOURCE_FILES),
            "allow_nonpy": list(NUMERICAL_SOURCE_NONPY),
            "exclude_parts": list(NUMERICAL_EXCLUDE_PARTS),
            "note": "只用于 config_hash/续跑身份；final deployment manifest 另用全包 hash（full_package_fingerprint）。"}


def full_package_fingerprint(root=None) -> dict:
    """**全包**指纹（含 tests/verification/generated）：只用于 final deployment manifest，不进 config_hash。"""
    root = Path(root) if root else PKG_DIR
    manifest = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(part in rel.split("/") for part in ("__pycache__", ".simdata")):
            continue
        manifest[rel] = sha256_file(p)
    return {"root": str(root), "kind": "full_package", "n_files": len(manifest),
            "aggregate": sha256_text(canonical_json(manifest)), "files": manifest,
            "note": "final deployment manifest 用；**不得**用于 config_hash（会被测试/生成物扰动）。"}


def source_fingerprint(root=None, extra_files=()) -> dict:
    """兼容入口：返回数值 source 允许清单指纹（= numerical_source_fingerprint）。"""
    return numerical_source_fingerprint(root=root)


def config_hash_for(protocol_dict, *, extra_config=None, root=None, extra_files=()) -> str:
    """config_hash = 协议哈希 + 数值 source hash + 额外配置（都不含真值）。"""
    src = numerical_source_fingerprint(root=root)
    return param_hash({"protocol": protocol_dict, "source_aggregate": src["aggregate"]},
                      {"extra_config": extra_config or {}})


def _resolve_contained(base: Path, child: Path) -> Path:
    base_r = base.resolve()
    child_r = child.resolve()
    try:
        child_r.relative_to(base_r)
    except ValueError:
        raise SimError("路径越界：%s 不在 %s 之内" % (child_r, base_r)) from None
    return child_r


def ensure_run_dir(data_root, runid: str, *, config_hash: str, extra: dict | None = None) -> dict:
    """建立/校验 runs/<runid>/。已存在且 config_hash 不同 -> 显式拒绝（不默用旧 run）。

    runid 必须是单一名字；run_dir 解析后须落在 data_root/runs/ 内（containment）。
    extra 只写入 metadata，**不得**覆盖 config_hash / runid / subdirs 等核心字段。
    """
    data_root = Path(data_root)
    name = validate_runid(runid)
    runs_base = data_root / "runs"
    run_dir = _resolve_contained(runs_base, runs_base / name)
    marker = run_dir / RUN_MARKER
    record = {
        "kind": "sim-run",
        "runid": name,
        "config_hash": str(config_hash),
        "created_at": now_iso(),
        "data_root": str(data_root),
        "subdirs": {k: str(run_dir / k) for k in ("cache", "results", "figures", "logs", "provenance")},
        "metadata": dict(extra or {}),
    }
    if run_dir.exists():
        if not marker.is_file():
            raise SimError("run 目录已存在但缺少 %s：拒绝覆写半成品 %s" % (RUN_MARKER, run_dir))
        have = json.loads(marker.read_text(encoding="utf-8"))
        if str(have.get("config_hash")) != str(config_hash):
            raise SimError(
                "runid=%s 已存在且 config_hash 不同（已有 %s != 本次 %s）：拒绝覆写，请换 --runid"
                % (runid, str(have.get("config_hash"))[:12], str(config_hash)[:12]))
        return have
    for sub in ("cache", "results", "figures", "logs", "provenance"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, record)
    return record


def run_paths(data_root, runid: str) -> dict:
    name = validate_runid(runid)
    data_root = Path(data_root)
    run_dir = data_root / "runs" / name
    return {
        "data_root": Path(data_root), "run_dir": run_dir,
        "cache": run_dir / "cache", "results": run_dir / "results",
        "figures": run_dir / "figures", "logs": run_dir / "logs",
        "provenance": run_dir / "provenance",
    }


def make_runid(config_hash: str, *, stamp: str | None = None) -> str:
    stamp = stamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return "%s_%s" % (stamp, config_hash[:10])


# --------------------------------------------------------------------------- #
# 确定性种子（只按 case/trial，与方法无关）
# --------------------------------------------------------------------------- #
SEED_SALT_CLEAN = "sim20260926:clean"
SEED_SALT_NOISE = "sim20260926:noise"


def seed_int(*parts) -> int:
    """稳定 64-bit 种子：sha256 前 8 字节。"""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2 ** 63 - 1)


def clean_seed(case_id: str) -> int:
    """clean 信号序列种子：只按 case_id（同 scene 的所有 trial 共享同一 clean 信号）。"""
    return seed_int(SEED_SALT_CLEAN, case_id)


def trial_seed(case_id: str, trial: int) -> int:
    """噪声 trial 种子：只按 (case_id, trial)，与方法无关。"""
    return seed_int(SEED_SALT_NOISE, case_id, int(trial))


def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & ((1 << 63) - 1))


# --------------------------------------------------------------------------- #
# 几何 / 网格 / 坐标
# --------------------------------------------------------------------------- #
def polar_to_xy(r_m: float, theta_deg: float) -> tuple:
    """本协议坐标：p = (r*sin(theta), -r*cos(theta))（视轴 -Y，theta>0 = +X）。"""
    th = math.radians(float(theta_deg))
    return (float(r_m) * math.sin(th), -float(r_m) * math.cos(th))


def xy_to_polar(x: float, y: float) -> tuple:
    return (math.hypot(float(x), float(y)), math.degrees(math.atan2(float(x), -float(y))))


def grid_axis(lo: float, hi: float, step: float) -> np.ndarray:
    """含端点的等步网格（数值稳健）。"""
    n = int(round((float(hi) - float(lo)) / float(step)))
    axis = np.round(float(lo) + float(step) * np.arange(n + 1), 9)
    if abs(float(axis[-1]) - float(hi)) > 1e-9:
        axis = np.append(axis, float(hi))
    return axis


def build_grid() -> tuple:
    return (grid_axis(GRID_X_MIN, GRID_X_MAX, GRID_STEP_M),
            grid_axis(GRID_Y_MIN, GRID_Y_MAX, GRID_STEP_M))


def dist_per_tap_m() -> float:
    return SAMPLE_INTERVAL_S * C0 / 2.0


def distance_ring_m() -> tuple:
    """距离环由 tap [crop_start, crop_start+窗长] 派生（v2: [0,30] -> [0, 4.49688687] m）。"""
    dpt = dist_per_tap_m()
    return (float(CROP_START) * dpt, float(CROP_START + N_TAPS_WINDOW) * dpt)


def crop_offset_seconds() -> float:
    """当前后端 crop_start 引入的固定延迟参考偏移（秒）= crop_start * Ts。"""
    return float(CROP_START) * float(SAMPLE_INTERVAL_S)


def crop_offset_range_m() -> float:
    """crop 偏移换算到双程距离（米）= v * crop_start * Ts / 2；本协议约 0.29979 m。

    说明：主协议 clean raw 的时间原点 = **物理零延迟**（raw tap 0 对应 tau = 0），
    目标回波写在 raw tap = tau/Ts；当前后端裁掉前 crop_start 个 tap 后，数据等效延迟
    为 tau - crop_offset_seconds，导向流形仍用绝对 tau，于是产生本固定偏移。
    **这是如实披露的口径差异，主协议不做任何补偿**；仅诊断 profile（compensated，
    非正式、默认关闭）在生成端移除该偏移用于对照。
    """
    return float(C0) * crop_offset_seconds() / 2.0


def domain_mask(grid_x=None, grid_y=None, *, fov_deg: float = FOV_DEGREES,
                ring_m: tuple | None = None) -> np.ndarray:
    """统一静态域掩膜（y<0、|theta|<=fov/2、距离环）。与硬件/各方法 _in_domain 同口径。"""
    gx = build_grid()[0] if grid_x is None else np.asarray(grid_x, dtype=float)
    gy = build_grid()[1] if grid_y is None else np.asarray(grid_y, dtype=float)
    Xg, Yg = np.meshgrid(gx, gy)
    r = np.hypot(Xg, Yg)
    ang = np.degrees(np.arctan2(Xg, -Yg))
    ring = distance_ring_m() if ring_m is None else (float(ring_m[0]), float(ring_m[1]))
    return ((Yg < 0.0) & (np.abs(ang) <= 0.5 * abs(fov_deg) + 1e-9)
            & (r >= ring[0] - 1e-9) & (r <= ring[1] + 1e-9))


def domain_rmax_m(grid_x=None, grid_y=None, *, fov_deg: float = FOV_DEGREES) -> float:
    """共同实际最大 r（掩膜内像素的 max |p|）——B1/B2 角步必须用此值，不用名义 3 m。"""
    gx = build_grid()[0] if grid_x is None else np.asarray(grid_x, dtype=float)
    gy = build_grid()[1] if grid_y is None else np.asarray(grid_y, dtype=float)
    mask = domain_mask(gx, gy, fov_deg=fov_deg)
    if not mask.any():
        raise SimError("统一静态域掩膜为空：检查 grid / fov / ring")
    Xg, Yg = np.meshgrid(gx, gy)
    return float(np.max(np.hypot(Xg[mask], Yg[mask])))


def angle_step_deg(*, target_tangential_half_m: float = GRID_STEP_M / 2.0,
                   grid_x=None, grid_y=None, fov_deg: float = FOV_DEGREES) -> float:
    """B1/B2 角步（度）：令 rmax * dtheta <= 2*target_tangential_half（半格切向 <= 2.5 cm）。

    与 q6_search_profiles 的 S1 口径一致，但 rmax 按本次统一静态域复算（不沿用旧全域值）。
    """
    rmax = domain_rmax_m(grid_x, grid_y, fov_deg=fov_deg)
    return math.degrees(2.0 * float(target_tangential_half_m) / rmax)


def b1b2_half_deg() -> float:
    """与 B1/B2 angle_grid_deg 的 half 口径一致：min(90, fov/2, geom)。"""
    gx, gy = build_grid()
    geom = math.degrees(math.atan2(float(np.max(np.abs(gx))),
                                   max(float(np.min(np.abs(gy))), 1e-12)))
    return float(min(90.0, 0.5 * abs(FOV_DEGREES), geom))


def b1b2_angle_actual(step_deg: float | None = None) -> dict:
    """B1/B2 冻结端点规则 n = ceil(2*half/step)+1；返回实际步长与半格切向位移。"""
    half = b1b2_half_deg()
    step = angle_step_deg() if step_deg is None else float(step_deg)
    span = 2.0 * half
    n = int(math.ceil(span / step - 1e-12)) + 1
    actual = span / (n - 1) if n > 1 else 0.0
    rmax = domain_rmax_m()
    return {"half_deg": float(half), "n_points": int(n), "requested_step_deg": float(step),
            "actual_step_deg": float(actual), "rmax_m": float(rmax),
            "half_cell_tangential_m": float(rmax * math.radians(actual) / 2.0)}


def b1b2_tau_actual(*, distance_step_m: float = GRID_STEP_M) -> dict:
    ring = distance_ring_m()
    span = (2.0 * ring[0] / C0, 2.0 * ring[1] / C0)
    req = 2.0 * float(distance_step_m) / C0
    n = int(math.ceil((span[1] - span[0]) / req - 1e-12))
    actual = (span[1] - span[0]) / n if n else 0.0
    return {"n_points": int(n + 1), "requested_step_s": float(req), "actual_step_s": float(actual),
            "actual_range_step_m": float(C0 * actual / 2.0)}


def native_range_bin_m() -> float:
    """C3 原生 range bin = v * 1/(F*df) / 2；df = 1/(fft_length*Ts)。"""
    nf = n_freq_bins()
    df = 1.0 / (float(FFT_LENGTH) * SAMPLE_INTERVAL_S)
    return float(C0) * (1.0 / (float(nf) * df)) / 2.0


def c3_zeropad_factor(*, target_range_step_m: float = GRID_STEP_M) -> int:
    return max(1, int(math.ceil(native_range_bin_m() / float(target_range_step_m) - 1e-12)))


# --------------------------------------------------------------------------- #
# 频率轴 / 带内频点（原始频序）
# --------------------------------------------------------------------------- #
def raw_freq_axis() -> np.ndarray:
    """硬件后端频率轴：np.fft.fftfreq(fft_length, Ts) + fc。"""
    return np.fft.fftfreq(int(FFT_LENGTH), d=float(SAMPLE_INTERVAL_S)) + float(FC_HZ)


def freq_range_si() -> tuple:
    return (float(FC_HZ) - 0.5 * float(BANDWIDTH_HZ), float(FC_HZ) + 0.5 * float(BANDWIDTH_HZ))


def in_band_indices() -> np.ndarray:
    """带内频点索引（按 fftfreq 原始索引序，非单调）。"""
    axis = raw_freq_axis()
    lo, hi = freq_range_si()
    return np.where((axis >= lo) & (axis <= hi))[0]


def frequencies_hz() -> np.ndarray:
    return raw_freq_axis()[in_band_indices()]


def n_freq_bins() -> int:
    return int(in_band_indices().size)


# --------------------------------------------------------------------------- #
# 合并前端：raw32 -> crop_start=0 窗 30 tap -> fft32 -> 带内 17 频点 -> X (L, F*C)
# --------------------------------------------------------------------------- #
def crop_raw(raw_cir: np.ndarray) -> np.ndarray:
    """raw [L,C,32] -> cropped [L,C,30]（cropped = raw[:,:,crop_start : crop_start+30]）。

    时间原点（主协议，硬性）：clean raw 的 raw tap 0 对应**物理零延迟** tau = 0；
    目标回波写在 raw tap = tau/Ts。2026-09-26 v2 口径 crop_start=0：**不裁前 2 个物理抽头**，
    数据等效延迟 = tau（v1 曾 crop_start=2，引入 -2*Ts ≈ -0.29979 m 双程参考偏移，已弃用）。
    窗长维持 N_TAPS_WINDOW=30：受冻结 c3 ref-delay 搜索窗 < 无模糊周期（32 ns）约束；
    被丢弃的 tap 30/31（距离 >4.5 m，超出 rmax=3.88 m）在本协议全部场景中无目标能量。
    """
    raw = np.asarray(raw_cir)
    if raw.ndim != 3 or raw.shape[-1] != RAW_TAPS:
        raise SimError("raw_cir 必须是 [L,C,%d]，收到 shape=%s" % (RAW_TAPS, raw.shape))
    cs, nw = int(CROP_START), int(N_TAPS_WINDOW)
    if cs + nw > RAW_TAPS:
        raise SimError("crop_start=%d + 窗长 %d 超出 raw_taps=%d" % (cs, nw, RAW_TAPS))
    return np.ascontiguousarray(raw[:, :, cs:cs + nw])


def x_from_cropped(cropped: np.ndarray, *, ant_calib_phase=None) -> np.ndarray:
    """cropped [L,C,30] -> X [L,F*C]（fft(n=32) 补零 -> 带内 17 频点，频率优先通道其次）。

    与 B0 包 build_observation_matrix / 硬件后端逐字等价；ant_calib_phase 为 None 时不校相位
    （解析层无实测校准矢量）。等价性由 adapter/frontend.py 的 verify_vs_b0 逐格核对。
    """
    cropped = np.asarray(cropped, dtype=np.complex128)
    if cropped.ndim != 3:
        raise SimError("cropped 必须是 [L,C,n_taps]，收到 shape=%s" % (cropped.shape,))
    L, C, _n = cropped.shape
    if int(C) != int(N_CHANNELS):
        raise SimError("cropped 通道数 %d != 协议 C=%d" % (C, N_CHANNELS))
    if not np.all(np.isfinite(cropped)):
        raise SimError("cropped 含 NaN/Inf（应显式记为 failed，不静默处理）")
    cfr = np.fft.fft(cropped, n=int(FFT_LENGTH), axis=-1)
    idx = in_band_indices()
    data = cfr[:, :, idx]
    if ant_calib_phase is not None:
        phi = np.asarray(ant_calib_phase, dtype=float).reshape(-1)
        if phi.size != C:
            raise SimError("ant_calib_phase 长度 %d != C=%d" % (phi.size, C))
        data = data * np.exp(-1j * phi)[None, :, None]
    X = data.transpose(0, 2, 1).reshape(L, -1)
    return np.ascontiguousarray(X)


def build_x_from_raw(raw_cir: np.ndarray, *, ant_calib_phase=None) -> np.ndarray:
    return x_from_cropped(crop_raw(raw_cir), ant_calib_phase=ant_calib_phase)


# --------------------------------------------------------------------------- #
# 噪声（复高斯；全 L x 4ch x 30tap 窗归一，起点 0）
# --------------------------------------------------------------------------- #
def complex_gaussian(shape, rng: np.random.Generator) -> np.ndarray:
    """标准复高斯（实/虚各 N(0,1/2)，故 E|x|^2 = 1）。"""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)


def signal_power(cropped_clean: np.ndarray) -> float:
    """Ps = 全 L x C x n_taps 的全局均方功率（新统一协议）。"""
    arr = np.asarray(cropped_clean)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) ** 2))


def noise_sigma(cropped_clean: np.ndarray, snr_db: float) -> float:
    """按目标 SNR 折算每复样本噪声标准差 sigma，使 E|n|^2 = Ps / SNR_lin。"""
    ps = signal_power(cropped_clean)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    if snr_lin <= 0.0:
        raise SimError("SNR(linear) 必须为正")
    return float(math.sqrt(ps / snr_lin))


def add_complex_noise(cropped_clean: np.ndarray, snr_db: float,
                      rng: np.random.Generator) -> np.ndarray:
    """在 cropped 30-tap CIR 上施加复高斯噪声（最终 FFT 前的实际输入 CIR）。"""
    arr = np.asarray(cropped_clean, dtype=np.complex128)
    sigma = noise_sigma(arr, snr_db)
    return arr + sigma * complex_gaussian(arr.shape, rng)


def noise_unit_for(l_available: int, l_actual: int, seed: int) -> np.ndarray:
    """噪声前缀策略：按**可用长度** l_available 抽标准复高斯，再切 [:l_actual]。

    这样 E4 的 L=10 与 L=128 共享同一前 10 快照噪声（complex_gaussian 的实/虚分别是两段
    standard_normal；若直接对较小 shape 抽样，imag 段起点不同 -> 前缀会被破坏）。
    """
    if int(l_actual) > int(l_available):
        raise SimError("l_actual=%d > l_available=%d" % (l_actual, l_available))
    full = complex_gaussian((int(l_available), N_CHANNELS, int(N_TAPS_WINDOW)), rng_from(seed))
    return np.ascontiguousarray(full[: int(l_actual)])


def assemble_noisy_x(raw_cir, L_requested: int, snr_db: float, seed: int, *,
                     allow_shorter: bool = False, validation_only: bool = False):
    """raw[L_avail,4,32] + 噪声 -> (X, L_actual, diag)。

    **绝不可复制快照**：噪声与数据必须严格同形状。
    * allow_shorter=False（正式）：L_avail < L_requested -> 抛 SimError（不足即 blocked 拒跑）。
    * allow_shorter=True（诊断/validation-only）：L_actual = min(L_requested, L_avail)，噪声按
      L_actual 生成（**不做 broadcast**），并记录 L_requested / L_actual / formal=False。
    * 噪声前缀：一律按 L_avail 抽 unit 再切 [:L_actual]，使跨 L 的 unit 前缀一致（E4）。
    * diag.formal = (not validation_only) and (L_actual == L_requested)（不是仅看 L 相等）。
    """
    raw = np.asarray(raw_cir)
    if raw.ndim != 3 or raw.shape[-1] != RAW_TAPS:
        raise SimError("raw_cir 必须是 [L,4,%d]，收到 %s" % (RAW_TAPS, raw.shape))
    l_avail = int(raw.shape[0])
    l_req = int(L_requested)
    if l_avail < l_req and not allow_shorter:
        raise SimError("clean 快照不足：可用 %d < 请求 %d（正式档拒绝；绝不可复制快照）"
                       % (l_avail, l_req))
    l_actual = min(l_req, l_avail)
    cropped = crop_raw(raw[:l_actual])
    unit = noise_unit_for(l_avail, l_actual, seed)
    if cropped.shape != unit.shape:
        raise SimError("数据与噪声形状不一致（禁止 broadcast 复制快照）：%s vs %s"
                       % (tuple(cropped.shape), tuple(unit.shape)))
    X = x_from_cropped(cropped + noise_sigma(cropped, snr_db) * unit)
    if int(X.shape[0]) != int(l_actual):
        raise SimError("X 行数 %d != 实际快照数 %d" % (int(X.shape[0]), int(l_actual)))
    diag = {"L_requested": l_req, "L_actual": int(l_actual), "L_available": l_avail,
            "shorter_than_requested": bool(l_actual < l_req),
            "validation_only": bool(validation_only),
            "formal": bool((not validation_only) and l_actual == l_req and l_req > 0),
            "noise_prefix_policy": "unit drawn for L_available then sliced [:L_actual]",
            "data_noise_shape_equal": True}
    return X, int(l_actual), diag


# --------------------------------------------------------------------------- #
# 线程口径快照 / provenance
# --------------------------------------------------------------------------- #
def blas_env() -> dict:
    return {"values": {v: os.environ.get(v) for v in BLAS_THREAD_VARS},
            "forced_to_1": list(BLAS_THREAD_VARS),
            "pre_existing_overridden": dict(BLAS_THREAD_OVERRIDES),
            "override_note": "线程变量在 import numpy 之前被强制设为 1；pre_existing_overridden 记录被覆盖的原值"}


def provenance_block(*, script: str | None = None) -> dict:
    return {"created_at": now_iso(), "script": script or (sys.argv[0] if sys.argv else None),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "blas_threads": blas_env(), "platform": sys.platform, "cwd": str(Path.cwd())}


def add_package_to_syspath() -> None:
    p = str(PKG_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


__all__ = [
    "PKG_DIR", "WS", "DEFAULT_DATA_ROOT", "default_data_root", "run_paths", "ensure_run_dir",
    "make_runid", "param_hash", "canonical_json", "sha256_file", "sha256_text", "sha256_bytes",
    "atomic_write_bytes", "atomic_write_json", "atomic_write_npz", "SimError",
    "validate_runid", "source_fingerprint", "config_hash_for",
    "numerical_source_fingerprint", "full_package_fingerprint",
    "NUMERICAL_SOURCE_DIRS", "NUMERICAL_SOURCE_FILES", "NUMERICAL_SOURCE_NONPY",
    "BLAS_THREAD_OVERRIDES",
    "seed_int", "clean_seed", "trial_seed", "rng_from",
    "polar_to_xy", "xy_to_polar", "grid_axis", "build_grid", "distance_ring_m", "dist_per_tap_m",
    "crop_offset_seconds", "crop_offset_range_m",
    "domain_mask", "domain_rmax_m", "angle_step_deg", "b1b2_half_deg", "b1b2_angle_actual",
    "b1b2_tau_actual", "native_range_bin_m", "c3_zeropad_factor",
    "raw_freq_axis", "freq_range_si", "in_band_indices", "frequencies_hz", "n_freq_bins",
    "crop_raw", "x_from_cropped", "build_x_from_raw",
    "complex_gaussian", "signal_power", "noise_sigma", "add_complex_noise",
    "noise_unit_for", "assemble_noisy_x",
    "blas_env", "provenance_block", "add_package_to_syspath",
    "C0", "FC_HZ", "BANDWIDTH_HZ", "SAMPLE_INTERVAL_S", "RAW_TAPS", "CROP_START",
    "N_TAPS_WINDOW", "FFT_LENGTH",
    "SNAPSHOTS", "N_CHANNELS", "EPSILON_R_BACKGROUND", "SYSTEM_DELAY_SI", "DIAG_LOAD_FACTOR",
    "IS_FORWARD_BACKWARD", "FOV_DEGREES", "GRID_STEP_M", "NMS_MIN_DIST_M", "GATE_M", "OSPA_C_M",
    "GRID_X_MIN", "GRID_X_MAX", "GRID_Y_MIN", "GRID_Y_MAX",
    "TX_POSITIONS_M", "RX_POSITIONS_M", "PAIRS", "VIRTUAL_X_M", "SPACING_M", "GEOMETRY_NOTE",
]
