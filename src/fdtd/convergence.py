#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛/时长前检：受控对照 + 判据冻结 + runner B 可核证据闸门。

纪律（2026-09-26 修正）
---------------------
* **只变 res / runtime**：domain、几何、目标、协议物理参数、链路参数全部保持与主档一致；
  每个组合写 ``scene_<tag>.json`` 并计算 ``controlled_hash``；所有组合的 controlled_hash 必须相同
  （否则直接报错），这样 runner B 才能把“前检结论”关联到具体冻结配置。
* **判据先冻结**：``criteria.json`` 已存在且与本次请求不同 → **拒覆**（不得事后改阈值）。
* **not_passed 是结论**：比较未达标时返回非零退出码，并原样写盘，不改判据、不改结论。
* ``check_evidence()``：给 runner B 的闸门入口 —— 正式 128/B 档求解前必须能证明“适用且 passed”。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import api, scenes as scenes_mod
from .protocol import PROTOCOL_VERSION, atomic_write_json, read_json, sha256_hex

DEFAULT_CRITERIA: Dict[str, Any] = {
    "criteria_version": 2,
    "max_peak_tap_shift": 1,
    "min_abs_correlation": 0.90,
    "max_shape_rel_l2": 0.30,
    "_note": ("判据运行前冻结（criteria_frozen_before_run=true）；不满足即 not_passed，不得改写。"
              "v2（2026-09-27，主代理诊断+用户批准）：raw rel_l2 对复 CIR 的全局复相位/幅度约定差"
              "敏感（两分辨率链路 dt/MF 参考对齐不同导致相位约定差），会把已收敛的场解形状误判为"
              "不收敛。诊断证据（3 场景缓存 CIR 实测）：raw rel_l2=1.76-2.12 而去单个最优复增益后 "
              "shape_rel_l2=0.207-0.214 全部<0.30，corr=0.977-0.978，峰位差=0，幅度比=0.89-1.17。"
              "故 v2 判据改用 shape_rel_l2（形状收敛性，物理主张所在），corr/峰移保留；原始 rel_l2 "
              "与对齐增益（幅度/相位）保留为披露字段，不删除不隐藏。"),
    "_v1": {"max_rel_l2": 0.30,
            "_note_v1": "v1 原始口径存档：max_rel_l2 对 raw 复 CIR 全局相位敏感，见 _note 修订原因"},
}


def controlled_hash(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any]) -> str:
    """受控配置指纹：含解析后的 domain/几何/目标/协议物理与链路参数，**不含 res / runtime**。"""
    meep = dict(scene_obj.get("meep", {}))
    meep.pop("resolution_px_per_m", None)
    meep.pop("run_time_meep", None)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "physics": {k: cfg[k] for k in sorted(cfg) if k in (
            "fc_hz", "bandwidth_hz", "sample_interval_s", "raw_taps", "crop_start", "fft_length",
            "aa_cutoff_hz", "aa_pad_min_factor", "match_filter_mode", "c0_m_per_s")},
        "geometry": dict(scene_obj.get("geometry", {})),
        "targets": dict(scene_obj.get("targets", {})),
        "domain": dict(scene_obj.get("domain", {})),
        "meep_without_res_runtime": meep,
        "dimensions": scene_obj.get("dimensions", 2),
        "length_unit_m": scene_obj.get("length_unit_m", 1.0),
    }
    return sha256_hex(payload)


def freeze_criteria(outdir: Path, criteria: Mapping[str, Any]) -> Dict[str, Any]:
    """写入判据；若已存在且不同 → 拒覆（返回最终生效的判据）。"""
    path = Path(outdir) / "criteria.json"
    payload = dict(criteria)
    payload["criteria_frozen_before_run"] = True
    if path.is_file():
        old = read_json(path)
        if json.dumps(old, sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise RuntimeError("已存在不同判据 %s：拒覆（不得事后改阈值）；如需新判据请换目录" % path)
        return old
    atomic_write_json(path, payload)
    return payload


def _combo_scene(base_scene: Mapping[str, Any], res: int, rt: float) -> Dict[str, Any]:
    scene = json.loads(json.dumps({k: v for k, v in base_scene.items()}))  # 深拷贝（保留已解析 domain）
    scene["meep"]["resolution_px_per_m"] = int(res)
    scene["meep"]["run_time_meep"] = float(rt)
    return scene


def dryrun_plan(protocol_path, scene, outdir, res_list: Sequence[int], rt_list: Sequence[float],
                max_states: int | None, background_mode: str | None, criteria_in) -> Dict[str, Any]:
    """只出预算与清单：写 criteria.json / scene_<tag>.json / convergence_plan.json，不求解。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    criteria = dict(DEFAULT_CRITERIA)
    if criteria_in:
        criteria = dict(read_json(criteria_in)) if Path(criteria_in).is_file() else json.loads(criteria_in)
    criteria = freeze_criteria(outdir, criteria)
    base = scenes_mod.load_scene(scene)
    cfg = api._load_cfg(protocol_path, base)
    base["domain"] = scenes_mod.resolve_domain(base, fc_hz=float(cfg["fc_hz"]))
    chash = controlled_hash(cfg, base)
    items: List[Dict[str, Any]] = []
    for res in res_list:
        for rt in rt_list:
            combo = _combo_scene(base, res, rt)
            if controlled_hash(cfg, combo) != chash:
                raise RuntimeError("受控配置被 res/runtime 之外的因素改变（res=%s rt=%s）" % (res, rt))
            tag = "res%d_rt%g" % (int(res), float(rt))
            p = outdir / ("scene_%s.json" % tag)
            atomic_write_json(p, combo)
            pl = api.plan_run(protocol_path, str(p), max_states=max_states, background_mode=background_mode)
            items.append({"tag": tag, "resolution_px_per_m": int(res), "run_time_meep": float(rt),
                          "scene_file": str(p), "cost": pl["cost"], "controlled_hash": chash,
                          "delay_reference_2way_3m_s": 2.0 * 3.0 / 2.99792458e8,
                          "delay_note": "3 m 双程约 20 ns；runtime 需覆盖 20 ns + 脉冲 4 ns"})
    plan = {"scene": base.get("scene_id"), "criteria": criteria, "controlled_hash": chash,
            "baseline": items[0]["tag"], "items": items, "n_runs": len(items), "dryrun": True,
            "note": "只是 250/350 与时长对照清单与预算；正式运行不在本轮",
            "generated_at": api._now_iso()}
    atomic_write_json(outdir / "convergence_plan.json", plan)
    return plan


# ---------------------------------------------------------------------------
# validation_scope / 报告 schema（2026-09-26 增补：供 adapter 与 runner gate 消费）
# ---------------------------------------------------------------------------
def validation_scope(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """adapter 可消费的最小适用范围：几何 profile + 覆盖范围 + 已测数值设置。"""
    return {
        "geometry_profile": geometry_profile(scene_obj),
        "k_targets": int(len(scene_obj["targets"].get("nominal_centers_m", []))),
        "max_path_m": float(scenes_mod.meep_target_support_m(scene_obj)),
        "domain_bounds_m": {"x": scene_obj["domain"]["inner_bounds_m"]["x"],
                            "y": scene_obj["domain"]["inner_bounds_m"]["y"],
                            "size_m": scene_obj["domain"]["size_m"]},
        "resolution_px_per_m": float(scene_obj["meep"]["resolution_px_per_m"]),
        "run_time_meep": float(scene_obj["meep"]["run_time_meep"]),
        "background_mode": str(scene_obj.get("background", {}).get("mode", "empty_field")),
    }


def physics_block(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """完整物理口径（不是只 scene 名）：频率/带宽/采样/截窗/几何/脉冲/抗混叠/背景。"""
    return {
        "fc_hz": float(cfg["fc_hz"]), "bandwidth_hz": float(cfg["bandwidth_hz"]),
        "sample_interval_s": float(cfg["sample_interval_s"]),
        "raw_taps": int(cfg["raw_taps"]), "crop_start": int(cfg["crop_start"]),
        "fft_length": int(cfg["fft_length"]), "aa_cutoff_hz": float(cfg["aa_cutoff_hz"]),
        "aa_pad_min_factor": float(cfg["aa_pad_min_factor"]),
        "match_filter_mode": str(cfg["match_filter_mode"]), "c0_m_per_s": float(cfg["c0_m_per_s"]),
        "pulse_formula_id": __import__("fdtd.protocol", fromlist=["x"]).PULSE_FORMULA_ID,
        "geometry": dict(scene_obj["geometry"]),
        "background": dict(scene_obj.get("background", {})),
    }


def canonical_hash(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any]) -> str:
    """canonical hash：physics + meep 实测设置 + domain + 链路版本（与 controlled_hash 并列保留）。"""
    from .protocol import CHAIN_VERSION
    return sha256_hex({
        "chain_version": CHAIN_VERSION, "physics": physics_block(cfg, scene_obj),
        "meep": dict(scene_obj["meep"]), "domain": dict(scene_obj["domain"]),
        "length_unit_m": scene_obj.get("length_unit_m", 1.0),
    })


def formal_report_fields(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any], *,
                         formal: bool, controlled: str) -> Dict[str, Any]:
    unit = __import__("fdtd.solver", fromlist=["x"])
    return {
        "scene_id": str(scene_obj.get("scene_id")),
        "scene_file_stem_is_not_id": True,
        "formal": bool(formal),
        "validation_scope": validation_scope(cfg, scene_obj),
        "physics": physics_block(cfg, scene_obj),
        "meep_tested": {
            "resolution_px_per_m": float(scene_obj["meep"]["resolution_px_per_m"]),
            "run_time_meep": float(scene_obj["meep"]["run_time_meep"]),
            "courant": float(scene_obj["meep"].get("courant", 0.5)),
            "pml_thickness_lambda": float(scene_obj["meep"].get("pml_thickness_lambda", 2.0)),
        },
        "controlled_hash": controlled,
        "canonical_hash": canonical_hash(cfg, scene_obj),
    }


def strict_params(cfg: Mapping[str, Any], scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """必须**严格相等**的部分：只含物理 profile（不含目标标称坐标）。

    目标标称坐标**不进** strict（否则每个 geometry 都得各做一次收敛，违背“少量预检”）；
    坐标差异只由 representative_coverage 的 max_path/domain 范围声明覆盖。
    数值方向不做推断：测过 350 不能证明 250，res/runtime 走 tested 组合 strict member。
    """
    tgt = target_profile(scene_obj)
    return {
        "physics": physics_block(cfg, scene_obj),
        "geometry": dict(scene_obj["geometry"]),
        "meep_strict": {"courant": float(scene_obj["meep"].get("courant", 0.5)),
                        "pml_thickness_lambda": float(scene_obj["meep"].get("pml_thickness_lambda", 2.0)),
                        "dimensions": int(scene_obj.get("dimensions", 2)),
                        "length_unit_m": float(scene_obj.get("length_unit_m", 1.0))},
        "target_profile": tgt,
    }


def geometry_profile(scene_obj: Mapping[str, Any]) -> str:
    """几何 profile：**tx/rx/pairs + 投影模式**的 canonical hash（不用 scene_id，故同几何 A/B 可共用一份报告）。"""
    g = dict(scene_obj["geometry"])
    return sha256_hex({"tx": g.get("tx_positions_m"), "rx": g.get("rx_positions_m"),
                       "pairs": g.get("pairs"),
                       "projection_mode": (g.get("projection") or {}).get("mode")})[:16]


def target_profile(scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """目标物理 profile：**去掉 seed / L / 标称坐标**（否则每 scene 一份，违背“少量预检”）。

    保留 model / sigma / radius（数组按“全部相同则归一为标量，否则给 min-max 声明”）/ 材料 / 余量；
    K（目标数）与坐标范围另由 validation_scope 的 k_targets / max_path_m / domain 作为**范围声明**。
    """
    tgt = dict(scene_obj.get("targets", {}))
    mm = dict(tgt.get("micro_motion", {}))

    def _norm(v):
        if isinstance(v, (list, tuple)):
            vals = [float(x) for x in v]
            return vals[0] if len(set(vals)) == 1 else [min(vals), max(vals)]
        return v

    return {"model": mm.get("model"), "sigma_m": _norm(mm.get("sigma_m")),
            "motion_radius_m": _norm(mm.get("radius_m")),
            "target_radius_m": tgt.get("radius_m"), "material": tgt.get("material"),
            "overlap_margin_m": tgt.get("overlap_margin_m")}


def formal_ok(scene_obj: Mapping[str, Any], resolution: float, run_time_meep: float,
              notphysical: bool | None = None) -> Tuple[bool, List[str]]:
    """formal 只能在“真实、非 tiny、主分辨率、足够 runtime”时成立（不得被低 res smoke 刷成 formal）。"""
    why: List[str] = []
    sid = str(scene_obj.get("scene_id", "")).lower()
    if ("smoke" in sid) or ("tiny" in sid):
        why.append("scene_id 含 smoke/tiny：禁止 formal")
    np_flag = scene_obj.get("notphysical")
    if np_flag is True or notphysical is True:
        why.append("notphysical=true：禁止 formal")
    if float(resolution) not in (250.0, 350.0):
        why.append("主分辨率必须是 250 或 350（得到 %s）" % resolution)
    if float(run_time_meep) + 1e-9 < 20.0:
        why.append("runtime 不足 20 meep（得到 %s）" % run_time_meep)
    return (len(why) == 0), why


def tested_combos(items: Sequence[Mapping[str, Any]], passed_tags: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    """报告里**实际比较且通过**的 (res, runtime) 组合；主 run 必须是其中之一（strict member）。"""
    by_tag = {str(i["tag"]): (int(i["resolution_px_per_m"]), float(i["run_time_meep"])) for i in items}
    keys = [str(t) for t in passed_tags] if passed_tags is not None else list(by_tag)
    return [{"tag": t, "resolution_px_per_m": by_tag[t][0], "run_time_meep": by_tag[t][1]}
            for t in keys if t in by_tag]


def representative_coverage(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """代表覆盖（只是合理范围声明，**不是万能证明**）：记录各代表场景的 profile/max_path/domain。"""
    return {
        "kind": "representative_range_only",
        "note": "代表场景覆盖只用于范围合理性判断；不可替代 tested 组合的 strict member 判定",
        "entries": [{"scene_id": e.get("scene_id"),
                     "geometry_profile": (e.get("validation_scope") or {}).get("geometry_profile"),
                     "max_path_m": (e.get("validation_scope") or {}).get("max_path_m"),
                     "domain_bounds_m": (e.get("validation_scope") or {}).get("domain_bounds_m"),
                     "tested_combos": e.get("tested_combos")} for e in entries],
    }


def range_covers(entry_scope: Mapping[str, Any], want_scope: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """范围覆盖判定（必须真判，不能只报字段）：want 的最远路径与域必须落在代表的声明范围内。"""
    why: List[str] = []
    e_path = entry_scope.get("max_path_m")
    w_path = want_scope.get("max_path_m")
    if e_path is None or w_path is None:
        why.append("缺 max_path_m 声明（无法判覆盖）")
    elif float(w_path) > float(e_path) + 1e-9:
        why.append("want max_path_m=%.4f 超出代表声明 %.4f（更远目标需另做前检）" % (w_path, e_path))
    eb = (entry_scope.get("domain_bounds_m") or {})
    wb = (want_scope.get("domain_bounds_m") or {})
    for ax in ("x", "y"):
        e_iv, w_iv = eb.get(ax), wb.get(ax)
        if not e_iv or not w_iv:
            why.append("缺 domain_bounds_m.%s" % ax)
            continue
        if float(w_iv[0]) < float(e_iv[0]) - 1e-9 or float(w_iv[1]) > float(e_iv[1]) + 1e-9:
            why.append("want 域 %s=[%.3f,%.3f] 超出代表域 [%.3f,%.3f]"
                       % (ax, w_iv[0], w_iv[1], e_iv[0], e_iv[1]))
    return (len(why) == 0), why




def compare_cir(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        raise ValueError("比较对象形状不一致：%s vs %s" % (a.shape, b.shape))
    va, vb = a.reshape(-1), b.reshape(-1)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    corr = float(abs(np.vdot(va, vb)) / (na * nb)) if na > 0 and nb > 0 else None
    rel = float(np.linalg.norm(va - vb) / na) if na > 0 else None
    shift = np.abs(np.abs(a).argmax(axis=-1) - np.abs(b).argmax(axis=-1))
    # shape 口径（v2）：移除单个最优复增益（幅度+全局相位）后的残差。
    # s* = <a,b>/||a||^2 使 ||b - s*a|| 最小；CIR 绝对相位/幅度是链路约定（dt/MF 对齐），
    # 不是网格收敛属性，故形状收敛性用去增益残差度量；增益本身如实披露。
    shape_rel_l2 = None
    gain = None
    if na > 0 and nb > 0:
        g = complex(np.vdot(va, vb) / (na * na))
        shape_rel_l2 = float(np.linalg.norm(vb - g * va) / na)
        gain = {"complex": {"real": float(g.real), "imag": float(g.imag)},
                "amplitude": float(abs(g)), "phase_rad": float(np.angle(g))}
    return {"abs_correlation": corr, "rel_l2": rel, "max_peak_tap_shift": int(shift.max()),
            "mean_peak_tap_shift": float(shift.mean()),
            "shape_rel_l2": shape_rel_l2, "gain_disclosure": gain}


def run_convergence(protocol_path, scene, outdir, res_list, rt_list, max_states, background_mode,
                    criteria_in, quiet: bool = False) -> Tuple[Dict[str, Any], int]:
    """真正跑对照；返回 (report, exit_code)，not_passed → 退出码 2。"""
    plan = dryrun_plan(protocol_path, scene, outdir, res_list, rt_list, max_states, background_mode, criteria_in)
    outdir = Path(outdir)
    criteria = plan["criteria"]
    runs: Dict[str, np.ndarray] = {}
    for item in plan["items"]:
        tag = item["tag"]
        run_dir = outdir / tag
        api.generate_clean(protocol_path, item["scene_file"], run_dir, resume=True, max_states=max_states,
                           background_mode=background_mode, progress=(None if quiet else None))
        with np.load(run_dir / "clean_cir.npz", allow_pickle=False) as npz:
            runs[tag] = npz["raw_cir"]
    base_tag = plan["baseline"]
    comparisons: Dict[str, Any] = {}
    flags: List[bool] = []
    for tag, cir in runs.items():
        if tag == base_tag:
            continue
        diff = compare_cir(runs[base_tag], cir)
        if "max_shape_rel_l2" in criteria:
            # v2：形状收敛判据（去单个最优复增益残差）；raw rel_l2 仅披露
            ok = (diff["max_peak_tap_shift"] <= criteria["max_peak_tap_shift"]
                  and diff["abs_correlation"] is not None and diff["abs_correlation"] >= criteria["min_abs_correlation"]
                  and diff["shape_rel_l2"] is not None and diff["shape_rel_l2"] <= criteria["max_shape_rel_l2"])
        else:
            # v1 存档口径（不再默认使用）
            ok = (diff["max_peak_tap_shift"] <= criteria["max_peak_tap_shift"]
                  and diff["abs_correlation"] is not None and diff["abs_correlation"] >= criteria["min_abs_correlation"]
                  and diff["rel_l2"] is not None and diff["rel_l2"] <= criteria["max_rel_l2"])
        comparisons["%s_vs_%s" % (tag, base_tag)] = {"diff": diff, "passed": bool(ok)}
        flags.append(bool(ok))
    verdict = "passed" if flags and all(flags) else "not_passed"
    scene_obj = scenes_mod.load_scene(plan["items"][0]["scene_file"])
    cfg2 = api._load_cfg(protocol_path, scene_obj)
    scene_obj["domain"] = scenes_mod.resolve_domain(scene_obj, fc_hz=float(cfg2["fc_hz"]))
    ok_formal, formal_why = formal_ok(scene_obj, scene_obj["meep"]["resolution_px_per_m"],
                                      scene_obj["meep"]["run_time_meep"])
    report = {"scene_id": str(scene_obj.get("scene_id")),
              "formal": bool(ok_formal and verdict == "passed"),
              "formal_reasons": formal_why, "notphysical": False,
              "criteria": criteria, "controlled_hash": plan["controlled_hash"],
              "baseline": base_tag, "comparisons": comparisons, "verdict": verdict,
              "strict_params": strict_params(cfg2, scene_obj),
              "tested_combos": tested_combos(plan["items"]) if verdict == "passed" else [],
              "validation_scope": validation_scope(cfg2, scene_obj),
              "physics": physics_block(cfg2, scene_obj),
              "meep_tested": {"resolution_px_per_m": float(scene_obj["meep"]["resolution_px_per_m"]),
                              "run_time_meep": float(scene_obj["meep"]["run_time_meep"]),
                              "courant": float(scene_obj["meep"].get("courant", 0.5)),
                              "pml_thickness_lambda": float(scene_obj["meep"].get("pml_thickness_lambda", 2.0))},
              "items": plan["items"],
              "representative_coverage": representative_coverage([{"scene_id": scene_obj.get("scene_id"),
                                                                   "validation_scope": validation_scope(cfg2, scene_obj),
                                                                   "tested_combos": tested_combos(plan["items"])}]),
              "exit_code": 0 if verdict == "passed" else 2,
              "note": "verdict=not_passed 必须如实报告，不得改判据；tested_combos 只在 verdict=passed 时给出",
              "generated_at": api._now_iso()}
    atomic_write_json(outdir / "convergence.json", report)
    return report, (0 if verdict == "passed" else 2)


def check_evidence(evidence_dir, *, scene, protocol_path=None) -> Dict[str, Any]:
    """runner B 闸门：判断该目录下的收敛证据是否“适用且 passed”。

    ``applicable`` 要求 ``controlled_hash`` 与当前主档一致（同一 domain/几何/目标/协议/链路，
    只差 res/runtime）。返回 dict 供 runner 直接判断；**不得**在没有适用 passed 证据时开跑 128/B 档。
    """
    d = Path(evidence_dir)
    out: Dict[str, Any] = {"evidence_dir": str(d), "applicable": False, "passed": False, "reasons": []}
    rep_p = d / "convergence.json"
    idx_p = d / "convergence_manifest.json"
    tj_p = d / "task_convergence.json"
    if not rep_p.is_file() and not idx_p.is_file() and not tj_p.is_file():
        out["reasons"].append("缺少 convergence.json / convergence_manifest.json / task_convergence.json（未做过前检）")
        return out
    base = scenes_mod.load_scene(scene)
    cfg = api._load_cfg(protocol_path, base)
    base["domain"] = scenes_mod.resolve_domain(base, fc_hz=float(cfg["fc_hz"]))
    want_strict = strict_params(cfg, base)
    want_scope = validation_scope(cfg, base)
    want_res = float(base["meep"]["resolution_px_per_m"])
    want_rt = float(base["meep"]["run_time_meep"])
    out["want"] = {"resolution_px_per_m": want_res, "run_time_meep": want_rt,
                   "scene_id": base.get("scene_id")}
    # 任务级（QoI）证据路径：目录含 task_convergence.json 时按任务判据验收
    # （判据先冻结；strict_params 与当前场景一致；res_a == want_res；verdict=passed）。
    tj_p = d / "task_convergence.json"
    if tj_p.is_file():
        try:
            tj = read_json(tj_p)
        except Exception as exc:
            out["reasons"].append("task_convergence.json 读取失败：%s" % exc)
            return out
        crit = tj.get("criteria") or {}
        checks = {
            "frozen": crit.get("criteria_frozen_before_run") is True,
            "strict_params_match": (json.dumps(tj.get("strict_params"), sort_keys=True)
                                    == json.dumps(want_strict, sort_keys=True)),
            "res_a_eq_want": float(tj.get("res_a", -1)) == want_res,
            "verdict_passed": str(tj.get("verdict")) == "passed",
        }
        out["task_convergence"] = {"checks": checks,
                                   "agreement": tj.get("detection_pattern_agreement"),
                                   "mean_shift_m": tj.get("mean_pred_shift_m"),
                                   "criteria_version": crit.get("criteria_version")}
        if not all(checks.values()):
            out["reasons"].append("task_convergence 未通过验收：%s"
                                  % [k for k, v in checks.items() if not v])
            return out
        out["applicable"] = True
        out["passed"] = True
        out["matched_entry"] = "task_convergence"
        out["semantics"] = "任务级（QoI）网格收敛证据：定位预测一致性判据（判据先冻结）"
        return out
    entries = []
    if idx_p.is_file():
        entries = list(read_json(idx_p).get("entries", []))
    if rep_p.is_file():
        entries = [read_json(rep_p)] + entries
    for i, rep in enumerate(entries):
        tag = "entry[%d:%s]" % (i, rep.get("scene_id"))
        if not bool(rep.get("formal", False)):
            out["reasons"].append("%s formal=false（tiny/validation-only 或未达 formal 条件：%s）"
                                  % (tag, rep.get("formal_reasons")))
            continue
        crit = rep.get("criteria") or {}
        if crit.get("criteria_frozen_before_run") is not True:
            out["reasons"].append("%s criteria.criteria_frozen_before_run != true（判据未经冻结）" % tag)
            continue
        comps = rep.get("comparisons") or {}
        if not comps:
            out["reasons"].append("%s comparisons 为空（空报告不得通过）" % tag)
            continue
        if not all(bool((v or {}).get("passed")) for v in comps.values()):
            out["reasons"].append("%s 存在未通过的 comparison（不能只看 verdict 字符串）" % tag)
            continue
        if json.dumps(rep.get("strict_params"), sort_keys=True) != json.dumps(want_strict, sort_keys=True):
            out["reasons"].append("%s strict_params 不严格一致（physics/几何/PML/单位/目标已变）" % tag)
            continue
        combos = rep.get("tested_combos") or []
        hit = [c for c in combos
               if float(c["resolution_px_per_m"]) == want_res and float(c["run_time_meep"]) == want_rt]
        if not hit:
            out["reasons"].append("%s 目标组合 res=%g rt=%g 不在 tested_combos（strict member 不通过；"
                                  "测过更细/更长也不算，除非另行显式声明）" % (tag, want_res, want_rt))
            continue
        if str(rep.get("verdict")) != "passed":
            out["reasons"].append("%s verdict=%s" % (tag, rep.get("verdict")))
            continue
        ok_range, why_range = range_covers(rep.get("validation_scope") or {}, want_scope)
        if not ok_range:
            out["reasons"].append("%s 代表覆盖不足（范围声明不是通过证明）：%s" % (tag, why_range))
            continue
        out["applicable"] = True
        out["passed"] = True
        out["matched_entry"] = tag
        out["matched_combo"] = hit[0]
        break
    out["representative_coverage"] = representative_coverage(entries)
    out["semantics"] = "strict member：目标 res/runtime 必须在该报告 tested_combos 内且 verdict=passed；"
    out["semantics"] += "physics/geometry/PML/单位 严格相等；不做分辨率/时长方向推断"
    return out
