# -*- coding: utf-8 -*-
"""本地复现 run_task_convergence 崩溃：stub 生成器 + 真实评估路径，输出完整堆栈。"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

PKG = Path(r"D:\论文\小论文修改\大修执行_20260923\06_仿真实验_20260926")
sys.path.insert(0, str(PKG))

from adapter import scene_export
import analytic.scenes as scenes
import fdtd.api as fapi
import fdtd.task_convergence as ftc
from fdtd.protocol import atomic_write_npz

TMP = Path(r"D:\论文\小论文修改\大修执行_20260923\06_仿真实验_20260926\.simdata\tmp_repro_qoi")
TMP.mkdir(parents=True, exist_ok=True)

specs = scenes.specs_for("E2_B", trials=5)
smap = scene_export.scenes_for_specs(specs)
cid = "cln_3e0ec7541705"
s250 = TMP / (cid + "_250.json")
s350 = TMP / (cid + "_350.json")
sc250 = smap[cid]
sc350 = json.loads(json.dumps(sc250))
sc350["meep"]["resolution_px_per_m"] = 350.0
s250.write_text(json.dumps(sc250, ensure_ascii=False), encoding="utf-8")
s350.write_text(json.dumps(sc350, ensure_ascii=False), encoding="utf-8")

N = 16


def fake_generate_clean(config, scene, outdir, resume=True, max_states=None,
                        background_mode=None, workers=1, progress=None, **kw):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(scene).read_text(encoding="utf-8")) if not isinstance(scene, dict) else dict(scene)
    res = float(doc["meep"]["resolution_px_per_m"])
    rng = np.random.default_rng(int(res))
    K = len(doc["targets"]["nominal_centers_m"])
    raw = np.zeros((N, 4, 32), dtype=complex)
    for l in range(N):
        for ch in range(4):
            for t in range(K):
                d = 1.5 + 0.3 * t + 0.01 * l
                tap = int(d / 0.1499)
                if tap < 32:
                    raw[l, ch, tap] = np.exp(-1j * 2 * np.pi * 0.05 * l * ch)
    raw += 0.01 * (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))
    npz = outdir / "clean_cir.npz"
    atomic_write_npz(npz, raw_cir=raw.astype(np.complex128),
                     t_origin_s=np.array(0.0), cache_key=np.array("x" * 64))
    print("[fake_gen] res=%g -> %s" % (res, npz))
    return {"clean_npz": str(npz), "stats": {"solves_actual": 0, "cache_hits": {}},
            "outdir": str(outdir)}


fapi.generate_clean = fake_generate_clean

try:
    rep, rc = ftc.run_task_convergence(None, str(s250), str(s350), TMP / "task_out_v2",
                                       workers=1, quiet=False)
    print("verdict =", rep.get("verdict"), "| rc =", rc)
    print("agreement =", rep.get("detection_pattern_agreement"),
          "| mean_shift =", rep.get("mean_pred_shift_m"))
except Exception:
    traceback.print_exc()
    print("REPRODUCED: 崩溃复现成功（上方即完整堆栈）")
