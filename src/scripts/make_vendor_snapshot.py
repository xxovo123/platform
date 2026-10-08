#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把冻结的旧 pipeline / 基线方法**逐字节**复制进 vendor/，并记录来源路径 + sha256。

纪律：vendor 副本必须与本地源**byte-identical**（sha256 相等）；路径差异由
adapter/methods_loader.py 的 bootstrap/常量覆盖处理，**不改 vendored 源文件内容**。

用法::
    py = .venv/Scripts/python.exe
    & $py -B 大修执行_20260923/06_仿真实验_20260926/scripts/make_vendor_snapshot.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
WS = PKG_DIR.parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import sim_common as sc  # noqa: E402

VENDOR = PKG_DIR / "vendor"

# (源相对 WS, vendor 内相对路径) —— 只列数值/合同相关文件
SOURCES = [
    ("大修执行_20260923/04_Q6/common/q6_contract.py", "common/q6_contract.py"),
    ("大修执行_20260923/04_Q6/common/q6_search_profiles.py", "common/q6_search_profiles.py"),
    ("大修执行_20260923/04_Q6/common/q6_case_cache.py", "common/q6_case_cache.py"),
    ("大修执行_20260923/04_Q6/common/q6_methods_this_method.py", "common/q6_methods_this_method.py"),
    ("大修执行_20260923/04_Q6/common/q6_methods_b0_standard_music.py",
     "common/q6_methods_b0_standard_music.py"),
    # 方法放 hardware/methods/：冻结模块用 parents[2]/common/q6_contract.py 定位公共合同，
    # vendor 布局须与此相对深度一致（vendor/hardware/methods -> vendor/common）。
    ("大修执行_20260923/04_Q6/hardware/methods/b1_deng.py", "hardware/methods/b1_deng.py"),
    ("大修执行_20260923/04_Q6/hardware/methods/b2_kakouche.py", "hardware/methods/b2_kakouche.py"),
    ("大修执行_20260923/04_Q6/hardware/methods/c1_jade.py", "hardware/methods/c1_jade.py"),
    ("大修执行_20260923/04_Q6/hardware/methods/c3_kim.py", "hardware/methods/c3_kim.py"),
    ("大修执行_20260923/04_Q6/B0_同模型校验/b0/__init__.py", "b0/__init__.py"),
    ("大修执行_20260923/04_Q6/B0_同模型校验/b0/cfr_replay.py", "b0/cfr_replay.py"),
    ("大修执行_20260923/04_Q6/B0_同模型校验/b0/comparison.py", "b0/comparison.py"),
    ("大修执行_20260923/04_Q6/B0_同模型校验/b0/geometry.py", "b0/geometry.py"),
    ("大修执行_20260923/04_Q6/B0_同模型校验/b0/standard_music.py", "b0/standard_music.py"),
    ("大修执行_20260923/01_投稿口径/p0b/protocol/knownk_protocol.json",
     "protocol/knownk_protocol.json"),
    ("Q2 数据采集/uwb-occupancy-detection-main/radar_algorithm.py", "src/radar_algorithm.py"),
]


def _knownk_eval_sources() -> list:
    src_dir = WS / "大修执行_20260923" / "01_投稿口径" / "p0b" / "driver" / "knownk_eval"
    if not src_dir.is_dir():
        raise SystemExit("knownk_eval 目录缺失：%s" % src_dir)
    return [(str(p.relative_to(WS)).replace("\\", "/"), "knownk_eval/%s" % p.name)
            for p in sorted(src_dir.glob("*.py"))]


PKG_FILES = ("大修执行_20260923/06_仿真实验_20260926/sim_common.py",
             "大修执行_20260923/06_仿真实验_20260926/protocol/sim_protocol.py",
             "大修执行_20260923/06_仿真实验_20260926/analytic/forward.py",
             "大修执行_20260923/06_仿真实验_20260926/analytic/scenes.py",
             "大修执行_20260923/06_仿真实验_20260926/metrics/sim_metrics.py")


def main() -> int:
    entries = []
    all_sources = SOURCES + _knownk_eval_sources()
    for rel_src, rel_dst in all_sources:
        src = WS / rel_src
        dst = VENDOR / rel_dst
        if not src.is_file():
            raise SystemExit("源文件缺失：%s" % src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        src_sha, dst_sha = sc.sha256_file(src), sc.sha256_file(dst)
        entries.append({"src": rel_src, "dst": "vendor/%s" % rel_dst,
                        "src_sha256": src_sha, "dst_sha256": dst_sha,
                        "byte_identical": bool(src_sha == dst_sha),
                        "bytes": int(src.stat().st_size)})
    if not all(e["byte_identical"] for e in entries):
        raise SystemExit("存在非 byte-identical 的 vendor 副本：拒绝")
    own = [{"path": p, "sha256": sc.sha256_file(WS / p)} for p in PKG_FILES if (WS / p).is_file()]
    manifest = {
        "kind": "sim-vendor-manifest",
        "created_at": sc.now_iso(),
        "discipline": "旧 pipeline 最终移植必须 byte-identical vendor 并记录 sha256；路径由 bootstrap 覆盖。",
        "n_vendored": len(entries),
        "all_byte_identical": True,
        "vendored": entries,
        "package_own_sources": own,
        "aggregate_sha256": sc.sha256_text(sc.canonical_json(entries)),
    }
    sc.atomic_write_json(VENDOR / "vendor_manifest.json", manifest)
    print("vendored %d files -> %s" % (len(entries), VENDOR))
    for e in entries:
        print("  %s  %s" % (e["dst_sha256"][:12], e["dst"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
