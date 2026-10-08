#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""写出 scripts/status.json：供部署/接手方判断核心代码冻结状态、已验/未跑与剩余阻断。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import sim_common as sc  # noqa: E402


def main() -> int:
    import protocol
    proto, ph = protocol.PROTOCOL, protocol.protocol_hash()
    status = {
        "kind": "sim-package-status",
        "package": "大修执行_20260923/06_仿真实验_20260926",
        "date": "2026-09-26",
        "core_code_frozen": True,
        "protocol_hash": ph,
        "protocol_json_sha256": sc.sha256_file(PKG / "protocol" / "sim_protocol.json"),
        "numerical_source_files": sc.numerical_source_fingerprint()["n_files"],
        "numerical_source_aggregate": sc.numerical_source_fingerprint()["aggregate"],
        "vendor_manifest_aggregate": json.loads(
            (PKG / "vendor" / "vendor_manifest.json").read_text(encoding="utf-8"))["aggregate_sha256"],
        "verified_local_smoke": [
            "scripts/selftest.py 全 PASS（前端与 vendor b0 逐格 bit-identical；6 法实际运行；case 只读+指纹不变）",
            "run --family E1_A --limit 1 --trials 2 -> 10 行；run --family E2_A --limit 1 --trials 1 -> 5 行",
            "run 未给 --resume 且 rows 已存在 -> 拒绝覆盖（exit 1）；--resume -> 跳过已完成（exit 0）",
            "run 层 B 无 --fdtd-validation -> 拒绝（exit 1）；--validation-only 走 tiny smoke 且标非正式",
            "verify -> E0 六法实际运行 + 同模型一致性 + clean(无噪)/noisy 区分 + crop 披露 + 零偏移 CFR",
            "e6 -> cold/warm median+p95 + 隔离进程 RSS（区分 peak/current）",
            "verification/run_all.py（独立验证代理）182 PASS / 0 FAIL；E6_STATUS=PASS",
        ],
        "not_run_formal": [
            "E1/E2/E4 正式全量矩阵（解析层与 FDTD 层）——未运行，需另行授权",
            "FDTD 正式生成（层 B）需远程 mp 环境 + 先通过收敛前检（fdtd convergence）",
            "figures 正式出图（依赖正式 rows）；本轮只落 schema 与 trial0 样例",
        ],
        "remaining": [
            "层 B 正式 run：先 fdtd convergence（Hypatia），再 run --family E1_B/E2_B/E4_B --fdtd-validation <report>",
            "远程部署与实测（远程 code/ 与 runs/ 分离；见 deployment/ 与 README 启动顺序）",
            "Q4 跨域（q4 代理消费 runner schema；build_case 已支持 geometry override）",
            "figures 按 results/samples + rows 出图（C1 只画点，不伪造 heatmap）",
        ],
        "blocking": [
            "FDTD 收敛报告未产出/未通过 -> 正式 B 被门禁拒绝（设计如此，非缺陷）",
            "本地无 Meep -> 层 B 正式生成只能在远程 mp 环境执行",
        ],
        "warm_scope": {
            "warm_supported": ["this_method", "b0_standard_music"],
            "warm_not_applicable": ["b1_deng", "b2_kakouche", "c3_kim", "c1_jade"],
            "statement": "E6 不宣称 5 法都有 warm；仅 2 法给出真正预构 warm。",
        },
        "last_fixes": [
            {
                "id": "raw_L_no_broadcast",
                "where": "sim_common.assemble_noisy_x / noise_unit_for；run_experiments.py（run 与 _rep_cases）",
                "problem": ("此前 raw[:L]（如 FDTD L_actual=1）与 unit[:L_clean=128] 会被 numpy broadcast 成 "
                            "128 快照，等于把 1 个静态回波复制成 128，违反绝不可复制快照。"),
                "fix": ("噪声与数据严格同形状：L_actual=min(L_requested,L_available)；unit 按 L_available 抽再切 "
                        "[:L_actual]（保 E4 前缀）；cropped.shape != unit.shape 即抛错；X 行数必须 == L_actual。"),
                "policy": ("正式（非 validation-only）：L_available < L_requested -> 抛错并记 blocked（退出非零）；"
                           "validation-only：用 L_actual，row 标 L_requested/L_actual/L_available 且 formal=false；"
                           "validation-only 先把 fdtd spec 的 L_clean 改成 max_states 以保持 cache/noise 签名一致。"),
                "diag": ("diag.formal = (not validation_only) and (L_actual == L_requested)，不是仅看 L 相等；"
                         "row 的 formal 字段可用于覆盖与标注。"),
                "tested_boundaries_local": [
                    "selftest: raw1 -> assemble_noisy_x(128, allow_shorter=True) => X 行数=1，L_actual=1，formal=false",
                    "selftest: raw1 -> assemble_noisy_x(128, allow_shorter=False) => SimError（正式拒跑）",
                    "selftest: noise_unit_for(128,10) == noise_unit_for(128,128)[:10]（E4 前缀不变）",
                    "selftest: assemble_noisy_x(10, validation_only=False) => L_actual=10 且 formal=true",
                    "selftest: freq 字段改为真实频点比对（frequencies_abs_diff_max=0.0）",
                ],
                "not_yet_verified": [
                    "端到端 runner_B 与短 FDTD cache（1 快照）联调：fixture 探针在“先建 run 目录再放 fixture”"
                    "的顺序上被打断，未完成；Hypatia 部署时可用 --validation-only --fdtd-max-states 1 复验"
                    "（预期 rows L_actual=1、formal=false、无广播）。",
                ],
            },
        ],
        "raw_L_guard": {
            "entry": "sim_common.assemble_noisy_x",
            "invariant": "cropped.shape == unit.shape 且 X.shape[0] == L_actual（禁止 broadcast 复制快照）",
            "formal_short_policy": "blocked 且退出非零（不静默丢行伪完整）",
            "validation_only_policy": "L_actual=min(requested,available)，row 标 requested/actual，formal=false",
        },
    }
    sc.atomic_write_json(PKG / "scripts" / "status.json", status)
    print(json.dumps({k: status[k] for k in ("core_code_frozen", "protocol_hash",
                                             "numerical_source_aggregate")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
