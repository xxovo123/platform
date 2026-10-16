# -*- coding: utf-8 -*-
"""协议加载与严格校验（P0b known-K 统一评估协议）。

设计要点：
  * 全部参数来自协议文件；运行时不接受与协议不一致的覆盖值（check_override），
    避免代码常量与文档口径漂移。
  * canonical SHA256 = sha256(canonical_text(payload 去掉 hash 字段))，
    canonical_text = json.dumps(..., ensure_ascii=False, indent=1, sort_keys=True)。
    加载时复算并与文件中的 protocol_canonical_sha256 比对，不一致即失败。
  * 协议文件记录上游历史 frozen_protocol.json 的 revision 与 canonical sha256，
    但**不修改**它。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

HASH_FIELD = "protocol_canonical_sha256"
SUPPORTED_SCHEMAS = ("p0b-knownk-protocol/1",)


class ProtocolError(RuntimeError):
    """协议缺失、schema 不符、SHA256 漂移或参数非法。"""


def canonical_text(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != HASH_FIELD}
    return json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_text(payload).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class PeakRule:
    threshold_rel: float
    min_dist_m: float
    neighbourhood: int
    include_center: bool
    strict: bool
    boundary: str
    plateau_connectivity: int
    plateau_representative: str
    plateau_tie_break: str
    nms_accept_rule: str
    count_cap: str
    allow_fewer_than_k: bool
    truth_used_for_peak_selection: bool


@dataclass(frozen=True)
class MatchingRule:
    gate_m: float
    gate_tolerance_m: float
    max_n_truth: int
    max_n_pred: int
    tie_break: str


@dataclass(frozen=True)
class Protocol:
    path: Path
    payload: dict
    protocol_id: str
    protocol_revision: int
    canonical_sha256_value: str
    file_sha256: str
    peaks: PeakRule
    matching: MatchingRule

    @property
    def effective_gate_m(self) -> float:
        return self.matching.gate_m + self.matching.gate_tolerance_m


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ProtocolError(msg)


def _positive_float(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ProtocolError("%s 必须是数值，收到 %r" % (name, value)) from None
    _require(math.isfinite(out), "%s 必须是有限值，收到 %r" % (name, value))
    return out


def load_protocol(path, *, expected_sha256: str | None = None, self_check: bool = True) -> Protocol:
    """装载协议并做两级校验：

      1) canonical SHA256（改一个字符就失败）；
      2) 语义校验（selfcheck）：描述字段逐字核对 + 最小算例行为自检——
         防止“改描述/改参数再重签 hash”绕过第 1 级。
    """
    path = Path(path)
    if not path.is_file():
        raise ProtocolError("协议文件不存在：%s" % path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError("协议文件不是合法 JSON：%s (%s)" % (path, exc)) from None
    _require(isinstance(payload, dict), "协议根必须是对象")
    schema = payload.get("schema")
    _require(schema in SUPPORTED_SCHEMAS,
             "协议 schema 不支持：%r（支持 %s）" % (schema, list(SUPPORTED_SCHEMAS)))

    recorded = payload.get(HASH_FIELD)
    _require(isinstance(recorded, str) and len(recorded) == 64,
             "协议缺少 %s（请先运行 scripts/freeze_protocol.py）" % HASH_FIELD)
    recomputed = canonical_sha256(payload)
    _require(recomputed == recorded,
             "协议 canonical SHA256 漂移：文件记录 %s，复算 %s；协议一旦冻结不得手改"
             % (recorded, recomputed))
    if expected_sha256 is not None:
        _require(recomputed == expected_sha256,
                 "协议 SHA256 与调用方期望不一致：期望 %s，实得 %s"
                 % (expected_sha256, recomputed))

    peaks_raw = payload.get("peaks") or {}
    nb = peaks_raw.get("neighbourhood") or {}
    plateau = peaks_raw.get("plateau") or {}
    threshold = peaks_raw.get("threshold") or {}
    nms = peaks_raw.get("nms") or {}
    count = peaks_raw.get("count") or {}
    window = nb.get("window") or []
    _require(list(window) == [3, 3], "取峰邻域必须为 3x3，收到 %r" % (window,))
    _require(bool(nb.get("include_center")) is True, "取峰邻域必须含中心点")
    _require(bool(nb.get("strict")) is False, "取峰邻域必须是非严格极大")
    _require(str(nb.get("boundary")) == "edge", "取峰边界必须按 edge 复制")
    _require(int(plateau.get("connectivity")) == 8, "平台去重必须为 8 连通")
    _require(str(plateau.get("representative")) == "plateau_max", "平台代表点必须是域内最大值")
    _require(str(plateau.get("tie_break")) == "min_row_then_col", "平台并列必须取最小 (row, col)")
    _require(str(threshold.get("mode")) == "linear_relative_to_global_finite_max",
             "阈值必须是线性相对全局最大有限值")
    _require(str(nms.get("metric")) == "euclidean_xy_m", "NMS 距离必须是 xy 欧氏距离（米）")
    _require(str(count.get("cap")) == "K", "峰数上限必须是给定 K")
    _require(bool(count.get("truth_used_for_peak_selection")) is False, "取峰不得使用真值")

    matching_raw = payload.get("matching") or {}
    limits = matching_raw.get("enumerate_limits") or {}
    peak_rule = PeakRule(
        threshold_rel=_positive_float(threshold.get("threshold_rel"), "peaks.threshold.threshold_rel"),
        min_dist_m=_positive_float(nms.get("min_dist_m"), "peaks.nms.min_dist_m"),
        neighbourhood=3,
        include_center=True,
        strict=False,
        boundary="edge",
        plateau_connectivity=8,
        plateau_representative=str(plateau.get("representative")),
        plateau_tie_break=str(plateau.get("tie_break")),
        nms_accept_rule=str(nms.get("accept_rule")),
        count_cap=str(count.get("cap")),
        allow_fewer_than_k=bool(count.get("allow_fewer_than_K")),
        truth_used_for_peak_selection=bool(count.get("truth_used_for_peak_selection")),
    )
    _require(peak_rule.min_dist_m > 0.0, "NMS 门限必须为正")
    _require(0.0 < peak_rule.threshold_rel <= 1.0, "相对阈值必须在 (0, 1] 内")

    matching_rule = MatchingRule(
        gate_m=_positive_float(matching_raw.get("gate_m"), "matching.gate_m"),
        gate_tolerance_m=_positive_float(matching_raw.get("gate_tolerance_m"),
                                        "matching.gate_tolerance_m"),
        max_n_truth=int(limits.get("max_n_truth", 8)),
        max_n_pred=int(limits.get("max_n_pred", 16)),
        tie_break=str(matching_raw.get("tie_break")),
    )
    _require(matching_rule.max_n_truth >= 1 and matching_rule.max_n_pred >= 1,
             "匹配枚举上限必须为正整数")

    protocol = Protocol(
        path=path,
        payload=payload,
        protocol_id=str(payload.get("protocol_id")),
        protocol_revision=int(payload.get("protocol_revision")),
        canonical_sha256_value=recomputed,
        file_sha256=sha256_file(path),
        peaks=peak_rule,
        matching=matching_rule,
    )
    from .selfcheck import behavioral_self_check, validate_semantic_fields  # 延迟导入避免循环

    validate_semantic_fields(payload)
    if self_check:
        behavioral_self_check(protocol)
    return protocol


def check_override(name: str, value, protocol_value, *, tol: float = 0.0) -> None:
    """严格校验：运行期覆盖值必须与协议一致，否则失败（防止静默漂移）。"""
    if isinstance(protocol_value, float) or isinstance(value, float):
        try:
            same = abs(float(value) - float(protocol_value)) <= tol
        except (TypeError, ValueError):
            same = False
    else:
        same = value == protocol_value
    if not same:
        raise ProtocolError(
            "参数 %s 与冻结协议不一致：运行期 %r vs 协议 %r（协议驱动参数，不允许静默覆盖）"
            % (name, value, protocol_value))
