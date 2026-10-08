# -*- coding: utf-8 -*-
"""本地真实 CIR 重放（B0 用）：从原始 .bin 重建与历史评估同一份 CFR/协方差输入。

依据（只读）：
  * 帧格式（记录在 Q2/Q3 冻结协议与本工作区文档）：4 B 起始标志 + 2 B 天线信息
    (tx_idx, rx_idx) + 4*ft_len B 小端 int16 的 I/Q 交错 + 4 B 结束标志；
    ft_len=32 时帧长 138 B。
  * 背景：与 GUI 同口径的在线指数参考（alpha = 1/bg_m_factor），首样本输出原值并
    令 ref = 样本，跨窗连续、不每窗重置；慢时间按每通道 ordinal。
  * 窗：每通道 ordinal 区间 [end_frame-ring, end_frame)，end_frame = round(t * rate)。

本模块是独立实现（不调用历史 evaluate.py 的 stream_captures）；其比特级等价性由
“重放谱 == 历史缓存 npz 谱”这一验收项在结果里显式核对，而不是假定成立。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

FRAME_START = bytes([0xFF, 0x00, 0xFF, 0x00])
FRAME_STOP = bytes([0xF0, 0x00, 0xF0, 0x00])
HEADER_LEN = 4
ANTENNA_INFO_LEN = 2
FOOTER_LEN = 4


class ReplayError(RuntimeError):
    """bin/meta 不合规或窗不可用（显式失败，不静默跳过）。"""


def frame_length(ft_len: int) -> int:
    return HEADER_LEN + ANTENNA_INFO_LEN + int(ft_len) * 4 + FOOTER_LEN


class FrameParser:
    """按文档化的帧格式做同步/重同步；语义与历史 StreamFrameParser 一致。"""

    def __init__(self, ft_len: int = 32):
        self.ft_len = int(ft_len)
        self.frame_len = frame_length(self.ft_len)
        self.reset()

    def reset(self):
        self._buf = bytearray()
        self.dropped_bytes = 0
        self.resync_count = 0
        self.bad_footer = 0
        self.frames_parsed = 0

    def feed(self, chunk):
        if chunk:
            self._buf.extend(bytes(chunk))
        frames = []
        while True:
            if not self._buf:
                break
            index = self._buf.find(FRAME_START)
            if index < 0:
                keep = len(FRAME_START) - 1
                if len(self._buf) > keep:
                    self.dropped_bytes += len(self._buf) - keep
                    del self._buf[: len(self._buf) - keep]
                break
            if index > 0:
                self.dropped_bytes += index
                del self._buf[:index]
            if len(self._buf) < self.frame_len:
                break
            frame = bytes(self._buf[: self.frame_len])
            if frame.endswith(FRAME_STOP):
                frames.append(frame)
                self.frames_parsed += 1
                del self._buf[: self.frame_len]
                if not self._buf:
                    break
            else:
                self.bad_footer += 1
                self.resync_count += 1
                del self._buf[:1]
        return frames

    def stats(self):
        return {
            "frames_parsed": int(self.frames_parsed),
            "dropped_bytes": int(self.dropped_bytes),
            "resync_count": int(self.resync_count),
            "bad_footer": int(self.bad_footer),
            "buffered_bytes": len(self._buf),
        }


def parse_frame_cir(frame: bytes, ft_len: int) -> np.ndarray:
    raw = np.frombuffer(frame[6:6 + 4 * ft_len], dtype="<i2").reshape(ft_len, 2)
    return raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)


def load_meta(bin_path) -> dict:
    meta_path = Path(str(bin_path) + ".meta")
    if not meta_path.is_file():
        raise ReplayError("缺少 meta：%s" % meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("ft_len", "udp_tx_list", "udp_rx_list", "tx_positions", "rx_positions",
                "tap_interval_si", "imaging_grid_spec"):
        if meta.get(key) in (None, []):
            raise ReplayError("meta 缺少必需字段 %s：%s" % (key, meta_path))
    if not (meta.get("recording") or {}).get("snapshot_rate_estimate_hz"):
        raise ReplayError("meta 缺少 recording.snapshot_rate_estimate_hz：%s" % meta_path)
    return meta


def replay_windows(bin_path, meta, t_list, *, ring: int = 128, chunk_bytes: int = 65536,
                   bg_m_factor=None):
    """一次扫描重放整段 bin，同时抓取 t_list 中所有可满足的窗。

    语义与历史 stream_captures 一致（逐帧检查 min(各通道计数) >= end_frame，命中即抓取；
    环长 ring、跨窗连续背景 EMA）。返回 {t_end_s: capture_dict}；不可满足的 t 不出现，
    由调用方显式记录为失败/跳过，不静默丢弃。

    capture_dict：
      arr: (n_rx, n_tx, ft_len, ring) complex64，通道排列 = arr[rx_index, tx_index]
      end_frame / frame_start / planned_end_frame / channel_frame_counts /
      channel_count_delta / sync_exact / parser_stats / unknown_frames / bytes / t_end_s
    """
    ft_len = int(meta["ft_len"])
    tx_list = [int(v) for v in meta["udp_tx_list"]]
    rx_list = [int(v) for v in meta["udp_rx_list"]]
    rate = float(meta["recording"]["snapshot_rate_estimate_hz"])
    if bg_m_factor is None:
        bg_m_factor = float(meta.get("bg_m_factor") or 4.0)
    alpha = 1.0 / float(bg_m_factor)
    channels = [(t, r) for t in tx_list for r in rx_list]
    tx_map = {v: i for i, v in enumerate(tx_list)}
    rx_map = {v: i for i, v in enumerate(rx_list)}
    parser = FrameParser(ft_len=ft_len)
    state = {key: {"ref": None, "first": True, "buf": deque(maxlen=int(ring)), "n": 0}
             for key in channels}
    unknown_frames = 0
    pending = []
    for t_value in t_list:
        end = int(round(float(t_value) * rate))
        if end >= int(ring):
            pending.append((float(t_value), end))
    pending.sort(key=lambda item: item[1])
    captures = {}
    bin_path = Path(bin_path)
    stopped_early = False
    with bin_path.open("rb") as fh:
        while not stopped_early:
            chunk = fh.read(int(chunk_bytes))
            if not chunk:
                break
            for frame in parser.feed(chunk):
                key = (frame[4], frame[5])
                if key not in state:
                    unknown_frames += 1
                    continue
                cur = parse_frame_cir(frame, ft_len)
                item = state[key]
                if item["first"]:
                    item["ref"] = cur.astype(np.complex128).copy()
                    out = cur
                    item["first"] = False
                else:
                    out = (cur - item["ref"]).astype(np.complex64)
                    item["ref"] = (1.0 - alpha) * item["ref"] + alpha * cur
                item["buf"].append(out)
                item["n"] += 1
                while pending:
                    t_value, end_frame = pending[0]
                    counts = [s["n"] for s in state.values()]
                    if min(counts) < end_frame:
                        break
                    pending.pop(0)
                    frame_end = int(min(counts))
                    arr = np.zeros((len(rx_list), len(tx_list), ft_len, int(ring)),
                                   dtype=np.complex64)
                    deltas = {}
                    for (t_, r_), s in state.items():
                        arr[rx_map[r_], tx_map[t_], :, :] = np.stack(list(s["buf"]), axis=1)
                        deltas["%d-%d" % (t_, r_)] = int(s["n"] - frame_end)
                    captures[float(t_value)] = {
                        "arr": arr,
                        "end_frame": frame_end,
                        "frame_start": int(frame_end - int(ring)),
                        "planned_end_frame": end_frame,
                        "channel_frame_counts": {"%d-%d" % k: int(v["n"]) for k, v in state.items()},
                        "channel_count_delta": deltas,
                        "sync_exact": bool(frame_end == end_frame and all(d == 0 for d in deltas.values())),
                        "parser_stats": parser.stats(),
                        "unknown_frames": int(unknown_frames),
                        "t_end_s": float(t_value),
                    }
                if not pending:
                    stopped_early = True
                    break
    for t_value, end_frame in pending:
        captures[float(t_value)] = {
            "unavailable": True,
            "reason": "insufficient_frames",
            "detail": "min(channel_frames)=%d < end_frame=%d"
                      % (min(s["n"] for s in state.values()), end_frame),
            "t_end_s": float(t_value),
            "planned_end_frame": int(end_frame),
            "channel_frame_counts": {"%d-%d" % k: int(v["n"]) for k, v in state.items()},
        }
    for capture in captures.values():
        capture["bytes"] = int(bin_path.stat().st_size)
        capture["bin_path"] = str(bin_path)
    return captures


def replay_window(bin_path, meta, t_end_s, *, ring: int = 128, chunk_bytes: int = 65536,
                  bg_m_factor=None):
    """单窗重放（replay_windows 的薄包装）；窗不可用或 end_frame < ring 时显式抛错。"""
    end_frame = int(round(float(t_end_s) *
                          float(meta["recording"]["snapshot_rate_estimate_hz"])))
    if end_frame < int(ring):
        raise ReplayError("end_frame=%d < ring=%d：该窗在历史协议下会被跳过" % (end_frame, ring))
    captures = replay_windows(bin_path, meta, [float(t_end_s)], ring=ring,
                              chunk_bytes=chunk_bytes, bg_m_factor=bg_m_factor)
    capture = captures[float(t_end_s)]
    if capture.get("unavailable"):
        raise ReplayError("窗不可用（%s）：%s" % (capture["reason"], capture["detail"]))
    return capture


def cropped_snapshots(arr, cir_offset: int):
    """按历史口径裁剪 tap（[:, :, cir_offset:, :]），返回快照列表 (n_rx, n_tx, n_taps-off)。"""
    cropped = arr[:, :, int(cir_offset):, :]
    return [cropped[..., i] for i in range(cropped.shape[-1])], cropped.shape[-1]
