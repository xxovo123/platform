# Original hardware CIR / 硬件原始 CIR

This directory contains exactly the **205 evaluation segments** used in the revised manuscript: 103 indoor and 102 outdoor segments, extracted from 31 original recordings. There are 338 true-target instances across these segments, with one to three people per segment.

The primary array is **`raw_cir`**, taken from the chip-recorded binary I/Q samples **before background subtraction, tap cropping, FFT or channel phase calibration**. It is not a processed snapshot matrix, response spectrum or newly simulated measurement. Raw here means chip-produced CIR; the recordings do not contain pre-chip ADC samples.

## Files

| Path | Contents |
|---|---|
| `segments/<segment_id>.npz` | One original CIR segment and its numeric metadata |
| `index.csv`, `index.json` | All 205 identifiers, recording/scenario association, target counts, evaluation times, relative paths, dimensions, dtypes and SHA-256 |
| `recordings/<recording_id>.json` | Acquisition settings, source-file identity and channel mapping for each of the 31 recordings |
| `schema.json` | Machine-readable field definitions and timing conventions |

Each NPZ can be opened with `allow_pickle=False`. Metadata strings use Unicode arrays; object arrays and serialized Python objects are not required.

## Array definitions

| Field | Shape / type | Meaning |
|---|---|---|
| `raw_cir` | `[128,8,32]`, complex64 | `[slow-time sample, transmit–receive channel, CIR tap]`. Real = recorded signed int16 I; imaginary = recorded signed int16 Q. Native chip counts, without amplitude normalization |
| `channel_pairs` | `[8,2]`, int64 | Zero-based physical coordinate indices `(tx_idx,rx_idx)` in transmitter-major, receiver-minor order |
| `device_channel_pairs` | `[8,2]`, int64 | Channel numbers encoded in the recording: transmitter numbers 1/2 and receiver numbers 4/5/6/7 |
| `tx_positions_m` | `[2,3]` | Physical transmitter coordinates `(x,y,z)` in metres |
| `rx_positions_m` | `[4,3]` | Physical receiver coordinates `(x,y,z)` in metres |
| `selected_pair_indices` | `[4]`, int64 | `[2,3,6,7]`: the four channels used by the manuscript estimator; all eight raw channels remain available |
| `source_frame_indices` | `[128,8]`, int64 | Zero-based ordinal of the valid parsed frame in the source binary stream |
| `source_channel_frame_indices` | `[128,8]`, int64 | Zero-based ordinal within each channel's valid frames |
| `estimated_frame_timestamps_s` | `[128,8]`, float64 | Nominal relative time reconstructed as per-channel frame index divided by the recorded estimated sample rate |
| `channel_frame_counts_at_capture` | `[8]`, int64 | Cumulative per-channel frame count when the formal segment was captured |
| `planned_end_frame` | scalar | `round(t_end_s * snapshot_rate_estimate_hz)` |
| `snapshot_rate_estimate_hz` | scalar | Source metadata estimate from channel counts and recording duration |
| `t_end_s` | scalar | Formal segment evaluation end time, in seconds |
| `tap_interval_s` | scalar | 1 ns; original tap indices 0 through 31 are retained |
| `center_frequency_hz`, `bandwidth_hz` | scalars | 7.9872 GHz and 500 MHz |
| `K` | scalar, int64 | Supplied true target count for the segment |
| `truth_xy_m` | `[K,2]` | Nominal target positions `(x,y)` in metres, for evaluation |
| `segment_id`, `recording_id`, `environment`, `config_id` | Unicode scalars | Segment identity, source recording, indoor/outdoor and target configuration |

complex64 exactly represents the integer I and Q values in these recordings. It introduces no loss of the original signed int16 sample values.

The eight raw channels are `(T0,R0), (T0,R1), (T0,R2), (T0,R3), (T1,R0), (T1,R1), (T1,R2), (T1,R3)`. Receivers R0/R1 correspond to the two TRX elements, followed by the two receive-only elements R2/R3. The manuscript selects `(T0,R2), (T0,R3), (T1,R2), (T1,R3)`.

Coordinates retain the original physical array geometry, including its z component. Nominal targets lie in the horizontal plane, with the forward evaluation region at `y < 0`; azimuth is `atan2(x,-y)`. Raw sample amplitudes are chip counts and are not calibrated electric-field units.

## Segment and timing conventions

The exported identifiers exactly match the frozen 205-segment evaluation list. For each formal end time, the original replay captures each channel's current final 128 samples when the minimum cumulative count across all eight channels reaches `round(t_end_s * snapshot_rate_estimate_hz)`. The same trigger and per-channel ring-buffer convention are used in this export.

The binary recordings contain **no measured per-frame timestamps or hardware sequence counters**. The exported time arrays are clearly named estimates, and the frame-index arrays are software ordinals recovered from the original record order. They do not assert cross-channel simultaneous acquisition. `t_end_s` is the formal evaluation time, not an independently measured timestamp for every CIR sample.

The 205 identifiers refer to temporal segments in the recordings. Their original recording association is preserved in the index; the count does not imply 205 separate acquisition sessions.

## Reading example

```python
from pathlib import Path
import json
import numpy as np

base = Path('data/hardware')
index = json.loads((base / 'index.json').read_text(encoding='utf-8'))
entry = index['segments'][0]
with np.load(base / entry['file'], allow_pickle=False) as z:
    raw = z['raw_cir']                          # [128,8,32], unchanged I + 1j*Q
    paper_channels = raw[:, z['selected_pair_indices'], :]  # [128,4,32], still raw
    truth = z['truth_xy_m']                     # [K,2], metres
    pairs = z['channel_pairs']                  # full eight-channel mapping
    nominal_time = z['estimated_frame_timestamps_s']
```

The example selects channels only. It performs no background processing, tap cropping or phase correction.

## Verification and scope

All 205 segment identifiers are unique and match the evaluation set exactly. The raw I/Q values match the original binary parser. During export, continuous recording-level background processing, the original finite window and the observation-matrix construction were reconstructed in memory; their outputs matched the frozen evaluation cache for all 205 segments. These checks establish segment/frame identity without publishing the processed arrays. NPZ readback, per-file hashes and recording identities were also verified.

Only original CIR and acquisition/evaluation metadata are published here. No processed CIR, covariance, MUSIC spectrum, localization outputs or hardware execution scripts are stored in this directory. Processed correspondence spectra, where needed for reviewing the simulation comparison, are separately identified in `../q4/`.

中文说明：这里的 `raw_cir` 是从原始采集文件中直接解码的芯片复数 CIR，维度为 **128 个慢时间样本 × 8 个收发通道 × 32 个抽头**，保留原始整数 I/Q 幅值。尚未进行背景扣除、截窗、FFT、归一化或通道相位校准。论文的四通道可通过 `selected_pair_indices=[2,3,6,7]` 读取；真值坐标和通道物理位置分别保存。索引完整覆盖 205 段评估数据，室内 103 段、室外 102 段。原文件没有逐帧实测时间戳，发布的时间字段是明确标注的重建估计值，不把它们冒称为芯片时间戳。硬件代码和处理后快照不随本目录发布。
