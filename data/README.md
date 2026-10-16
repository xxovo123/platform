# Data structure / 数据结构

[Platform overview / 平台原理](../README.md) · [中文平台介绍](../README_zh.md) · [Original hardware CIR / 硬件原始 CIR](hardware/README.md)

This directory groups simulation data and the 205-segment hardware evaluation dataset. Numerical inputs, ground truth, saved predictions and summaries are distinct files. Hardware data are original chip CIR; simulation CIR are outputs of the numerical receiver chain.

## Controlled simulation CIR

`cache/fdtd/<clean_id>/` contains the 14 controlled FDTD geometries. `cache/analytic/<clean_id>/` contains 29 parametric geometries. `cache_index.json` maps their identifiers, target counts, coordinates, seeds and array shapes.

| File / array | Meaning |
|---|---|
| `clean_cir.npz: raw_cir` | Complex128 array `[snapshot, channel, tap] = [128, 4, 32]`; numerical receiver-chain CIR, after the model's background processing |
| `clean_meta.json` | Scene, nominal truth, clean seed, physics identity, original SHA-256, geometry and processing settings |
| `truth.json` | Independent nominal target positions; evaluation input, not an estimator input |
| `protocol.json`, `plan.json`, `provenance/` (FDTD) | Resolved physical scene, source, receiver-chain parameters and actual bounded-motion trajectory |

Controlled channels are one transmitter at the origin and four receivers at x = −0.038, −0.019, 0 and 0.019 m. The tap interval is 1 ns, carrier 7.9872 GHz and bandwidth 500 MHz. The simulation window uses raw taps `[0:30]`, then FFT length 32 and 17 in-band bins. Channel–frequency features are ordered by frequency first, then channel: feature index = `f*4 + channel`. The stored snapshot matrix is `[L,68]`, the transpose convention of the manuscript's `[68,L]` matrix.

The `raw_cir` name in simulation caches means the 32-tap input before the algorithm's finite window. It does **not** mean a hardware recording or an unprocessed electromagnetic field.

```python
from pathlib import Path
import numpy as np
folder = Path('data/cache/fdtd/cln_de385a3b2982')
with np.load(folder / 'clean_cir.npz', allow_pickle=False) as z:
    cir = z['raw_cir']             # [128, 4, 32]
    first_16 = cir[:16]            # genuine prefix; no repeated snapshots
```

## Original field examples

`field_samples/<cache_key>/` contains `background.npz` and `state_00000.npz` for the fixed S1/S2/S5 localization scenes. `field_samples/index.json` associates those physical cache keys with clean IDs.

| Array | Meaning |
|---|---|
| `records` | Real-valued recorded field samples `[receiving channel, field-time sample]`; before the receiver processing chain |
| `t_s` | Physical time in seconds, with one entry per field-time sample |
| `meta_json` | Unicode JSON scalar giving transmitter, source, solver and physical target-state information |
| `cache_key`, `kind`, `index` | Physics cache identity, empty/target state and state index |

Only these representative field records are supplied; the complete controlled CIR, full trajectories and all formal trials are supplied separately. Regenerating missing field states requires actual Meep solves.

## Formal simulation results

`results/runs/` contains six completed groups:

| Family | Conditions | Trials per condition | Saved records |
|---|---:|---:|---:|
| E1_A | Five layouts × nine SNR parameters | 200 | 9,000 |
| E1_B | Same layouts and SNRs, FDTD | 50 | 2,250 |
| E2_A | Three directions × eight separations × three SNRs | 100 | 7,200 |
| E2_B | Three directions × 0.30/0.50/1.00 m, middle SNR | 50 | 450 |
| E4_A | Five prefix lengths × nine SNRs | 100 | 4,500 |
| E4_B | Five prefix lengths × three SNRs, FDTD | 50 | 750 |

Each run has `tasks.jsonl` (geometry and requested settings), `truth.json`, `rows.jsonl` (one record per trial), `pairs.csv`, `source_manifest.json`, `config.json`, `checks.json`, `status.json`, `summary.csv` and fixed spectrum `samples/`. `_predictions` records x/y in metres and response score; `_errors` records position/range error in metres and azimuth error in degrees for the same assignment. `seed` is the actual noise seed. `missing` means unassigned truth, whereas `fn` is a detection miss under the separate 0.30 m matching rule.

All trial records, including missing outputs, are retained. Cross-family shared inputs are listed in `tables/shared_input_aliases.csv`; 24,150 task records are not 24,150 independent physical trajectories.

Use `tables/formal/` for the final exported statistics rather than the historical run `summary.csv`; the final export includes the corrected strict two-target-resolution aggregation. `tables/manuscript/` adds `snr_expected_db = snr_db + 10*log10(2)` without changing saved noise or predictions. The raw-run SNR tag remains the original generator parameter. A fixed clean ID and trial share the noise realization across SNR and prefix lengths.

## Simulation–measurement correspondence

`q4/sim_cir/<config_id>/clean_cir.npz` contains fifteen configuration-specific numerical CIR sequences with the hardware geometry. Its `raw_cir` array is `[128,4,32]`; `burnin` records the startup count, `snapshots` the retained count, and `cache_key` the physical cache identity. The 32 startup states were used in receiver background processing before the retained 128 snapshots; they are not additional rows of the stored array. Geometry and estimator settings are in `q4/generator_settings.json`, and the effective protocol specifies the Q4 trajectory and background settings.

`q4/spectra/` contains fifteen simulated response spectra and all 205 hardware response spectra used in the manuscript's correspondence comparison. These are **processed spectra**, distinct from the original CIR in `hardware/`. The historical filename prefix `uncorrected_` means the original operational hardware processing, with no additional registration applied in this comparison.

Spectrum NPZ files contain `spectrum`, `normalized`, `grid_x`, `grid_y`, `mask`, `predictions` and `truth`. Both spectra have shape `[len(grid_y), len(grid_x)] = [74,89]`; grid coordinates are metres. `mask` identifies the common valid grid points. `predictions` and `truth` are x/y pairs in metres.

`q4/per_window.csv` records direct spectrum correlation, identity-matched peak displacement and missing identities for each of the 205 segments. `tables/Q4_environment_config.csv` aggregates environment/configuration groups. `tables/Q4_config_equal_weight.csv` gives the equal-configuration aggregate: FDTD correlation 0.696 after rounding. `q4/parametric/` supplies the twenty-seed plane-wave reference results for each of the fifteen configurations, with rounded correlation 0.474. Its three saved trial-0 examples retain `X`, source phases, delays, spectra and predictions. `q4/generator_settings.json` supplies the actual physical array geometry and estimator settings for this reference.

## Hardware raw CIR

See [hardware/README.md](hardware/README.md) for the complete 205 original CIR segments, their dimensions, channel order, timestamps, truth and relationship to the frozen evaluation identifiers. Hardware CIR are not simulated data and are not background-removed snapshots.

## Integrity and scope

`source_manifest.json` records source and released hashes for the selected simulation files. Original numerical arrays and saved trial predictions are unchanged. Published run configuration JSON removes internal conversation/agent fields; the original numerical values and source hashes are retained. Derived correspondence tables select the published processing branch. The hardware export has its own per-file hashes and extraction verification.

`settings/` records the effective experiment protocol and environment. Source-pulse arrays are reference samples of the simulation excitation, not hardware measurements. Full intermediate FDTD field caches are represented by examples and regeneration code. The revision adds the complete hardware evaluation software and continuous EMA states under `../hardware_reproduction/`, and the revised separation inputs and trials under `../spacing_revision/`.

中文说明：仿真 `raw_cir` 是数值接收链输出的 32 抽头 CIR，原始场在 `field_samples/`；硬件芯片采集的原始 CIR 单独存放于 `hardware/`。所有数组读取示例均使用 `allow_pickle=False`。坐标与距离单位为米，时延与时间单位为秒，频率单位为 Hz，方位角单位为度。论文信噪比标注、原始运行参数和有限样本实现值分别保留，不能混用。
