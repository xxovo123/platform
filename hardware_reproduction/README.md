# Hardware reproduction

The complete release workflow is supported on **Linux**, using the root `environment.yml`: Python 3.12.11, NumPy 2.3.0 and Conda package `pymeep=1.30.1=nompi_py312hfc0f314_104` (the module reports `meep.__version__=1.30.0`). This exact pymeep build was used in the previous conditional server run. Follow [the Linux guide](../docs/linux_validation.md) for all 5330 replay runs and comparisons with the approved manuscript. The current candidate is `revision-20261019-candidate`, archive `platform_validation_20261019.zip`. Reported issues have been repaired without a new validation run; previous conditional numerical reproduction passed. See [validation status](../docs/validation_status_20261019.md) for the evidence and conditions.

This integrated package covers the full **205-segment evaluation dataset**: 103 indoor and 102 outdoor segments, 31 recordings and 338 target instances. Six estimator implementations and twenty ablation configurations each cover all 205 segments. Every estimator receives K. Baselines are adaptations to shared CIR, not reproductions of the cited papers' acquisition systems or a general ranking.

For reproduction, use the fresh Linux environment specified above. The archived standalone full replay used Python 3.13.3, NumPy 2.5.3 and SciPy 1.18.1; these are historical provenance, not the current environment specification. From this directory:

```text
python -B reproduce_hardware.py verify --out generated/check
python -B reproduce_hardware.py all-saved --out generated/check
python -B validate_release.py
python -B reproduce_hardware.py replay --methods this_method --limit 1 --out generated/smoke
python -B reproduce_hardware.py replay --methods all --out generated/full_replay
python -B reproduce_hardware.py tables --replay-input generated/full_replay/replay_predictions.json --out generated/new_tables
```

`verify` retains strict existing raw CIR file, continuous EMA state and source hash checks. It compares all **205 FFT arrays** with the frozen `data/preprocessing_reference.npz`, using elementwise **max_abs<=1e-12 and `rtol=0`**. All 205 reference arrays were matched byte for byte to their existing `X_sha256` values during archival preparation, without executing estimators. Original `X_hash` results remain diagnostics. The previous Linux run had 205/205 FFT hash mismatches; three checked examples in the report had a maximum absolute difference of about `1.6e-13`, within `1e-12`. Raw and EMA state hashes matched. A stale source-manifest entry also blocked the original check. These issues have been addressed; the revised workflow has not been revalidated.

`all-saved` independently recomputes assignments, statistics, printed-table checks and four review plots from frozen predictions. `replay` starts from raw CIR plus continuous EMA state and executes estimators. `all` means six methods **plus twenty configurations**, including the ablation reference. These are distinct reproduction levels. The preprocessing repair changes neither paper metrics nor estimators or their output rules.

The six IDs are `this_method` (JSF-MUSIC), `b1_deng`, `b2_kakouche`, `c1_jade`, `c3_kim` and `b5_wang_full2d` (traditional joint 2D MUSIC, not Wang's successive estimator). `data/config_matrix.json` defines the twenty hardware ablations; `data/result_map.json` indexes all 26 frozen sets. Adaptation notes remain in `docs/`.

| Path | Contents |
|---|---|
| `data/raw/` | All original complex CIR segments `[128,8,32]`, truth, indices, 31 recording metadata records |
| `data/ema_states/` | Pre-segment background state and channel initialization masks |
| `data/preprocessing_reference.npz`, `data/preprocessing_reference.json` | Frozen FFT arrays and archival metadata; all 205 arrays matched existing X_sha256 values |
| `data/contexts.json`, `source/` | Geometry, calibration, settings, protocol and estimator/metric/preprocessing dependencies |
| `data/frozen/` | Six methods and twenty ablation prediction sets |
| `data/reference_selected_measurements.csv` | Fixed four-repeat selection in 30 groups |
| `data/paper_reference/` | Unchanged historical numerical table snapshots; current caption wording can differ |
| `full_replay/replay_comparison.csv`, `replay_verification.json` | Historical 5,330-run full replay evidence, positions agree within `1e-8 m` |
| `outputs/` | Archived independent statistics, 102 baseline/ablation checks and numerical review plots |
| `integration_manifest.json` | Imported hashes/sizes and intentional omissions |

The continuous EMA is not reset per segment; raw frames alone cannot reproduce recording-level background subtraction without the released initial state. Arrays were checked against historical observations and are not synthesized from truth.

Large full-replay prediction/diagnostic JSON, smoke duplicates and caches are omitted; all independent inputs and compact full-replay evidence remain. New outputs belong under ignored `generated/`. `validate_release.py` checks imported hashes, all 5,330 archived segment/configuration pairs, complete frozen coverage, all 120 fixed temporal measurements and 110 additional group/stability table values.

The comparison, ablation, stability and position plots are numerical review figures, not the original manuscript layouts. Original non-simulation schematics, approved anonymized photographs and regeneration recipes are in `../manuscript_assets/`. See [the figure/table map](../docs/hardware_reproduction_map.md) for statistics versus photography, static artwork and mixed spectra. These hardware commands execute no simulation.

For case construction, `this_method.estimate(case)`, array ordering and exporting all 205 measured spectra from raw, see [the raw-to-case interface](docs/raw_case_interface.md). `python -B export_hardware_spectra.py --out generated/raw_spectra` exports hardware spectra and per-input hashes without simulation. Q4 spectrum matching uses the separate root-level `q4_reproduce.py verify-spectra` command.
