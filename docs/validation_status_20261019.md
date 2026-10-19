# Validation status — 2026-10-19

The current candidate is **`revision-20261019-candidate`**, archive **`platform_validation_20261019.zip`**. **Reported issues have been repaired; the revised candidate has not been revalidated.** Previous conditional numerical reproduction passed. Linux remains required. This document summarizes the evidence and conditions from the 2026-10-19 independent validation report.

## Previous candidate

The tested archive was `platform_validation_20261018.zip`, identifier `revision-20261018-candidate`. **It did not pass as supplied.** Environment creation and extraction required corrections, and the unmodified saved-data stage failed its hardware preprocessing check.

A conditional run completed all four stages and passed their numerical assertions. It used Conda `pymeep=1.30.1=nompi_py312hfc0f314_104`, UTF-8-aware extraction and a separate driver that skipped the failing hardware-preprocessing check and replaced the command that repeated that check with its table and figure subcommands. The report records no changes to the packaged files, reference results or downstream numerical tolerances. These exceptions prevent an unconditional acceptance claim.

## Three reported issues and repairs

| Issue | Reported evidence | Repair |
|---|---|---|
| Conda package pin | `pymeep=1.30.0` could not resolve with Python 3.12.11 and NumPy 2.3.0. The working build reports `meep.__version__=1.30.0`. | Use the exact previously installed Linux build `pymeep=1.30.1=nompi_py312hfc0f314_104`; retain Python 3.12.11 and NumPy 2.3.0. Package and module versions are documented separately. |
| ZIP filename decoding | Bare `unzip` corrupted 26 Chinese filenames. UTF-8-aware extraction restored all filenames, and the original release check passed 1514/1514 files. | Extraction instructions use `unzip -O UTF-8`; where `-O` is unavailable, use `python3 -m zipfile -e platform_validation_20261019.zip .`. |
| Hardware preprocessing verification | One source-manifest hash was stale. FFT hashes differed for 205/205 segments across platforms, while existing raw and EMA state hash checks passed. Three checked examples in the report had a maximum absolute FFT difference of about `1.6e-13`, within `1e-12`; this is not a measured maximum over all 205 segments. | Source and release manifests are refreshed. Use all 205 frozen FFT arrays in `hardware_reproduction/data/preprocessing_reference.npz`, with elementwise **max_abs<=1e-12, rtol=0**. All 205 reference arrays were matched byte for byte to their existing `X_sha256` values during archival preparation, without executing estimators. Retain strict existing raw file, EMA state and source hashes, with original `X_hash` results as diagnostics. |

The FFT reference identity check preserves the existing reference arrays before introducing a numerical comparison across platforms. Its archival metadata is in `hardware_reproduction/data/preprocessing_reference.json`; it is not an experimental rerun. These repairs change neither paper metrics, estimators, output rules nor downstream acceptance thresholds.

## Conditional evidence already available

| Stage or review | Reported result for the previous candidate under the conditions above |
|---|---|
| 1. Saved-data reconstruction | Completed release checks, saved statistics and figure reconstruction, with the failing hardware preprocessing check skipped. |
| 2. Replay from released inputs | Passed 16,500 controlled trials, 2400 FDTD spacing trials, 5330 hardware estimates, 205 hardware spectra, 15 FDTD spectra, 300 parametric references and 4100 spectrum comparisons. |
| 3. Fresh input generation | Passed full coverage: 5 parametric sequences and 44 FDTD sequences (29 controlled and 15 correspondence). All 44 new FDTD CIR sequences were bitwise equal to their references. |
| 4. Replay from fresh inputs | Passed the full downstream replay using newly generated simulation CIR, with the same hardware-preprocessing exception. |
| Tables and figures | Tables V–IX met the fixed numerical checks. All eight figure pairs passed human review. The stage-2 and stage-4 comparison PNGs had matching hashes for all eight pairs. |

This is evidence for the reported conditional run of the previous candidate. It is not a validation result for the revised archive. Failed checks and their logs remain part of the validation record.

## Revised candidate: no new validation run

| Item | Status |
|---|---|
| Reported issues | Repaired as described above |
| New environment installation | Not rerun; the pin uses the build installed in the previous conditional run |
| Hardware preprocessing and stages 1–4 | Not rerun |
| Numerical tables and eight-figure review | Previous conditional evidence retained; no new review claimed |

The [Linux guide](linux_validation.md) documents independent reproduction. The previous conditional outcome does not establish an unconditional pass for this revised candidate.
