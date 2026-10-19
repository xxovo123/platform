# Single-anchor IR-UWB radar: FDTD, CIR processing and JSF-MUSIC

Materials for **CIR-Based Joint Range–Azimuth Localization for Single-Anchor IR-UWB Radar: Implementation and Validation**, by Shangyou Zeng and Lin Sun. Candidate identifier: `revision-20261019-candidate`; archive: `platform_validation_20261019.zip`.

**Linux is required for the documented reproduction workflow.** Create a new Linux Conda environment from `environment.yml`, including Meep for electromagnetic field generation. Native Windows is not a supported environment for the complete workflow. Reported issues have been repaired without a new validation run; previous conditional numerical reproduction passed. See [validation status](docs/validation_status_20261019.md) for the scope of that evidence.

Start with [the full validation guide](docs/linux_validation.md), [expected results and tolerances](validation/expected_results.json), and [the manuscript-to-code map](docs/manuscript_reproduction.md). [中文入口](README_zh.md).

## Released experiments and data

All **205 evaluation segments** of original chip-recorded complex channel impulse response (CIR) are included, with truth, geometry, calibration, recording identifiers and continuous-background state. Hardware code includes JSF-MUSIC, five adapted baselines, twenty ablation settings and the fixed repeated-measurement selection. Every estimator receives the target count.

The current target-spacing experiment contains only **24 FDTD conditions / 2400 noise trials**: transverse, radial and oblique pairs at **0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75 and 1.00 m**. The position grid is fixed at 0.05 m. No parametric target-spacing dataset or supplementary-material package is included. Parametric noise-localization and simulation–measurement reference experiments remain included.

| Directory | Contents |
|---|---|
| `src/` | Frozen FDTD, receiver-chain, signal-model and estimator dependencies |
| `data/cache/` | Five FDTD and five parametric clean geometries for noise/prefix experiments |
| `data/results/runs/` | Four non-spacing families, 150 conditions and 16,500 trials |
| `spacing_revision/` | Current 24 FDTD spacing conditions, clean inputs, original outputs and spectral references |
| `data/q4/` | 15 FDTD configurations, 205 hardware spectra, 300 parametric reference trials |
| `data/hardware/` | Original 205 raw CIR segments and acquisition metadata |
| `hardware_reproduction/` | Raw/EMA inputs, six methods, twenty ablations, metrics and tables |
| `manuscript_assets/` | Explanatory artwork, approved anonymized photos and drawing recipes |
| `validation/reference/` | Eight approved figure PDFs, nine table environments and fixed comparison data |
| `validation/` | Full replay, fresh input generation and acceptance scripts |

Non-spacing trial counts are E1_A=9000, E1_B=2250, E4_A=4500 and E4_B=750. E4_A is a retained parametric prefix reference; Fig. 5 uses E4_B. All retained families enter the full replay. Trials sharing a clean geometry are not independent physical environments. Zero-success and failed detections remain in the data.

## Linux environment

Run in a new directory and environment; do not clone an existing research environment or import its caches. Choose an absolute environment prefix:

```bash
# From a new working directory containing the supplied archive:
unzip -O UTF-8 platform_validation_20261019.zip
cd platform
```

If your `unzip` does not support `-O`, use `python3 -m zipfile -e platform_validation_20261019.zip .` instead, then enter `platform/`. Preserve UTF-8 filenames; bare `unzip` produced corrupted Chinese filenames in the previous validation.

```bash
conda env create --prefix /absolute/path/jsf-validation-env --file environment.yml
conda activate /absolute/path/jsf-validation-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
python -c "import sys, meep, numpy, scipy, matplotlib; print(sys.executable); print(meep.__version__, numpy.__version__, scipy.__version__, matplotlib.__version__)"
python validation/check_release.py
```

The required versions are Python 3.12.11, the Conda package **`pymeep=1.30.1=nompi_py312hfc0f314_104`**, NumPy 2.3.0, SciPy 1.16.0 and Matplotlib 3.10.3. This exact Linux pymeep build was used in the previous conditional server run and reports **`meep.__version__ == '1.30.0'`**; the package and module version strings differ. Python and NumPy remain unchanged. Save the resolved Conda package list and BLAS configuration. Do not install the unrelated PyPI package named `meep`.

## Full reproduction

Commands start at the repository root. New results use separate directories; original inputs and references remain unchanged.

```bash
# 1. Hashes, preprocessing, saved metrics, numerical figures and artwork.
python validation/run_validation.py --mode saved --out generated_validation/saved

# 2. All estimators from released CIR and raw hardware inputs.
python validation/run_validation.py --mode replay --out generated_validation/replay --workers 4

# 3. Fresh inputs: 5 analytic geometries, 29 controlled FDTD geometries
#    (5 noise + 24 spacing), and 15 measured-geometry FDTD sequences.
python validation/regenerate_inputs.py --out generated_validation/fields --workers 4

# 4. Downstream replay using ONLY the newly generated simulation inputs.
python validation/run_validation.py --mode replay --out generated_validation/from_fields --workers 4 \
  --generated-index generated_validation/fields/generated_index.json
```

Step 1 reconstructs saved data. Step 2 runs 16,500 controlled trials, 2400 spacing trials, 5330 hardware method/configuration runs, 205 hardware spectra, 15 FDTD spectra and 300 parametric reference trials. It does not solve electromagnetic fields. Steps 3–4 test field generation and its downstream results. Report these stages separately. Full field generation is expensive; `--workers` parallelizes states within one scene. Each worker has one BLAS thread; inspect RAM before increasing four workers.

Reports record commands, logs, environment, identities and numerical differences. `--resume` can continue only a run with the same package identity. It never substitutes archived CIR for missing new inputs. Smoke examples and shortened demonstrations are not full acceptance. See [the guide](docs/linux_validation.md) for pass criteria and visual review.

Hardware preprocessing uses frozen FFT arrays for all 205 evaluation segments in `hardware_reproduction/data/preprocessing_reference.npz`. All 205 reference arrays were matched byte for byte to their existing `X_sha256` values during archival preparation, without executing estimators. Elementwise comparisons require **max_abs<=1e-12, rtol=0**. Existing raw CIR file, continuous EMA state and source hash checks remain strict; original `X_hash` results remain diagnostics. This comparison does not change paper metrics or estimators.

Individual commands:

```bash
python reproduce.py verify
python reproduce.py tables --out outputs/tables
python reproduce.py figures --out outputs/figures
python spacing_reproduce.py verify
python spacing_reproduce.py replay --task E2_B_transverse_d300_snr+10 --trial 1000
python spacing_reproduce.py figures --out outputs/spacing
python q4_reproduce.py verify-spectra
python q4_figures.py --out outputs/q4
```

Both plot entry points generate the **same current FDTD-only Fig. 4**. Full replay additionally plots newly computed values. Compare numerical data and visual content; PDF timestamps/font serialization can differ across environments.

## Processing and interpretation

The two-dimensional Meep solver computes `Ez` for all scatterers together. Controlled scenes use one transmitter, four receivers, 128 actual slow-time states and empty-scene subtraction. Correspondence uses projected measured geometry, an exponential moving average of 0.25, 32 startup states and 128 retained states. Simulated metal cylinders do not establish measured close-person resolving capability.

| Controlled setting | Value |
|---|---|
| Carrier / nominal bandwidth | 7.9872 GHz / 500 MHz |
| Transmitter | (0,0,0) m |
| Receiver x coordinates | −0.038, −0.019, 0, 0.019 m; y=z=0 |
| Field grid / Courant factor | 350 cells/m / 0.5 |
| Duration / PML thickness | 20 Meep time units / two carrier wavelengths |
| Motion step SD / bound | 0.002 m / 0.01 m |
| Numerical CIR | complex128, [128 states, 4 channels, 32 taps] |

The source is the derivative of a Blackman–Harris-windowed carrier, duration `2/B`. One Meep time unit is `1 m/c0`; 20 units are about 66.7 ns. The receiver applies background removal, analytic-signal conversion, baseband mixing, filtering, sampling and reference correlation. The 400 MHz antialias cutoff is distinct from the nominal bandwidth and ±250 MHz frequency selection.

Controlled processing retains taps `[0:30]`, adds complex noise, computes a length-32 fast-time FFT, selects 17 in-band bins and stacks frequency then channel. `X` is stored as `[128,68]` with row snapshots; the paper uses column snapshots. Hardware raw CIR is `[128,8,32]`, with selected pairs `[2,3,6,7]`, historical tap window `[2:32]`, channel calibration and restored continuous EMA state. See [data structures and examples](data/README.md).

Covariance, forward–backward processing, diagonal loading and the MUSIC criterion follow the frozen code. Steering uses transmit–target–receive path length and configured system delay. Ordinary localization outputs use 3×3 maxima, eight-connected plateau handling, relative height 0.30, minimum separation 0.10 m and at most supplied K outputs. These output rules remain unchanged.

Fig. 4 instead reports **truth-referenced two-peak correspondence before final peak selection**, not final localization success. Two distinct local maxima must match the two truths within `min(0.10 m,d/3)`. Truth is used after spectrum formation, never as an estimator input. The exact metric and original outputs are documented [here](spacing_revision/README.md).

Historical `snr_db` labels remain generator parameters. Expected CIR SNR is `snr_db + 10*log10(2)` because of the executed noise scaling; scaling and seeds are unchanged. RMSE pools ungated assigned-pair squared errors; detection uses a separate 0.30 m one-to-one gate. Unassigned targets are excluded from RMSE and retained in coverage/miss statistics.

## Traceability

`data/release_manifest.json` hashes every packaged file except itself. Source and run manifests retain numerical provenance. Historical identifiers in shared source modules describe that source version; the public execution set is enumerated by `validation/expected_results.json`, retained run tasks and `spacing_revision/conditions.json`.

`validation/geometry_gate.py` isolates the executed small-spacing preflight correction: nominal bounded motion is counted once, and actual states are checked for overlap. It changes neither field solver inputs nor trajectories, noise, estimator rules or numerical modules. This is not a grid-convergence study.

Photos and the original CIR-chain vector PDF are released static sources. The CIR-chain diagram has no confirmed editable authoring source. Other explanatory figures have drawing recipes. See [hardware reproduction](hardware_reproduction/README.md), [raw hardware data](data/hardware/README.md) and [hardware/asset mapping](docs/hardware_reproduction_map.md).
