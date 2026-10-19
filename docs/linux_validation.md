# Independent Linux validation

This guide documents the reproduction workflow for `revision-20261019-candidate` (`platform_validation_20261019.zip`). Use **Linux**, a new directory and a new Conda environment. Reported issues have been repaired without a new validation run; previous conditional numerical reproduction passed. See [the dated status](validation_status_20261019.md) for the evidence and conditions. This guide does not authorize replacing the reference results when a comparison fails.

## 1. Install and record the environment

Verify the supplied archive's external SHA-256, unpack it into a new working directory, and enter `platform/`. Keep the original archive and `data/release_manifest.json` unchanged.

```bash
# In a new working directory, after checking the supplied archive hash:
unzip -O UTF-8 platform_validation_20261019.zip
cd platform
```

If `unzip` lacks `-O`, extract with `python3 -m zipfile -e platform_validation_20261019.zip .` instead, then enter `platform/`. Bare `unzip` corrupted 26 Chinese filenames in the previous validation. A filename mismatch is a failed extraction, not permission to skip a release hash check.

```bash
conda env create --prefix /absolute/path/jsf-validation-env --file environment.yml
conda activate /absolute/path/jsf-validation-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
mkdir -p generated_validation/environment
python -c "import sys,platform,meep,numpy,scipy,matplotlib; print(platform.platform()); print(sys.executable); print(meep.__version__,numpy.__version__,scipy.__version__,matplotlib.__version__); numpy.show_config()" > generated_validation/environment/runtime.txt
conda list --explicit > generated_validation/environment/conda-explicit.txt
conda env export --prefix /absolute/path/jsf-validation-env > generated_validation/environment/conda-resolved.yml
python validation/check_release.py
python validation/regenerate_inputs.py --plan --out generated_validation/fields
```

Use `environment.yml`, not an existing environment clone. The Conda package is pinned to **`pymeep=1.30.1=nompi_py312hfc0f314_104`**, the exact Linux build used in the previous conditional server run; its module reports **`meep.__version__=1.30.0`**. Record both the installed package/build and the module string. Keep Python **3.12.11** and NumPy **2.3.0**; pinning the Conda package to `pymeep=1.30.0` was incompatible with that combination in the previous validation. The unrelated PyPI package named `meep` is not the electromagnetic solver. Do not use `mpirun` with the Python process pool: the documented driver uses one scene at a time and `--workers` parallel state processes. Record CPU, RAM, disk availability, OS and Conda package/build versions. Do not change the published resolution, duration, seeds, thresholds or matching rules to make a comparison pass.

## 2. Saved-data reconstruction

```bash
python validation/run_validation.py --mode saved --out generated_validation/saved
```

This checks all release hashes, non-spacing records, 2400 spacing records, raw hardware preprocessing, recorded source fingerprints, hardware table calculations and all 205 saved correspondence comparisons. It reconstructs the four numerical figures and three editable schematics/photo layouts. Fig. 1 remains a hash-checked static vector source. No estimator replay or new field solve is implied by this stage.

The hardware validation restores continuous EMA state for every segment. A reset EMA at each segment is incorrect. The 205-segment set is not replaced by the 120-segment temporal subset. Original integer I/Q values, selected channels, calibration and known K are retained.

The hardware check uses all **205 FFT arrays** frozen in `hardware_reproduction/data/preprocessing_reference.npz`. All 205 reference arrays were matched byte for byte to their existing `X_sha256` values during archival preparation, without executing estimators. Runtime elementwise comparison requires **max_abs<=1e-12, `rtol=0`**. Existing raw CIR file, continuous EMA state and source hashes retain strict checks, as do release files. Original `X_hash` results remain diagnostic records; a cross-platform FFT hash mismatch alone is not a numerical failure. This changes the preprocessing comparison, not any estimator, paper metric or downstream tolerance.

## 3. Full estimator replay

```bash
python validation/run_validation.py --mode replay --out generated_validation/replay --workers 4
```

| Work | Required coverage | Comparison |
|---|---:|---|
| Controlled noise/prefix processing | 150 conditions, 16500 trials | Seeds/status/counts exact; output coordinates, assignment metrics, pooled RMSE/DR |
| Current FDTD spacing | 24 conditions, 2400 trials | All identities and noise seeds, original predictions and strict-output flags; matched local maxima and spectral correspondence flags |
| Hardware methods/ablations | 205×26=5330 estimates | Every method/configuration and segment, position sets with assignment; then new assignments and statistics |
| Hardware spectra | All 205 | Raw CIR + EMA → normalized spectrum and peaks |
| FDTD correspondence | All 15 | Released CIR → spectrum and peaks |
| Parametric correspondence | 15×20=300 runs | 4100 spectrum/identity comparisons with the hardware set |
| Tables/figures from new outputs | Tables V–IX and Figs. 3–5,8 | Approved numeric values and visual content |

The driver generates new prediction files, reconstructs hardware tables from those files, writes new Fig. 4 rates from the replayed spectra, and draws numerical figures from new results. `numeric_comparison.json` compares all hardware summary/group/stability/selection CSV cells and all 24 Fig. 4 data points with pre-run references.

## 4. Generate fresh simulation inputs and use them downstream

```bash
python validation/regenerate_inputs.py --out generated_validation/fields --workers 4
python validation/run_validation.py --mode replay --out generated_validation/from_fields --workers 4 \
  --generated-index generated_validation/fields/generated_index.json
```

The required generation set is **5 parametric sequences, 29 controlled FDTD sequences and 15 correspondence FDTD sequences**. Controlled FDTD means the five noise-layout geometries plus the 24 Fig. 4 geometries. Each retains 128 actual states. Correspondence additionally solves 32 EMA startup states and its two-transmitter geometry. The field settings are 350 cells/m and 20 Meep units; the receiver chain and trajectory seeds are fixed.

Start with four state workers, single-thread BLAS, one scene at a time. Monitor memory and CPU rather than assuming more workers are faster. Field states and provenance remain in this new output directory for resume. Do not copy old field caches or use an old `mp` environment. `generated_index.json` explicitly points downstream at new CIR; missing entries fail rather than falling back to released data. Running only a shortened scene or one configuration does not meet this stage's full coverage.

Fresh complex CIR comparisons use both relative L2 error and maximum absolute error divided by the reference peak magnitude, each <=1e-7. No amplitude fitting, phase alignment, time shifting or truth-fitted correction is applied. The subsequent complete estimator replay must still meet exact count/flag and position criteria. A close CIR comparison alone does not guarantee full downstream acceptance.

## 5. Fixed expectations and tolerances

The machine-readable authority is [`validation/expected_results.json`](../validation/expected_results.json), supported by original trial files, `validation/reference/` and approved table extracts. It is prepared before remote runs.

| Quantity | Acceptance |
|---|---|
| Released file hash, segment/trial identity, seed, number of trials, flags, TP/FN/FP | Exact |
| Existing hardware raw CIR file, continuous EMA state and source hashes | Exact |
| Frozen hardware FFT reference identity, all 205 segments | Matched byte for byte to existing X_sha256 values during archival preparation |
| Hardware FFT arrays, all 205 segments | max_abs<=1e-12, rtol=0, against frozen arrays; original X_hash results retained as diagnostics |
| Simulation output position | <=1e-10 m, same number of outputs |
| Hardware output position | <=1e-8 m after one-to-one matching, same number of outputs |
| Normalized spectrum maximum absolute difference | <1e-7 |
| Spectral correlation difference | <1e-7 |
| Pairwise SSE difference | <=1e-9 |
| Derived hardware CSV floating values | <=1e-8; integer/count/selection fields exact |
| Printed table values | Same rounding at the published precision |
| PDF/PNG | Numeric data, axes, series, markers, examples and layout checked; PDF byte equality is not required |

Fig. 4 counts out of 100 in the spacing order `.15,.20,.25,.30,.40,.50,.75,1.00`:

| Direction | Two-peak correspondence |
|---|---|
| Transverse | 0, 0, 0, 87, 100, 100, 100, 100 |
| Radial | 0, 0, 0, 0, 100, 100, 100, 100 |
| Oblique | 0, 0, 100, 100, 100, 100, 100, 100 |

These are spectral diagnostic counts. The original final-output counts, including the radial zeros, are separate reference fields and must remain unchanged. Do not substitute output success for the Fig. 4 ordinate.

Hardware JSF-MUSIC reference: position RMSE 0.1789976248462499 m, range RMSE 0.06870345272668464 m, azimuth RMSE 6.14794670188026 degrees, TP/FN/FP=315/23/19, assigned pairs=334, DR=93.19526627218934%. All six methods and twenty ablations—not just this row—must pass. Fixed temporal selection: 30 groups, 120 identity/rank pairs.

Table IX rows (correlation, displacement m, pairs, missing): indoor `(0.674,0.179,162,2)`, outdoor `(0.722,0.162,172,2)`, combined `(0.696,0.171,334,4)`. Parametric macro correlation rounds to 0.474. Configuration averaging differs from pooling all segments; use the released code and definitions.

## 6. Inspect all figures

The driver produces `visual_comparison/fig01_*.png` through `fig08_*.png`, showing approved and generated figures side by side. Its `manifest.json` remains **human review pending** until the operator inspects them. `verification.json` passing numerically does not sign off visual review.

Inspect all eight pairs: no omitted conditions or zero-success points; Fig. 4 has eight markers per direction, circle/square/triangle symbols and correct ordinates; all other series, fixed examples, color limits, axes, units, legends and panels correspond; no stretched markers, clipping or overlaps. Fig. 8 uses the fixed identifier-order examples. Confirm static photos and Fig. 1 by hash, not by claiming to regenerate measurements. Table I–IV declarations should be checked against the released settings/geometry, with exact approved extracts supplied for that review.

Record findings in `validation/acceptance_report_template.md` copied into the run folder. Keep rendered comparisons and the checklist with the report.

## 7. Monitor, resume and report

Each stage writes `commands.json`, per-command logs, environment and verification JSON. The driver prints `START` and `DONE`. Inspect the active command log with `tail -f`; do not infer a stall only from a quiet parent log. Field generation prints per-scene/state activity, writes `checks.json`, `generated_index.json` and field provenance. Monitor process count, RSS, CPU and output timestamps.

To resume the same package, append `--resume` to the interrupted driver/generation command. Completed driver steps are skipped only with the same package identity and command; partial field states are reused only in this independent run directory. Final numeric comparison re-reads outputs. If code or references change, create a new package identity and a new output directory; do not mask the change as a resume.

On any mismatch, stop that stage, keep the failed records, exit code, environment and all logs, and report condition/segment IDs plus actual/reference values. Do not relax tolerances, tune thresholds, remove scenes, replace generated arrays, or assert full success from a partial run. Diagnose version/build differences first. Any change affecting paper values requires author review and a new sealed version.

Retain the archive hash, release-manifest hash, resolved environment, all four stage reports, commands/logs, full numerical comparisons, new figure pairs, human visual checklist, fresh field coverage/provenance and unresolved issues. Packaging and historical conditional evidence do not establish that the revised candidate passed full validation.
