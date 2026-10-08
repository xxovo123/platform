# Single-anchor IR-UWB radar: simulation code and CIR data

Materials for **Joint Range-Azimuth Estimation for Single-Anchor IR-UWB Radar: Simulation, Algorithm, and Experimental Validation**, by Shangyou Zeng and Lin Sun.

This release contains the simulation code used for the major revision, selected simulation data, and the complete **205 evaluation segments** of chip-recorded hardware CIR. Hardware acquisition and evaluation software is outside this release. The simulation numerical core and saved predictions retain their original values; portable entry points are provided at the repository root.

See [data/README.md](data/README.md) for array definitions, [data/hardware/README.md](data/hardware/README.md) for the original hardware CIR, and [README_zh.md](README_zh.md) for Chinese instructions.

## Contents

| Path | Contents |
|---|---|
| `src/fdtd/` | Meep solver, simultaneous-target scenes, receiver chain, background processing and CIR construction |
| `src/analytic/` | Parametric signal-model CIR generation |
| `src/exec2/jsfsim/` | Final simulation layouts, deterministic seeds, noise, estimation and metrics |
| `src/vendor/` | Frozen estimator dependencies used by the simulation package |
| `data/cache/` | 14 FDTD and 29 parametric clean CIR sequences, with independent truth and settings |
| `data/field_samples/` | Six original solver records: empty scene and nominal target state for S1, S2 and S5 |
| `data/results/runs/` | All 24,150 saved formal simulation trial records, task definitions and fixed spectrum examples |
| `data/q4/` | Simulation–measurement correspondence inputs, saved spectra, fixed-seed reference results and summaries |
| `data/hardware/` | Exactly 205 raw, unprocessed hardware CIR segments and acquisition metadata |
| `data/settings/` | Effective protocol, recorded environment and source-pulse reference |
| `data/tables/` | Final numerical tables and manuscript SNR labels |

The large collection of intermediate FDTD field states is represented by fixed examples rather than copied in full. All 128-snapshot controlled CIR sequences are included. Missing intermediate field states can be regenerated with Meep.

## Environment

Saved-data checks, plots and estimator replay need Python, NumPy, SciPy and Matplotlib:

```bash
python -m pip install -r requirements.txt
```

The original remote simulation environment used Python 3.12.11, Meep 1.30.0, NumPy 2.3.0 and SciPy 1.16.0. For FDTD regeneration, use a Linux Conda environment:

```bash
conda env create -f environment.yml
conda activate jsf-music
```

## Reproduction

Run from the repository root. These commands read the released data and leave the original arrays unchanged:

```bash
python reproduce.py verify
python reproduce.py tables --out outputs/tables
python reproduce.py figures --out outputs/figures
python reproduce.py replay --layout S2
python q4_reproduce.py summarize
python q4_reproduce.py replay --config G-420 --trial 0
```

`verify` checks source/data hashes, all clean CIR shapes and the saved assignment/detection metrics for all 24,150 formal records. `tables` recomputes pooled position, range and azimuth RMSE, detection and resolution counts from the saved predictions. `figures` regenerates the three simulation curve figures from the final tables. `replay` regenerates noise and runs JSF-MUSIC for one archived trial, comparing the resulting positions with its saved result. No command above invokes the FDTD solver.

To regenerate one controlled 128-state FDTD scene, choose a new output directory:

```bash
python regenerate_fdtd.py --clean-id cln_de385a3b2982 --out outputs/new_fdtd_S2 --workers 1
```

This command performs actual Meep simulations. `--max-states` can make a shortened demonstration; its output is not the full sequence used in the paper. The archived production settings are 350 cells/m and 20 Meep time units.

## Interpretation and review coverage

The simulation package evaluates JSF-MUSIC. The five external estimators were compared on hardware in the manuscript; their complete hardware evaluation pipeline is not included here.

| Review item | Released evidence |
|---|---|
| Q1 | Joint channel–frequency observation construction, transmit–target–receive steering and MUSIC implementation |
| Q2–Q3 | All 205 raw hardware CIR segments, segment identifiers, acquisition settings and target truth |
| Q4 | Fifteen simulation configurations, all 205 saved measurement spectra, correspondence records and fixed-seed parametric reference |
| Q5 | Complete controlled simulation records for SNR, directional separation and snapshot sensitivity; archived hardware factor tables |
| Q6 | The same hardware input data are released; hardware baseline execution software is outside this release |
| Q7 | Simulation source, parameters, seeds, data structures and verified reproduction entry points |

**SNR labels:** historical `snr_db` in the saved run records is a generator parameter. Because the recorded complex-noise scaling gives half of its nominal target noise power, the expected CIR SNR is `snr_db + 10*log10(2)`, approximately `snr_db + 3.0103 dB`. The manuscript and `data/tables/manuscript/` use this expected SNR. Reproduction preserves the executed noise scaling and seeds.

RMSE pools squared errors over all assigned pairs, without a distance gate. Unassigned targets do not enter RMSE. Detection uses a separate 0.30 m one-to-one match; its missed-target count differs from the unassigned-target count. Noise trials reuse their layout's clean sequence, and snapshot tests use genuine prefixes of that sequence.

The provided plotting scripts reproduce the numerical content of the simulation figures; PDF byte identity across environments is not asserted. Arrays and the original numerical source are tracked by SHA-256 in `data/source_manifest.json` and the hardware export index. Internal SSH utilities, credentials and conversation records are excluded.
