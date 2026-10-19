# Manuscript-to-code and data map

Approved 2026-10-18 manuscript, English 12 pages. **Run the complete workflow on Linux in a new environment from `environment.yml`.** Commands start at repository root. [Full validation guide](linux_validation.md).

| Item | Input / implementation | Output and comparison |
|---|---|---|
| Fig. 1, CIR chain | `manuscript_assets/figures/CIR_Time_Domain_Signal_Acquisition_Procedure.pdf` | Static vector source, exact hash; no confirmed editable source. |
| Fig. 2, workflow | `manuscript_assets/source/` | `python manuscript_assets/regenerate.py`; `algo_flowchart_restored.pdf`; schematic spectrum. |
| Tables I–II | `src/exec2/jsfsim/protocol.py`, tasks and clean metadata | `reproduce.py verify`; approved table text in `validation/reference/manuscript_tables/`. |
| Fig. 3, noise localization | E1_A 9000, E1_B 2250 trials; `validation/control_replay.py` | New predictions, metrics and CSV; `reproduce.py figures --tables <new tables>`. |
| Fig. 4, spacing sensitivity | `spacing_revision/`: 24 FDTD conditions, 2400 trial records | `spacing_reproduce.py replay-all`; `figures --results <new JSON>`; all 24 spectral rates, unchanged original outputs. |
| Fig. 5, snapshots | E4_B 750 trials, genuine prefixes of a 128-state sequence | `validation/control_replay.py`; `reproduce.py figures`. E4_A adds 4500 checked reference trials, not plotted in Fig. 5. |
| Fig. 6, arrays | `manuscript_assets/regenerate.py`, hardware geometry | Generated `antenna_layout.pdf`; four selected pairs and seven virtual positions. |
| Fig. 7, scenes | Approved crops, `manuscript_assets/data/nominal_positions.json` | Generated `hardware_setup.pdf`; static photos plus reconstructed position diagram. |
| Tables III–IV | Hardware raw index, recording metadata, contexts, EMA states | All 205 IDs, 31 recordings, 15 configurations; raw preprocessing and exact truth. |
| Table V | Raw CIR → new JSF-MUSIC predictions → assignments | Full replay and `tables --replay-input ...`; `groups.csv`. |
| Table VI | Fixed 30 groups × 4 temporal selections | New `stability.csv`, `selected_measurements.csv`; all 120 identities/ranks, sample SD. |
| Table VII | Six implementations, each using all 205 segments | `replay --methods all`; new `summary.csv` and `paper_checks.json`. |
| Table VIII | Twenty hardware configurations, including reference | Same full replay; 4100 executions and all printed values. |
| Fig. 8 | 15 FDTD + 205 raw-derived hardware spectra | `q4_hardware_replay.py`, `validation/q4_replay_all.py`, then `q4_figures.py --spectra <new spectra> --records <new CSV>`. |
| Table IX | All 205 spectral/identity comparisons, equal configuration weighting | New `table_IX.csv`: indoor/outdoor/combined correlation, displacement, pairs and missing. |
| Parametric correspondence | Geometry, deterministic seeds, direct frequency model, 300 trials | `validation/q4_replay_all.py`; all 4100 comparisons and 0.474 macro correlation. Not the Deng estimator. |
| Fresh simulation inputs | `validation/regenerate_inputs.py` | 5 analytic, 29 controlled FDTD, 15 measured-geometry FDTD; 128 retained states, Q4 also 32 startup states. |

Both root figure entry points produce the current FDTD-only Fig. 4. All 24 conditions remain irrespective of success. Parametric target-spacing data and dedicated execution are excluded; shared frozen numerical modules retain their original identities and support the other published parametric experiments.

The full driver checks saved data, runs all estimators and reconstructs figures. Fresh Meep generation and downstream replay from the new inputs are separate stages. Historical `hardware_reproduction/full_replay/` evidence is not proof that this candidate passed a new Linux environment.

The [hardware map](hardware_reproduction_map.md) details interfaces and source paths. Check numerical equivalence and visual content, not cross-version PDF binary identity. Static photography cannot be reconstructed by an estimator.
