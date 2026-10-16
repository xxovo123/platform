# Hardware reproduction and manuscript asset map

Paths are relative to the repository root (`platform/` locally). Numbers follow `Mandatory_Corrections_Working_20261016.aux`; TeX labels take precedence after renumbering. This integration changes no manuscript or IEEE template.

| Paper item | Source and entry | Output / scope |
|---|---|---|
| Fig. 1, CIR receive chain | `manuscript_assets/figures/CIR_Time_Domain_Signal_Acquisition_Procedure.pdf` | Original static vector PDF: mixer/filter/ADC/CIR/FFT diagram. No confirmed editable authoring source found. |
| Fig. 2, four-stage workflow | `manuscript_assets/figures/algo_flowchart_restored.svg`; `source/draw_algorithm_flowchart.py` + `source/restore_workflow.py` under that package | Editable original SVG/PDF. `python manuscript_assets/regenerate.py` rebuilds tensor/matrix artwork; drawn spectrum is schematic. |
| Fig. 6, `fig:antenna_array` | `manuscript_assets/regenerate.py`, ported original `build_manuscript_assets.py` antenna section | `generated/antenna_layout.pdf/png`: physical and virtual geometry in mm; eight ordered pairs, seven virtual positions, four selected channels. Geometry is also preserved in hardware contexts/recording metadata. |
| Fig. 7, `fig:hardware_setup` | Approved `manuscript_assets/photos/*_anonymized_crop.jpg`, `data/nominal_positions.json`, `regenerate.py` | `generated/hardware_setup.pdf/png`: static acquisition photos plus regenerated layout and position schematic. Seven positions form a union across configurations, not seven simultaneous people. |
| Table III, `tab:hardware_settings` | `hardware_reproduction/data/raw/recordings/*.json`, `data/contexts.json`, preserved protocol/source; `reproduce_hardware.py verify` | Settings, not empirical performance. Continuous EMA state must be restored. |
| Table IV, `tab:configurations` | Raw index and each segment's `truth_xy_m`; `manuscript_assets/data/nominal_positions.json` | Actual one-, two-, three-person configurations. Raw arrays retain precise coordinates; schematic labels round to three decimals. |
| Table V, `tab:groups` | `reproduce_hardware.py all-saved` | `groups.csv`: environment and count partitions of all 205 evaluation segments. `validate_release.py` checks 20 printed numeric cells. |
| Table VI, `tab:stability` | Same; fixed `data/reference_selected_measurements.csv` | `stability.csv`, `selected_measurements.csv`: 30 groups, four temporal measurements each, sample SD (`ddof=1`). Validator checks all 120 IDs/ranks and 90 printed cells. |
| Table VII, `tab:baseline` | Six frozen sets indexed by `data/result_map.json`; `all-saved`; raw estimator replay via `replay --methods all` | `summary.csv`, `per_segment.csv`, `paper_checks.json`. All six methods cover all 205 segments. RMSE uses ungated assigned pairs; DR/FN/FP use the protocol detection gate. K is supplied. |
| Table VIII, `tab:ablation` | Twenty `data/config_matrix.json` configurations, frozen predictions and raw replay | Same outputs: frequency count, digital bandwidth, snapshots, channel subsets, FB averaging. Twenty includes the reference. |
| Table IX, `tab:q4_comparison`, Fig. 8, `fig:q4_comparison` | `data/q4/`, `q4_reproduce.py`, `q4_figures.py` | Mixed simulation/measurement material; see `manuscript_reproduction.md` for saved-spectrum verification, replay and figure commands. |

All abbreviated hardware data/source/output paths above are relative to `hardware_reproduction/`. The four archived `outputs/hardware_*.pdf/png` plots are **numerical review plots**: comparison, ablation, stability and nominal positions. They do not reproduce the manuscript layouts, photography or original spectrum panels. The position plot uses truth only, not localization estimates. Original non-simulation manuscript assets live in `manuscript_assets/figures/`.

## Six implementations and twenty ablations

| CLI ID | Display name | Implementation inside `hardware_reproduction/source/大修执行_20260923/` |
|---|---|---|
| `this_method` | JSF-MUSIC | `04_Q6/common/q6_methods_this_method.py` |
| `b1_deng` | Deng adaptation | `04_Q6/hardware/methods/b1_deng.py` |
| `b2_kakouche` | Kakouche adaptation | `04_Q6/hardware/methods/b2_kakouche.py` |
| `c1_jade` | JADE adaptation | `04_Q6/hardware/methods/c1_jade.py` |
| `c3_kim` | Kim adaptation | `04_Q6/hardware/methods/c3_kim.py` |
| `b5_wang_full2d` | Traditional 2D MUSIC | `04_Q6/hardware/methods/b5_wang_full2d.py`; conventional joint spectrum, not Wang's successive estimator |
| `ablation/<ID>` | Twenty hardware configurations | `03_Q5/hardware_ablation_20260926/scripts/q5_backend.py` |

Adaptation/formula notes remain in `hardware_reproduction/docs/`. Baselines share a CIR interface, not the original papers' acquisition systems; results do not establish a general ranking.

## Commands and retained evidence

From `hardware_reproduction/`, with NumPy, SciPy and Matplotlib installed:

```text
python -B reproduce_hardware.py verify --out generated/check
python -B reproduce_hardware.py all-saved --out generated/check
python -B validate_release.py
python -B reproduce_hardware.py replay --methods this_method --limit 1 --out generated/smoke
python -B reproduce_hardware.py replay --methods all --out generated/full_replay
```

`verify` checks raw CIR, restored EMA/FFT features and source hashes. `all-saved` independently recomputes assignments, metrics and review plots; it is not estimator replay. Historical `full_replay/replay_verification.json` and `replay_comparison.csv` document all 205 × 26 = 5,330 executions and position agreement within `1e-8 m` after permutation matching. Integration reruns preprocessing, saved metrics and a short hardware smoke check, without claiming another complete estimator replay.

All 205 raw segments, 205 initial EMA states, 31 recording metadata records, contexts, protocol, dependencies and 26 frozen result sets remain. The continuous EMA is not reset at segment boundaries. `integration_manifest.json` inventories copied files and omissions. The omitted 28,433,548-byte full-replay prediction/diagnostic JSON duplicates frozen positions and regenerable diagnostics; its original hash/size remain recorded. Compact per-run comparisons and the verification summary remain. Archived timing observations are not timing guarantees. Cache and smoke duplicates are excluded.

The pre-existing `data/hardware/` raw release remains for compatibility; the hardware supplement is independently portable. Archived `data/paper_reference/` TeX files are unchanged historical numerical references; captions can differ from the October 16 manuscript. Archived `outputs/` evidence is explicitly included in Git; new runs use ignored `generated/`.

Only two previously approved anonymized submission crops are copied, at their frozen hashes. Prior anonymization covered faces, indoor personal notices, phone and visible plates, and removed GPS. No original photos or internal full-frame derivatives are included. Static photos cannot be reconstructed from estimator outputs. The CIR diagram's editable authoring source remains unavailable. The revision release identifier is `revision-20261016`.
