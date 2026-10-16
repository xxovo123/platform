# Manuscript-to-code and data map

Revision: `revision-20261016`; title: *CIR-Based Joint Range–Azimuth Localization for Single-Anchor IR-UWB Radar: Implementation and Validation*.

Commands below run from the repository root. Install `requirements.txt` for simulation processing and `hardware_reproduction/requirements.txt` for the hardware pipeline. Only field regeneration needs Linux/Meep (`environment.yml`). A complete from-scratch field solve is expensive; cached CIR replay and saved-prediction metric recomputation are separate supported levels of reproduction.

| Manuscript item | Released input and code | Reproduction |
|---|---|---|
| Fig. 1, receiver/CIR chain; Fig. 2, processing workflow | Static explanatory diagrams and source assets under `manuscript_assets/` | See `docs/hardware_reproduction_map.md`; these are diagrams, not computed experimental observations |
| Tables I–II, controlled settings/layouts | `src/exec2/jsfsim/protocol.py`, `data/settings/`, `data/results/runs/*/tasks.jsonl` | `python reproduce.py verify`; fixed geometry/parameter declarations, not estimated statistics |
| Fig. 3, SNR curves | `data/results/runs/E1_*`, `data/tables/`, `reproduce.py` | `python reproduce.py tables --out outputs/tables`; `python reproduce.py figures --out outputs/figures` |
| Fig. 4, revised directional separation | `spacing_revision/cache`, `conditions.json`, `rows`, `summaries`, `spacing_reproduce.py` | `python spacing_reproduce.py verify`; `python spacing_reproduce.py figures --out outputs/spacing`; estimator replay as in its README |
| Fig. 5, snapshot sensitivity | `data/results/runs/E4_B*`, `data/tables/`, `reproduce.py` | Same `tables` and `figures` commands; genuine prefixes of the released 128-state sequence |
| Fig. 6, antenna geometry; Fig. 7, environments and positions | Static geometry and anonymized photographs in `manuscript_assets/` | See `docs/hardware_reproduction_map.md`; photos are recorded source assets, not algorithm outputs |
| Tables III–IV, hardware settings/configurations | `hardware_reproduction/data/raw/`, acquisition metadata, processing contexts and truth | Hardware README/map; 205 segments, 31 recordings, 15 configurations, supplied K |
| Table V, grouped hardware performance | Raw segments, EMA initial states, saved predictions, hardware metric code | Hardware saved-metric verification and table regeneration; full estimator replay is separately available |
| Table VI, repeated measurements | The defined 30 groups and 120 temporal measurements | Hardware reproduction entry points; this subset is not substituted for the full 205-segment method comparison |
| Table VII, six-method comparison | Six implementations plus all 205 raw segments | Hardware replay and tables; comparisons concern adapted implementations |
| Table VIII, ablations | Twenty configurations with the same reconstruction and metric definitions | Hardware replay and tables; 20 × 205 evaluations |
| Fig. 8, nine-panel correspondence spectra | `data/q4/spectra/`, `data/q4/per_window.csv`, `q4_figures.py` | `python q4_figures.py --out outputs/q4`; deterministic first identifier per configuration/environment |
| Table IX, correspondence statistics | All 205 saved measurement spectra, 15 simulated spectra, correspondence code | `python q4_reproduce.py verify-spectra`; `python q4_reproduce.py summarize` |
| FDTD correspondence estimates | `data/q4/sim_cir/`, `generator_settings.json`, `src/exec2/jsfsim/q4_pipeline.py` | `python q4_reproduce.py replay-fdtd --config all` |
| Hardware correspondence spectra | All 205 raw segments and continuous EMA states; `q4_hardware_replay.py` | `python q4_hardware_replay.py`; verifies all normalized measurement spectra against the released archive |
| Parametric correspondence reference | `data/q4/parametric/`, fixed-seed frequency-domain model | `python q4_reproduce.py replay --config G-420 --trial 0`; 20 trials per configuration in the saved output |
| Controlled FDTD fields | `src/fdtd/`, protocol, clean metadata and example field records | `python regenerate_fdtd.py --clean-id cln_de385a3b2982 --out outputs/new_fdtd_S2 --workers 1` |
| Measured-geometry FDTD fields | Same solver, measured horizontal geometry and background-state protocol | `python q4_regenerate_fdtd.py --config G-420 --out outputs/new_q4_G420 --workers 1` |

`reproduce.py figures` retains the original separation plot for archive compatibility. **Use `spacing_reproduce.py figures` for the revised manuscript's Fig. 4.** All revised fixed-grid conditions are included; none are selected according to success. The 0.025 m paired diagnostics in the original executed protocol are not the manuscript's analysis.

Frozen numerical modules retain their source fingerprints. Wrapper paths are portable, while recorded absolute paths in original provenance are historical metadata. Numeric regeneration is verified; byte-identical PDF generation across Matplotlib/font versions is not asserted. Intermediate FDTD field states are not all packaged; all required clean CIR sequences are, and solver regeneration code is supplied.
