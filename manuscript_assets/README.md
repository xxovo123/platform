# Non-simulation manuscript assets

Original artwork is preserved in `figures/`. The manuscript and original IEEE template remain unchanged.

From the repository root, with NumPy and Matplotlib:

```text
python -B manuscript_assets/regenerate.py
python -B manuscript_assets/validate_assets.py --compare-rendered
```

Outputs go to ignored `manuscript_assets/generated/`. The code ports the existing antenna/scene sections of `build_manuscript_assets.py` and four-stage restoration recipe. It executes no estimator or simulation.

Asset validation checks original SHA-256, photo metadata and optional rendered PDF comparisons (install PyMuPDF for `--compare-rendered`). Photo originals are the exact approved crops; no new image synthesis is used.

| Asset | Status and source |
|---|---|
| `figures/antenna_layout.pdf` | Original physical/virtual array schematic. Rebuild with `regenerate.py`; original geometry retained. |
| `figures/hardware_setup.pdf` | Original composition of approved anonymous photos and nominal positions. Layout rebuild uses `photos/` and `data/nominal_positions.json`. |
| `figures/algo_flowchart_restored.svg` / `.pdf` | Original editable four-stage tensor/matrix artwork. Rebuild from `source/draw_algorithm_flowchart.py` plus `source/restore_workflow.py`. Drawn spectrum is schematic. |
| `figures/CIR_Time_Domain_Signal_Acquisition_Procedure.pdf` | Original static vector receive-chain drawing. No confirmed editable authoring source found. |
| `photos/*_anonymized_crop.jpg` | Two static photographic records, exact approved submission crops. No original photographs or full-frame derivatives included. Faces, personal notices, phone and visible plates were anonymized; GPS removed. |

`source_manifest.json` records origins, hashes and adaptations. PDF dates, SVG IDs and renderer versions can change regenerated file hashes; compare rendered content. Photos are not generated from statistics. [The figure/table map](../docs/hardware_reproduction_map.md) distinguishes review plots and original paper figures. Simulation SNR, separation, snapshots and Q4 spectra use the root-level reproduction commands documented in `../docs/manuscript_reproduction.md`.
