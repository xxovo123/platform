# Independent Linux validation report

- Date / operator:
- Host / Linux version / CPU / RAM:
- Archive SHA-256:
- `data/release_manifest.json` SHA-256:
- Environment path and explicit Conda package file:
- Python / Meep / NumPy / SciPy / Matplotlib / BLAS:
- BLAS threads / state workers:

| Stage | Coverage actually completed | Pass / fail / pending | Evidence path | Elapsed |
|---|---|---|---|---|
| Saved-data reconstruction | | | | |
| Complete replay from released CIR | | | | |
| Fresh analytic + 44 FDTD sequences | | | | |
| Complete replay from new simulation inputs | | | | |

| Check | Required | Actual / maximum difference | Status |
|---|---:|---|---|
| Hardware segments / methods+configs / executions | 205 / 26 / 5330 | | |
| Temporal groups / selected IDs | 30 / 120 | | |
| Controlled conditions / trials | 150 / 16500 | | |
| FDTD spacing conditions / trials | 24 / 2400 | | |
| Spacing flags and peak identities | Exact, all trials | | |
| Q4 hardware / FDTD / parametric | 205 / 15 / 300 | | |
| Parametric-to-hardware comparisons | 4100 | | |
| Fresh input coverage | 5 analytic + 29 controlled FDTD + 15 Q4 | | |
| CIR relative L2 / normalized max error | <=1e-7 / <=1e-7 | | |
| Hardware / simulation coordinates | <=1e-8 m / <=1e-10 m | | |
| Full table V–IX comparison | All rows, printed values | | |

| Figure | All content present, labels correct, no clipping? | Evidence / issue |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 4: 3×8 points, correct diagnostic and symbols | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8: fixed examples, normalized colors | | |

Tables I–IV settings/geometry review:

Failures or deviations (include identities and actual/reference values):

Final decision (separately state numerical, field-generation and visual status):

Changes made during validation, if any; new package identity required:

Publication performed: **No** (this report alone does not publish the repository).
