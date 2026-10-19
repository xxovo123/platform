# FDTD target-spacing sensitivity: current Fig. 4

Exactly **24 FDTD conditions and 2400 trials**, fixed **0.05 m search grid**. Directions: transverse, radial, oblique. Distances: **0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00 m**. Supplied K=2, L=128, expected CIR SNR=13.0103 dB (`snr_db=10` in the historical generator). Trial IDs 1000–1099 in every condition. No failures are removed.

Fig. 4 reports **truth-referenced two-peak correspondence before final peak selection**. Form a finite, positive 3×3 local-maximum mask, merge eight-connected plateau components, and represent each by its largest value (ties by row, then column). Two distinct maxima must match the two truths one-to-one within `min(0.10 m,d/3)`. This diagnostic precedes the original relative-height threshold, minimum-distance suppression and K-output cap. It is not final localization success. Extra maxima do not invalidate correspondence.

| Path | Content |
|---|---|
| `conditions.json` | All identities, geometry, clean seeds and settings |
| `cache/fdtd/<clean_id>/` | Original clean CIR, independent truth and metadata |
| `rows/*_grid0.05.json` | Original predictions, statuses, seeds and metrics, 100 per condition |
| `summaries/` | Original output-success summaries, **not** the Fig. 4 ordinate |
| `spectral_reference.json` | All 2400 spectral records and 24 summaries, matching peaks and original-output flags |
| `manifest.json` | File hashes excluding the manifest itself |

No parametric target-spacing experiment, 0.025 m grid study or distance above 1.00 m is included. Noise repetitions use one clean sequence per condition, not 100 independent environments.

From the root in Linux:

```bash
python spacing_reproduce.py verify
python spacing_reproduce.py replay --task E2_B_transverse_d300_snr+10 --trial 1000
python spacing_reproduce.py replay-all --workers 4 --out generated_validation/spacing
python spacing_reproduce.py figures --out outputs/spacing \
  --results generated_validation/spacing/replay_verification.json
```

`verify` checks stored data without estimator execution. `replay-all` reconstructs spectra and output positions from CIR and seeds, comparing all 2400 flags, positions and matched peak identities. Plotting consumes the new full report. Both figure commands reproduce the single panel with circle/square/triangle markers and no overlaid error bars.

Fresh field generation uses `validation/regenerate_inputs.py`, 128 actual states per controlled scene. Its isolated preflight overlay counts the nominal motion margin once and checks actual state clearance; solver parameters, trajectories and estimator remain unchanged.

Expected successes out of 100 in the eight-distance order:

| Direction | Counts |
|---|---|
| Transverse | 0, 0, 0, 87, 100, 100, 100, 100 |
| Radial | 0, 0, 0, 0, 100, 100, 100, 100 |
| Oblique | 0, 0, 100, 100, 100, 100, 100, 100 |

These conditional simulation results establish neither measured close-person resolving capability nor a universal hardware resolution bound.
