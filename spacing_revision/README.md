# Revised separation sensitivity analysis

This package supplies all 65 conditions and 6,500 noise trials used in the revised manuscript, at a fixed **0.05 m position-search grid**. The estimator, supplied target count (K=2), snapshot count (128), peak rule and expected CIR SNR (13.0103 dB) match the original controlled study. The stored generator `snr_db=10` follows the noise scaling explained in the root README.

The three directions are transverse, radial and oblique. There are 48 parametric and 17 FDTD conditions, with trials 1000–1099 per condition. Nine FDTD sequences reuse the original archive and eight were newly solved. Every condition's clean CIR, truth, settings, seed and prediction records are included. Exactly two outputs and both one-to-one position errors <= min(0.10 m,d/3) are required for success. Failures remain in the denominator.

Run from the repository root:

```bash
python spacing_reproduce.py verify
python spacing_reproduce.py figures --out outputs/spacing
python spacing_reproduce.py replay --task E2_B_transverse_d875_snr+10 --trial 1000
```

`verify` checks input hashes, all 6,500 seed identities and assignment metrics; `figures` regenerates the manuscript's six-panel figure and complete numerical CSV; `replay` reconstructs noise and runs the estimator against a saved trial. All commands use released inputs and require no Meep installation.

For actual regeneration in the documented Linux/Meep environment, the executed driver and full execution protocol are preserved as `original_run_spacing.py` and `original_protocol.json`:

```bash
python spacing_revision/original_run_spacing.py --source src --archive data --out outputs/spacing_new --protocol spacing_revision/original_protocol.json --execute
```

This original execution protocol additionally runs paired 0.025 m diagnostic checks. They are not used in the manuscript figure, reported rates or formal 6,500-trial analysis. The original completion report accordingly counts 10,200 trials; it is a provenance record, not the size of the published fixed-grid analysis. No diagnostic result replaces a primary-grid result. Full diagnostic outputs remain in the authors' execution archive.

The manifest covers all packaged clean sequences, primary trial records and summaries. The original source manifest records numerical-source hashes. Original absolute paths in provenance records describe the execution machine; reproduction commands resolve the released relative paths.

The curves are conditional on one clean sequence per condition. They do not establish a universal separation threshold, hardware close-person resolution, or numerical convergence of the FDTD solver. Nonmonotonic parametric points and unsuccessful radial conditions are included without selection.
