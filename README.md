# FDTD-CIR Simulation Platform

**A general-purpose electromagnetic simulation platform for generating Channel Impulse Response (CIR) data from FDTD simulations using Meep.**

This repository accompanies a manuscript under review. It provides a clean, modular pipeline that converts raw FDTD field recordings into baseband CIR matrices — the input format expected by downstream localization and imaging algorithms. No algorithm implementations (MUSIC, ESPRIT, BP, etc.) are included; the platform stops at CIR generation, serving as a reproducible and reusable simulation frontend.

---

## Repository Structure

```
platform/
├── fdtd_simulator.py      # FDTD simulation core (Meep)
├── signal_converter.py    # Passband MDM → Baseband CIR conversion
├── antenna_config.py      # Antenna array configuration (4 modes)
├── data_manager.py        # Data caching, export, and logging
├── simulation_config.py   # Unified entry point & parameter management
├── __init__.py            # Package index
└── requirements.txt       # Python dependencies
```

---

## Pipeline Overview

The platform implements a three-stage pipeline:

```
  ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
  │  1. FDTD (Meep) │ ───→ │  2. IQ Demod +   │ ───→ │  3. Differential │
  │  MDM Acquisition │      │  Downsampling    │      │  CIR Snapshots   │
  └─────────────────┘      └──────────────────┘      └─────────────────┘
```

### Stage 1: FDTD Simulation & MDM Acquisition

We use [Meep](https://meep.readthedocs.io/) (a finite-difference time-domain solver) to simulate electromagnetic wave propagation in a 2D domain.

**Excitation signal.** Each transmitting antenna emits a Blackman-Harris derivative pulse, chosen for its compact spectral support and low sidelobes. The pulse is defined in `fdtd_simulator.py` as:

$$
s(t) = w'(t) \cos(2\pi f_c (t - t_0)) - 2\pi f_c \, w(t) \sin(2\pi f_c (t - t_0))
$$

where $w(t)$ is the Blackman-Harris window and $f_c$ is the center frequency (default: 7.9872 GHz).

**Multi-Dimensional Matrix (MDM).** For each transmit (TX) antenna, a separate Meep simulation is run. The $E_z$ field is recorded at every receive (RX) antenna location at every FDTD time step. The result is stored as a 3D array:

```
MDM[n_rx, n_tx, n_timesteps]
```

**Scene geometry.** The simulation domain includes a simplified vehicle interior with metallic pillars (A/B/C pillars) and seats, as defined in `make_car_interior_geometry()`. Multipath scattering can be toggled on or off via the `no_multipath` flag.

**Moving targets.** Target motion is modeled via random walk paths constrained within a circular radius, as defined in `generate_random_walk_path()`. Each step produces a new MDM snapshot representing the channel state at that position.

### Stage 2: IQ Demodulation & Downsampling (Passband → Baseband CIR)

The raw MDM signal is a passband recording sampled at the FDTM-native rate (typically very high). We convert it to a complex baseband CIR through the following steps, implemented in `signal_converter.py`:

1. **Analytic signal construction.** The Hilbert transform is applied along the time axis to obtain the analytic representation of each TX-RX channel.

2. **Digital down-conversion.** The analytic signal is multiplied by a complex local oscillator $e^{-j 2\pi f_c t}$ to shift the spectrum from passband to baseband.

3. **Downsampling.** The baseband signal is resampled to the target sampling rate (default: 1 GSa/s) using `scipy.signal.resample`.

4. **Matched filtering (optional).** When a reference pulse is provided, a matched filter $h_{\text{mf}}(t) = p^*(-t)$ is applied via FFT-based convolution to maximize the output SNR. The filter delay is compensated so the CIR time origin aligns with $t = 0$.

5. **Truncation.** The CIR is truncated to a fixed number of taps (default: 64 or configurable via `cir_points`) to form a compact representation.

The output is a complex-valued CIR matrix:

```
CIR[n_rx, n_tx, n_cir_points]
```

### Stage 3: Differential CIR Snapshots

Each CIR frame is obtained through **differential measurement**: the CIR at step $k$ is derived from the difference of two consecutive MDM snapshots:

$$
\text{CIR}_k = \mathcal{T}\big( \text{MDM}_{\text{step } k+1} - \text{MDM}_{\text{step } k} \big)
$$

where $\mathcal{T}(\cdot)$ denotes the Stage 2 processing (IQ demodulation + downsampling + matched filtering). This differential operation suppresses static background reflections (pillars, seats) and isolates the contribution of the moving targets, which is the signal of interest for downstream algorithms.

When random walk is disabled (`enable_random_walk=False`), a single MDM snapshot is used directly for the static-target case.

---

## Antenna Array Configuration

Four antenna modes are supported, all configured in `antenna_config.py` via the `build_antenna_arrays()` function:

| Mode | Description |
|------|-------------|
| `separated` | TX and RX form two parallel ULAs separated by one wavelength |
| `colocated` | TX and RX share the same antenna positions |
| `hybrid_tdm` | Two-row array; each antenna alternately transmits while all others receive |
| `hybrid_tdm_tx_top` | Two-row array; only the top row transmits, both rows receive |

All arrays use half-wavelength ($\lambda/2$) element spacing at the center frequency. Optionally, RX antenna Y-positions can be randomized within $[0, \lambda/2]$ via `randomize_rx_y_pos`.

---

## Usage

### Quick Start

```python
from simulation_config import quick_sim

result = quick_sim(
    n_antennas=4,
    scenarios=['rear_right_seat'],
    antenna_mode='separated',
    enable_random_walk=True,
    num_movements=10,
    cir_points=32,
    output_dir='./output',
)

# Access results
cir_snapshots = result['cir_snapshots']  # list of [n_rx, n_tx, n_cir_points]
time_axis     = result['time_axis']       # baseband time axis
freq_axis     = result['freq_axis']       # baseband frequency axis
antenna_info  = result['antenna_info']    # TX/RX positions
```

### Command Line

```bash
python simulation_config.py \
  --antennas 4 \
  --mode separated \
  --scenarios rear_right_seat front_right_seat \
  --random-walk \
  --movements 10 \
  --step-cm 1.0 \
  --radius-cm 3.0 \
  --cir-points 32 \
  --output ./sim_output
```

### Advanced: Custom Pipeline

```python
from simulation_config import run_simulation_pipeline, register_scenario
import meep as mp

# Register custom target positions
register_scenario('my_target', mp.Vector3(0.5, -1.0), mp.Vector3(0.5, -1.05))

result = run_simulation_pipeline({
    'n_antennas': 8,
    'fc_si': 7.9872e9,
    'bandwidth_si': 500e6,
    'antenna_mode': 'hybrid_tdm',
    'active_scenarios': ['my_target'],
    'enable_random_walk': True,
    'num_movements': 20,
    'cir_points': 64,
    'output_dir': './my_experiment',
})
```

---

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
meep>=1.25.0
matplotlib>=3.5.0      # optional, for visualization
```

Install: `pip install -r requirements.txt`

---

## Design Notes for Reviewers

- **Reproducibility.** All random number generators use NumPy's default state. Results can be exactly reproduced by caching MDM data (enabled by default via `use_cache=True`). Cached `.npz` files include antenna positions and target paths.
- **Modularity.** Each stage (FDTD, signal conversion, antenna config, data I/O) is a self-contained module with a documented public API. The pipeline can be repurposed for different geometries, frequencies, or antenna layouts by modifying only the relevant module.
- **No algorithm leakage.** This platform intentionally excludes all imaging, localization, and parameter estimation algorithms. It produces only the CIR data that serves as input to such algorithms, ensuring a clean separation between simulation and processing.
- **Computational cost.** FDTD simulations are the dominant cost. For $N$ TX antennas, $N$ separate Meep runs are executed per snapshot. MDM caching avoids recomputation when only downstream processing parameters change.

---

## License

This code is provided for peer-review purposes accompanying a submitted manuscript.
