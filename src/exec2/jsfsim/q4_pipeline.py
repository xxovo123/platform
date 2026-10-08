"""Q4 shared-grid comparison from verified saved CIRs; never starts Meep.

The main branch uses the user's declared +2 tap hardware CFR shift. The
uncorrected branch is retained. Neither branch is selected by its accuracy.
Numerical validation is separate from successful execution of this pipeline.
"""
from __future__ import annotations

import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import data as D, estimate, metrics as M, paths, protocol as P
from . import q4_compare as Q


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_inputs(project_root=None):
    root = Path(project_root) if project_root else Q.workspace_root()
    source = root / Q.HW_WINDOWS_CSV
    man = Q.frozen_205(source)
    rows = [r for r in csv.DictReader(source.open(encoding='utf-8-sig'))
            if r['method'] == 'fb_on' and r['in_205'] == 'True']
    if len(rows) != 205 or len({r['window_id'] for r in rows}) != 205:
        raise RuntimeError('Frozen list contains duplicates')
    by_wid = {r['window_id']: r for r in rows}
    if set(man['by_config']) != set(P.Q4_TARGETS):
        raise RuntimeError('Frozen configurations differ from the protocol')
    hw = Q.hardware_windows(root / Q.HW_CIR_REL, source)
    gx_ref = gy_ref = None
    cfg_ref = None
    evidence = []
    calibration_errors = []
    for rid, group in _groups(hw, 'recording_id').items():
        p = root / '04_Q6/hardware/cache/cases' / (rid + '.npz')
        # Trusted frozen local cache: only window_ids is object dtype. The Q4
        # export itself is pickle-free; no external NPZ files are accepted here.
        with np.load(p, allow_pickle=True) as z:
            gx, gy = np.asarray(z['grid_x']), np.asarray(z['grid_y'])
            config = json.loads(str(z['config_json']))
            model = json.loads(str(z['model_json']))
            search = json.loads(str(z['search_json']))
            if gx_ref is None:
                gx_ref, gy_ref, cfg_ref = gx.copy(), gy.copy(), copy.deepcopy(config)
            elif not np.array_equal(gx, gx_ref) or not np.array_equal(gy, gy_ref):
                raise RuntimeError('Hardware grids are inconsistent: ' + rid)
            for w in group:
                wid = w['window_id']
                canonical = by_wid[wid]
                if w['environment'] != canonical['environment']:
                    raise RuntimeError('Hardware environment differs from the frozen list')
                w['config_id'] = canonical['config_id']
                truth = np.asarray(w['truth_xy'])
                order = M.match_rmse(P.Q4_TARGETS[w['config_id']], truth.tolist())
                if order['tp'] != len(truth) or any(d > 1e-9 for _, _, d in order['pairs']):
                    raise RuntimeError('Truth coordinates differ from fixed configuration: ' + wid)
                # C2-5/C3-3 hardware JSON order differs from the display table.
                # Preserve raw trajectories; assign both prediction sets to the
                # authoritative hardware truth identities in the same order.
                w['simulation_nominal_to_hardware_truth_index'] = [int(p) for _, p, _ in order['pairs']]
                if w['cir_offset'] != 2 or w['ft_len'] != 32 or w['cir'].shape != (128, 4, 30):
                    raise RuntimeError('Unexpected hardware timing or shape: ' + wid)
                X = np.asarray(z['X__' + wid], dtype=np.complex128)
                rebuilt = Q.hw_x(w['cir'])
                phase = config.get('ant_calib_phase')
                if phase is not None:
                    virtual = [int(t) * len(w['rx_positions']) + int(r) for t, r in w['pairs']]
                    phase_selected = np.asarray(phase)[virtual]
                    rebuilt *= np.tile(np.exp(-1j * phase_selected), len(P.in_band_indices()))[None, :]
                err = float(np.linalg.norm(rebuilt - X) / max(np.linalg.norm(X), np.finfo(float).tiny))
                calibration_errors.append(err)
                if err > 1e-10:
                    raise RuntimeError('Exported CIR does not reproduce calibrated hardware X: %s rel=%g' % (wid, err))
                if X.shape != (128, 68) or not np.all(np.isfinite(X)):
                    raise RuntimeError('Invalid hardware features')
                if not np.array_equal(w['pairs'], z['pairs']):
                    raise RuntimeError('Channel order differs between hardware sources')
                if not np.allclose(z['frequencies_hz'], P.frequencies_hz(), rtol=0, atol=.01):
                    raise RuntimeError('Frequency order differs from the protocol')
                for key in ('fft_length', 'tap_interval_si', 'center_freq_si', 'bandwidth_si',
                            'system_delay_si', 'diag_load_factor', 'is_forward_backward',
                            'fov_degrees', 'cir_window', 'epsilon_r_background'):
                    if config[key] != cfg_ref[key]:
                        raise RuntimeError('Hardware configuration varies: ' + key)
                w.update({'X_cached': X, 'config': copy.deepcopy(config), 'model': copy.deepcopy(model),
                          'search': copy.deepcopy(search), 'grid_x': gx, 'grid_y': gy,
                          'cache_npz': str(p)})
        evidence.append({'recording_id': rid, 'cache_npz': str(p), 'sha256': digest(p),
                         'n_selected_windows': len(group)})
    grid = {'source': 'Frozen hardware cache grid arrays and config_json, audited for all 205 windows',
            'n_windows': 205, 'n_recordings': len(evidence), 'all_identical': True,
            'x': [float(gx_ref[0]), float(gx_ref[-1]), int(len(gx_ref))],
            'y': [float(gy_ref[0]), float(gy_ref[-1]), int(len(gy_ref))],
            'step_x_m': float(gx_ref[1] - gx_ref[0]), 'step_y_m': float(gy_ref[1] - gy_ref[0]),
            'arrays_sha256': hashlib.sha256(gx_ref.tobytes() + gy_ref.tobytes()).hexdigest(),
            'config': cfg_ref, 'hardware_sources': evidence,
            'calibrated_X_max_relative_reconstruction_error': max(calibration_errors),
            'list_source_sha256': digest(source)}
    return man, hw, grid


def _groups(rows, key):
    out = collections.defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def shift_x(X, taps):
    idx = np.asarray(P.in_band_indices())
    factors = np.repeat(np.exp(-2j * np.pi * idx * int(taps) / P.FFT_LENGTH), P.N_CHANNELS)
    return np.ascontiguousarray(np.asarray(X) * factors[None, :])


def alignment_tests():
    tests = []
    # Absolute physical integer delays, including taps moved to 30/31. With raw
    # taps 0/1 zero, crop[2:32]+declared shift must recover the full raw32 CFR.
    for tap in (2, 5, 12, 27, 30, 31):
        raw = np.zeros((16, 4, 32), dtype=complex)
        raw[:, :, tap] = np.array([1, 1j, -1, -1j])[None, :]
        full = np.fft.fft(raw, axis=-1)[:, :, P.in_band_indices()].transpose(0, 2, 1).reshape(16, 68)
        shifted = Q.hw_x(raw[:, :, 2:32], shift_taps=2)
        err = float(np.max(np.abs(shifted - full)))
        tests.append({'physical_delay_ns': tap, 'max_CFR_error': err, 'passed': err < 1e-12})
    rng = np.random.default_rng(20260930)
    X = rng.normal(size=(16, 68)) + 1j * rng.normal(size=(16, 68))
    roundtrip = shift_x(shift_x(X, 2), -2)
    tests.append({'name': 'shift_inverse', 'passed': bool(np.allclose(roundtrip, X, rtol=0, atol=2e-15))})
    ratio = X.reshape(16, 17, 4)[:, :, 0] / X.reshape(16, 17, 4)[:, :, 1]
    shifted = shift_x(X, 2).reshape(16, 17, 4)
    tests.append({'name': 'channel_relative_phase_preserved', 'passed': bool(np.allclose(
        shifted[:, :, 0] / shifted[:, :, 1], ratio, rtol=1e-13, atol=1e-13))})
    tests.append({'name': 'zero_pixels_in_correlation', 'passed': abs(Q.pearson([0, 1, 2], [0, 2, 4]) - 1) < 1e-12})
    # Cross-domain identity assignment must retain a large discrepancy.
    ids = Q.match_identity([[0, -1], [1, -1]], [[0, -1], [4, -1]])
    tests.append({'name': 'ungated_identity_keeps_large_error', 'passed': ids == {0: 0, 1: 1}})
    return {'passed': all(r['passed'] for r in tests), 'tests': tests,
            'shift_taps': 2, 'shift_source': 'DECISIONS_20260929 user decision and hardware cir_offset=2 metadata',
            'absolute_reference_limit': 'These are synthetic processing-chain tests; metadata is authoritative. '
                'Channel-relative phase is insensitive to a common absolute shift and is not absolute calibration.',
            'support_limit': 'Hardware raw taps 0/1 were discarded and cannot be recovered; the main branch '
                'preserves all 30 observed taps with a CFR phase, without a second crop or truth fitting.'}


def make_case(w, X, *, identifier, sim_geometry=None):
    geom = sim_geometry or {'tx_positions_m': w['tx_positions'].tolist(),
                           'rx_positions_m': w['rx_positions'].tolist(), 'pairs': w['pairs'].tolist()}
    case = paths.adapter('case_build').build_case(case_id=identifier, clean_id=identifier, X=X, k=w['k'],
        grid_x=w['grid_x'], grid_y=w['grid_y'], layer='B', source='fdtd', scene='Q4',
        tags=[], snr_db=None, trial=0, seed=0, L=len(X), geometry=geom)
    # Shared settings/mask from the frozen hardware configuration. Calibration
    # is already in hardware X; the ideal simulated channels have no hardware phase.
    case['config'] = copy.deepcopy(w['config'])
    case['search'].update(copy.deepcopy(w['search']))
    case['search']['grid_x_m'] = case['grid_x']
    case['search']['grid_y_m'] = case['grid_y']
    case['search']['n_grid_points'] = len(w['grid_x']) * len(w['grid_y'])
    case['model'] = copy.deepcopy(w['model'])
    case['model']['time_origin'] = 'physical zero after declared hardware +2 tap phase; uncorrected diagnostic separately'
    case['model']['channel_calibration'] = {'applied': 'hardware X already calibrated; simulation ideal channels'}
    return case


def result(case):
    r = estimate.run_single(case)
    sp = r.get('spectrum')
    return {'status': r['status'], 'predictions': [[float(p['x']), float(p['y'])] for p in r['predictions']],
            'spectrum': None if sp is None else np.asarray(sp, dtype=float)}


def common_mask(w):
    xx, yy = np.meshgrid(w['grid_x'], w['grid_y'])
    radius = np.hypot(xx, yy)
    theta = np.degrees(np.arctan2(xx, -yy))
    start, end = w['config']['distance_ring_m']
    return (np.abs(theta) <= w['config']['fov_degrees'] / 2 + 1e-12) & \
           (radius >= start - 1e-12) & (radius <= end + 1e-12)


def normalized(sp):
    m = float(np.max(sp)) if sp.size else 0
    return sp / m if m > 0 else np.zeros_like(sp)


def compare_identities(truth, ps, ph):
    si, hi = Q.match_identity(truth, ps), Q.match_identity(truth, ph)
    deltas, joint, sim_only, hw_only, both_missing = [], [], [], [], []
    for k in range(len(truth)):
        if k in si and k in hi:
            joint.append(k)
            deltas.append(float(np.linalg.norm(np.asarray(ps[si[k]]) - ph[hi[k]])))
        elif k in si:
            sim_only.append(k)
        elif k in hi:
            hw_only.append(k)
        else:
            both_missing.append(k)
    return {'delta_m': deltas, 'n_joint_identity': len(joint), 'joint_identity': joint,
            'only_sim_identity': sim_only, 'only_hw_identity': hw_only,
            'both_missing_identity': both_missing,
            'peak_difference_sse_m2': sum(d * d for d in deltas),
            'peak_difference_rmse_m': math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else None}


def aggregate(rows):
    if not rows:
        return {}
    delta_sse = sum(r['peak_difference_sse_m2'] for r in rows)
    n = sum(r['n_joint_identity'] for r in rows)
    rhos = [r['spectral_rho'] for r in rows if r['spectral_rho'] is not None]
    return {'n_windows': len(rows), 'n_joint_identities': n,
            'peak_difference_rmse_m': math.sqrt(delta_sse / n) if n else None,
            'peak_difference_sse_m2': delta_sse,
            'only_sim_identities': sum(len(r['only_sim_identity']) for r in rows),
            'only_hw_identities': sum(len(r['only_hw_identity']) for r in rows),
            'both_missing_identities': sum(len(r['both_missing_identity']) for r in rows),
            'spectral_rho_mean': float(np.mean(rhos)) if rhos else None,
            'spectral_rho_sd': float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None,
            'n_spectral_rho': len(rhos), 'n_undefined_rho': len(rows) - len(rhos),
            'sim_truth_metrics_repeated_by_hardware_window': M.aggregate([r['sim_truth_metrics'] for r in rows], bootstrap=False),
            'hw_truth_metrics': M.aggregate([r['hw_truth_metrics'] for r in rows], bootstrap=False)}


def run_saved(sim_cir_root, output_dir, *, configs=None, project_root=None, numerical_validated=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    man, hw, grid = load_inputs(project_root)
    tests = alignment_tests()
    dump(out / 'frozen_205.json', man)
    dump(out / 'shared_grid.json', grid)
    dump(out / 'time_reference.json', tests)
    if not tests['passed']:
        raise RuntimeError('Synthetic time/metric validation failed')
    cfgs = list(configs or P.Q4_TARGETS)
    selected = [w for w in hw if w['config_id'] in cfgs]
    np.savez_compressed(out / 'shared_grid.npz', grid_x=hw[0]['grid_x'], grid_y=hw[0]['grid_y'], common_mask=common_mask(hw[0]))
    rows = []
    sim_details = []
    spectral_dir = out / 'spectra'
    spectral_dir.mkdir(exist_ok=True)
    for cfg in cfgs:
        windows = [w for w in selected if w['config_id'] == cfg]
        w0 = windows[0]
        p = Path(sim_cir_root) / cfg / 'clean_cir.npz'
        with np.load(p, allow_pickle=False) as z:
            raw = np.asarray(z['raw_cir'])
        if raw.shape != (128, 4, 32) or not np.all(np.isfinite(raw)):
            raise RuntimeError('Simulated CIR shape is invalid: ' + cfg)
        geom = {'tx_positions_m': P.Q4_TX_POSITIONS_M, 'rx_positions_m': P.Q4_RX_POSITIONS_M, 'pairs': P.Q4_PAIRS}
        sim = result(make_case(w0, D.clean_x(raw, L=128), identifier='q4_sim_' + cfg, sim_geometry=geom))
        if sim['status'] != 'ok' or sim['spectrum'] is None:
            raise RuntimeError('Simulation estimator failed: ' + cfg)
        ms = M.evaluate_row(w0['truth_xy'], sim['predictions'], status=sim['status'])
        sim_details.append({'config_id': cfg, 'source': str(p), 'sha256': digest(p),
                            'predictions': sim['predictions'], 'truth_metrics': ms,
                            'one_shared_trajectory': True, 'kept_states': 128, 'burnin_states': 32,
                            'canonical_truth_xy': w0['truth_xy'],
                            'simulation_nominal_to_hardware_truth_index': w0['simulation_nominal_to_hardware_truth_index']})
        np.savez_compressed(spectral_dir / ('sim_' + cfg + '.npz'), spectrum=sim['spectrum'],
            normalized=normalized(sim['spectrum']), grid_x=w0['grid_x'], grid_y=w0['grid_y'],
            predictions=np.asarray(sim['predictions']).reshape(-1, 2), truth=np.asarray(P.Q4_TARGETS[cfg]), mask=common_mask(w0))
        for w in windows:
            for branch, shift in (('declared_2tap', 2), ('uncorrected', 0)):
                hw_result = result(make_case(w, shift_x(w['X_cached'], shift), identifier='q4_' + branch + '_' + w['window_id']))
                if hw_result['status'] != 'ok' or hw_result['spectrum'] is None:
                    raise RuntimeError('Hardware estimator failed: ' + w['window_id'])
                ps, ph = sim['spectrum'], hw_result['spectrum']
                if ps.shape != ph.shape or np.any(ps < 0) or np.any(ph < 0):
                    raise RuntimeError('Spectrum domain/shape mismatch')
                valid = common_mask(w) & np.isfinite(ps) & np.isfinite(ph)
                # Valid zeros remain in the correlation population.
                rho = Q.pearson(normalized(ps)[valid], normalized(ph)[valid])
                row = {'window_id': w['window_id'], 'recording_id': w['recording_id'],
                       'environment': w['environment'], 'config_id': cfg, 'branch': branch,
                       'hw_shift_taps': shift, 'spectral_rho': rho,
                       'n_common_valid_pixels': int(valid.sum()), 'hw_predictions': hw_result['predictions'],
                       'sim_predictions': sim['predictions'], 'sim_truth_metrics': ms,
                       'hw_truth_metrics': M.evaluate_row(w['truth_xy'], hw_result['predictions'], status=hw_result['status'])}
                row.update(compare_identities(w['truth_xy'], sim['predictions'], hw_result['predictions']))
                rows.append(row)
                np.savez_compressed(spectral_dir / (branch + '_' + w['window_id'] + '.npz'),
                    spectrum=ph, normalized=normalized(ph), grid_x=w['grid_x'], grid_y=w['grid_y'],
                    mask=valid, predictions=np.asarray(hw_result['predictions']).reshape(-1, 2), truth=np.asarray(w['truth_xy']))
        print('Q4 READY', cfg, 'windows', len(windows), 'both branches saved', flush=True)
        dump(out / 'per_window.json', rows)
    summaries = []
    by_config = []
    for branch in ('declared_2tap', 'uncorrected'):
        for cfg in cfgs:
            subset = [r for r in rows if r['branch'] == branch and r['config_id'] == cfg]
            by_config.append({'branch': branch, 'config_id': cfg, **aggregate(subset)})
            for env in ('indoor', 'outdoor'):
                sub = [r for r in subset if r['environment'] == env]
                if sub:
                    summaries.append({'branch': branch, 'config_id': cfg, 'environment': env, **aggregate(sub)})
    equal_weight = {}
    for branch in ('declared_2tap', 'uncorrected'):
        cfg = [r for r in by_config if r['branch'] == branch]
        corr = [r['spectral_rho_mean'] for r in cfg if r['spectral_rho_mean'] is not None]
        rms = [r['peak_difference_rmse_m'] for r in cfg if r['peak_difference_rmse_m'] is not None]
        equal_weight[branch] = {'spectral_rho_mean': float(np.mean(corr)) if corr else None,
                                'n_configs_with_rho': len(corr), 'n_configs_with_joint_identity': len(rms),
                                'peak_difference_RMS_of_config_RMSE_m': float(np.sqrt(np.mean(np.square(rms)))) if rms else None,
                                'mean_of_config_peak_difference_RMSE_m': float(np.mean(rms)) if rms else None}
    expected = {w for c in cfgs for w in man['by_config'][c]}
    for branch in ('declared_2tap', 'uncorrected'):
        got = [r['window_id'] for r in rows if r['branch'] == branch]
        if set(got) != expected or len(got) != len(expected):
            raise RuntimeError('Q4 comparison coverage is incomplete')
    report = {'kind': 'q4-shared-grid-cross-domain', 'pipeline_passed': True,
              'n_configs': len(cfgs), 'n_windows_each_branch': len(expected),
              'frozen_hw_windows': 205, 'by_environment_config': summaries, 'by_config': by_config,
              'config_equal_weight': equal_weight, 'simulations': sim_details,
              'numerical_validated': bool(numerical_validated),
              'formal_similarity_claim_allowed': bool(numerical_validated and len(cfgs) == 15),
              'no_artificial_white_noise': True, 'declared_shift_fixed_before_comparison': True,
              'time_reference': tests, 'grid': grid,
              'scope': '2D metal scatterers and bounded random walk; hardware contains noise and unmodelled room/body effects',
              'independence': '15 simulated trajectories are reused across 205 hardware windows; windows are not independent FDTD solves.'}
    dump(out / 'report.json', report)
    flat = [{k: r[k] for k in ('window_id', 'recording_id', 'environment', 'config_id', 'branch',
                              'hw_shift_taps', 'spectral_rho', 'n_common_valid_pixels', 'n_joint_identity',
                              'peak_difference_rmse_m', 'peak_difference_sse_m2')}
            | {'only_sim_identities': len(r['only_sim_identity']), 'only_hw_identities': len(r['only_hw_identity']),
               'both_missing_identities': len(r['both_missing_identity']),
               'sim_truth_rmse_m': math.sqrt(r['sim_truth_metrics']['sse_m2'] / r['sim_truth_metrics']['pairs']) if r['sim_truth_metrics']['pairs'] else None,
               'hw_truth_rmse_m': math.sqrt(r['hw_truth_metrics']['sse_m2'] / r['hw_truth_metrics']['pairs']) if r['hw_truth_metrics']['pairs'] else None,
               'sim_dr': r['sim_truth_metrics']['dr'], 'hw_dr': r['hw_truth_metrics']['dr']}
            for r in rows]
    with (out / 'per_window.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return report
