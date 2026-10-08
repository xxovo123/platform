"""Read and verify actual numerical evidence before formal FDTD runs.

A nonempty filename, a partial scene check, or a bare passed=true is insufficient.
Evidence is verified before the runner creates output or starts any solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import convergence as C, paths, protocol as P


class ValidationError(RuntimeError):
    pass


def verify_user_waiver(path, *, profile='controlled'):
    """An explicit user decision authorizes execution without claiming convergence."""
    paths.ensure_paths()
    p = Path(path) if path else None
    if p is None or not p.is_file():
        raise ValidationError('Explicit numerical-waiver decision is missing')
    doc = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or doc.get('kind') != 'jsf-user-numerical-waiver-20260930' \
            or doc.get('user_authorized') is not True or not doc.get('user_instruction_verbatim'):
        raise ValidationError('Not an explicit recorded user decision')
    if profile not in doc.get('profiles', []) or doc.get('physics_by_profile', {}).get(profile) != physics(profile):
        raise ValidationError('Waiver does not cover this physical profile')
    if doc.get('selected_numerics') != {'resolution_px_per_m': 350.0, 'run_time_meep': 20.0}:
        raise ValidationError('Waiver permits only the original 350 px/m, 20 Meep settings')
    if doc.get('numerical_convergence_verified') is not False or doc.get('diagnostics_retained') is not True:
        raise ValidationError('User waiver must retain the failed diagnostics without claiming convergence')
    from . import estimate
    if doc.get('estimator_module_sha256') != estimate.module_hash():
        raise ValidationError('Estimator changed after the recorded decision')
    return {'kind': 'user_authorized_numerical_gate_waiver', 'report_path': str(p.resolve()),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'profile': profile,
            'selected_numerics': doc['selected_numerics'], 'execution_allowed': True,
            'numerical_convergence_verified': False, 'diagnostics_retained': True,
            'user_instruction_verbatim': doc['user_instruction_verbatim']}


def physics(profile):
    if profile not in ('controlled', 'q4'):
        raise ValidationError('Unsupported numerical-validation profile')
    from fdtd import protocol as FP
    return {'profile': profile, 'fc_hz': P.FC_HZ, 'bandwidth_hz': P.BANDWIDTH_HZ,
            'sample_interval_s': P.SAMPLE_INTERVAL_S, 'raw_taps': P.RAW_TAPS,
            'crop_start': P.CROP_START, 'fft_length': P.FFT_LENGTH,
            'frequency_indices': P.in_band_indices(),
            'metal_radius_m': P.SCATTERER_RADIUS_M, 'motion_sigma_m': P.MICRO_MOTION_SIGMA_M,
            'motion_radius_m': P.MICRO_MOTION_RADIUS_M, 'courant': P.FDTD_COURANT,
            'pml_thickness_lambda': P.PML_THICKNESS_LAMBDA,
            'tx_positions_m': P.TX_POSITIONS_M if profile == 'controlled' else P.Q4_TX_POSITIONS_M,
            'rx_positions_m': P.RX_POSITIONS_M if profile == 'controlled' else P.Q4_RX_POSITIONS_M,
            'pairs': P.PAIRS if profile == 'controlled' else P.Q4_PAIRS,
            'background_mode': 'empty_field' if profile == 'controlled' else 'ema_windowed',
            'burnin_states': 0 if profile == 'controlled' else P.Q4_BURNIN_STATES,
            'ema_alpha': None if profile == 'controlled' else P.Q4_EMA_ALPHA,
            'pulse_formula': FP.PULSE_FORMULA_ID, 'chain_version': FP.CHAIN_VERSION,
            'seed_scheme': 'jsf_sim_20260929_v1 canonical-json sha256'}


def verify(path, *, profile='controlled', output_root=None):
    paths.ensure_paths()
    p = Path(path) if path else None
    if p is None or not p.is_file():
        raise ValidationError('Formal FDTD requires an existing numerical validation report')
    try:
        doc = json.loads(p.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValidationError('Invalid numerical report: ' + str(exc)) from exc
    if not isinstance(doc, dict) or doc.get('kind') != 'jsf-numerical-validation-v2' or doc.get('passed') is not True:
        raise ValidationError('Numerical validation is missing, failed, or uses an unsupported evidence schema')
    if doc.get('criteria') != C.CRITERIA:
        raise ValidationError('Numerical criteria differ from the frozen protocol')
    if doc.get('physics') != physics(profile):
        raise ValidationError('Numerical evidence uses different physical inputs or processing')
    from . import estimate
    if doc.get('estimator_module_sha256') != estimate.module_hash():
        raise ValidationError('Numerical evidence uses a different estimator implementation')
    want_tags = {tag for tag, _, _ in C.representative_scenes(profile)}
    cases = doc.get('representatives', [])
    if {r.get('tag') for r in cases} != want_tags or len(cases) != len(want_tags):
        raise ValidationError('Representative scene coverage is incomplete or duplicated')
    selected = doc.get('selected_numerics', {})
    if not isinstance(selected.get('resolution_px_per_m'), (int, float)) or \
       not isinstance(selected.get('run_time_meep'), (int, float)):
        raise ValidationError('Selected numerical settings are missing')
    res, runtime = float(selected['resolution_px_per_m']), float(selected['run_time_meep'])
    if not math.isfinite(res) or not math.isfinite(runtime) or res <= 0 or runtime <= 0:
        raise ValidationError('Selected numerical settings are invalid')
    for representative in cases:
        comparisons = representative.get('comparisons', [])
        successful = set()
        for row in comparisons:
            axis = row.get('axis')
            if axis not in ('time', 'grid'):
                raise ValidationError('Evidence comparison has an invalid axis')
            if int(row.get('n_states_compared', 0)) < 16 or row.get('trajectory_exactly_equal') is not True \
               or row.get('physical_domain_equal') is not True:
                raise ValidationError('Evidence lacks 16 real matching states and an identical physical domain')
            diff, music = row.get('diff', {}), row.get('music', {})
            values = [diff.get('abs_correlation'), diff.get('shape_rel_l2'), diff.get('max_peak_tap_shift'),
                      music.get('max_peak_position_diff_m')]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValidationError('Undefined numerical metric cannot pass')
            actual_pass = diff['abs_correlation'] >= .90 and diff['shape_rel_l2'] <= .30 \
                and diff['max_peak_tap_shift'] <= 1 and music.get('same_output_count') is True \
                and music['max_peak_position_diff_m'] <= .05 + 1e-12 \
                and music.get('low', {}).get('status') == music.get('high', {}).get('status') == 'ok'
            if row.get('passed') is not True or not actual_pass:
                raise ValidationError('Failed numerical comparison: ' + representative['tag'] + '/' + axis)
            if axis == 'grid':
                if float(row['low']['time']) != runtime or float(row['high']['time']) != runtime \
                   or float(row['high']['res']) != res:
                    raise ValidationError('Selected mesh is not the highest tested passing grid level')
            else:
                if float(row['low']['res']) != float(row['high']['res']) or \
                   runtime not in (float(row['low']['time']), float(row['high']['time'])):
                    raise ValidationError('Selected run time is not covered by a time comparison')
            successful.add(axis)
            artifacts = row.get('artifacts', [])
            if len(artifacts) < 4:
                raise ValidationError('Evidence artifact hashes are missing')
            for artifact in artifacts:
                file = (Path(output_root) / artifact['relative_path']
                        if output_root and artifact.get('relative_path') else Path(artifact['path']))
                if output_root:
                    allowed = Path(output_root).resolve()
                    if not file.resolve().is_relative_to(allowed):
                        raise ValidationError('Numerical artifact is outside the recovery output root')
                if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != artifact['sha256']:
                    raise ValidationError('Numerical artifact is missing or changed: ' + str(file))
        if successful != {'time', 'grid'}:
            raise ValidationError('Both time and grid comparisons must pass: ' + representative['tag'])
    return {'report_path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'profile': profile, 'selected_numerics': selected, 'n_representatives': len(cases),
            'passed': True, 'local_representatives_only': True}
