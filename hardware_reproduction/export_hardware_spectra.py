"""Export measured JSF-MUSIC spectra reconstructed from raw CIR and restored EMA."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
import reproduce_hardware as hw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=hw.ROOT/'generated/raw_spectra')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 205:
        ap.error('--limit must be 1..205')
    out = args.out.resolve()
    if not out.is_relative_to(hw.ROOT):
        ap.error('Output must stay inside hardware_reproduction')
    out.mkdir(parents=True, exist_ok=True)
    contexts = hw.read(hw.DATA/'contexts.json')
    reference = hw.read(hw.ROOT/hw.read(hw.DATA/'result_map.json')['this_method']['file'])
    method = hw.module('raw_spectrum_this_method', hw.COMMON/'q6_methods_this_method.py')
    entries = hw.dataset()[:args.limit] if args.limit else hw.dataset()
    manifest = []
    for i, entry in enumerate(entries, 1):
        sid = entry['segment_id']
        case = hw.make_case(entry, contexts)
        result = method.estimate(case)
        if result['status'] != 'ok':
            raise RuntimeError((sid, result['diagnostics']))
        predicted = np.asarray([[p['x'], p['y']] for p in result['predictions']], float).reshape(-1, 2)
        truth, expected = hw.q5_metrics.extract_sets(reference[sid], where=sid)
        expected = np.asarray(expected, float).reshape(-1, 2)
        if predicted.shape != expected.shape:
            raise RuntimeError((sid, 'Prediction count differs from frozen reference'))
        distances = np.linalg.norm(predicted[:, None, :]-expected[None, :, :], axis=2)
        r, c = linear_sum_assignment(distances)
        delta = float(distances[r, c].max()) if len(r) else 0.
        if delta > 1e-8:
            raise RuntimeError((sid, 'Prediction mismatch', delta))
        with np.load(hw.DATA/'raw'/entry['file'], allow_pickle=False) as raw:
            truth = raw['truth_xy_m'].copy()
        path = out/(sid+'.npz')
        np.savez_compressed(path, spectrum=result['spectrum'], grid_x=case['grid_x'], grid_y=case['grid_y'],
                            predictions=predicted, truth=truth, segment_id=sid,
                            recording_id=entry['recording_id'], config_id=entry['config_id'], environment=entry['environment'])
        manifest.append({'segment_id': sid, 'file': path.name, 'sha256': hw.sha(path),
                         'raw_sha256': hw.sha(hw.DATA/'raw'/entry['file']),
                         'ema_sha256': hw.sha(hw.DATA/'ema_states'/(sid+'.npz')),
                         'X_sha256': hashlib.sha256(case['X'].tobytes()).hexdigest(),
                         'max_frozen_position_difference_m': delta})
        if i % 10 == 0 or i == len(entries):
            print(f'Exported hardware spectra {i}/{len(entries)}', flush=True)
    hw.write(out/'manifest.json', {'n_segments': len(entries), 'all_205': len(entries) == 205,
             'simulation_executed': False, 'estimator': 'this_method', 'spectrum_normalization': 'none',
             'Q4_pixel_comparison_performed': False, 'segments': manifest})


if __name__ == '__main__':
    main()
