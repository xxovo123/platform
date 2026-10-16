"""Validate release coverage and archived evidence without running any estimator."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'generated/release_validation.json')
    args = ap.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT), 'Output must stay inside hardware_reproduction')
    entries = read(ROOT/'data/raw/index.json')['segments']
    ids = {e['segment_id'] for e in entries}
    require(len(entries) == len(ids) == 205, 'Segment coverage')
    require(Counter(e['environment'] for e in entries) == {'indoor': 103, 'outdoor': 102}, 'Environment coverage')
    require(len({e['recording_id'] for e in entries}) == 31, 'Recording coverage')
    require(sum(e['K'] for e in entries) == 338, 'Truth count')
    result_map = read(ROOT/'data/result_map.json')
    configs = read(ROOT/'data/config_matrix.json')['configs']
    require(len(result_map) == 26 and len(configs) == 20, 'Six methods plus twenty ablations')
    require({n.split('/', 1)[1] for n in result_map if n.startswith('ablation/')} == {c['id'] for c in configs}, 'Config IDs')
    for name, item in result_map.items():
        records = read(ROOT/item['file'])
        require(set(records) == ids, 'Frozen coverage: '+name)
        require(all(r['status'] == 'ok' for r in records.values()), 'Frozen statuses: '+name)
    for entry in entries:
        require((ROOT/'data/ema_states'/(entry['segment_id']+'.npz')).is_file(), 'Missing EMA state')
        require((ROOT/'data/raw'/entry['recording_metadata_file']).is_file(), 'Missing metadata')
    comparison = rows(ROOT/'full_replay/replay_comparison.csv')
    require(len(comparison) == 5330, 'Replay row count')
    require({(r['method'], r['segment_id']) for r in comparison} == {(m, sid) for m in result_map for sid in ids}, 'Replay Cartesian coverage')
    require(all(r['pass'] == 'True' and float(r['max_position_difference_m']) <= 1e-8 and r['prediction_count'] == r['reference_count'] for r in comparison), 'Archived replay agreement')
    replay = read(ROOT/'full_replay/replay_verification.json')
    require(replay['pass'] and replay['n_estimator_runs'] == 5330 and set(replay['names']) == set(result_map), 'Archived replay summary')
    manifest = read(ROOT/'integration_manifest.json')
    modified = set(manifest.get('modified_after_copy', []))
    for item in manifest['copied_files']:
        if item['file'] in modified:
            continue
        path = ROOT/item['file']
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Imported hash: '+item['file'])
    selected = rows(ROOT/'outputs/selected_measurements.csv')
    reference = rows(ROOT/'data/reference_selected_measurements.csv')
    key = lambda r: (r['environment'], r['config_id'], r['window_id'])
    require(len(selected) == len(reference) == 120 and {key(r) for r in selected} == {key(r) for r in reference}, 'Fixed temporal selection')
    require(all(int(r['rank_zero_based']) == int(next(x for x in reference if key(x) == key(r))['index']) for r in selected), 'Temporal selection ranks')
    # Numerical checks omitted by the original baseline/ablation-only paper_checks.
    extra_checks = 0
    groups = rows(ROOT/'outputs/groups.csv')
    for line in (ROOT/'data/paper_reference/groups.tex').read_text(encoding='utf-8').splitlines():
        if not re.match(r'^(Indoor|Outdoor|\$K=)', line):
            continue
        label, *cells = line.split('&')
        record = next(r for r in groups if (r['environment'] == label.strip().lower() and r['K'] == '') or (label.startswith('$K=') and r['environment'] == 'all' and r['K'] == re.search(r'\d', label).group()))
        for field, cell, digits in zip(['rmse_pos_m', 'rmse_range_m', 'rmse_angle_deg', 'dr_percent'], cells, [4, 4, 3, 2]):
            value = float(re.search(r'\d+(?:\.\d+)?', cell).group())
            require(abs(float(record[field])-value) <= .5*10**(-digits)+1e-10, 'Group table: '+label+field)
            extra_checks += 1
    stability = rows(ROOT/'outputs/stability.csv')
    for line in (ROOT/'data/paper_reference/stability.tex').read_text(encoding='utf-8').splitlines():
        if not re.match(r'^(G-|C[23]-)', line):
            continue
        cfg, *cells = line.split('&')
        for i, cell in enumerate(cells):
            env = ['indoor', 'outdoor'][i//3]
            field = ['sd_rmse_pos_m', 'sd_rmse_range_m', 'sd_rmse_angle_deg'][i%3]
            record = next(r for r in stability if r['environment'] == env and r['config_id'] == cfg.strip())
            value = float(re.search(r'\d+(?:\.\d+)?', cell).group())
            require(abs(float(record[field])-value) <= .5*10**(-[4, 4, 3][i%3])+1e-10, 'Stability table: '+cfg+field)
            extra_checks += 1
    forbidden = []
    for path in ROOT.rglob('*'):
        if not path.is_file() or 'generated' in path.relative_to(ROOT).parts:
            continue
        if '__pycache__' in path.parts or 'mplconfig' in path.parts or path.suffix in {'.pyc', '.pyo', '.pem', '.key', '.pfx'} or path.name in {'.env', 'id_rsa', 'id_ed25519', 'remote_ssh_access.md'}:
            forbidden.append(path.relative_to(ROOT).as_posix())
    require(not forbidden, 'Forbidden files: '+str(forbidden))
    result = {'pass': True, 'segments': 205, 'recordings': 31, 'truth_instances': 338,
              'methods': 6, 'ablation_configs': 20, 'archived_replay_rows_checked': 5330,
              'imported_hashes_checked': len(manifest['copied_files'])-len(modified),
              'temporal_selection_rows_checked': 120, 'extra_printed_table_checks': extra_checks,
              'estimator_executed': False, 'simulation_code_executed': False,
              'note': 'Run reproduce_hardware.py verify/all-saved separately for raw/features/source and recomputed metrics.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
