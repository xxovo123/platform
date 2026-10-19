"""Check exact packaged file hashes, current public scope, and expected identities."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacing_reproduce import conditions,read

def main():
    manifest=read(ROOT/'data/release_manifest.json')
    failures=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:failures.append(item['path'])
    if failures:raise RuntimeError('Package changed: '+str(failures))
    recs=conditions();assert len({r['task_id'] for r in recs})==24
    assert not list((ROOT/'spacing_revision/cache').glob('analytic/*'))
    assert not list((ROOT/'data/results/runs').glob('E2_*'))
    assert not any(r['scene']=='E2' for r in read(ROOT/'data/cache_index.json'))
    reference=read(ROOT/'validation/expected_results.json')
    assert reference['spacing']['conditions']==24 and reference['spacing']['trials']==2400
    assert len(list((ROOT/'validation/reference/figures').glob('*.pdf')))==8
    assert len(read(ROOT/'validation/reference/manuscript_tables/index.json'))==9
    print(json.dumps({'passed':True,'files_checked':len(manifest['files']),'spacing_conditions':24,
                      'spacing_trials':2400,'reference_figures':8,'reference_tables':9}))

if __name__=='__main__':main()
