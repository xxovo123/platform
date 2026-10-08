"""Regenerate one controlled FDTD scene in a new output directory (requires Meep)."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(ROOT/'src/exec2'));sys.path.insert(0,str(ROOT/'src'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clean-id',default='cln_de385a3b2982')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-states',type=int,help='Optional shortened demonstration; not a full paper dataset.')
    args=ap.parse_args()
    dest=args.out.resolve()
    if dest==ROOT or dest.is_relative_to(ROOT/'data') or dest.is_relative_to(ROOT/'src'):
        raise ValueError('Choose a new output directory outside data/ and src/.')
    if dest.exists() and any(dest.iterdir()):raise ValueError('Output must be empty; existing data will not be overwritten.')
    from jsfsim import data as D,paths
    paths.ensure_paths()
    from fdtd import api
    specs=[json.loads(x) for p in (ROOT/'data/results/runs').glob('*_B_*/tasks.jsonl') for x in p.read_text().splitlines()]
    rec=next((r for r in specs if r['clean_id']==args.clean_id),None)
    if rec is None:raise ValueError('Unknown controlled FDTD clean ID')
    scene=D.build_fdtd_scene(rec,resolution_px_per_m=350,run_time_meep=20,background_mode='empty_field',snapshots=128)
    result=api.generate_clean(None,scene,dest/'clean',cache_root=dest/'field_cache',workers=args.workers,
                              max_states=args.max_states,background_mode='empty_field',progress=print)
    print(json.dumps({'output':str(dest),'full_128_snapshot_scene':args.max_states is None,
                      'resolution_cells_per_m':350,'duration_meep':20,
                      'note':'Regeneration at the archived experiment settings; no new convergence claim.'},indent=2))

if __name__=='__main__':main()
