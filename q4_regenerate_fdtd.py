"""Regenerate a measured-geometry FDTD sequence (requires Linux/Meep)."""
from pathlib import Path
import argparse
import os
import sys
import json

ROOT=Path(__file__).resolve().parent
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path[:0]=[str(ROOT/'src/exec2'),str(ROOT/'src')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='G-420');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args();dest=a.out.resolve()
    if dest==ROOT or dest.is_relative_to(ROOT/'data') or dest.is_relative_to(ROOT/'src') or dest.is_relative_to(ROOT/'spacing_revision'):
        raise ValueError('Use a new output directory outside archived data/source.')
    if dest.exists() and any(dest.iterdir()):raise ValueError('Output must be empty.')
    from jsfsim import q4_compare as Q, protocol as P
    if a.config not in P.Q4_TARGETS:raise ValueError('Unknown configuration')
    result=Q.sim_clean_for(a.config,root=dest,workers=a.workers,resolution=350,run_time=20)
    print(json.dumps({'config':a.config,'clean_id':result['clean_id'],'kept_states':128,'burnin_states':32,'out':str(dest)}))

if __name__=='__main__':main()
