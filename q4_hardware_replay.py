"""Regenerate all 205 Q4 measurement spectra from raw CIR and continuous EMA state."""
from pathlib import Path
import argparse
import json
import os
import sys
import importlib.util
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/q4_hardware')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'spectra').mkdir(exist_ok=True)
    path=ROOT/'hardware_reproduction/reproduce_hardware.py'
    spec=importlib.util.spec_from_file_location('hardware_replay',path)
    H=importlib.util.module_from_spec(spec);sys.modules[spec.name]=H;spec.loader.exec_module(H)
    contexts=H.read(H.DATA/'contexts.json')
    method=H.module('q4_hardware_reference',H.COMMON/'q6_methods_this_method.py')
    checked=[]
    for e in H.dataset():
        r=method.estimate(H.make_case(e,contexts));assert r['status']=='ok'
        with np.load(ROOT/'data/q4/spectra'/('uncorrected_'+e['segment_id']+'.npz'),allow_pickle=False) as z:
            archived={k:z[k] for k in z.files}
            p=np.array([[v['x'],v['y']] for v in r['predictions']]).reshape(-1,2)
            assert p.shape==z['predictions'].shape and np.allclose(p,z['predictions'],atol=1e-10,rtol=0)
            sp=np.asarray(r['spectrum']);ref=z['spectrum']
            err=float(np.max(np.abs(sp/sp.max()-ref/ref.max())))
            assert err<1e-7,(e['segment_id'],err)
        archived.update(spectrum=sp,predictions=p)
        np.savez_compressed(a.out/'spectra'/('uncorrected_'+e['segment_id']+'.npz'),**archived)
        checked.append({'segment_id':e['segment_id'],'normalized_spectrum_max_error':err})
    report={'passed':True,'segments':len(checked),'max_error':max(r['normalized_spectrum_max_error'] for r in checked),'checks':checked}
    (a.out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
