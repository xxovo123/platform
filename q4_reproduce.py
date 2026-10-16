"""Reproduce published simulation-to-measurement summaries and a fixed parametric trial."""
from pathlib import Path
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
import sys

ROOT=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
sys.path.insert(0,str(ROOT/'src/exec2'));sys.path.insert(0,str(ROOT/'src'))

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def summarize():
    rows=read(ROOT/'data/q4/per_window.csv')
    groups=defaultdict(list)
    for row in rows:groups[row['config_id']].append(row)
    rhos=[sum(float(r['spectral_rho']) for r in group)/len(group) for group in groups.values()]
    displacements=[sum(float(r['peak_difference_sse_m2']) for r in group)/sum(int(r['n_joint_identity']) for r in group) for group in groups.values()]
    params=read(ROOT/'data/q4/parametric/by_config.csv')
    paramrho=sum(float(r['rho']) for r in params)/len(params)
    result={'hardware_segments':len(rows),'configurations':len(groups),
            'fdtd_config_equal_spectral_correlation':sum(rhos)/len(rhos),
            'fdtd_config_equal_peak_displacement_RMS_m':math.sqrt(sum(displacements)/len(displacements)),
            'parametric_config_equal_spectral_correlation':paramrho,
            'new_fdtd_solves':0,'estimator_runs':0}
    if len(rows)!=205 or len(groups)!=15 or round(result['fdtd_config_equal_spectral_correlation'],3)!=.696 or round(paramrho,3)!=.474:
        raise RuntimeError('Published correspondence summary mismatch')
    print(json.dumps(result,indent=2))

def replay(config,trial):
    import numpy as np
    from jsfsim import estimate,paths
    paths.ensure_paths();CB=paths.adapter('case_build')
    settings=json.loads((ROOT/'data/q4/generator_settings.json').read_text())[config]
    with np.load(ROOT/'data/q4/spectra'/('sim_'+config+'.npz'),allow_pickle=False) as z:
        truth=z['truth'];gx=z['grid_x'];gy=z['grid_y'];mask=z['mask']
    physical_tx=np.array(settings['tx_positions']);physical_rx=np.array(settings['rx_positions'])
    tx=physical_tx.copy();rx=physical_rx.copy();tx[:,2]=0;rx[:,2]=0
    pairs=np.array(settings['pairs'],dtype=int);freq=np.array(settings['frequencies_hz'])
    q=np.column_stack([truth,np.zeros(len(truth))]);r=np.linalg.norm(q,axis=1)
    tau=(2*r[:,None]-(q/r[:,None])@(tx[pairs[:,0]]+rx[pairs[:,1]]).T)/299792458.
    A=np.exp(-2j*np.pi*freq[None,:,None]*tau[:,None,:]).reshape(len(truth),-1)
    seed=int.from_bytes(hashlib.sha256(f'q4_ideal_20260930|{config}|{trial}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed);beta=np.exp(2j*np.pi*rng.random((128,len(truth))))
    X=np.ascontiguousarray(beta@A)
    geom=dict(tx_positions_m=physical_tx.tolist(),rx_positions_m=physical_rx.tolist(),pairs=pairs.tolist())
    case=CB.build_case(case_id=f'control_{config}_plane_wave_{trial}',clean_id=config,X=X,k=len(truth),
             grid_x=gx,grid_y=gy,layer='A',source='analytic',scene='Q4_control',tags=[],snr_db=None,trial=trial,seed=seed,L=128,geometry=geom)
    case['config']=copy.deepcopy(settings['config']);case['search'].update(copy.deepcopy(settings['search']))
    case['model']=copy.deepcopy(settings['model'])
    case['model']['time_origin']='ideal direct frequency model, tau_ref=2r/c; no physical calibration claim'
    result=estimate.run_single(case)
    expected=next(r for r in read(ROOT/'data/q4/parametric/outputs.csv') if r['config_id']==config and int(r['trial'])==trial)
    input_hash=hashlib.sha256(X.tobytes()).hexdigest()
    # Eigenvector signs and response scores may vary with BLAS; positions are on the fixed grid.
    sample=ROOT/'data/q4/parametric/samples'/f'plane_wave_{config}_trial{trial:02d}.npz'
    comparison=None
    if sample.exists():
        with np.load(sample,allow_pickle=False) as z:
            pred=np.array([[p['x'],p['y']] for p in result['predictions']]).reshape(-1,2)
            comparison=pred.shape==z['predictions'].shape and np.allclose(pred,z['predictions'],atol=1e-12,rtol=0)
    ok=input_hash==expected['X_sha256'] and result['status']=='ok' and comparison is not False
    print(json.dumps({'pass':bool(ok),'config_id':config,'trial':trial,'seed':seed,
              'input_hash_matches':input_hash==expected['X_sha256'],'saved_predictions_match':comparison,
              'predictions':result['predictions'],'new_fdtd_solves':0,'estimator_runs':1},indent=2))
    if not ok:raise RuntimeError('Fixed parametric trial differs from the published input/output')

def verify_spectra():
    import numpy as np
    from jsfsim import q4_pipeline as Q
    records=read(ROOT/'data/q4/per_window.csv')
    for r in records:
        with np.load(ROOT/'data/q4/spectra'/('sim_'+r['config_id']+'.npz'),allow_pickle=False) as s, np.load(ROOT/'data/q4/spectra'/('uncorrected_'+r['window_id']+'.npz'),allow_pickle=False) as h:
            valid=s['mask'] & h['mask'] & np.isfinite(s['spectrum']) & np.isfinite(h['spectrum'])
            rho=np.corrcoef(Q.normalized(s['spectrum'])[valid],Q.normalized(h['spectrum'])[valid])[0,1]
            assert abs(rho-float(r['spectral_rho']))<1e-12,r['window_id']
            stats=Q.compare_identities(h['truth'],s['predictions'],h['predictions'])
            assert stats['n_joint_identity']==int(r['n_joint_identity'])
            for key in ['only_sim','only_hw','both_missing']:
                assert len(stats[key+'_identity'])==int(r[key+'_identities']),(r['window_id'],key)
            assert abs(stats['peak_difference_sse_m2']-float(r['peak_difference_sse_m2']))<1e-12
    print(json.dumps({'passed':True,'spectra_compared':len(records),'zero_pixels_retained':True}))
    summarize()


def replay_fdtd(config):
    import numpy as np
    from jsfsim import data as D, q4_pipeline as Q, protocol as P
    settings=json.loads((ROOT/'data/q4/generator_settings.json').read_text())[config]
    with np.load(ROOT/'data/q4/spectra'/('sim_'+config+'.npz'),allow_pickle=False) as z:
        expected={k:z[k] for k in z.files}
    with np.load(ROOT/'data/q4/sim_cir'/config/'clean_cir.npz',allow_pickle=False) as z:raw=z['raw_cir']
    w={**settings,'k':len(expected['truth']),'grid_x':expected['grid_x'],'grid_y':expected['grid_y']}
    for key in ['tx_positions','rx_positions','pairs']:w[key]=np.asarray(w[key])
    geom={'tx_positions_m':P.Q4_TX_POSITIONS_M,'rx_positions_m':P.Q4_RX_POSITIONS_M,'pairs':P.Q4_PAIRS}
    result=Q.result(Q.make_case(w,D.clean_x(raw,L=128),identifier='released_q4_'+config,sim_geometry=geom))
    assert result['status']=='ok'
    assert np.asarray(result['predictions']).shape==expected['predictions'].shape
    assert np.allclose(result['predictions'],expected['predictions'],rtol=0,atol=1e-10)
    err=float(np.max(np.abs(Q.normalized(result['spectrum'])-Q.normalized(expected['spectrum']))))
    assert err<1e-7,(config,err)
    print(json.dumps({'passed':True,'config':config,'normalized_spectrum_max_error':err,'new_fdtd_solves':0}))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('summarize');sub.add_parser('verify-spectra')
    f=sub.add_parser('replay-fdtd');f.add_argument('--config',default='all')
    p=sub.add_parser('replay');p.add_argument('--config',default='G-420');p.add_argument('--trial',type=int,default=0)
    args=ap.parse_args()
    if args.action=='summarize':summarize()
    elif args.action=='verify-spectra':verify_spectra()
    elif args.action=='replay-fdtd':
        for config in (sorted(json.loads((ROOT/'data/q4/generator_settings.json').read_text())) if args.config=='all' else [args.config]):replay_fdtd(config)
    else:replay(args.config,args.trial)

if __name__=='__main__':main()
