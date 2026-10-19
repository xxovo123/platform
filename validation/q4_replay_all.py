"""All 15 FDTD spectra, 300 parametric reference trials, and 205 new hardware spectra."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import math
import os
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/exec2'),str(ROOT/'src')]
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
from spacing_reproduce import read,write

def csv_read(p):
    return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))

def csv_write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--hardware-spectra',type=Path,required=True)
    ap.add_argument('--generated-index',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import q4_reproduce
    from jsfsim import data as D,q4_pipeline as Q,protocol as P
    specs=read(ROOT/'data/q4/generator_settings.json')
    oldrows=csv_read(ROOT/'data/q4/per_window.csv')
    assert len(oldrows)==205 and len({r['window_id'] for r in oldrows})==205
    output=a.out/'spectra';output.mkdir(exist_ok=True)
    hardware={};simulation={};checks=[]
    for r in oldrows:
        path=a.hardware_spectra/('uncorrected_'+r['window_id']+'.npz')
        with np.load(path,allow_pickle=False) as z:hardware[r['window_id']]={k:z[k] for k in z.files}
        np.savez_compressed(output/path.name,**hardware[r['window_id']])
    for cfg,settings in sorted(specs.items()):
        with np.load(ROOT/'data/q4/spectra'/('sim_'+cfg+'.npz'),allow_pickle=False) as z:ref={k:z[k] for k in z.files}
        path=Path(read(a.generated_index)['q4'][cfg]) if a.generated_index else ROOT/'data/q4/sim_cir'/cfg/'clean_cir.npz'
        with np.load(path,allow_pickle=False) as z:raw=z['raw_cir']
        assert raw.shape==(128,4,32)
        w={**settings,'k':len(ref['truth']),'grid_x':ref['grid_x'],'grid_y':ref['grid_y']}
        for key in ['tx_positions','rx_positions','pairs']:w[key]=np.asarray(w[key])
        geom={'tx_positions_m':P.Q4_TX_POSITIONS_M,'rx_positions_m':P.Q4_RX_POSITIONS_M,'pairs':P.Q4_PAIRS}
        result=Q.result(Q.make_case(w,D.clean_x(raw,L=128),identifier='validation_'+cfg,sim_geometry=geom))
        assert result['status']=='ok'
        pred=np.asarray(result['predictions']).reshape(-1,2)
        err=float(np.max(np.abs(Q.normalized(result['spectrum'])-Q.normalized(ref['spectrum']))))
        assert pred.shape==ref['predictions'].shape and np.allclose(pred,ref['predictions'],atol=1e-10,rtol=0)
        assert err<1e-7,(cfg,err)
        ref.update(predictions=pred,spectrum=result['spectrum']);simulation[cfg]=ref
        np.savez_compressed(output/('sim_'+cfg+'.npz'),**ref)
        checks.append({'config':cfg,'normalized_spectrum_max_error':err,'passed':True})
    newrows=[]
    for r in oldrows:
        s=simulation[r['config_id']];h=hardware[r['window_id']]
        valid=s['mask'] & h['mask'] & np.isfinite(s['spectrum']) & np.isfinite(h['spectrum'])
        rho=float(np.corrcoef(Q.normalized(s['spectrum'])[valid],Q.normalized(h['spectrum'])[valid])[0,1])
        stats=Q.compare_identities(h['truth'],s['predictions'],h['predictions'])
        assert abs(rho-float(r['spectral_rho']))<1e-7
        assert stats['n_joint_identity']==int(r['n_joint_identity'])
        assert abs(stats['peak_difference_sse_m2']-float(r['peak_difference_sse_m2']))<1e-9
        for key in ['only_sim','only_hw','both_missing']:assert len(stats[key+'_identity'])==int(r[key+'_identities'])
        row=dict(r);row.update(spectral_rho=rho,peak_difference_sse_m2=stats['peak_difference_sse_m2'])
        newrows.append(row)
    csv_write(a.out/'per_window.csv',newrows)
    refs={(r['config_id'],int(r['trial']),r['window_id']):r for r in csv_read(ROOT/'data/q4/parametric/comparisons.csv')}
    assert len(refs)==4100
    paramchecks=[];by_cfg=defaultdict(list);input_checks=[]
    for cfg in sorted(specs):
        for trial in range(20):
            result,inp=q4_reproduce.replay(cfg,trial,return_result=True);input_checks.append(inp)
            pred=[[p['x'],p['y']] for p in result['predictions']]
            for r in [r for r in newrows if r['config_id']==cfg]:
                h=hardware[r['window_id']];s=simulation[cfg]
                valid=s['mask'] & h['mask'] & np.isfinite(result['spectrum']) & np.isfinite(h['spectrum'])
                rho=float(np.corrcoef(Q.normalized(result['spectrum'])[valid],Q.normalized(h['spectrum'])[valid])[0,1])
                stats=Q.compare_identities(h['truth'],pred,h['predictions']);ref=refs[cfg,trial,r['window_id']]
                ok=(abs(rho-float(ref['rho']))<1e-7 and stats['n_joint_identity']==int(ref['joint'])
                    and abs(stats['peak_difference_sse_m2']-float(ref['sse']))<1e-9)
                assert ok,(cfg,trial,r['window_id'],rho,ref)
                paramchecks.append({'config_id':cfg,'trial':trial,'window_id':r['window_id'],'rho':rho,
                    'joint':stats['n_joint_identity'],'sse':stats['peak_difference_sse_m2'],'passed':ok})
                by_cfg[cfg].append(rho)
        print('Q4 complete',cfg,flush=True)
    csv_write(a.out/'parametric_comparisons.csv',paramchecks)
    csv_write(a.out/'parametric_input_checks.csv',input_checks)
    groups=defaultdict(list)
    for r in newrows:groups[r['config_id']].append(r)
    rho=float(np.mean([np.mean([float(r['spectral_rho']) for r in v]) for v in groups.values()]))
    rms=math.sqrt(float(np.mean([sum(float(r['peak_difference_sse_m2']) for r in v)/sum(int(r['n_joint_identity']) for r in v) for v in groups.values()])))
    param=float(np.mean([np.mean(v) for v in by_cfg.values()]))
    assert round(rho,3)==.696 and round(param,3)==.474
    table_ix=[]
    for env,want in [('indoor',(.674,.179,162,2)),('outdoor',(.722,.162,172,2)),('all',(.696,.171,334,4))]:
        selected=[r for r in newrows if env=='all' or r['environment']==env]
        grouped=defaultdict(list)
        for r in selected:grouped[r['config_id']].append(r)
        corr=float(np.mean([np.mean([float(r['spectral_rho']) for r in v]) for v in grouped.values()]))
        dis=math.sqrt(float(np.mean([sum(float(r['peak_difference_sse_m2']) for r in v)/sum(int(r['n_joint_identity']) for r in v) for v in grouped.values()])))
        pairs=sum(int(r['n_joint_identity']) for r in selected)
        missing=sum(len(hardware[r['window_id']]['truth'])-int(r['n_joint_identity']) for r in selected)
        assert (round(corr,3),round(dis,3),pairs,missing)==want,(env,corr,dis,pairs,missing)
        table_ix.append({'environment':env,'correlation':corr,'displacement_m':dis,'pairs':pairs,'missing':missing})
    csv_write(a.out/'table_IX.csv',table_ix)
    report={'passed':True,'full_scope':True,'hardware_spectra':205,'fdtd_spectra':15,'parametric_trials':300,
            'parametric_spectrum_comparisons':4100,'fdtd_config_equal_spectral_correlation':rho,
            'fdtd_config_equal_peak_displacement_RMS_m':rms,'parametric_config_equal_spectral_correlation':param,
            'input_hashes_identical':all(r['input_hash_matches'] for r in input_checks),'fdtd_checks':checks,
            'input_mode':'newly generated CIR' if a.generated_index else 'released clean CIR',
            'tolerance_note':'Compare numerical spectra and all 4100 identity/metric records; cross-library input byte hashes are recorded separately.'}
    write(a.out/'verification.json',report)
    print({k:v for k,v in report.items() if k!='fdtd_checks'})

if __name__=='__main__':main()
