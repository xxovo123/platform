"""Regenerate every public clean input in an isolated Linux directory, with resumable field states."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/exec2'),str(ROOT/'src'),str(ROOT/'validation')]
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
from spacing_reproduce import read,write,conditions
from jsfsim import data as D,protocol as P,paths
paths.ensure_paths()
import geometry_gate
geometry_gate.install()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory():
    controlled={};analytic={}
    for folder in sorted((ROOT/'data/results/runs').iterdir()):
        for line in (folder/'tasks.jsonl').read_text().splitlines():
            r=json.loads(line);kind='fdtd' if r['layer']=='B' else 'analytic'
            item={'rec':r,'reference':str(ROOT/'data/cache'/kind/r['clean_id']/'clean_cir.npz')}
            (controlled if r['layer']=='B' else analytic)[r['clean_id']]=item
    for r in conditions():
        controlled[r['clean_id']]={'rec':r,'reference':str(ROOT/'spacing_revision/cache/fdtd'/r['clean_id']/'clean_cir.npz')}
    return analytic,controlled

def compare(actual,reference):
    import numpy as np
    with np.load(actual,allow_pickle=False) as a,np.load(reference,allow_pickle=False) as b:
        aa=a['raw_cir'];bb=b['raw_cir']
    assert aa.shape==bb.shape==(128,4,32) and np.isfinite(aa).all()
    diff=float(np.max(np.abs(aa-bb))); scale=float(np.max(np.abs(bb)))
    relative=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb))
    normalized=diff/scale
    return {'passed':relative<=1e-7 and normalized<=1e-7,'relative_l2_error':relative,
            'max_error_over_reference_peak':normalized,'max_abs_error':diff,
            'actual_sha256':sha(actual),'reference_sha256':sha(reference),'shape':list(aa.shape)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--kind',choices=['all','analytic','controlled','q4'],default='all')
    ap.add_argument('--workers',type=int,default=4,help='Parallel FDTD states; one scene at a time')
    ap.add_argument('--id',help='Optional one clean ID or Q4 config for diagnosis; not full acceptance')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--plan',action='store_true',help='Read-only inventory; no Meep import or solve')
    a=ap.parse_args();analytic,controlled=inventory();q4=read(ROOT/'data/q4/generator_settings.json')
    if a.plan:
        print(json.dumps({'analytic':len(analytic),'controlled_fdtd':len(controlled),'q4_fdtd':len(q4),
                          'controlled_ids':sorted(controlled),'q4_configs':sorted(q4)},indent=2));return
    assert platform.system()=='Linux','Full reproduction requires Linux'
    assert a.workers>=1
    out=a.out.resolve()
    assert out!=ROOT and not any(out.is_relative_to(ROOT/p) for p in ['src','data','spacing_revision','validation/reference'])
    identity={'package_manifest_sha256':sha(ROOT/'data/release_manifest.json'),'mode':'independent-input-regeneration'}
    marker=out/'validation_identity.json'
    if out.exists() and any(out.iterdir()):
        assert a.resume and marker.exists() and read(marker)==identity,'Use a new directory or --resume for this package only'
    out.mkdir(parents=True,exist_ok=True);write(marker,identity)
    write(out/'verification.json',{'passed':False,'status':'running','full_scope':False,
        'source_manifest_sha256':identity['package_manifest_sha256']})
    index=read(out/'generated_index.json') if (out/'generated_index.json').exists() else {'analytic':{},'controlled':{},'q4':{}}
    checks=read(out/'checks.json') if (out/'checks.json').exists() else {}
    if a.kind!='analytic':
        import meep
        print('Meep',meep.__version__,'workers',a.workers,flush=True)
    from fdtd import api
    for kind,items in [('analytic',analytic),('controlled',controlled),('q4',q4)]:
        if a.kind not in ['all',kind]:continue
        for key,item in sorted(items.items()):
            if a.id and key!=a.id:continue
            t0=time.perf_counter();dest=out/kind/key
            if kind=='analytic':
                ref=Path(item['reference']);meta=read(ref.parent/'clean_meta.json')
                generated=D.build_clean_analytic(item['rec'],root=dest,protocol_hash=meta['protocol_hash'],source_aggregate=meta['source_aggregate'])
                path=dest/'cache/analytic'/key/'clean_cir.npz'
            else:
                if kind=='controlled':
                    rec=item['rec'];ref=Path(item['reference'])
                    scene=D.build_fdtd_scene(rec,resolution_px_per_m=350,run_time_meep=20,background_mode='empty_field',snapshots=128)
                    mode='empty_field'
                else:
                    truth=P.Q4_TARGETS[key];cid=P.clean_id_for('fdtd',len(truth),truth)
                    rec={'clean_id':cid,'k':len(truth),'targets_xy':truth,'clean_seed':P.clean_seed(cid)}
                    geom={'tx_positions_m':P.Q4_TX_POSITIONS_M,'rx_positions_m':P.Q4_RX_POSITIONS_M,'pairs':P.Q4_PAIRS}
                    scene=D.build_fdtd_scene(rec,resolution_px_per_m=350,run_time_meep=20,background_mode='ema_windowed',
                                            snapshots=128,burnin_states=32,geometry=geom)
                    mode='ema_windowed';ref=ROOT/'data/q4/sim_cir'/key/'clean_cir.npz'
                generated=api.generate_clean(None,scene,dest/'clean',cache_root=dest/'field_cache',resume=True,
                    workers=a.workers,background_mode=mode,progress=lambda s:print(kind,key,s,flush=True))
                path=Path(generated['clean_npz'])
            check=compare(path,ref)
            check.update(kind=kind,id=key,elapsed_s=time.perf_counter()-t0)
            if kind!='analytic':
                check.update(run_summary=generated['provenance_json'],stats=generated['stats'])
            checks[kind+'/'+key]=check;index[kind][key]=str(path)
            write(out/'checks.json',checks);write(out/'generated_index.json',index)
            print(json.dumps(check),flush=True)
            if not check['passed']:raise RuntimeError('New CIR differs; do not replace it with the archived CIR')
    expected={'analytic':set(analytic),'controlled':set(controlled),'q4':set(q4)}
    full=all(set(index[k])==v for k,v in expected.items())
    report={'passed':all(c['passed'] for c in checks.values()),'full_scope':full,
            'counts':{k:len(index[k]) for k in index},'expected':{k:len(v) for k,v in expected.items()},
            'source_manifest_sha256':identity['package_manifest_sha256'],'linux':True,
            'note':'Input regeneration is separate from downstream estimator replay; run the full suite using generated_index.json.'}
    write(out/'verification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
