"""Reproducible separation study using unchanged numerical modules.

Source, archived-data and output paths are explicit; no SSH or credentials.
The default invocation only audits. --execute performs the frozen protocol.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(p)


def bootstrap(source):
    source = Path(source).resolve()
    os.environ['JSF_FROZEN_BUNDLE'] = str(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source/'exec2'))
    from jsfsim import paths
    paths.ensure_paths()


def conditions(proto):
    from jsfsim import protocol as P
    specs = P.expand_e2('A', proto['trials'], proto['analytic_distances_m'], [10.0])
    for direction, distances in proto['fdtd_distances_m'].items():
        specs.extend(r for r in P.expand_e2('B', proto['trials'], distances, [10.0])
                     if r['direction'] == direction)
    return specs


def grids(rec, proto):
    d = rec['d_nominal_m']
    ans = [proto['primary_grid_m']]
    if d <= .30 or (rec['direction'] == 'transverse' and .50 <= d <= 1.):
        ans.append(proto['sensitivity_grid_m'])
    return ans


def old_clean(rec, archive):
    from jsfsim import data as D
    import numpy as np
    folder = Path(archive)/'cache'/('fdtd' if rec['layer']=='B' else 'analytic')/rec['clean_id']
    if not (folder/'clean_meta.json').exists():
        return None
    loaded = D.load_clean(archive, rec['layer'], rec['clean_id'])
    meta = loaded['meta']
    if sha(folder/'truth.json') != meta['truth_sha256']:
        raise RuntimeError('Archived truth hash mismatch')
    if not np.array_equal(meta['targets_xy_m'], rec['targets_xy']):
        raise RuntimeError('Archived nominal geometry mismatch')
    if loaded['raw_cir'].shape != (128,4,32) or not np.isfinite(loaded['raw_cir']).all():
        raise RuntimeError('Archived CIR incomplete')
    if meta['clean_seed'] != rec['clean_seed']:
        raise RuntimeError('Archived seed mismatch')
    if rec['layer']=='B':
        if meta['meep']['resolution_px_per_m'] != 350 or meta['meep']['run_time_meep'] != 20:
            raise RuntimeError('Archived numerical settings differ')
    return loaded


def evaluate(payload):
    bootstrap(payload['source'])
    from jsfsim import data as D, protocol as P, runner, estimate, metrics
    import numpy as np
    rec = payload['rec']
    trial = payload['trial']
    with np.load(payload['raw_path'], allow_pickle=False) as z:
        raw = z['raw_cir']
    built = D.build_x(raw, L_requested=128, snr_db=10., physical_id=rec['clean_id'], trial=trial)
    step = payload['grid_m']
    gx=P.grid_axis(P.GRID_X_MIN,P.GRID_X_MAX,step)
    gy=P.grid_axis(P.GRID_Y_MIN,P.GRID_Y_MAX,step)
    case=runner._build_case(rec,built['X'],grid_x=gx,grid_y=gy,trial=trial,
                            seed=P.noise_seed(rec['clean_id'],trial))
    result=estimate.run_single(case)
    pred=[[float(p['x']),float(p['y'])] for p in result['predictions']]
    m=metrics.evaluate_row(rec['targets_xy'],pred,resolution_distance_m=rec['d_actual_m'],
                           status=result['status'])
    return {'condition':rec['task_id'],'layer':rec['layer'],'direction':rec['direction'],
        'd_m':rec['d_nominal_m'],'grid_m':step,'trial':trial,
        'noise_seed':P.noise_seed(rec['clean_id'],trial),'clean_id':rec['clean_id'],
        'status':result['status'],'predictions':pred,'metrics':m,'noise':built['diag'],
        'strict_success':bool((m['resolution'] or {}).get('success',False))}


def summarize(rows):
    n=len(rows); k=sum(r['strict_success'] for r in rows)
    z=1.959963984540054; phat=k/n
    denom=1+z*z/n
    center=(phat+z*z/(2*n))/denom
    half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denom
    return {'n':n,'successful':k,'rate':phat,'wilson95':[center-half,center+half],
        'method_failed':sum(r['status']!='ok' for r in rows),
        'tp':sum(r['metrics']['tp'] for r in rows),
        'dr':sum(r['metrics']['tp'] for r in rows)/(2*n)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'))
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--layer',choices=['A','B','all'],default='all')
    ap.add_argument('--smoke',action='store_true',help='Replay one archived trial; no new fields.')
    args=ap.parse_args()
    bootstrap(args.source)
    from jsfsim import data as D,protocol as P
    import sim_common as sc
    import numpy as np
    from fdtd import api
    proto=read(args.protocol)
    specs=[r for r in conditions(proto) if args.layer=='all' or r['layer']==args.layer]
    out=args.out.resolve(); archive=args.archive.resolve(); source=args.source.resolve()
    if out==archive or out==source or archive/'cache' in out.parents or source in out.parents:
        raise ValueError('Output must be separate from archived inputs and numerical source')
    out.mkdir(parents=True,exist_ok=True)
    ph=sha(args.protocol)
    snapshot=out/'protocol.json'
    if snapshot.exists() and sha(snapshot)!=ph:
        # Compare canonical content because the output is formatted by dump().
        if read(snapshot)!=proto: raise RuntimeError('Output protocol changed')
    dump(snapshot,proto)
    fingerprint=sc.numerical_source_fingerprint(source)
    manifest={'numerical':fingerprint,'driver_sha256':sha(__file__),'protocol_sha256':ph,
        'jsfsim':{p.name:sha(p) for p in (source/'exec2/jsfsim').glob('*.py')}}
    if (out/'source_manifest.json').exists() and read(out/'source_manifest.json')!=manifest:
        raise RuntimeError('Numerical or driver source changed; select a fresh output directory')
    dump(out/'source_manifest.json',manifest)
    dump(out/'conditions.json',specs)
    checks=[]
    for rec in specs:
        existing=old_clean(rec,archive)
        item={'condition':rec['task_id'],'clean_id':rec['clean_id'],'layer':rec['layer'],
              'direction':rec['direction'],'d_m':rec['d_nominal_m'],'archived':existing is not None}
        gate=min(.10,rec['d_actual_m']/3)
        item['grid_check']=[]
        for step in grids(rec,proto):
            gx=P.grid_axis(P.GRID_X_MIN,P.GRID_X_MAX,step)
            gy=P.grid_axis(P.GRID_Y_MIN,P.GRID_Y_MAX,step)
            errors=[math.hypot(float(np.min(abs(np.asarray(gx)-x))),float(np.min(abs(np.asarray(gy)-y))))
                    for x,y in rec['targets_xy']]
            if max(errors)>gate: raise RuntimeError('Grid cannot meet strict gate')
            item['grid_check'].append({'step':step,'nearest_errors':errors,'gate':gate})
        if rec['layer']=='B':
            scene=D.build_fdtd_scene(rec,resolution_px_per_m=350,run_time_meep=20,
                                      background_mode='empty_field',snapshots=128)
            prep=api._prepare(None,scene,max_states=None,background_mode='empty_field')
            distances=np.linalg.norm(prep['traj'][:,0]-prep['traj'][:,1],axis=1)
            item['minimum_actual_center_distance_m']=float(distances.min())
            item['nonoverlap_required_m']=2*P.SCATTERER_RADIUS_M+.005
            if distances.min()<item['nonoverlap_required_m']: raise RuntimeError('Target overlap')
        checks.append(item)
    dump(out/('audit_'+args.layer+'.json'),{'conditions':checks,'new_fdtd_geometries':sum(
        x['layer']=='B' and not x['archived'] for x in checks),'passed':True})
    print('AUDIT',len(checks),'conditions; new FDTD scenes',sum(
        x['layer']=='B' and not x['archived'] for x in checks),flush=True)
    if args.smoke:
        rec=next(r for r in specs if r['layer']=='A' and r['direction']=='transverse' and r['d_nominal_m']==.3)
        payload={'source':str(source),'rec':rec,'trial':0,'grid_m':.05,
                 'raw_path':str(archive/'cache/analytic'/rec['clean_id']/'clean_cir.npz')}
        row=evaluate(payload)
        matches=[]
        for p in (archive/'results/runs').glob('*/rows.jsonl'):
            for line in p.read_text(encoding='utf-8').splitlines():
                r=json.loads(line)
                if r['task']==rec['task_id'] and r['trial']==0: matches.append(r)
        if not matches: raise RuntimeError('No archived smoke reference')
        for r in matches:
            if not np.allclose(row['predictions'],np.array(r['_predictions'])[:,:2],atol=1e-12,rtol=0):
                raise RuntimeError('Smoke replay predictions differ')
            if row['strict_success']!=r['resolution_success']: raise RuntimeError('Smoke metric differs')
        dump(out/'smoke.json',{'passed':True,'matches':len(matches),'row':row})
        print('SMOKE PASSED',flush=True)
    if not args.execute: return
    status={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stage':'running','completed':[]}
    status_path=out/('status_'+args.layer+'.json')
    dump(status_path,status)
    try:
        for rec in specs:
            status.update(current=rec['task_id'],stage='clean');dump(status_path,status)
            loaded=old_clean(rec,archive)
            if loaded is not None:
                raw_root=archive
            else:
                raw_root=out
                kwargs={'root':out,'protocol_hash':ph,'source_aggregate':fingerprint['aggregate']}
                if rec['layer']=='A': loaded=D.build_clean_analytic(rec,**kwargs)
                else:
                    loaded=D.build_clean_fdtd(rec,**kwargs,workers=proto['fdtd_workers'],
                                              resolution_px_per_m=350,run_time_meep=20)
            kind='fdtd' if rec['layer']=='B' else 'analytic'
            folder=raw_root/'cache'/kind/rec['clean_id']
            dump(out/'inputs'/(rec['clean_id']+'.json'),{'source':str(folder),
                'archived':raw_root==archive,'npz_sha256':sha(folder/'clean_cir.npz'),
                'truth_sha256':sha(folder/'truth.json'),'meta':loaded['meta']})
            for step in grids(rec,proto):
                tag=rec['task_id']+'_grid'+str(step)
                destination=out/'rows'/(tag+'.json')
                expected_trials=list(range(proto['trial_start'],proto['trial_start']+proto['trials']))
                if destination.exists():
                    rows=read(destination)
                    if [r['trial'] for r in rows]!=expected_trials: raise RuntimeError('Partial row checkpoint')
                else:
                    status.update(stage='noise_trials',current=tag);dump(status_path,status)
                    jobs=[{'source':str(source),'rec':rec,'trial':t,'grid_m':step,
                           'raw_path':str(folder/'clean_cir.npz')} for t in expected_trials]
                    with concurrent.futures.ProcessPoolExecutor(max_workers=proto['trial_workers'],
                            mp_context=multiprocessing.get_context('spawn')) as pool:
                        rows=list(pool.map(evaluate,jobs,chunksize=4))
                    dump(destination,rows)
                summary={'condition':rec,'grid_m':step,**summarize(rows)}
                dump(out/'summaries'/(tag+'.json'),summary)
                status['completed'].append({'condition':tag,'rate':summary['rate']})
                dump(status_path,status)
                print('DONE',tag,summary['successful'],'/',summary['n'],flush=True)
        status.update(stage='complete',finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        dump(status_path,status)
    except Exception as exc:
        status.update(stage='failed',error=repr(exc));dump(status_path,status)
        raise


if __name__=='__main__':
    main()
