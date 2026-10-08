"""Portable, cache-only reproduction for the released JSF-MUSIC simulations."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
sys.path.insert(0,str(ROOT/'src/exec2'))
sys.path.insert(0,str(ROOT/'src'))
OFFSET=10*math.log10(2)

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def records(run):
    return [json.loads(x) for x in (run/'rows.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

def verify():
    import numpy as np
    from jsfsim import metrics
    import sim_common as sc
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    release_manifest=ROOT/'data/release_manifest.json'
    if release_manifest.exists():
        complete=json.loads(release_manifest.read_text())
        for item in complete['files']:
            p=ROOT/item['path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Release file hash mismatch: '+item['path'])
    for item in manifest['files']:
        p=ROOT/item['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('SHA-256 mismatch: '+item['path'])
    index=json.loads((ROOT/'data/cache_index.json').read_text())
    for item in index:
        kind='fdtd' if item['layer']=='B' else 'analytic'
        p=ROOT/'data/cache'/kind/item['clean_id']
        meta=json.loads((p/'clean_meta.json').read_text())
        for name,key in [('clean_cir.npz','npz_sha256'),('truth.json','truth_sha256')]:
            if hashlib.sha256((p/name).read_bytes()).hexdigest()!=meta[key]:
                raise RuntimeError('Original clean-cache hash mismatch: '+str(p/name))
        with np.load(p/'clean_cir.npz',allow_pickle=False) as z:
            if z['raw_cir'].shape!=(128,4,32) or not np.isfinite(z['raw_cir']).all():
                raise RuntimeError('Invalid CIR: '+str(p))
    total=0;counts={}
    for run in sorted((ROOT/'data/results/runs').iterdir()):
        tasks={x['task_id']:x for x in [json.loads(s) for s in (run/'tasks.jsonl').read_text().splitlines()]}
        rows=records(run);counts[run.name]=len(rows);total+=len(rows)
        source=json.loads((run/'source_manifest.json').read_text())
        actual=sc.numerical_source_fingerprint(ROOT/'src')
        if source['frozen_source_aggregate']!=actual['aggregate']:
            raise RuntimeError('Numerical source fingerprint differs from formal run: '+run.name)
        for r in rows:
            t=tasks[r['task']]
            pred=[x[:2] for x in r['_predictions']]
            check=metrics.evaluate_row(t['targets_xy'],pred,resolution_distance_m=t['resolution_distance_m'],status=r['status'])
            for key in ['pairs','missing','spurious','tp','fn','fp']:
                if check[key]!=r[key]:raise RuntimeError('Saved metric mismatch: '+r['task']+' '+key)
            if not math.isclose(check['sse_m2'],r['sse_m2'],abs_tol=1e-11,rel_tol=1e-11):
                raise RuntimeError('Saved SSE mismatch: '+r['task'])
    if total!=24150:raise RuntimeError('Formal row count is not 24150')
    hardware=ROOT/'data/hardware/index.json'
    hardware_report=None
    if hardware.exists():
        doc=json.loads(hardware.read_text())
        entries=doc['segments'];ids={s['segment_id'] for s in entries}
        frozen=json.loads((ROOT/'data/q4/evaluation_segments_205.json').read_text())
        if len(entries)!=205 or ids!=set(frozen['window_ids']):raise RuntimeError('Hardware identifiers differ from the frozen 205 segments')
        for item in entries:
            p=hardware.parent/item['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Hardware raw CIR hash mismatch: '+item['segment_id'])
            with np.load(p,allow_pickle=False) as z:
                if z['raw_cir'].shape!=(128,8,32):raise RuntimeError('Invalid hardware raw CIR shape')
        hardware_report={'raw_segments':len(entries),'by_environment':doc['by_environment'],
                         'shape':[128,8,32],'all_segment_hashes_match':True}
    report={'pass':True,'files_hashed':len(manifest['files']),'controlled_fdtd_caches':14,
            'analytic_caches':29,'formal_rows':total,'rows_by_run':counts,
            'numerical_source_fingerprint':actual['aggregate'],'hardware':hardware_report,
            'new_fdtd_solves':0,'estimator_runs':0}
    print(json.dumps(report,indent=2))
    return report

def tables(out):
    out.mkdir(parents=True,exist_ok=True)
    total=0
    for run in sorted((ROOT/'data/results/runs').iterdir()):
        family=json.loads((run/'config.json').read_text())['task']
        groups=defaultdict(list)
        for row in records(run):groups[row['task']].append(row)
        result=[]
        reference={r['task']:r for r in csv_rows(ROOT/'data/tables/formal'/(family+'.csv'))}
        for task,rows in sorted(groups.items()):
            first=rows[0];pairs=sum(r['pairs'] for r in rows);ntrue=sum(r['n_truth'] for r in rows)
            errors=[e for r in rows for e in r['_errors']]
            rmse=math.sqrt(sum(r['sse_m2'] for r in rows)/pairs) if pairs else None
            dr=sum(r['tp'] for r in rows)/ntrue if ntrue else None
            if not math.isclose(rmse,float(reference[task]['rmse_pooled_m']),abs_tol=1e-11):
                raise RuntimeError('Published pooled RMSE differs: '+task)
            if not math.isclose(dr,float(reference[task]['dr']),abs_tol=1e-12):
                raise RuntimeError('Published DR differs: '+task)
            resolution=[r['resolution_success'] for r in rows if r['resolution_success'] is not None]
            result.append(dict(task=task,family=family,layout=first['layout'],direction=first['direction'],
                separation_m=first['d_nominal_m'],L=first['L'],snr_parameter_db=first['snr_db'],
                snr_expected_db=first['snr_db']+OFFSET,trials=len(rows),pairs=pairs,
                position_rmse_m=rmse,range_rmse_m=math.sqrt(sum(e['e_r']**2 for e in errors)/pairs) if pairs else None,
                azimuth_rmse_deg=math.sqrt(sum(e['e_theta']**2 for e in errors)/pairs) if pairs else None,
                DR=dr,TP=sum(r['tp'] for r in rows),FN=sum(r['fn'] for r in rows),FP=sum(r['fp'] for r in rows),
                unassigned_truth=sum(r['missing'] for r in rows),
                resolution_success=sum(bool(x) for x in resolution) if resolution else None,
                resolution_trials=len(resolution) if resolution else None))
        with (out/(family+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
        total+=len(result)
    print(json.dumps({'conditions':total,'output':str(out),'all_position_RMSE_and_DR_match':True,'estimator_runs':0}))

def replay(args):
    import numpy as np
    from jsfsim import runner
    run=ROOT/'data/results/runs'/args.run
    tasks=[json.loads(x) for x in (run/'tasks.jsonl').read_text().splitlines()]
    eligible=[t for t in tasks if (not args.layout or t.get('layout')==args.layout)
              and abs(t['snr_db']-args.snr_parameter_db)<1e-9 and t['L']==args.snapshots]
    if not eligible:raise ValueError('No matching saved task')
    rec=sorted(eligible,key=lambda t:t['task_id'])[0]
    source=json.loads((run/'source_manifest.json').read_text())
    kind='fdtd' if rec['layer']=='B' else 'analytic'
    with np.load(ROOT/'data/cache'/kind/rec['clean_id']/'clean_cir.npz',allow_pickle=False) as z:raw=z['raw_cir']
    saved=next(r for r in records(run) if r['task']==rec['task_id'] and r['trial']==args.trial)
    payload=dict(rec=rec,raw=raw,trial=args.trial,run_id=args.run,layer=rec['layer'],
                 truth_sha=saved['truth_sha256'],protocol_hash=source['protocol_hash'],save_sample=False)
    result=runner._trial_worker(payload)
    actual=np.asarray(result['_predictions']);expected=np.asarray(saved['_predictions'])
    same_shape=actual.shape==expected.shape
    same_xy=same_shape and np.allclose(actual[:,:2],expected[:,:2],atol=1e-12,rtol=0)
    ok=same_xy and all(result[k]==saved[k] for k in ['pairs','tp','fn','fp','missing','spurious'])
    print(json.dumps({'pass':bool(ok),'task':rec['task_id'],'trial':args.trial,
            'snr_parameter_db':rec['snr_db'],'snr_expected_db':rec['snr_db']+OFFSET,
            'predictions':result['_predictions'],'saved_predictions':saved['_predictions'],
            'new_fdtd_solves':0,'estimator_runs':1},indent=2))
    if not ok:raise RuntimeError('Replayed predictions differ from the saved trial')

def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
    out.mkdir(parents=True,exist_ok=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#6B5B00'];marks=['o','s','^','D','v']
    fig,axs=plt.subplots(2,2,figsize=(7.05,4.1),layout='constrained')
    for col,family in enumerate(['E1_A','E1_B']):
        rows=csv_rows(ROOT/'data/tables/manuscript'/(family+'.csv'))
        for i,layout in enumerate(['S1','S2','S3','S4','S5']):
            group=sorted([r for r in rows if r['layout']==layout],key=lambda r:float(r['snr_expected_db']))
            x=[float(r['snr_expected_db']) for r in group]
            axs[0,col].plot(x,[float(r['rmse_pooled_m']) for r in group],marker=marks[i],color=colors[i],label=layout)
            axs[1,col].plot(x,[100*float(r['dr']) for r in group],marker=marks[i],color=colors[i])
        axs[0,col].set(title=['Parametric signal model','FDTD'][col],ylabel='Position RMSE (m)');axs[0,col].legend(ncol=3)
        axs[1,col].set(xlabel='Expected CIR SNR (dB)',ylabel='Detection rate (%)',ylim=(-3,103))
    for ax in axs.flat:ax.grid(True,ls='--',alpha=.4)
    save_figure(fig,out,'simulation_snr')
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.3),layout='constrained')
    for col,family in enumerate(['E2_A','E2_B']):
        rows=csv_rows(ROOT/'data/tables/manuscript'/(family+'.csv'))
        for i,direction in enumerate(['transverse','radial','oblique']):
            group=sorted([r for r in rows if r['direction']==direction and float(r['snr_db'])==10],key=lambda r:float(r['d_nominal_m']))
            axs[col].plot([float(r['d_nominal_m']) for r in group],[100*float(r['resolution_success_rate']) for r in group],marker=marks[i],color=colors[i],label=direction)
        axs[col].set(title=['Parametric signal model','FDTD'][col],xlabel='Separation (m)',ylabel='Resolved trials (%)',ylim=(-3,103));axs[col].legend();axs[col].grid(True,ls='--',alpha=.4)
    save_figure(fig,out,'simulation_separation')
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.3),layout='constrained')
    rows=csv_rows(ROOT/'data/tables/manuscript/E4_B.csv')
    for i,snr in enumerate([0,10,20]):
        group=sorted([r for r in rows if float(r['snr_db'])==snr],key=lambda r:int(r['L']))
        x=[int(r['L']) for r in group];label=f'{snr+OFFSET:.2f} dB'
        axs[0].plot(x,[float(r['rmse_pooled_m']) for r in group],marker=marks[i],color=colors[i],label=label)
        axs[1].plot(x,[100*float(r['dr']) for r in group],marker=marks[i],color=colors[i],label=label)
    axs[0].set(ylabel='Position RMSE (m)');axs[1].set(ylabel='Detection rate (%)',ylim=(-3,103))
    for ax in axs:ax.set(xlabel='Snapshots');ax.legend();ax.grid(True,ls='--',alpha=.4)
    save_figure(fig,out,'simulation_snapshots')
    print(json.dumps({'figures':3,'output':str(out),'estimator_runs':0,'new_fdtd_solves':0}))

def save_figure(fig,out,name):
    import matplotlib.pyplot as plt
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('verify')
    for action in ['tables','figures']:
        p=sub.add_parser(action);p.add_argument('--out',type=Path,default=ROOT/'outputs'/action)
    p=sub.add_parser('replay');p.add_argument('--run',default='E1_B_formal350_20260930')
    p.add_argument('--layout',default='S2');p.add_argument('--snr-parameter-db',type=float,default=10)
    p.add_argument('--snapshots',type=int,default=128);p.add_argument('--trial',type=int,default=0)
    args=ap.parse_args()
    if args.action=='verify':verify()
    elif args.action=='tables':tables(args.out.resolve())
    elif args.action=='figures':figures(args.out.resolve())
    else:replay(args)

if __name__=='__main__':main()
