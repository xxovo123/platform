"""Reproduce Fig. 4: 24 FDTD conditions and 2,400 truth-referenced trials."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import json
import math
import os
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'spacing_revision'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
sys.path[:0]=[str(ROOT/'src/exec2'),str(ROOT/'src')]
DIRS=['transverse','radial','oblique']
POINTS=[.15,.20,.25,.30,.40,.50,.75,1.00]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def conditions():
    records=read(DATA/'conditions.json')
    assert len(records)==24
    assert {(r['layer'],r['direction'],r['d_nominal_m']) for r in records}=={
        ('B',d,s) for d in DIRS for s in POINTS}
    return records

def verify():
    import numpy as np
    from jsfsim import metrics,protocol as P
    for item in read(DATA/'manifest.json')['files']:
        assert hashlib.sha256((DATA/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    refs=read(DATA/'spectral_reference.json')
    lut={(r['condition'],r['trial']):r for r in refs['trials_detail']}
    assert len(lut)==len(refs['trials_detail'])==2400
    seen=set(); counts=[]
    for rec in conditions():
        tid=rec['task_id']; folder=DATA/'cache/fdtd'/rec['clean_id']
        meta=read(folder/'clean_meta.json')
        for name,key in [('clean_cir.npz','npz_sha256'),('truth.json','truth_sha256')]:
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==meta[key]
        with np.load(folder/'clean_cir.npz',allow_pickle=False) as z:
            assert z['raw_cir'].shape==(128,4,32) and np.isfinite(z['raw_cir']).all()
        rows=read(DATA/'rows'/(tid+'_grid0.05.json'))
        assert [r['trial'] for r in rows]==list(range(1000,1100))
        spectral=0
        for r in rows:
            key=(tid,r['trial']); seen.add(key)
            assert r['grid_m']==.05 and r['noise_seed']==P.noise_seed(rec['clean_id'],r['trial'])
            m=metrics.evaluate_row(rec['targets_xy'],r['predictions'],resolution_distance_m=rec['d_actual_m'],status=r['status'])
            assert m==r['metrics'] and bool(m['resolution']['success'])==r['strict_success']
            ref=lut[key]; assert ref['noise_seed']==r['noise_seed']
            assert ref['original_output_success']==r['strict_success']
            peaks=ref['matched_local_peaks'];gate=min(.10,rec['d_actual_m']/3)
            for truth,peak in zip(rec['targets_xy'],peaks):
                if peak is not None:
                    assert math.dist(truth,[peak['x'],peak['y']])<=gate+1e-12
            ok=all(p is not None for p in peaks)
            if ok: assert (peaks[0]['row'],peaks[0]['col'])!=(peaks[1]['row'],peaks[1]['col'])
            assert ok==ref['spectral_two_peak_correspondence']
            spectral+=int(ok)
        summary=next(s for s in refs['summaries'] if s['condition']==tid)
        assert spectral==summary['spectral_correspondence_count']
        assert sum(r['strict_success'] for r in rows)==summary['original_output_success_count']
        counts.append({'condition':tid,'spectral':spectral,'trials':100})
    assert seen==set(lut)
    return {'passed':True,'conditions':24,'trials':2400,'estimator_runs':0,'field_solves':0,'counts':counts}

def correspondence(spectrum,gx,gy,truth,d):
    """The approved pre-selection 3x3 / eight-connected plateau diagnostic."""
    import numpy as np
    from jsfsim import estimate
    picker=estimate.load_estimator().load('this_method').knownk_peaks
    mask=(spectrum>=picker.neighbourhood_max3x3(spectrum)) & np.isfinite(spectrum) & (spectrum>0)
    _, groups=picker.label_plateaus_8conn(mask)
    peaks=[]
    for group in groups:
        row,col=min(group,key=lambda rc:(-spectrum[rc],rc[0],rc[1]))
        peaks.append({'row':int(row),'col':int(col),'x':float(gx[col]),'y':float(gy[row]),'value':float(spectrum[row,col])})
    peaks.sort(key=lambda p:(-p['value'],p['row'],p['col']))
    gate=min(.10,d/3)
    matched=[next((p for p in peaks if math.dist(t,[p['x'],p['y']])<=gate+1e-12),None) for t in truth]
    ok=all(p is not None for p in matched)
    if ok: assert (matched[0]['row'],matched[0]['col'])!=(matched[1]['row'],matched[1]['col'])
    return bool(ok),matched

def replay_group(payload):
    import numpy as np
    from jsfsim import data as D,protocol as P,runner,estimate,metrics
    rec,trials,out,index=payload; tid=rec['task_id'];out=Path(out)
    if index:
        path=Path(read(index)['controlled'][rec['clean_id']])
    else:
        path=DATA/'cache/fdtd'/rec['clean_id']/'clean_cir.npz'
    with np.load(path,allow_pickle=False) as z: raw=z['raw_cir']
    assert raw.shape==(128,4,32)
    saved={r['trial']:r for r in read(DATA/'rows'/(tid+'_grid0.05.json'))}
    refs={r['trial']:r for r in read(DATA/'spectral_reference.json')['trials_detail'] if r['condition']==tid}
    gx,gy=P.build_grid();rows=[];fail=[]
    for trial in trials:
        built=D.build_x(raw,L_requested=128,snr_db=10.,physical_id=rec['clean_id'],trial=trial)
        result=estimate.run_single(runner._build_case(rec,built['X'],grid_x=gx,grid_y=gy,trial=trial,seed=P.noise_seed(rec['clean_id'],trial)))
        pred=[[float(p['x']),float(p['y'])] for p in result['predictions']]
        ref=saved[trial]
        same=result['status']==ref['status'] and np.asarray(pred).shape==np.asarray(ref['predictions']).shape
        same=bool(same and np.allclose(pred,ref['predictions'],rtol=0,atol=1e-10))
        ok,peaks=correspondence(result['spectrum'],gx,gy,rec['targets_xy'],rec['d_actual_m'])
        metric=metrics.evaluate_row(rec['targets_xy'],pred,resolution_distance_m=rec['d_actual_m'],status=result['status'])
        peak_xy=[None if p is None else [p['x'],p['y']] for p in peaks]
        ref_xy=[None if p is None else [p['x'],p['y']] for p in refs[trial]['matched_local_peaks']]
        matched=ok==refs[trial]['spectral_two_peak_correspondence'] and peak_xy==ref_xy
        passed=same and matched and bool(metric['resolution']['success'])==ref['strict_success']
        rows.append({'condition':tid,'trial':trial,'noise_seed':P.noise_seed(rec['clean_id'],trial),
                     'status':result['status'],'predictions':pred,'spectral_two_peak_correspondence':ok,
                     'original_output_success':bool(metric['resolution']['success']),
                     'matched_local_peaks':peaks,'passed':passed})
        if not passed:fail.append(trial)
    write(out/'rows'/(tid+'.json'),rows)
    summary={'model':'B','condition':tid,'direction':rec['direction'],'spacing_m':rec['d_nominal_m'],
             'trials':len(rows),'spectral_correspondence_count':sum(r['spectral_two_peak_correspondence'] for r in rows),
             'original_output_success_count':sum(r['original_output_success'] for r in rows),
             'passed':not fail,'failed_trials':fail,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    summary['spectral_correspondence_rate']=summary['spectral_correspondence_count']/len(rows)
    write(out/'conditions'/(tid+'.json'),summary)
    return summary

def replay_all(out,workers=1,index=None,task=None,trial=1000):
    records=[r for r in conditions() if task is None or r['task_id']==task]
    assert records,'Unknown condition'
    trials=list(range(1000,1100)) if task is None else [trial]
    assert all(t in range(1000,1100) for t in trials)
    payloads=[(r,trials,str(out),str(index) if index else None) for r in records]
    if workers==1:summaries=list(map(replay_group,payloads))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:summaries=list(pool.map(replay_group,payloads))
    full=task is None
    result={'passed':all(s['passed'] for s in summaries),'full_scope':full,'conditions':len(summaries),
            'trials':sum(s['trials'] for s in summaries),'summaries':summaries,
            'input_mode':'newly generated CIR' if index else 'released clean CIR','field_solves_in_this_command':0}
    write(out/'replay_verification.json',result)
    if not result['passed']:raise RuntimeError('Spacing replay mismatch; see per-condition records')
    return result

def figures(out,results=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=read(results or DATA/'spectral_reference.json');groups=report['summaries']
    assert len(groups)==24 and all(s['trials']==100 for s in groups)
    assert {(s['direction'],s['spacing_m']) for s in groups}=={(d,p) for d in DIRS for p in POINTS}
    os.environ.setdefault('MPLCONFIGDIR',str(out.resolve()/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.5,2.65),layout='constrained')
    for direction,color,marker in zip(DIRS,['#0072B2','#D55E00','#009E73'],['o','s','^']):
        rr=sorted([s for s in groups if s['direction']==direction],key=lambda s:s['spacing_m'])
        ax.plot([s['spacing_m'] for s in rr],[100*s['spectral_correspondence_rate'] for s in rr],
                marker=marker,linestyle='-',linewidth=.9,color=color,markerfacecolor=color,
                markeredgecolor=color,markersize=3.5,markeredgewidth=.5,label=direction.capitalize(),clip_on=False)
    ax.set_xlim(.125,1.025);ax.set_ylim(-5,105)
    ax.set_xticks(POINTS,[f'{p:.2f}' for p in POINTS],rotation=45,ha='right',fontsize=7)
    ax.set_yticks([0,20,40,60,80,100]);ax.grid(linestyle='--',linewidth=.4,alpha=.4)
    ax.set_title('FDTD target-separation sensitivity',fontsize=8)
    ax.set_ylabel('Two-peak correspondence (%)\n(truth-referenced diagnostic)')
    ax.set_xlabel('Target separation (m)');ax.legend(loc='center right',fontsize=7)
    for ext in ['pdf','png']:fig.savefig(out/('simulation_separation.'+ext),dpi=250)
    plt.close(fig)
    with (out/'spacing_primary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['direction','spacing_m','successes','trials','rate']);w.writeheader()
        for s in groups:w.writerow(dict(direction=s['direction'],spacing_m=s['spacing_m'],
            successes=s['spectral_correspondence_count'],trials=s['trials'],rate=s['spectral_correspondence_rate']))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['verify','figures','replay','replay-all'])
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/spacing')
    ap.add_argument('--task',default='E2_B_transverse_d300_snr+10')
    ap.add_argument('--trial',type=int,default=1000)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--generated-index',type=Path)
    ap.add_argument('--results',type=Path,help='Complete replay_verification.json for figures from new results')
    a=ap.parse_args()
    if a.workers<1:ap.error('workers must be positive')
    if a.command=='figures':figures(a.out,a.results)
    elif a.command=='verify':
        result=verify();write(a.out/'verify_verification.json',result);print(json.dumps(result))
    else:
        result=replay_all(a.out,a.workers,a.generated_index,a.task if a.command=='replay' else None,a.trial)
        print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))

if __name__=='__main__':main()
