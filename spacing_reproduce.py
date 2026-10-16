"""Verify, replay and plot the revision's fixed-5-cm separation sensitivity study."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'spacing_revision'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
sys.path[:0]=[str(ROOT/'src/exec2'),str(ROOT/'src')]

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def groups():
    return [read(p) for p in sorted((DATA/'summaries').glob('*.json'))]

def verify():
    import numpy as np
    from jsfsim import metrics, protocol as P
    for item in read(DATA/'manifest.json')['files']:
        assert hashlib.sha256((DATA/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    count=0; failure=Counter(); stats=[]
    for g in groups():
        rec=g['condition']; assert g['grid_m']==.05
        rows=read(DATA/'rows'/(rec['task_id']+'_grid0.05.json'))
        assert [r['trial'] for r in rows]==list(range(1000,1100))
        folder=DATA/'cache'/('fdtd' if rec['layer']=='B' else 'analytic')/rec['clean_id']
        meta=read(folder/'clean_meta.json')
        for name,key in [('clean_cir.npz','npz_sha256'),('truth.json','truth_sha256')]:
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==meta[key]
        with np.load(folder/'clean_cir.npz',allow_pickle=False) as z:
            assert z['raw_cir'].shape==(128,4,32) and np.isfinite(z['raw_cir']).all()
        reasons=Counter()
        for r in rows:
            assert r['grid_m']==.05 and r['noise_seed']==P.noise_seed(rec['clean_id'],r['trial'])
            m=metrics.evaluate_row(rec['targets_xy'],r['predictions'],resolution_distance_m=rec['d_actual_m'],status=r['status'])
            assert m==r['metrics']
            assert bool(m['resolution']['success'])==r['strict_success']
            assert r['status']=='ok'
            reason='success' if r['strict_success'] else ('output_count' if len(r['predictions'])!=2 else 'position_gate')
            reasons[reason]+=1; failure[reason]+=1; count+=1
        assert sum(r['strict_success'] for r in rows)==g['successful']
        stats.append({'model':rec['layer'],'direction':rec['direction'],'d_m':rec['d_nominal_m'],**dict(reasons)})
    assert count==6500 and len(groups())==65
    return {'passed':True,'conditions':65,'trials':count,'failure_categories':dict(failure),'condition_counts':stats}

def figures(out):
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out.resolve()/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(7.05,3.5),sharex=True,sharey=True,layout='constrained')
    rows=[]
    for i,layer in enumerate(['A','B']):
        for j,direction in enumerate(['transverse','radial','oblique']):
            ax=axes[i,j]
            gg=sorted([g for g in groups() if g['condition']['layer']==layer and g['condition']['direction']==direction],key=lambda g:g['condition']['d_nominal_m'])
            x=[g['condition']['d_nominal_m'] for g in gg]; y=[100*g['rate'] for g in gg]
            ax.errorbar(x,y,yerr=[[max(0,100*(g['rate']-g['wilson95'][0])) for g in gg],[max(0,100*(g['wilson95'][1]-g['rate'])) for g in gg]],fmt='o-',lw=.9,ms=3,capsize=2,color='#1f77b4' if layer=='A' else '#d95f02')
            ax.set_title(('Parametric' if layer=='A' else 'FDTD')+' / '+direction,fontsize=8)
            ax.set_ylim(-5,105);ax.set_xlim(.1,1.55);ax.set_xticks([.25,.5,.75,1,1.25,1.5]);ax.grid(alpha=.3)
            if i==1:ax.set_xlabel('Separation (m)')
            if j==0:ax.set_ylabel('Success rate (%)')
            for g in gg:
                rows.append({'model':layer,'direction':direction,'d_m':g['condition']['d_nominal_m'],'grid_m':.05,'successes':g['successful'],'trials':g['n'],'rate':g['rate'],'ci_low':g['wilson95'][0],'ci_high':g['wilson95'][1]})
    for ext in ['pdf','png']:fig.savefig(out/('simulation_separation.'+ext),dpi=220)
    plt.close(fig)
    with (out/'spacing_primary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def replay(task, trial):
    import numpy as np
    from jsfsim import data as D,protocol as P,runner,estimate
    rec=next(r for r in read(DATA/'conditions.json') if r['task_id']==task)
    rawpath=DATA/'cache'/('fdtd' if rec['layer']=='B' else 'analytic')/rec['clean_id']/'clean_cir.npz'
    with np.load(rawpath,allow_pickle=False) as z:raw=z['raw_cir']
    built=D.build_x(raw,L_requested=128,snr_db=10.,physical_id=rec['clean_id'],trial=trial)
    gx,gy=P.build_grid()
    case=runner._build_case(rec,built['X'],grid_x=gx,grid_y=gy,trial=trial,seed=P.noise_seed(rec['clean_id'],trial))
    result=estimate.run_single(case)
    pred=[[float(p['x']),float(p['y'])] for p in result['predictions']]
    saved=next(r for r in read(DATA/'rows'/(task+'_grid0.05.json')) if r['trial']==trial)
    assert result['status']==saved['status']
    assert np.asarray(pred).shape==np.asarray(saved['predictions']).shape
    assert np.allclose(pred,saved['predictions'],rtol=0,atol=1e-10)
    return {'passed':True,'task':task,'trial':trial,'predictions':pred}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['verify','figures','replay'])
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/spacing')
    ap.add_argument('--task',default='E2_B_transverse_d875_snr+10')
    ap.add_argument('--trial',type=int,default=1000)
    a=ap.parse_args()
    if a.command=='figures': figures(a.out); print('Saved fixed-5-cm figure and CSV to',a.out)
    else:
        result=verify() if a.command=='verify' else replay(a.task,a.trial)
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/(a.command+'_verification.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='condition_counts'}))

if __name__=='__main__': main()
