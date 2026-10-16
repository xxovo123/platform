"""Portable hardware-only reproduction. Never imports or executes simulation code."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
SOURCE=ROOT/'source'
COMMON=SOURCE/'大修执行_20260923/04_Q6/common'
METHODS=SOURCE/'大修执行_20260923/04_Q6/hardware/methods'
Q5=SOURCE/'大修执行_20260923/03_Q5/hardware_ablation_20260926/scripts'
DRIVER=SOURCE/'大修执行_20260923/01_投稿口径/p0b/driver'
for p in [COMMON,Q5,DRIVER,SOURCE/'Q2 数据采集/uwb-occupancy-detection-main']:
    sys.path.insert(0,str(p))
import q5_metrics
import q6_search_profiles
from knownk_eval import load_protocol, peaks
PROTOCOL=load_protocol(SOURCE/'大修执行_20260923/01_投稿口径/p0b/protocol/knownk_protocol.json')
LABELS={'this_method':'JSF-MUSIC','b1_deng':'Deng','b2_kakouche':'Kakouche','c1_jade':'JADE','c3_kim':'Kim','b5_wang_full2d':'Traditional 2D MUSIC'}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def py(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [py(v) for v in x]
    return x

def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(py(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def table(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    return m

def dataset():
    entries=read(DATA/'raw/index.json')['segments']
    assert len(entries)==len({e['segment_id'] for e in entries})==205
    assert sum(e['K'] for e in entries)==338
    return entries

def features(entry):
    """Continuous-EMA replay using separately archived complex128 initial states."""
    sid=entry['segment_id']; rec=entry['recording_id']
    meta=read(DATA/'raw/recordings'/(rec+'.json'))
    with np.load(DATA/'raw'/entry['file'],allow_pickle=False) as z:
        raw=z['raw_cir'].copy()
    with np.load(DATA/'ema_states'/(sid+'.npz'),allow_pickle=False) as z:
        ref=z['ema_reference_before_segment'].copy(); first=z['ema_first_sample_mask'].copy()
    out=np.empty_like(raw); alpha=1./meta['bg_m_factor']
    for l in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            cur=raw[l,c]
            if first[c]:
                out[l,c]=cur; ref[c]=cur.astype(np.complex128); first[c]=False
            else:
                out[l,c]=(cur-ref[c]).astype(np.complex64)
                ref[c]=(1-alpha)*ref[c]+alpha*cur
    crop=out[:,:,meta['crop_tap_start_inclusive']:meta['crop_tap_stop_exclusive']]
    sel=np.asarray(meta['selected_pair_indices'])
    X=np.fft.fft(crop.astype(np.complex128),n=meta['fft_length'],axis=-1)[:,sel,:][:,:,meta['fft_bin_indices']]
    if meta['phase_calibration_enabled']:
        X=X*np.exp(-1j*np.asarray(meta['channel_phase_calibration_rad'])[sel])[None,:,None]
    return X.transpose(0,2,1).reshape(128,-1)

def make_case(entry,contexts):
    ctx=copy.deepcopy(contexts[entry['recording_id']]); X=features(entry)
    case={'case_schema_version':1,'id':entry['segment_id'],'recording_id':entry['recording_id'],
          'environment':entry['environment'],'config_id':entry['config_id'],'t_end_s':entry['endtime'],
          'X':X,'k':entry['K'],'feature_order':'frequency-major then channel (index = f*C + ch)',
          'n_snapshots':128,'n_features':68,'n_freq_bins':17,'n_channels':4,
          'snapshots_available':False,'snapshots':None,
          'config':ctx['config'],'model':ctx['model'],'search':ctx['search']}
    for key in ['grid_x','grid_y','frequencies_hz','pairs','tx_positions','rx_positions']:
        case[key]=np.asarray(ctx[key])
    case['search']['grid_x_m']=case['grid_x']; case['search']['grid_y_m']=case['grid_y']
    return case

def gated(T,P):
    """Frozen max cardinality, then min total distance, then lexicographic tie."""
    d=np.linalg.norm(T[:,None,:]-P[None,:,:],axis=2)
    best=((float('inf'),float('inf'),()),[])
    def visit(i,used,pairs,cost):
        nonlocal best
        if i==len(T):
            key=(-len(pairs),cost,tuple(pairs))
            if key<best[0]: best=(key,pairs.copy())
            return
        visit(i+1,used,pairs,cost)
        for j in range(len(P)):
            if j not in used and d[i,j]<=.30+1e-12:
                visit(i+1,used|{j},pairs+[(i,j)],cost+float(d[i,j]))
    visit(0,set(),[],0.)
    return best[1]

def metrics(T,P):
    T=np.asarray(T,float).reshape(-1,2); P=np.asarray(P,float).reshape(-1,2)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(P))
    cost=((T[:,None,:]-P[None,:,:])**2).sum(axis=2)
    r,c=linear_sum_assignment(cost)
    pos=cost[r,c]
    rt=np.linalg.norm(T,axis=1); rp=np.linalg.norm(P,axis=1)
    at=np.degrees(np.arctan2(T[:,0],-T[:,1])); ap=np.degrees(np.arctan2(P[:,0],-P[:,1]))
    dr=rp[c]-rt[r]; da=(ap[c]-at[r]+180)%360-180
    gp=gated(T,P); n=len(r)
    return {'pairs':n,'missing':len(T)-n,'n_truth':len(T),'n_pred':len(P),
            'sse_pos':float(pos.sum()),'sse_range':float((dr**2).sum()),'sse_angle':float((da**2).sum()),
            'tp':len(gp),'fn':len(T)-len(gp),'fp':len(P)-len(gp),
            'ospa_sq':q5_metrics.ospa_squared(T,P),
            'rmse_pos_m':math.sqrt(float(pos.sum())/n) if n else None,
            'rmse_range_m':math.sqrt(float((dr**2).sum())/n) if n else None,
            'rmse_angle_deg':math.sqrt(float((da**2).sum())/n) if n else None}

def aggregate(rows):
    n=sum(r['pairs'] for r in rows); nt=sum(r['n_truth'] for r in rows)
    result={'n_segments':len(rows),'n_truth':nt,'pairs':n,'missing':sum(r['missing'] for r in rows)}
    for key in ['tp','fn','fp']:
        result[key]=sum(r[key] for r in rows)
    for kind,unit in [('pos','m'),('range','m'),('angle','deg')]:
        result['rmse_'+kind+'_'+unit]=math.sqrt(sum(r['sse_'+kind] for r in rows)/n) if n else None
    result['dr_percent']=100*result['tp']/nt if nt else None
    result['segments_with_fp']=sum(r['fp']>0 for r in rows)
    result['fp_per_segment']=result['fp']/len(rows) if rows else None
    result['rms_ospa_m']=math.sqrt(sum(r['ospa_sq'] for r in rows)/len(rows)) if rows else None
    return result

def frozen_rows(name,entries,result_map):
    records=read(ROOT/result_map[name]['file']); rows=[]
    for e in entries:
        sid=e['segment_id']; rec=records[sid]
        assert rec['status']=='ok',(name,sid,rec['status'])
        T,P=q5_metrics.extract_sets(rec,where=name+'/'+sid)
        with np.load(DATA/'raw'/e['file'],allow_pickle=False) as raw:
            assert np.array_equal(np.asarray(T),raw['truth_xy_m']),(name,sid,'truth mismatch')
        m=metrics(T,P)
        assert (m['tp'],m['fn'],m['fp'])==(rec['tp'],rec['fn'],rec['fp']),(name,sid,'detection mismatch')
        rows.append({'method':name,'window_id':sid,'recording_id':e['recording_id'],'environment':e['environment'],
                     'config_id':e['config_id'],'K':e['K'],**m})
    return rows

def paper_checks(summaries):
    """Compare with frozen copy of working paper, at printed precision."""
    checks=[]
    baseline=(DATA/'paper_reference/baseline.tex').read_text('utf-8')
    ids=list(LABELS)
    rows=[line for line in baseline.splitlines() if re.match(r'^(JSF-MUSIC|Deng et|Kakouche et|JADE |Kim et|Traditional 2D)',line)]
    assert len(rows)==6
    for name,line in zip(ids,rows):
        cells=line.split('&')[1:]
        nums=[float(re.findall(r'-?\d+(?:\.\d+)?',cell)[0]) for cell in cells]
        fields=['rmse_pos_m','rmse_range_m','rmse_angle_deg','dr_percent','fn','fp','pairs']
        for f,expected in zip(fields,nums):
            value=summaries[name][f]; tol=.005000001 if f=='dr_percent' else .000050001 if f.startswith('rmse') else 0
            checks.append({'table':'baseline','method':name,'field':f,'paper':expected,'recomputed':value,'atol':tol,'pass':abs(value-expected)<=tol})
    matrix=read(DATA/'config_matrix.json')['configs']
    # Paper rows follow this explicit ordering; no reliance on execution order.
    order=['REF','A_F09','A_F05','B_span250','B_span125','S_span250_F9','C_L016','C_L032','C_L064',
           'D_C2_01','D_C2_02','D_C2_03','D_C2_12','D_C2_13','D_C2_23',
           'D_C3_012','D_C3_013','D_C3_023','D_C3_123','E_FB0']
    text=(DATA/'paper_reference/ablation.tex').read_text('utf-8')
    lines=[l for l in text.splitlines() if re.match(r'^(Reference|\d+ bins|\d+ snapshots|Channels |Without FB)',l)]
    available={c['id'] for c in matrix}
    assert set(order)==available
    assert len(lines)==len(order)==20
    for cid,line in zip(order,lines):
        vals=[float(re.findall(r'\d+(?:\.\d+)?',v)[0]) for v in line.split('&')[1:]]
        s=summaries['ablation/'+cid]
        for f,expected,tol in [('rmse_pos_m',vals[1],.000050001),('dr_percent',vals[2],.005000001),('missing',vals[3],0)]:
            value=s[f]; checks.append({'table':'ablation','method':cid,'field':f,'paper':expected,'recomputed':value,'atol':tol,'pass':abs(value-expected)<=tol})
    return checks

def saved(out):
    entries=dataset(); result_map=read(DATA/'result_map.json'); all_rows=[]; summaries={}
    for name in result_map:
        rows=frozen_rows(name,entries,result_map); all_rows+=rows; summaries[name]=aggregate(rows)
    table(out/'summary.csv',[{'method':n,**s} for n,s in summaries.items()]); table(out/'per_segment.csv',all_rows)
    groups=[]
    main=[r for r in all_rows if r['method']=='this_method']
    for env in ['indoor','outdoor','all']:
        for k in [None,1,2,3]:
            rows=[r for r in main if (env=='all' or r['environment']==env) and (k is None or r['K']==k)]
            groups.append({'environment':env,'K':k,**aggregate(rows)})
    table(out/'groups.csv',groups)
    grouped=defaultdict(list)
    for r in main: grouped[r['environment'],r['config_id']].append(r)
    stability=[]; selected=[]
    for (env,cfg),rows in sorted(grouped.items()):
        rows.sort(key=lambda r:(r['recording_id'],float(r['window_id'].split('__t')[-1])))
        assert len(rows)>=4
        ranks=[int(i*(len(rows)-1)/3+.5) for i in range(4)]
        chosen=[rows[i] for i in ranks]
        s={'environment':env,'config_id':cfg,'source_n':len(rows),'n_selected':4}
        for f in ['rmse_pos_m','rmse_range_m','rmse_angle_deg']:
            assert all(r[f] is not None for r in chosen)
            s['sd_'+f]=float(np.std([r[f] for r in chosen],ddof=1))
        stability.append(s)
        selected.extend({'environment':env,'config_id':cfg,'rank_zero_based':rank,'window_id':r['window_id']} for rank,r in zip(ranks,chosen))
    table(out/'stability.csv',stability); table(out/'selected_measurements.csv',selected)
    reference=list(csv.DictReader((DATA/'reference_stability.csv').open(encoding='utf-8-sig',newline='')))
    deltas=[]
    for s,r in zip(stability,reference):
        assert (s['environment'],s['config_id'])==(r['environment'],r['config_id'])
        for f in ['sd_rmse_pos_m','sd_rmse_range_m','sd_rmse_angle_deg']:
            deltas.append(abs(s[f]-float(r[f])))
    assert len(stability)==len(reference)==30 and len(selected)==120
    checks=paper_checks(summaries)
    write(out/'paper_checks.json',checks)
    result={'scope':'saved predictions -> independent assignments, statistics and hardware plot inputs',
            'n_prediction_records':len(all_rows),'n_methods_and_configs':len(summaries),'n_stability_groups':30,
            'n_selected_temporal_measurements':120,'stability_max_abs_difference':max(deltas),
            'all_paper_numeric_checks_pass':all(c['pass'] for c in checks),
            'paper_checks':len(checks),'estimator_rerun':False,'source_snapshot':'data/paper_reference'}
    write(out/'saved_verification.json',result)
    assert max(deltas)<1e-10 and result['all_paper_numeric_checks_pass'],result
    print(json.dumps(result,ensure_ascii=False),flush=True)

def verify(out):
    entries=dataset(); expected={r['segment_id']:r for r in read(DATA/'preprocessing_verification.json')['checks']}; checks=[]
    for e in entries:
        sid=e['segment_id']; X=features(e)
        checks.append({'segment_id':sid,'raw_hash_pass':sha(DATA/'raw'/e['file'])==e['sha256'],
                       'X_hash_pass':hashlib.sha256(X.tobytes()).hexdigest()==expected[sid]['X_sha256'],
                       'state_hash_pass':sha(DATA/'ema_states'/(sid+'.npz'))==expected[sid]['state_sha256']})
    sources=read(ROOT/'source_manifest.json')
    source_checks=[{'file':s['file'],'pass':sha(ROOT/s['file'])==s['sha256']} for s in sources if 'file' in s]
    report={'n_segments':205,'n_recordings':31,'checks':checks,'source_checks':source_checks,
            'pass':all(all(r[k] for k in ['raw_hash_pass','X_hash_pass','state_hash_pass']) for r in checks) and all(r['pass'] for r in source_checks)}
    write(out/'verify.json',report)
    assert report['pass']
    print('Verified 205 raw segments, restored EMA/FFT features and copied source hashes.',flush=True)

def estimate(out,names,limit):
    entries=dataset()
    if limit: entries=entries[:limit]
    contexts=read(DATA/'contexts.json'); rm=read(DATA/'result_map.json'); cfgs={c['id']:c for c in read(DATA/'config_matrix.json')['configs']}
    mods={}; baseline={}; rows=[]; reports=[]
    for name in names:
        assert name in rm,name
        baseline[name]=read(ROOT/rm[name]['file'])
        if name.startswith('ablation/'):
            mods[name]=module('hardware_q5_backend',Q5/'q5_backend.py')
        elif name=='this_method':
            mods[name]=module('hardware_reference_method',COMMON/'q6_methods_this_method.py')
        else:
            mods[name]=module('hardware_'+name,METHODS/(name+'.py'))
    # All processing below begins with released raw CIR and restored EMA states.
    for i,e in enumerate(entries,1):
        case=make_case(e,contexts); sid=e['segment_id']
        for name in names:
            begin=time.perf_counter(); m=mods[name]
            if name.startswith('ablation/'):
                cfg=cfgs[name.split('/')[1]]
                X,freqs,pairs,selection=m.apply_config(case,cfg,df_hz=31.25e6)
                image,diag=m.music_image_param(cfg=case['config'],X=X,selected_freqs=freqs,pairs=pairs,
                    tx_positions=case['tx_positions'],rx_positions=case['rx_positions'],grid_x=case['grid_x'],grid_y=case['grid_y'],k=case['k'],
                    is_forward_backward=cfg['is_forward_backward'])
                pp,_=peaks.nms_peaks(image,case['grid_x'],case['grid_y'],PROTOCOL.peaks,k=case['k'])
                prediction=[[p['x'],p['y']] for p in pp]; diagnostics={'selection':selection,'backend':diag}
            else:
                params=q6_search_profiles.params_for('grid5cm',name)
                result=m.estimate(case) if name=='this_method' else m.estimate(case,params=params)
                assert result['status']=='ok',(name,sid,result)
                prediction=[[p['x'],p['y']] for p in result['predictions']]
                diagnostics=result['diagnostics']
            elapsed=time.perf_counter()-begin
            T,ref=q5_metrics.extract_sets(baseline[name][sid],name+'/'+sid)
            # Assignment makes prediction order irrelevant for the comparison.
            A=np.asarray(prediction,float).reshape(-1,2); B=np.asarray(ref,float).reshape(-1,2)
            if A.shape==B.shape:
                d=np.linalg.norm(A[:,None,:]-B[None,:,:],axis=2); r,c=linear_sum_assignment(d)
                delta=float(d[r,c].max()) if len(r) else 0.
            else: delta=None
            match=delta is not None and delta<=1e-8
            rows.append({'method':name,'segment_id':sid,'prediction_count':len(A),'reference_count':len(B),'max_position_difference_m':delta,'pass':match,'elapsed_s':elapsed})
            reports.append({'method':name,'segment_id':sid,'predictions_xy_m':prediction,'metrics':metrics(T,prediction),'diagnostics':diagnostics})
            if not match:
                write(out/'replay_predictions.json',reports); table(out/'replay_comparison.csv',rows)
                raise AssertionError((name,sid,'prediction mismatch',delta))
        if i%5==0 or i==len(entries): print(f'hardware replay {i}/{len(entries)} x {len(names)}',flush=True)
    table(out/'replay_comparison.csv',rows); write(out/'replay_predictions.json',reports)
    write(out/'replay_verification.json',{'n_segments':len(entries),'n_methods_and_configs':len(names),'n_estimator_runs':len(rows),
          'all_205':len(entries)==205,'pass':all(r['pass'] for r in rows),'position_comparison_atol_m':1e-8,
          'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'names':names})

def figures(out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats=list(csv.DictReader((out/'summary.csv').open(encoding='utf-8-sig')))
    lookup={r['method']:r for r in stats}; names=list(LABELS)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,key,ylabel in zip(axes,['rmse_pos_m','dr_percent'],['Position RMSE (m)','Detection rate (%)']):
        ax.bar(range(6),[float(lookup[n][key]) for n in names],color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'])
        ax.set_xticks(range(6),[LABELS[n] for n in names],rotation=30,ha='right'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.3)
    for ext in ['pdf','png']: fig.savefig(out/('hardware_comparison.'+ext),dpi=200)
    plt.close(fig)
    abl=[r for r in stats if r['method'].startswith('ablation/')]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax.bar(range(len(abl)),[float(r['rmse_pos_m']) for r in abl],color='#1f77b4')
    ax.set_xticks(range(len(abl)),[r['method'].split('/')[1] for r in abl],rotation=60,ha='right'); ax.set_ylabel('Position RMSE (m)'); ax.grid(axis='y',alpha=.3)
    for ext in ['pdf','png']: fig.savefig(out/('hardware_ablation.'+ext),dpi=200)
    plt.close(fig)
    st=list(csv.DictReader((out/'stability.csv').open(encoding='utf-8-sig')))
    fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
    configs=sorted({r['config_id'] for r in st}); x=np.arange(len(configs))
    for env,shift,color in [('indoor',-.2,'#1f77b4'),('outdoor',.2,'#ff7f0e')]:
        lut={r['config_id']:r for r in st if r['environment']==env}
        ax.bar(x+shift,[float(lut[c]['sd_rmse_pos_m']) for c in configs],width=.4,label=env,color=color)
    ax.set_xticks(x,configs,rotation=40,ha='right'); ax.set_ylabel('SD of segment position RMSE (m)'); ax.legend(); ax.grid(axis='y',alpha=.3)
    for ext in ['pdf','png']: fig.savefig(out/('hardware_stability.'+ext),dpi=200)
    plt.close(fig)
    # A numerical hardware layout from true coordinates; no attempt to synthesize acquisition photos.
    fig,ax=plt.subplots(figsize=(4,4),layout='constrained'); pts=[]
    for e in dataset():
        with np.load(DATA/'raw'/e['file'],allow_pickle=False) as z: pts.extend(z['truth_xy_m'].tolist())
    pts=np.unique(np.asarray(pts),axis=0)
    ax.scatter(pts[:,0],pts[:,1],marker='x',label='Nominal target positions'); ax.scatter([0],[0],marker='s',color='black',label='Radar')
    ax.set(xlabel='x (m)',ylabel='y (m)'); ax.set_aspect('equal'); ax.grid(); ax.legend(fontsize=8)
    for ext in ['pdf','png']: fig.savefig(out/('hardware_positions.'+ext),dpi=200)
    plt.close(fig)
    write(out/'figure_manifest.json',{'scope':'hardware numerical review figures from recomputed saved predictions and released truth; not pixel-identical manuscript layouts',
          'figures':['hardware_comparison','hardware_ablation','hardware_stability','hardware_positions'],
          'formats':['pdf','png'],'simulation_code_executed':False})

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['verify','tables','figures','replay','all-saved'])
    ap.add_argument('--out',type=Path,default=ROOT/'outputs')
    ap.add_argument('--methods',default='this_method',help='comma-separated method IDs; all = 6 methods + 20 ablations')
    ap.add_argument('--limit',type=int)
    args=ap.parse_args(); args.out=args.out.resolve()
    # This supplement only writes inside itself, protecting originals and the publishing repository.
    assert args.out.is_relative_to(ROOT), 'Output must be inside hardware_reproduction'
    if args.limit is not None and not 1<=args.limit<=205: ap.error('--limit must be 1..205')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command in ['verify','all-saved']: verify(args.out)
    if args.command in ['tables','all-saved']: saved(args.out)
    if args.command in ['figures','all-saved']: figures(args.out)
    if args.command=='replay':
        rm=read(DATA/'result_map.json'); names=list(rm) if args.methods=='all' else args.methods.split(',')
        estimate(args.out,names,args.limit)

if __name__=='__main__': main()
