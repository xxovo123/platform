"""Run every released non-spacing trial; compare outputs and recompute tables."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
import shutil
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/exec2'),str(ROOT/'src')]
os.environ['JSF_FROZEN_BUNDLE']=str(ROOT/'src')
from spacing_reproduce import read,write

def task(payload):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from jsfsim import runner
    run,rec,saved,out,index=payload
    kind='fdtd' if rec['layer']=='B' else 'analytic'
    path=(Path(read(index)['controlled' if kind=='fdtd' else 'analytic'][rec['clean_id']]) if index
          else ROOT/'data/cache'/kind/rec['clean_id']/'clean_cir.npz')
    with np.load(path,allow_pickle=False) as z:raw=z['raw_cir']
    source=read(ROOT/'data/results/runs'/run/'source_manifest.json')
    rows=[];checks=[]
    for ref in saved:
        actual=runner._trial_worker(dict(rec=rec,raw=raw,trial=ref['trial'],run_id=run,
            layer=rec['layer'],truth_sha=ref['truth_sha256'],protocol_hash=source['protocol_hash'],save_sample=False))
        aa=np.asarray([p[:2] for p in actual.get('_predictions',[])],float).reshape(-1,2)
        bb=np.asarray([p[:2] for p in ref['_predictions']],float).reshape(-1,2)
        delta=0.0
        if aa.shape!=bb.shape:delta=float('inf')
        elif len(aa):
            d=np.linalg.norm(aa[:,None,:]-bb[None,:,:],axis=2);ii,jj=linear_sum_assignment(d);delta=float(d[ii,jj].max())
        ok=delta<=1e-10 and all(actual.get(k)==ref[k] for k in ['status','seed','pairs','tp','fn','fp','missing','spurious','resolution_success'])
        ok=ok and abs(actual.get('sse_m2',float('inf'))-ref['sse_m2'])<=1e-9
        checks.append({'task':rec['task_id'],'trial':ref['trial'],'passed':bool(ok),'max_xy_difference_m':delta})
        rows.append(actual)
    write(Path(out)/'tasks'/(rec['task_id']+'.json'),{'rows':rows,'checks':checks})
    return {'task':rec['task_id'],'trials':len(rows),'passed':all(c['passed'] for c in checks),'rows':rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--generated-index',type=Path)
    ap.add_argument('--smoke',action='store_true',help='One trial per task; explicitly not a full validation')
    a=ap.parse_args(); assert a.workers>0
    summaries=[]
    for run in sorted((ROOT/'data/results/runs').iterdir()):
        tasks=[json.loads(x) for x in (run/'tasks.jsonl').read_text().splitlines()]
        by={r['task_id']:[] for r in tasks}
        for line in (run/'rows.jsonl').read_text().splitlines():
            r=json.loads(line);by[r['task']].append(r)
        for rec in tasks:
            assert len(by[rec['task_id']])==rec['trials']
            assert len({r['trial'] for r in by[rec['task_id']]})==rec['trials']
        out=a.out/'runs'/run.name;out.mkdir(parents=True,exist_ok=True)
        for name in ['tasks.jsonl','config.json','source_manifest.json']:shutil.copy2(run/name,out/name)
        payloads=[(run.name,r,by[r['task_id']][:1] if a.smoke else by[r['task_id']],str(out),
                   str(a.generated_index) if a.generated_index else None) for r in tasks]
        with ProcessPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(task,payloads))
        with (out/'rows.jsonl').open('w',encoding='utf-8') as f:
            for r in results:
                for row in r.pop('rows'):f.write(json.dumps(row)+'\n')
        summaries.extend(results)
        print(json.dumps({'run':run.name,'completed_trials':sum(r['trials'] for r in results),'passed':all(r['passed'] for r in results)}),flush=True)
    report={'passed':all(r['passed'] for r in summaries),'full_scope':not a.smoke,
            'trials':sum(r['trials'] for r in summaries),'tasks':len(summaries),
            'input_mode':'newly generated CIR' if a.generated_index else 'released clean CIR','checks':summaries}
    if not a.smoke:assert report['trials']==16500 and report['tasks']==150
    write(a.out/'verification.json',report)
    if not report['passed']:raise RuntimeError('Controlled replay mismatch; retain failed trials')
    if not a.smoke:
        import reproduce
        reproduce.tables(a.out/'tables',a.out/'runs')

if __name__=='__main__':main()
