"""Linux validation driver: distinct saved-data and full estimator-replay modes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacing_reproduce import read,write

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['saved','replay'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--generated-index',type=Path,help='After full fresh field generation; missing inputs fail, never fall back')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();assert platform.system()=='Linux','Run the release validation on Linux'
    assert a.workers>0
    out=a.out.resolve();assert out.is_relative_to(ROOT/'generated_validation'),'Output must be under generated_validation/'
    digest=hashlib.sha256((ROOT/'data/release_manifest.json').read_bytes()).hexdigest()
    identity={'mode':a.mode,'release_manifest_sha256':digest,
              'generated_index':str(a.generated_index.resolve()) if a.generated_index else None,
              'generated_index_sha256':hashlib.sha256(a.generated_index.read_bytes()).hexdigest() if a.generated_index else None}
    if out.exists() and any(out.iterdir()):
        assert a.resume and (out/'identity.json').exists() and read(out/'identity.json')==identity,'Use a fresh output or resume the same package'
    out.mkdir(parents=True,exist_ok=True);write(out/'identity.json',identity)
    write(out/'verification.json',{'passed':False,'status':'running','identity':identity})
    if a.generated_index:
        field=read(a.generated_index.parent/'verification.json')
        assert field['passed'] and field['full_scope'] and field['source_manifest_sha256']==digest
    import numpy,scipy,matplotlib
    environment={'system':platform.platform(),'python':sys.version,'executable':sys.executable,
                 'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
                 'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    write(out/'environment.json',environment)
    hw=ROOT/'hardware_reproduction/generated'/out.name
    art=ROOT/'manuscript_assets/generated'/out.name
    commands=[]
    def add(name,*args):commands.append((name,[sys.executable,'-B',*map(str,args)]))
    add('release-integrity','validation/check_release.py')
    add('simulation-integrity','reproduce.py','verify')
    add('scene-preflight','validation/preflight_inputs.py','--out',out/'scene_preflight.json')
    add('spacing-integrity','spacing_reproduce.py','verify','--out',out/'spacing_saved')
    add('hardware-preprocessing','hardware_reproduction/reproduce_hardware.py','verify','--out',hw/'saved')
    add('hardware-saved','hardware_reproduction/reproduce_hardware.py','all-saved','--out',hw/'saved')
    add('hardware-archive-check','hardware_reproduction/validate_release.py')
    add('correspondence-saved','q4_reproduce.py','verify-spectra')
    add('saved-tables','reproduce.py','tables','--out',out/'tables')
    add('saved-figures','reproduce.py','figures','--out',out/'figures')
    add('saved-figure8','q4_figures.py','--out',out/'figures')
    add('artwork','manuscript_assets/regenerate.py','--out',art)
    add('artwork-integrity','manuscript_assets/validate_assets.py')
    if a.mode=='replay':
        gen=['--generated-index',a.generated_index.resolve()] if a.generated_index else []
        add('control-replay','validation/control_replay.py','--out',out/'control','--workers',a.workers,*gen)
        add('spacing-replay','spacing_reproduce.py','replay-all','--out',out/'spacing','--workers',a.workers,*gen)
        add('hardware-replay','hardware_reproduction/reproduce_hardware.py','replay','--methods','all','--out',hw/'replay')
        add('hardware-new-tables','hardware_reproduction/reproduce_hardware.py','tables','--replay-input',hw/'replay/replay_predictions.json','--out',hw/'new_tables')
        add('hardware-spectra','q4_hardware_replay.py','--out',out/'hardware_spectra')
        add('correspondence-replay','validation/q4_replay_all.py','--out',out/'q4','--hardware-spectra',out/'hardware_spectra/spectra',*gen)
        add('new-figures','reproduce.py','figures','--out',out/'new_figures','--tables',out/'control/tables','--spacing-results',out/'spacing/replay_verification.json')
        add('new-figure8','q4_figures.py','--out',out/'new_figures','--spectra',out/'q4/spectra','--records',out/'q4/per_window.csv')
    add('numeric-comparison','validation/compare_results.py','--run',out,'--mode',a.mode)
    add('figure-comparison','validation/compare_figures.py','--run',out,'--mode',a.mode)
    log=out/'commands.json';history=read(log) if log.exists() and a.resume else []
    prior={r['name']:r for r in history if r['returncode']==0}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name,command in commands:
        if name not in ['release-integrity','numeric-comparison','figure-comparison'] and name in prior and prior[name]['argv']==command:
            print('RESUME completed',name,flush=True);continue
        print('START',name,flush=True);start=time.time();path=out/(name+'.log')
        with path.open('a',encoding='utf-8') as f:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record={'name':name,'argv':command,'returncode':proc.returncode,'elapsed_s':time.time()-start,
                'started_epoch':start,'log':str(path)}
        history.append(record);write(log,history)
        print('DONE',name,'returncode',proc.returncode,'seconds',round(record['elapsed_s'],1),flush=True)
        if proc.returncode:
            write(out/'verification.json',{'passed':False,'failed_command':name,'identity':identity})
            raise SystemExit(proc.returncode)
    checks={}
    if a.mode=='replay':
        checks['control']=read(out/'control/verification.json')
        checks['spacing']=read(out/'spacing/replay_verification.json')
        checks['hardware']=read(hw/'replay/replay_verification.json')
        checks['q4']=read(out/'q4/verification.json')
        assert checks['control']['passed'] and checks['control']['full_scope'] and checks['control']['trials']==16500
        assert checks['spacing']['passed'] and checks['spacing']['full_scope'] and checks['spacing']['trials']==2400
        assert checks['hardware']['pass'] and checks['hardware']['n_estimator_runs']==5330
        assert checks['q4']['passed'] and checks['q4']['parametric_trials']==300
        assert read(hw/'new_tables/saved_verification.json')['estimator_rerun']
    figure_root=out/('new_figures' if a.mode=='replay' else 'figures')
    for stem in ['simulation_snr','simulation_separation','simulation_snapshots','spatial_spectra']:
        assert (figure_root/(stem+'.pdf')).is_file() and (figure_root/(stem+'.png')).is_file()
    report={'passed':True,'mode':a.mode,'linux':True,'full_estimator_replay':a.mode=='replay',
            'fresh_field_inputs':bool(a.generated_index),'release_manifest_sha256':digest,
            'figures':str(figure_root),'artwork':str(art),'hardware':str(hw),
            'visual_review':'pending: compare the 8 manuscript figures with validation/reference/figures; record checklist',
            'field_regeneration':'check separate verification.json and generated_index.json; cache replay alone does not establish field regeneration',
            'commands_completed':len(commands),'environment':environment}
    if a.generated_index:
        field=read(a.generated_index.parent/'verification.json')
        assert field['passed'] and field['full_scope'] and field['source_manifest_sha256']==digest
        report['field_verification']=field
    write(out/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
