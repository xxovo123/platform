"""Compare newly produced tables and figure inputs with fixed pre-run references."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacing_reproduce import read,write

def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))

def compare_csv(actual,reference,atol=1e-8):
    aa=rows(actual);bb=rows(reference)
    assert len(aa)==len(bb) and len(aa)>0,(actual,'row count')
    assert aa[0].keys()==bb[0].keys(),(actual,'columns')
    cells=0;maxdelta=0.0
    for i,(a,b) in enumerate(zip(aa,bb)):
        for key,value in a.items():
            if value==b[key]:cells+=1;continue
            try: x,y=float(value),float(b[key])
            except ValueError:raise AssertionError((actual,i,key,value,b[key]))
            delta=abs(x-y)
            tolerance=0 if key in ['n_segments','n_truth','pairs','missing','tp','fn','fp','K','rank_zero_based','n_selected','source_n'] else atol
            assert math.isfinite(delta) and delta<=tolerance,(actual,i,key,value,b[key])
            maxdelta=max(maxdelta,delta);cells+=1
    return {'file':actual.name,'rows':len(aa),'cells':cells,'max_numeric_delta':maxdelta,'passed':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--mode',choices=['saved','replay'],required=True)
    ap.add_argument('--hardware-out',type=Path,help='Optional explicit hardware table directory for component checks')
    a=ap.parse_args();out=a.run.resolve()
    hardware=a.hardware_out or ROOT/'hardware_reproduction/generated'/out.name/('new_tables' if a.mode=='replay' else 'saved')
    checks=[]
    for name in ['summary.csv','groups.csv','stability.csv','selected_measurements.csv']:
        checks.append(compare_csv(hardware/name,ROOT/'validation/reference/hardware'/name))
    figure_dir=out/('new_figures' if a.mode=='replay' else 'figures')
    actual={(r['direction'],float(r['spacing_m'])):r for r in rows(figure_dir/'spacing_primary.csv')}
    reference=rows(ROOT/'validation/reference/fig4_counts.csv')
    assert len(actual)==len(reference)==24
    for r in reference:
        x=actual[r['direction'],float(r['spacing_m'])]
        for key in ['successes','trials','rate']:assert float(x[key])==float(r[key]),(r,x)
    if a.mode=='replay':
        q4=read(out/'q4/verification.json');assert q4['passed'] and q4['parametric_spectrum_comparisons']==4100
        assert read(out/'control/verification.json')['trials']==16500
        assert read(out/'spacing/replay_verification.json')['trials']==2400
    report={'passed':True,'hardware_checks':checks,'fig4_points':24,'mode':a.mode,
            'table_IX_checked_by':'validation/q4_replay_all.py in replay mode',
            'full_estimator_replay':a.mode=='replay','visual_review_required':True}
    write(out/'numeric_comparison.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
