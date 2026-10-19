"""Validate all scene geometries/trajectories without importing Meep or solving fields."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
import regenerate_inputs as R
from spacing_reproduce import read,write

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    import numpy as np
    from fdtd import api
    analytic,controlled=R.inventory();records=[]
    for cid,item in sorted(controlled.items()):
        scene=R.D.build_fdtd_scene(item['rec'],resolution_px_per_m=350,run_time_meep=20,background_mode='empty_field',snapshots=128)
        prep=api._prepare(None,scene,max_states=None,background_mode='empty_field')
        truth=read(Path(item['reference']).parent/'truth.json')
        assert prep['n_snapshots']==prep['n_total']==128 and prep['burnin']==0
        assert np.allclose(prep['traj'],truth['targets_xy_m'],atol=1e-14,rtol=0),cid
        records.append({'kind':'controlled','id':cid,'states':128,'geometry_ok':True,'trajectory_matches':True})
    for cfg in sorted(read(ROOT/'data/q4/generator_settings.json')):
        truth=R.P.Q4_TARGETS[cfg];cid=R.P.clean_id_for('fdtd',len(truth),truth)
        rec={'clean_id':cid,'k':len(truth),'targets_xy':truth,'clean_seed':R.P.clean_seed(cid)}
        geom={'tx_positions_m':R.P.Q4_TX_POSITIONS_M,'rx_positions_m':R.P.Q4_RX_POSITIONS_M,'pairs':R.P.Q4_PAIRS}
        scene=R.D.build_fdtd_scene(rec,resolution_px_per_m=350,run_time_meep=20,background_mode='ema_windowed',snapshots=128,burnin_states=32,geometry=geom)
        prep=api._prepare(None,scene,max_states=None,background_mode='ema_windowed')
        assert prep['n_snapshots']==128 and prep['n_total']==160 and prep['burnin']==32
        records.append({'kind':'q4','id':cfg,'states':160,'geometry_ok':True})
    assert len(records)==44 and len(analytic)==5
    report={'passed':True,'scenes':44,'field_solves':0,'checks':records}
    write(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
