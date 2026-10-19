"""Rebuild the manuscript's nine-panel correspondence figure from released spectra."""
from pathlib import Path
import argparse
import csv
import json
import os
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/q4')
    ap.add_argument('--spectra',type=Path,default=ROOT/'data/q4/spectra')
    ap.add_argument('--records',type=Path,default=ROOT/'data/q4/per_window.csv')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(a.out.resolve()/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    with a.records.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    fig,axs=plt.subplots(3,3,figsize=(7.05,5.5),layout='constrained'); selected=[]
    for i,cfg in enumerate(['G-420','C2-1','C3-1']):
        files=[a.spectra/('sim_'+cfg+'.npz')]
        for env in ['indoor','outdoor']:
            r=min([r for r in rows if r['config_id']==cfg and r['environment']==env],key=lambda r:r['window_id'])
            files.append(a.spectra/('uncorrected_'+r['window_id']+'.npz'));selected.append(r)
        for j,p in enumerate(files):
            with np.load(p,allow_pickle=False) as d:
                ax=axs[i,j];z=d['spectrum']/np.max(d['spectrum'])
                im=ax.pcolormesh(d['grid_x'],d['grid_y'],z,cmap='jet',vmin=0,vmax=1,shading='nearest',rasterized=True)
                ax.scatter(*d['truth'].T,marker='x',color='white',s=26,lw=1.1,label='True position',path_effects=[pe.Stroke(linewidth=2,foreground='black'),pe.Normal()])
                ax.scatter(*d['predictions'].T,facecolors='none',edgecolors='black',s=29,lw=1,label='Estimated peak')
                ax.set(xlim=(-2.2,2.2),ylim=(-4.2,-.55),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f'({chr(97+i*3+j)}) {cfg}: '+['FDTD','Indoor','Outdoor'][j])
                ax.grid(True,ls='--',lw=.45,alpha=.4)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.colorbar(im,ax=axs,shrink=.65,label='Normalized pseudo-spectrum')
    for ext in ['pdf','png']:fig.savefig(a.out/('spatial_spectra.'+ext),bbox_inches='tight',pad_inches=.04,dpi=170)
    (a.out/'fixed_examples.json').write_text(json.dumps(selected,indent=2))
    print('Rebuilt all nine panels from fixed identifier-order examples.')

if __name__=='__main__': main()
