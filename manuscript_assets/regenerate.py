"""Regenerate non-simulation manuscript illustrations from their original plotting recipes."""
from pathlib import Path
import argparse, json, os, runpy, hashlib
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,default=ROOT/'generated')
args=ap.parse_args(); O=args.out.resolve()
if not O.is_relative_to(ROOT): ap.error('Output must stay inside manuscript_assets')
O.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(O/'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':8,
 'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'lines.linewidth':1.25,'lines.markersize':3.5,'pdf.fonttype':42,'ps.fonttype':42})
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#6B5B00']
marks=['o','s','^','D','v']
def save(fig,name):
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(O/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
def grid(ax): ax.grid(True,ls='--',lw=.45,alpha=.4)

# Physical and virtual arrays: all ordered channels and their duplicated location.
fig,axs=plt.subplots(1,2,figsize=(7.05,2.45),layout='constrained')
physical=[(-19,0,'TRX1'),(19,0,'TRX2'),(-19,-20,'RX1'),(0,-20,'RX2')]
for x,z,lab in physical:
    axs[0].scatter(x,z,s=110,marker='s',facecolors='#BBD8E9',edgecolors='#40576B');axs[0].annotate(lab,(x,z),xytext=(0,10),textcoords='offset points',ha='center')
axs[0].set(xlim=(-33,33),ylim=(-31,17),xlabel='x (mm)',ylabel='z (mm)',title='(a) Physical elements');grid(axs[0])
virtual={}
for tx,name in [(-19,'TRX1'),(19,'TRX2')]:
    for x,z,lab in physical: virtual.setdefault((x+tx,z),[]).append(name+' → '+lab)
for (x,z),labs in virtual.items():
    chosen=z==-20
    axs[1].scatter(x,z,s=70,marker='s',facecolors='#BBD8E9' if chosen else 'white',edgecolors='#40576B')
    txt='\n'.join(labs).replace('TRX','T').replace('RX','R')
    axs[1].annotate(txt,(x,z),xytext=(0,8 if z==0 else -11),textcoords='offset points',ha='center',va='bottom' if z==0 else 'top',fontsize=6.5)
axs[1].set(xlim=(-49,49),ylim=(-38,18),xlabel='Virtual x (mm)',ylabel='Virtual z (mm)',title='(b) Seven virtual positions');grid(axs[1])
save(fig,'antenna_layout')

# Scene photos and position map use a single, consistent panel layout.
fig=plt.figure(figsize=(7.05,2.7),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1.05,1])
for i,env in enumerate(['indoor','outdoor']):
    ax=fig.add_subplot(gs[i,0]);ax.imshow(plt.imread(ROOT/'photos'/(env+'_anonymized_crop.jpg')));ax.axis('off');ax.set_title(f'({chr(97+i)}) {env.capitalize()}')
ax=fig.add_subplot(gs[:,1]);positions=json.loads((ROOT/'data/nominal_positions.json').read_text()); coords=list(positions.values())
ids=list(positions)
for (x,y),lab in zip(coords,ids):
    ax.scatter(x,y,marker='x',s=30,color='#0072B2');ax.annotate(lab,(x,y),xytext=(4,-10),textcoords='offset points',fontsize=6.5)
ax.scatter(0,0,marker='s',c='black',s=23);ax.text(.12,0,'Radar',va='center',fontsize=7)
ax.set(xlim=(-1.65,1.9),ylim=(-3.3,.25),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='(c) Tested positions');grid(ax);save(fig,'hardware_setup')


os.environ['MANUSCRIPT_ASSET_OUT']=str(O)
runpy.run_path(str(ROOT/'source/restore_workflow.py'),run_name='__main__')
report={'simulation_executed':False,'estimator_executed':False,'regenerated':['antenna_layout','hardware_setup','algo_flowchart_restored'],'static':['CIR_Time_Domain_Signal_Acquisition_Procedure.pdf','indoor_anonymized_crop.jpg','outdoor_anonymized_crop.jpg'],'note':'Original plotting recipes; PDF dates and SVG identifiers can differ. Static originals are not synthesized.'}
(O/'regeneration_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
