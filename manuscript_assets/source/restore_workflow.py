"""Retain the original four-panel artwork; align notation and stacking with the paper.

This edits manuscript illustration only. It does not run an estimator or a simulation.
The earlier source and its generated figures remain unchanged.
"""
from pathlib import Path
import hashlib,json,os

P=Path(__file__).resolve().parent
original=P/'draw_algorithm_flowchart.py'
s=original.read_text(encoding='utf-8')
for a,b in [
    ('N_{rx}','C'),('N_{taps}','N'),('N_{freq}','F'),
    (r'\mathbf{x}_l',r'\mathbf{z}_\ell'),(r'\mathbf{X}',r'\mathbf{Z}'),
    ('$H_l$',r'$H_\ell$'),('K_{\\mathrm{alg}}=N_{\\mathrm{true}}','K'),
    ('(r,\\theta)','(\\mathbf{p})'),
    ('"Stage 1: Data Acquisition & FFT"','"Stage 1: CIR & FFT"'),
    ('"Stage 2: Channel-Frequency Stacking"','"Stage 2: Snapshot Stacking"'),
    ('"Stage 3: Covariance & Forward-Backward Averaging"','"Stage 3: Covariance Processing"'),
    ('"Stage 4: Subspace Processing & Pseudo-Spectrum"','"Stage 4: Subspace & Joint Spectrum"'),
    ('"Flatten"','"Stack rows"'),
    ('"$r$", size=6.0','"$x$", size=6.0'),
    ('"$\\theta$", size=6.0','"$y$", size=6.0'),
]:
    assert a in s,a
    s=s.replace(a,b)
old='''    for k in range(4):
        rect(sx0, sy0 + (3 - k) * sh / 4.0, sw, sh / 4.0, CL[k], ec="none",
             lw=0.0, z=3)'''
new='''    # Four channel colors repeat for each frequency, top to bottom.
    for k in range(16):
        rect(sx0, sy0 + (15 - k) * sh / 16.0, sw, sh / 16.0, CL[k % 4],
             ec=INK, lw=0.25, z=3)'''
assert old in s
s=s.replace(old,new)
s=s.replace('band_rows(12, CL)','[CL[(11-r) % 4] for r in range(12)]')
s=s.replace('10^{-5}',r'\epsilon')
s=s.replace('"Stack rows", size=6.8','"Stack", size=6.2')
s=s.replace('"Forward-Backward", size=6.8','"Forward-Backward", size=5.8')
s=s.replace('"unified", size=6.0','"Peak", size=5.6')
s=s.replace('"peak picking", size=6.0','"picking", size=5.6')
s=s.replace('size=6.5)\n\n    block_arrow(216.0','size=5.8)\n\n    block_arrow(216.0')
s=s.replace('size=6.2)\n\n    block_arrow(320.0','size=5.7)\n\n    block_arrow(320.0')
# Leave visible clearance at process-box and matrix borders, including the EVD input.
for a,b in [
    ('block_arrow(382.0, 418.0','block_arrow(388.0, 418.0'),
    ('block_arrow(468.0, 492.0','block_arrow(474.0, 492.0'),
    ('block_arrow(78.0, 126.0','block_arrow(84.0, 126.0'),
    ('block_arrow(216.0, 246.0','block_arrow(222.0, 244.0'),
    ('block_arrow(320.0, 390.0','block_arrow(320.0, 378.0'),
    ('block_arrow(454.0, 482.0','block_arrow(456.0, 478.0'),
    ('block_arrow(454.0, 500.0','block_arrow(456.0, 496.0'),
    ('block_arrow(608.0, 640.0','block_arrow(610.0, 640.0'),
    ('(514.0, 45.0)','(514.0, 42.0)'),
    ('"signal subspace", size=5.2','"signal", size=5.2'),
    ('"noise subspace", size=5.2','"noise", size=5.2'),
]:
    assert a in s,a
    s=s.replace(a,b)
# Color contours are schematic, with the same blue-to-red progression as jet maps.
s=s.replace('"#F6D9B8"','"#00CFFF"').replace('"#EFA45E"','"#FFDB00"').replace('"#CE5A17"','"#D00000"')
ns={'__name__':'restored_manuscript_artwork','__file__':str(original)}
exec(compile(s,str(original),'exec'),ns)
ns['mpl'].rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'dejavuserif'})
original_arrow=ns['block_arrow']
def spaced_arrow(x0,x1,y,w=3.2,hw=8.6,hl=8.6,color=ns['INK'],z=6):
    # A lighter, consistent connector keeps its head legible in narrow gaps.
    return original_arrow(x0,x1,y,w=1.6,hw=5.8,hl=5.8,color=color,z=z)
ns['block_arrow']=spaced_arrow
for stage in ['stage1','stage2','stage3','stage4']:ns[stage]()
out=Path(os.environ['MANUSCRIPT_ASSET_OUT'])
out.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','svg','png']:
    ns['fig'].savefig(out/f'algo_flowchart_restored.{ext}',dpi=300)
ns['plt'].close(ns['fig'])
manifest={
    'source':str(original),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'changes':['retain original four-stage tensor/matrix composition',
               'align C,N,F,L,Z,z,K,p notation',
               'repeat channel colors inside each frequency block',
               'shorten connectors at box borders and use consistent lighter arrowheads',
               'joint Cartesian position spectrum; range/azimuth output'],
    'retained_original_figure1':'CIR_Time_Domain_Signal_Acquisition_Procedure.pdf',
    'figure1_sha256':hashlib.sha256((P.parent/'figures/CIR_Time_Domain_Signal_Acquisition_Procedure.pdf').read_bytes()).hexdigest(),
    'estimator_runs':0,'solver_runs':0,
}
(out/'restored_figure_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Restored workflow saved; original figure assets unchanged.')
