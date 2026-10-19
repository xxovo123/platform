"""Render all eight reference/generated figure pairs for mandatory human inspection."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacing_reproduce import write

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--mode',choices=['saved','replay'],required=True)
    a=ap.parse_args();run=a.run.resolve()
    import pymupdf
    from PIL import Image,ImageDraw
    output=run/'visual_comparison';output.mkdir(parents=True,exist_ok=True)
    computed=run/('new_figures' if a.mode=='replay' else 'figures')
    art=ROOT/'manuscript_assets/generated'/run.name
    order=['CIR_Time_Domain_Signal_Acquisition_Procedure','algo_flowchart_restored','simulation_snr',
           'simulation_separation','simulation_snapshots','antenna_layout','hardware_setup','spatial_spectra']
    reports=[]
    for number,name in enumerate(order,1):
        reference=ROOT/'validation/reference/figures'/(name+'.pdf')
        actual=(ROOT/'manuscript_assets/figures' if number==1 else art if number in [2,6,7] else computed)/(name+'.pdf')
        images=[]
        for path in [reference,actual]:
            with pymupdf.open(path) as doc:
                assert len(doc)==1
                pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(1000/doc[0].rect.width,1000/doc[0].rect.width),alpha=False)
                images.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
        h=max(im.height for im in images)+45
        canvas=Image.new('RGB',(2020,h),'white');d=ImageDraw.Draw(canvas)
        for x,im,label in [(0,images[0],'Approved reference'),(1020,images[1],'Regenerated')]:
            canvas.paste(im,(x,40));d.text((x+10,10),f'Fig. {number}: {label}',fill='black')
        filename=f'fig{number:02d}_{name}.png';canvas.save(output/filename)
        reports.append({'figure':number,'reference':str(reference),'generated':str(actual),'side_by_side':filename,
            'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'generated_sha256':hashlib.sha256(actual.read_bytes()).hexdigest(),'human_review':'pending'})
    write(output/'manifest.json',{'pairs':reports,'status':'rendered; human inspection pending',
        'checklist':['all curves and points present','axis units and ranges','marker shapes and legend',
                     'all panels and fixed examples','no clipping/overlap','schematics/photographs match sources']})
    print(json.dumps({'figure_pairs_rendered':8,'output':str(output),'human_review':'pending'}))

if __name__=='__main__':main()
