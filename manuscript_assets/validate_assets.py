"""Check original asset hashes, approved photo metadata and regenerated rendering."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compare-rendered', action='store_true', help='Requires PyMuPDF and prior regenerate.py run')
    args = ap.parse_args()
    manifest = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Asset hash mismatch: '+item['file'])
    photo_checks = []
    for path in sorted((ROOT/'photos').glob('*.jpg')):
        with Image.open(path) as image:
            exif = image.getexif()
            nested = exif.get_ifd(34665) if 34665 in exif else {}
            if 34853 in exif or any(k in exif or k in nested for k in [42032, 42033, 42035, 37510]):
                raise RuntimeError('Unexpected GPS/owner/serial/comment metadata: '+path.name)
            photo_checks.append({'file':path.name,'size':image.size,'GPS_owner_serial_comment_present':False})
    renders = []
    if args.compare_rendered:
        import pymupdf
        for name in ['antenna_layout','hardware_setup','algo_flowchart_restored']:
            arrays = []
            for path in [ROOT/'figures'/(name+'.pdf'), ROOT/'generated'/(name+'.pdf')]:
                with pymupdf.open(path) as doc:
                    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    arrays.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n))
            same_shape = arrays[0].shape == arrays[1].shape
            delta = np.abs(arrays[0].astype(float)-arrays[1]) if same_shape else None
            renders.append({'file':name+'.pdf','same_shape':same_shape,
                            'pixel_equal':same_shape and bool(np.array_equal(*arrays)),
                            'mean_abs_channel_difference':float(delta.mean()) if same_shape else None})
    result = {'original_asset_hashes_pass':True,'original_assets_checked':len(manifest['files']),
              'photo_metadata_checks':photo_checks,'render_comparisons':renders,
              'note':'Rendering differences are reported, not silently treated as byte-identical reproduction.'}
    out = ROOT/'generated/asset_validation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
