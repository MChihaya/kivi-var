import json, os.path as osp, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='./results/failure2/perclass')
ap.add_argument('--configs', default='fp16,int2,early_b3,late_b3')
a = ap.parse_args()

hdr = ['config', 'txt_chars', 'txt_conf', 'txt_fid', 'face_det', 'face_fid', 'fine_fid']
print('  '.join(f'{h:>9}' for h in hdr))
for c in a.configs.split(','):
    p = osp.join(a.dir, f'table_{c}.json')
    if not osp.exists(p):
        print(f'{c:>9}  (missing)'); continue
    rows = json.load(open(p))['rows']
    txt = [r for r in rows if r['group'] == 'text']
    chars = np.mean([r['ocr_gen']['chars'] for r in txt])
    conf = np.mean([r['ocr_gen']['conf'] for r in txt])
    tfid = np.mean([r['fid_gen_vs_real'] for r in txt])
    sk = [r for r in rows if r['idx'] == 796][0]
    fine = [r for r in rows if r['group'] == 'fine_structure']
    ffid = np.mean([r['fid_gen_vs_real'] for r in fine])
    vals = [c, f'{chars:.2f}', f'{conf:.3f}', f'{tfid:.1f}',
            f'{sk["face_gen"]["det_rate"]:.3f}', f'{sk["fid_gen_vs_real"]:.1f}', f'{ffid:.1f}']
    print('  '.join(f'{v:>9}' for v in vals))
