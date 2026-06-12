"""High-frequency content of real vs VAR-generated images, per ImageNet class.
Tests whether "VAR fails on high-frequency classes": measures real high-freq energy
per class, compares with VAR's generated high-freq energy (is VAR blurring?), and
correlates real high-freq with the per-class FID (does high-freq predict the failure?).
Run in VAR venv (.venv). CPU only."""
import os, os.path as osp, glob, json
import numpy as np
from PIL import Image

ROOT = '.'
VAL = os.environ['IMAGENET_VAL']
GEN_FAIL = osp.join(ROOT, 'results/failure1/gen')
GEN_GOOD = osp.join(ROOT, 'results/good_classes/fp16')
CUTOFF = 0.25; N_REAL = 50; SIZE = 256

WNIDS = sorted([d for d in os.listdir(VAL) if d.startswith('n') and osp.isdir(osp.join(VAL, d))])
fc = json.load(open(osp.join(ROOT, 'analysis/failure_classes.json')))
classes = []
for grp in ['text', 'fine_structure', 'animal_detail', 'human_face', 'reference_easy']:
    for k, v in fc.get(grp, {}).items():
        classes.append((int(k), v, grp))
GOOD = {207: 'golden retriever', 291: 'lion', 340: 'zebra', 388: 'giant panda',
        779: 'school bus', 817: 'sports car', 963: 'pizza', 985: 'daisy'}
for k, v in GOOD.items():
    classes.append((k, v, 'good_object'))
# per-class FID (gen vs real train), from report table 5
PCFID = {489: 116.7, 488: 88.0, 595: 42.4, 921: 107.8, 917: 88.2, 919: 67.7, 916: 39.0,
         281: 57.8, 259: 39.8, 562: 90.0, 437: 41.1, 980: 40.6}

w1 = np.hanning(SIZE); WIN = np.outer(w1, w1)
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
R = np.sqrt((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) / (SIZE / 2)
HIMASK = R > CUTOFF


def hf_ratio(path):
    try:
        a = np.asarray(Image.open(path).convert('L').resize((SIZE, SIZE)), dtype=np.float64)
    except Exception:
        return None
    a = (a - a.mean()) * WIN
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    P[SIZE // 2, SIZE // 2] = 0.0
    t = P.sum()
    return float(P[HIMASK].sum() / t) if t > 0 else None


def mean_hf(paths):
    v = [x for x in (hf_ratio(p) for p in paths) if x is not None]
    return (float(np.mean(v)), len(v)) if v else (float('nan'), 0)


rows = []
for idx, name, grp in classes:
    wnid = WNIDS[idx]
    real = sorted(glob.glob(osp.join(VAL, wnid, '*.JPEG')))[:N_REAL]
    gd = osp.join(GEN_FAIL, '%04d' % idx)
    if not osp.isdir(gd):
        gd = osp.join(GEN_GOOD, '%04d' % idx)
    gen = sorted(glob.glob(osp.join(gd, '*.png')))
    rhf, rn = mean_hf(real); ghf, gn = mean_hf(gen)
    rows.append(dict(idx=idx, name=name, group=grp, real_hf=round(rhf, 4), gen_hf=round(ghf, 4),
                     deficit=round(rhf - ghf, 4), fid=PCFID.get(idx), n_real=rn, n_gen=gn))
    print('%-17s %-15s real=%.4f gen=%.4f deficit=%+.4f fid=%s' %
          (name[:17], grp, rhf, ghf, rhf - ghf, PCFID.get(idx)), flush=True)

print('\n=== group means (real high-freq, generated high-freq, deficit) ===')
gm = {}
for g in ['text', 'fine_structure', 'animal_detail', 'human_face', 'reference_easy', 'good_object']:
    rs = [r for r in rows if r['group'] == g]
    rh = np.mean([r['real_hf'] for r in rs]); gh = np.mean([r['gen_hf'] for r in rs])
    gm[g] = dict(real_hf=round(float(rh), 4), gen_hf=round(float(gh), 4), deficit=round(float(rh - gh), 4), n=len(rs))
    print('%-16s real=%.4f gen=%.4f deficit=%+.4f (n=%d)' % (g, rh, gh, rh - gh, len(rs)))

fid_rows = [r for r in rows if r['fid'] is not None]
rh = np.array([r['real_hf'] for r in fid_rows]); fid = np.array([r['fid'] for r in fid_rows])
allrh = np.array([r['real_hf'] for r in rows]); allgh = np.array([r['gen_hf'] for r in rows])
print('\ncorr(real_hf, per-class FID) over %d classes = %.3f' % (len(fid_rows), np.corrcoef(rh, fid)[0, 1]))
print('corr(real_hf, gen_hf) over all = %.3f  (VAR matches the real spectrum)' % np.corrcoef(allrh, allgh)[0, 1])
print('mean deficit (real-gen) over all = %+.4f' % float(np.mean(allrh - allgh)))
json.dump(dict(cutoff=CUTOFF, rows=rows, group_means=gm), open(osp.join(ROOT, 'results/hf_analysis.json'), 'w'), indent=2)
print('SAVED results/hf_analysis.json'); print('HF_DONE')
