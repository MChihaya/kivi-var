"""LPIPS/PSNR to FP16 on VAR good-class generations (isolates quantization damage,
no confound from VAR's inherent text/face weakness). Run in VAR venv (.venv).
Reads results/good_classes/{config}/{ci:04d}/{idx:03d}.png."""
import os, os.path as osp, sys, glob, json
import numpy as np
from PIL import Image
import torch, lpips

ROOT = '.'
base = osp.join(ROOT, 'results/good_classes')
REF = 'fp16'
CONFIGS = ['uni_int3', 'kivi_int3', 'uni_int2', 'kivi_int2']
fn = lpips.LPIPS(net='alex').cuda().eval()


def t(p):
    x = np.asarray(Image.open(p).convert('RGB'))
    return torch.from_numpy(x).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).cuda()


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m)


# enumerate reference images
refglob = sorted(glob.glob(osp.join(base, REF, '*', '*.png')))
keys = [osp.relpath(p, osp.join(base, REF)) for p in refglob]
print('ref images: %d (classes: %d)' % (len(keys), len(set(k.split(os.sep)[0] for k in keys))))
rows = []
for c in CONFIGS:
    lp, ps = [], []
    for k in keys:
        rp, cp = osp.join(base, REF, k), osp.join(base, c, k)
        if not osp.exists(cp):
            continue
        with torch.no_grad():
            lp.append(float(fn(t(rp), t(cp)).item()))
        ps.append(psnr(np.asarray(Image.open(rp).convert('RGB')), np.asarray(Image.open(cp).convert('RGB'))))
    a = np.array(lp)
    rows.append(dict(config=c, n=len(lp), lpips_mean=round(float(a.mean()), 4), lpips_std=round(float(a.std(ddof=1)), 4),
                     psnr=round(float(np.mean(ps)), 2)))
    print('%-10s n=%d LPIPS %.4f±%.4f  PSNR %.2f' % (c, len(lp), a.mean(), a.std(ddof=1), np.mean(ps)))
# paired: KIVI vs uniform per image at each bit
for bit in ['int3', 'int2']:
    u, kv = [], []
    for k in keys:
        up, kp = osp.join(base, 'uni_' + bit, k), osp.join(base, 'kivi_' + bit, k)
        rp = osp.join(base, REF, k)
        if not (osp.exists(up) and osp.exists(kp)):
            continue
        with torch.no_grad():
            u.append(float(fn(t(rp), t(up)).item()))
            kv.append(float(fn(t(rp), t(kp)).item()))
    u, kv = np.array(u), np.array(kv)
    wins = int((kv < u).sum())
    print('%s: KIVI closer to FP16 than uniform in %d/%d images (mean diff uni-kivi=%.4f)' % (bit, wins, len(u), float((u - kv).mean())))
json.dump(rows, open(osp.join(base, 'lpips.json'), 'w'), indent=2)
print('VAR_GOOD_LPIPS_DONE')
