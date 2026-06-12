"""Compare Infinity quant configs against FP16 reference (run in VAR venv).
LPIPS + PSNR to fp16, and easyocr text legibility on the text-prompt image.
Usage: python inf_metrics.py results/infinity"""
import os, os.path as osp, sys, json, glob
import numpy as np
from PIL import Image

ROOT = '.'
base = sys.argv[1] if len(sys.argv) > 1 else osp.join(ROOT, 'results/infinity')
REF = 'fp16'
CONFIGS = ['uni_int3', 'kivi_int3', 'uni_int2', 'kivi_int2']
TEXT_KEY = 'explore_more'   # prompt containing rendered text


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


# LPIPS (optional)
lpips_fn = None
try:
    import torch, lpips
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    def lpips_d(a, b):
        def t(x):
            x = torch.from_numpy(x).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).cuda()
            return x
        with torch.no_grad():
            return float(lpips_fn(t(a), t(b)).item())
except Exception as e:
    print('lpips unavailable:', e)

    def lpips_d(a, b):
        return float('nan')

# easyocr (optional)
ocr = None
try:
    import easyocr
    ocr = easyocr.Reader(['en'], gpu=True)
except Exception as e:
    print('easyocr unavailable:', e)


def text_chars(p):
    if ocr is None or not osp.exists(p):
        return -1, -1.0
    res = ocr.readtext(p)
    chars = sum(len(t) for _, t, _ in res)
    conf = float(np.mean([c for _, _, c in res])) if res else 0.0
    return chars, round(conf, 3)


refdir = osp.join(base, REF)
keys = [osp.splitext(osp.basename(p))[0] for p in sorted(glob.glob(osp.join(refdir, '*.jpg')))]
print('reference images:', keys)
rows = []
for cfg in CONFIGS:
    cdir = osp.join(base, cfg)
    if not osp.isdir(cdir):
        print('skip (missing):', cfg); continue
    lp, ps = [], []
    for k in keys:
        rp, cp = osp.join(refdir, k + '.jpg'), osp.join(cdir, k + '.jpg')
        if not (osp.exists(rp) and osp.exists(cp)):
            continue
        a, b = load(rp), load(cp)
        lp.append(lpips_d(a, b)); ps.append(psnr(a, b))
    tc, tconf = text_chars(osp.join(cdir, TEXT_KEY + '.jpg'))
    rows.append(dict(config=cfg, n=len(lp),
                     lpips_to_fp16=round(float(np.nanmean(lp)), 4),
                     psnr_to_fp16=round(float(np.mean(ps)), 2),
                     text_chars=tc, text_conf=tconf))
# fp16 self text
rtc, rtconf = text_chars(osp.join(refdir, TEXT_KEY + '.jpg'))
print('\n=== INFINITY QUANT METRICS (vs FP16) ===')
print('fp16 text_chars=%d conf=%.3f' % (rtc, rtconf))
hdr = '%-12s %4s %14s %13s %10s %9s' % ('config', 'n', 'lpips_to_fp16', 'psnr_to_fp16', 'txt_chars', 'txt_conf')
print(hdr)
for r in rows:
    print('%-12s %4d %14.4f %13.2f %10d %9.3f' % (
        r['config'], r['n'], r['lpips_to_fp16'], r['psnr_to_fp16'], r['text_chars'], r['text_conf']))
with open(osp.join(base, 'metrics.json'), 'w') as f:
    json.dump(dict(ref_text=dict(chars=rtc, conf=rtconf), rows=rows), f, indent=2)
print('\nSAVED', osp.join(base, 'metrics.json'))
print('INF_METRICS_DONE')
