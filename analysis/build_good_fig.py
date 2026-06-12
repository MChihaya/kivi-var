"""Good-class quant comparison figure (rows=classes, cols=configs) with baked labels."""
import os.path as osp
from PIL import Image, ImageDraw, ImageFont
ROOT = '.'
BASE = osp.join(ROOT, 'results/good_classes')
OUT = osp.join(ROOT, 'figures_out/fig5_goodclass_quant.png')
COLS = [('fp16', 'FP16 (baseline)'), ('uni_int3', 'Uniform INT3'), ('kivi_int3', 'KIVI INT3 (ours)'),
        ('uni_int2', 'Uniform INT2'), ('kivi_int2', 'KIVI INT2 (ours)')]
ROWS = [('0207', 'golden retriever'), ('0388', 'giant panda'), ('0980', 'volcano'),
        ('0817', 'sports car'), ('0963', 'pizza')]
IDX = '000'
CELL = 300; PADL = 255; PADT = 54; GAP = 5


def font(sz):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if osp.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell(cfg, ci):
    p = osp.join(BASE, cfg, ci, IDX + '.png')
    if osp.exists(p):
        return Image.open(p).convert('RGB').resize((CELL, CELL))
    im = Image.new('RGB', (CELL, CELL), (40, 40, 40)); ImageDraw.Draw(im).text((CELL//2-20, CELL//2), 'N/A', fill=(220,220,220), font=font(20)); return im


W = PADL + len(COLS)*(CELL+GAP); H = PADT + len(ROWS)*(CELL+GAP)
cv = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(cv)
fh, fr = font(31), font(26)
for cj, (cfg, lab) in enumerate(COLS):
    d.text((PADL+cj*(CELL+GAP)+8, 10), lab, fill='black', font=fh)
for ri, (ci, rlab) in enumerate(ROWS):
    y = PADT+ri*(CELL+GAP)
    d.text((8, y+CELL//2-16), rlab, fill='black', font=fr)
    for cj, (cfg, _) in enumerate(COLS):
        cv.paste(cell(cfg, ci), (PADL+cj*(CELL+GAP), y))
cv.save(OUT); print('SAVED', OUT, cv.size)
