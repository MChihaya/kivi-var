"""Build Infinity quant comparison figure (rows=prompts, cols=configs) with baked labels.
Run in any venv with PIL. Output: figures_out/fig6_infinity_quant.png"""
import os, os.path as osp
from PIL import Image, ImageDraw, ImageFont

ROOT = '.'
BASE = osp.join(ROOT, 'results/infinity')
OUT = osp.join(ROOT, 'figures_out/fig6_infinity_quant.png')
COLS = [('fp16', 'FP16 (baseline)'), ('uni_int3', 'Uniform INT3'),
        ('kivi_int3', 'KIVI INT3 (ours)'), ('uni_int2', 'Uniform INT2'),
        ('kivi_int2', 'KIVI INT2 (ours)')]
ROWS = [('explore_more', 'text: "Explore More"'), ('corgi_dog', 'Corgi w/ sunglasses'),
        ('cat_fashion', 'cats fashion (B&W)'), ('toy_car', 'diecast toy car (night)')]
CELL = 320
PADL = 300   # left label gutter
PADT = 54    # top header
GAP = 6


def font(sz):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if osp.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load_cell(cfg, key):
    p = osp.join(BASE, cfg, key + '.jpg')
    if osp.exists(p):
        return Image.open(p).convert('RGB').resize((CELL, CELL))
    im = Image.new('RGB', (CELL, CELL), (40, 40, 40))
    ImageDraw.Draw(im).text((CELL // 2 - 30, CELL // 2), 'N/A', fill=(200, 200, 200), font=font(20))
    return im


W = PADL + len(COLS) * (CELL + GAP)
H = PADT + len(ROWS) * (CELL + GAP)
canvas = Image.new('RGB', (W, H), (255, 255, 255))
d = ImageDraw.Draw(canvas)
fh, fr = font(31), font(24)
for ci, (cfg, label) in enumerate(COLS):
    x = PADL + ci * (CELL + GAP)
    d.text((x + 8, 10), label, fill=(0, 0, 0), font=fh)
for ri, (key, rlabel) in enumerate(ROWS):
    y = PADT + ri * (CELL + GAP)
    d.text((8, y + CELL // 2 - 14), rlabel, fill=(0, 0, 0), font=fr)
    for ci, (cfg, _) in enumerate(COLS):
        x = PADL + ci * (CELL + GAP)
        canvas.paste(load_cell(cfg, key), (x, y))
canvas.save(OUT)
print('SAVED', OUT, canvas.size)
