import os, os.path as osp, glob
import PIL.Image as I, PIL.ImageDraw as D, PIL.ImageFont as F
ROOT = '.'
FIG = osp.join(ROOT, 'figures_out')


def font(sz):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if osp.exists(p):
            return F.truetype(p, sz)
    return F.load_default()


def tlen(dr, t, ft):
    try:
        return dr.textlength(t, font=ft)
    except Exception:
        return len(t) * sz * 0.5


def add_margins(grid, top_labels=None, left_labels=None, cell=256, lm=160, tm=44):
    W, H = grid.size
    canvas = I.new('RGB', (W + lm, H + tm), (255, 255, 255))
    canvas.paste(grid, (lm, tm))
    dr = D.Draw(canvas)
    if top_labels:
        ft = font(32)
        for k, t in enumerate(top_labels):
            x = lm + k * cell + cell // 2
            w = tlen(dr, t, ft)
            dr.text((x - w / 2, max(4, tm // 2 - 16)), t, fill=(0, 0, 0), font=ft)
    if left_labels:
        ft = font(28)
        for k, t in enumerate(left_labels):
            y = tm + k * cell + cell // 2
            dr.text((8, y - 16), t, fill=(0, 0, 0), font=ft)
    return canvas


def strip(d, n=6, sz=160):
    fs = sorted(glob.glob(d + '/*.png'))[:n]
    ims = [I.open(f).convert('RGB').resize((sz, sz)) for f in fs]
    r = I.new('RGB', (sz * len(ims), sz), (255, 255, 255))
    for k, im in enumerate(ims):
        r.paste(im, (k * sz, 0))
    return r


# fig1: baseline samples with class labels under each pair
f1 = I.open(osp.join(ROOT, 'results/baseline/sanity_d16.png')).convert('RGB')  # 2048x256
c1 = I.new('RGB', (f1.width, f1.height + 52), (255, 255, 255)); c1.paste(f1, (0, 0))
dr = D.Draw(c1); ft = font(32)
for k, t in enumerate(['Volcano', 'Lighthouse', 'Bald eagle', 'Fountain']):
    x = k * 512 + 256; w = tlen(dr, t, ft)
    dr.text((x - w / 2, f1.height + 12), t, fill=(0, 0, 0), font=ft)
c1.save(osp.join(FIG, 'fig1_baseline_samples.png'))

# fig2: failure-1 text (book jacket) real/gen, left labels
f2 = I.open(osp.join(FIG, 'fig2_text_bare.png')).convert('RGB')  # 1536x512, 2 rows x6, cell256
c2 = add_margins(f2, left_labels=['Real', 'VAR (FP16)'], cell=256, lm=205, tm=10)
c2.save(osp.join(FIG, 'fig2_failure1_text.png'))

# fig3: failure-2 bit sweep, top + left labels
f3 = I.open(osp.join(FIG, 'fig3_bits_bare.png')).convert('RGB')  # 1024x768, 4col x3row, cell256
c3 = add_margins(f3, top_labels=['FP16', 'INT4', 'INT3', 'INT2'],
                 left_labels=['Street sign', 'Digital clock', 'Volcano'], cell=256, lm=235, tm=62)
c3.save(osp.join(FIG, 'fig3_failure2_bits.png'))

# fig4: case study, left labels
for ci, nm in [(921, 'book_jacket'), (796, 'ski_mask')]:
    rows = [('Real', osp.join(ROOT, 'data/ref_perclass/%04d' % ci)),
            ('VAR FP16', osp.join(ROOT, 'results/failure1/gen/%04d' % ci)),
            ('Uniform INT3', osp.join(ROOT, 'results/cs2/unif/%04d' % ci)),
            ('KIVI INT3', osp.join(ROOT, 'results/cs2/kivi/%04d' % ci))]
    strips = [strip(d) for _, d in rows]
    W = max(s.width for s in strips); cell = 160
    bare = I.new('RGB', (W, cell * len(strips)), (255, 255, 255))
    for k, s in enumerate(strips):
        bare.paste(s, (0, k * cell))
    c4 = add_margins(bare, left_labels=[r[0] for r in rows], cell=cell, lm=170, tm=10)
    c4.save(osp.join(FIG, 'fig4_casestudy_%s.png' % nm))
    print('fig4', nm)
print('FIGS_DONE')
