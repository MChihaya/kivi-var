"""Combine the VAR good-class panel (build_good_fig.py) and the Infinity panel
(build_inf_fig.py) side by side -> the report's qualitative-comparison figure (Fig. 5)."""
from PIL import Image

A = 'figures_out/fig5_goodclass_quant.png'
B = 'figures_out/fig6_infinity_quant.png'
OUT = 'figures_out/fig8_9_combined.png'

a, b = Image.open(A), Image.open(B)
h = max(a.height, b.height)


def rs(im):
    if im.height == h:
        return im
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


a, b = rs(a), rs(b)
c = Image.new('RGB', (a.width + b.width, h), 'white')
c.paste(a, (0, 0))
c.paste(b, (a.width, 0))
c.save(OUT)
print('saved', OUT, c.size)
