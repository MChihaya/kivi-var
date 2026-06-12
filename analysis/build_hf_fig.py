"""Two-panel figure for the high-frequency failure analysis.
(a) real high-freq content vs per-class FID (high-freq predicts the failure, r=0.70).
(b) real vs generated high-freq (VAR matches the spectrum, r=0.97)."""
import os.path as osp, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '.'
d = json.load(open(osp.join(ROOT, 'results/hf_analysis.json')))
rows = d['rows']
PT = '#3477b0'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

# (a) real high-freq vs per-class FID
fr = [r for r in rows if r['fid'] is not None]
x = np.array([r['real_hf'] for r in fr]); y = np.array([r['fid'] for r in fr])
ax1.scatter(x, y, c=PT, s=70, edgecolor='k', lw=0.5, zorder=3)
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax1.plot(xs, m * xs + b, '--', color='k', lw=1.2, zorder=2)
ax1.text(0.05, 0.92, '$r = %.2f$' % np.corrcoef(x, y)[0, 1], transform=ax1.transAxes,
         fontsize=17, fontweight='bold')
ax1.set_xlabel('high-frequency content of REAL images\n(spectral energy ratio)', fontsize=13)
ax1.set_ylabel('per-class FID  (higher = worse)', fontsize=13)
ax1.set_title('(a)  high-freq content predicts the failure', fontsize=13.5, fontweight='bold', loc='left')
ax1.grid(alpha=0.25)

# (b) real vs generated high-freq
rx = np.array([r['real_hf'] for r in rows]); ry = np.array([r['gen_hf'] for r in rows])
ax2.scatter(rx, ry, c=PT, s=56, edgecolor='k', lw=0.4, zorder=3)
lim = [0, max(rx.max(), ry.max()) * 1.08]
ax2.plot(lim, lim, '-', color='gray', lw=1.0, zorder=1)
ax2.text(0.05, 0.92, '$r = %.2f$' % np.corrcoef(rx, ry)[0, 1],
         transform=ax2.transAxes, fontsize=17, fontweight='bold')
ax2.text(0.46, 0.08, '$y=x$', transform=ax2.transAxes, fontsize=13, color='#444444')
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel('high-frequency content of REAL images', fontsize=13)
ax2.set_ylabel('high-frequency content of\nVAR-GENERATED images', fontsize=13)
ax2.set_title('(b)  VAR reproduces the amount', fontsize=13.5, fontweight='bold', loc='left')
ax2.grid(alpha=0.25)
ax1.tick_params(labelsize=11); ax2.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(osp.join(ROOT, 'figures_out/fig2_failure1_highfreq.png'), dpi=160,
            bbox_inches='tight', facecolor='white')
print('SAVED fig2_failure1_highfreq.png')
