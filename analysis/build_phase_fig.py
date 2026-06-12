"""Visual proof for the phase/magnitude experiment: one text image under 5 conditions,
with OCR char counts. Output: figures_out/fig5_phase_structure.png"""
import os, os.path as osp, glob
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import easyocr

ROOT='.'; VAL=os.environ['IMAGENET_VAL']; GEN=osp.join(ROOT,'results/failure1/gen')
SIZE=256; IDX=921  # book jacket (large clear title text)
WNIDS=sorted([d for d in os.listdir(VAL) if d.startswith('n')])
reader=easyocr.Reader(['en'],gpu=True,verbose=False); rng=np.random.default_rng(2)

def gray(p): return np.asarray(Image.open(p).convert('L').resize((SIZE,SIZE)),dtype=np.float64)
def recon(mag,ph):
    im=np.real(np.fft.ifft2(mag*np.exp(1j*ph))); im=im-im.min(); return (im/(im.max()+1e-9)*255).astype(np.uint8)
def ocr(u8):
    return sum(len(t) for _,t,c in reader.readtext(np.stack([u8]*3,-1)) if c>0.4)
def avgmag(paths):
    a=None
    for p in paths: m=np.abs(np.fft.fft2(gray(p))); a=m if a is None else a+m
    return a/len(paths)

rpaths=sorted(glob.glob(osp.join(VAL,WNIDS[IDX],'*.JPEG')))[:30]
gpaths=sorted(glob.glob(osp.join(GEN,'%04d'%IDX,'*.png')))[:30]
magR=avgmag(rpaths); magV=avgmag(gpaths)
# pick a real image with the most OCR text, and a VAR sample
_ro=sorted(rpaths[:30], key=lambda q: ocr(gray(q).astype(np.uint8))); real=_ro[len(_ro)//2]  # median-legibility real (representative)
g=gray(real); F=np.fft.fft2(g); magr=np.abs(F); phr=np.angle(F)
_go=sorted(gpaths[:30], key=lambda q: ocr(gray(q).astype(np.uint8))); genp=_go[len(_go)//2]
gg=gray(genp); Fg=np.fft.fft2(gg); phg=np.angle(Fg)

panels=[
 ('Real image', g.astype(np.uint8)),
 ('Real: magnitude kept,\nPHASE randomized', recon(magr, phr+rng.uniform(-np.pi,np.pi,phr.shape))),
 ('Real: phase kept,\nmagnitude <- VAR', recon(magV, phr)),
 ('VAR-generated', gg.astype(np.uint8)),
 ('VAR: phase kept,\nmagnitude <- Real', recon(magR, phg)),
]
fig,axes=plt.subplots(1,5,figsize=(13,3.0))
for i,(ax,(title,img)) in enumerate(zip(axes,panels)):
    ax.imshow(img,cmap='gray'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(%d)'%(i+1),fontsize=17)
    ax.set_xlabel('OCR chars: %d'%ocr(img),fontsize=15,fontweight='bold')
fig.tight_layout()
fig.savefig(osp.join(ROOT,'figures_out/fig5_phase_structure.png'),dpi=150,bbox_inches='tight',facecolor='white')
print('SAVED fig5_phase_structure.png')
