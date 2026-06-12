"""Controlled experiment: is VAR''s text failure a deficit of high-frequency ENERGY
(magnitude spectrum) or of STRUCTURE (Fourier phase)? We manipulate magnitude and phase
of real / VAR-generated text images independently and read the result with OCR (a clean
legibility = structure probe).

Decisive predictions if "energy OK, structure (phase) broken":
 (1) real image, magnitude kept, phase randomized  -> OCR collapses   (legibility lives in phase)
 (2) real image, phase kept, magnitude replaced by VAR''s class-average -> OCR preserved
                                                       (VAR''s magnitude/energy is adequate)
 (3) VAR image, phase kept, magnitude corrected to the REAL class-average -> OCR stays low
                                                       (fixing energy does not fix VAR''s failure
                                                        -> the broken part is the phase/structure)
Run in VAR venv (.venv)."""
import os, os.path as osp, glob, json
import numpy as np
from PIL import Image
import easyocr

ROOT='.'; VAL=os.environ['IMAGENET_VAL']
GEN=osp.join(ROOT,'results/failure1/gen'); SIZE=256
TEXT={919:'street sign',921:'book jacket',916:'web site',922:'menu',917:'comic book',664:'monitor',530:'digital clock'}
WNIDS=sorted([d for d in os.listdir(VAL) if d.startswith('n')])
reader=easyocr.Reader(['en'],gpu=True,verbose=False)
rng=np.random.default_rng(0)

def gray(p):
    return np.asarray(Image.open(p).convert('L').resize((SIZE,SIZE)),dtype=np.float64)

def recon(mag,phase):
    img=np.real(np.fft.ifft2(mag*np.exp(1j*phase)))
    img=img-img.min(); img=img/(img.max()+1e-9)*255
    return img.astype(np.uint8)

def nchars(img_u8):
    res=reader.readtext(np.stack([img_u8]*3,-1))
    res=[r for r in res if r[2]>0.4]
    return sum(len(t) for _,t,_ in res)

# class-average magnitude for real and gen
def avg_mag(paths):
    acc=None;n=0
    for p in paths:
        m=np.abs(np.fft.fft2(gray(p)))
        acc=m if acc is None else acc+m; n+=1
    return acc/max(n,1)

cond={k:[] for k in ['real','real_phaserand','real_magVAR','gen','gen_magREAL']}
for idx,name in TEXT.items():
    rpaths=sorted(glob.glob(osp.join(VAL,WNIDS[idx],'*.JPEG')))[:30]
    gpaths=sorted(glob.glob(osp.join(GEN,'%04d'%idx,'*.png')))[:30]
    magR=avg_mag(rpaths); magV=avg_mag(gpaths)
    for p in rpaths:
        g=gray(p); F=np.fft.fft2(g); mag=np.abs(F); ph=np.angle(F)
        cond['real'].append(nchars(g.astype(np.uint8)))
        cond['real_phaserand'].append(nchars(recon(mag, ph+rng.uniform(-np.pi,np.pi,ph.shape))))  # keep magnitude, randomize phase
        cond['real_magVAR'].append(nchars(recon(magV, ph)))                                        # keep phase, magnitude<-VAR avg
    for p in gpaths:
        g=gray(p); F=np.fft.fft2(g); ph=np.angle(F)
        cond['gen'].append(nchars(g.astype(np.uint8)))
        cond['gen_magREAL'].append(nchars(recon(magR, ph)))                                        # keep VAR phase, magnitude<-real avg
    print('done', name, flush=True)

res={k:round(float(np.mean(v)),2) for k,v in cond.items()}
print('\n=== phase/magnitude OCR experiment (chars per image, text classes) ===')
print('real images                         : %.2f'%res['real'])
print('real, magnitude kept, PHASE random  : %.2f   (legibility lives in phase if this collapses)'%res['real_phaserand'])
print('real, phase kept, magnitude<-VAR    : %.2f   (VAR energy adequate if this stays high)'%res['real_magVAR'])
print('VAR-generated images                : %.2f'%res['gen'])
print('VAR, phase kept, magnitude<-REAL    : %.2f   (failure is phase/structure if this stays low)'%res['gen_magREAL'])
json.dump(res,open(osp.join(ROOT,'results/phase_structure.json'),'w'),indent=2)
print('PHASE_EXP_DONE')
