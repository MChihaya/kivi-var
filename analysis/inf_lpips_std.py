"""Per-image LPIPS to FP16 with mean/std/min/max per config (run in VAR venv)."""
import os.path as osp, sys, glob, json
import numpy as np
from PIL import Image
import torch, lpips
ROOT='.'; base=osp.join(ROOT,'results/infinity')
REF='fp16'; CONFIGS=['uni_int3','kivi_int3','uni_int2','kivi_int2']
fn=lpips.LPIPS(net='alex').cuda().eval()
def t(p):
    x=np.asarray(Image.open(p).convert('RGB'))
    return torch.from_numpy(x).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
keys=[osp.splitext(osp.basename(p))[0] for p in sorted(glob.glob(osp.join(base,REF,'*.jpg')))]
print('n_prompts=%d'%len(keys))
out={}
for c in CONFIGS:
    vals=[]
    for k in keys:
        rp,cp=osp.join(base,REF,k+'.jpg'),osp.join(base,c,k+'.jpg')
        with torch.no_grad(): vals.append(float(fn(t(rp),t(cp)).item()))
    a=np.array(vals)
    out[c]=dict(mean=round(a.mean(),4),std=round(a.std(ddof=1),4),min=round(a.min(),4),max=round(a.max(),4))
    print('%-10s mean=%.4f std=%.4f min=%.4f max=%.4f'%(c,a.mean(),a.std(ddof=1),a.min(),a.max()))
for bit in ['int3','int2']:
    u=[];kv=[]
    for k in keys:
        u.append(float(fn(t(osp.join(base,REF,k+'.jpg')),t(osp.join(base,'uni_'+bit,k+'.jpg'))).item()))
        kv.append(float(fn(t(osp.join(base,REF,k+'.jpg')),t(osp.join(base,'kivi_'+bit,k+'.jpg'))).item()))
    wins=sum(1 for x,y in zip(u,kv) if y<x)
    print('%s: KIVI closer than uniform in %d/%d prompts (mean diff uni-kivi=%.4f)'%(bit,wins,len(keys),float(np.mean(np.array(u)-np.array(kv)))))
json.dump(out,open(osp.join(base,'lpips_std.json'),'w'),indent=2)
print('LPIPS_STD_DONE')
