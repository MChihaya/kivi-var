"""Follow-up to quant_evict_combo: the naive combination (int3_evict) lost to pure discard
because INT3 degrades the CAUSALLY-CRITICAL early scales (cf. the failure-2 decomposition,
which showed early-scale quantization is what breaks generation). Hypothesis: a PRINCIPLED
combination that PROTECTS the early scales at fp16 and applies quant+discard ONLY to the late
scales should be competitive. We test it here.

Configs at ~12% budget (good classes, LPIPS to full FP16, lower=better). Reuses the fp16,
evict_only, int3_evict images already generated under results/quant_evict_combo/.
  smart_combo: early scales 0-4 fp16 (full), late scales 5-9 KIVI-INT2 + stride 3
               budget = (55*16 + 211*2)/(680*16) = 0.120
If smart_combo < evict_only, the principled combination (protect early, compress late two ways)
reaches a tighter budget than pure discard. If not, pure discard remains best. Run in VAR .venv."""
import os, os.path as osp, sys, json, types
sys.path.insert(0,'.'); sys.path.insert(0,'VAR'); sys.path.insert(0,'analysis')
import numpy as np, torch
import torch.nn.functional as F
from kv_compression.quant import fake_quant
from models.basic_var import slow_attn
CKPT='checkpoints'
setattr(torch.nn.Linear,'reset_parameters',lambda s:None); setattr(torch.nn.LayerNorm,'reset_parameters',lambda s:None)
from models import build_vae_var
PN=(1,2,3,4,5,6,8,10,13,16); L2SI={p*p:i for i,p in enumerate(PN)}
vae,var=build_vae_var(V=4096,Cvae=32,ch=160,share_quant_resi=4,device='cuda',patch_nums=PN,num_classes=1000,depth=16,shared_aln=False)
vae.load_state_dict(torch.load(osp.join(CKPT,'vae_ch160v4096z32.pth'),map_location='cpu'),strict=True)
var.load_state_dict(torch.load(osp.join(CKPT,'var_d16.pth'),map_location='cpu'),strict=True)
vae.eval(); var.eval()
for p in var.parameters(): p.requires_grad_(False)
EARLY_MAX=4; LATE_BITS=2; STRIDE=3; KIVI=True

def _qblock(bsi, x, is_value):
    if bsi is not None and bsi<=EARLY_MAX: return x           # early: fp16 (protected)
    rd = -1 if is_value else (-2 if KIVI else -1)             # late: value per-token, key per-channel
    return fake_quant(x, LATE_BITS, rd, False, False)

def pf(self,x,attn_bias):
    B,L,C=x.shape
    qkv=F.linear(x,self.mat_qkv.weight,torch.cat((self.q_bias,self.zero_k_bias,self.v_bias))).view(B,L,3,self.num_heads,self.head_dim)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
    if self.attn_l2_norm:
        sm=self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q=F.normalize(q,dim=-1).mul(sm); k=F.normalize(k,dim=-1)
    si=L2SI.get(L,None)
    if not hasattr(self,'_kb') or si==0: self._kb=[]; self._vb=[]   # already-evicted past blocks
    # attend to: past evicted blocks (quantized by scale) + current full (quantized by its scale)
    kparts=[_qblock(b,x2,False) for (b,x2) in self._kb]+[_qblock(si,k,False)]
    vparts=[_qblock(b,x2,True)  for (b,x2) in self._vb]+[_qblock(si,v,True)]
    o=slow_attn(query=q,key=torch.cat(kparts,dim=2),value=torch.cat(vparts,dim=2),scale=self.scale,attn_mask=attn_bias,dropout_p=0).transpose(1,2).reshape(B,L,C)
    # retain current for the future (evict late by stride)
    if STRIDE>1 and si is not None and si>EARLY_MAX: ks,vs=k[:,:,::STRIDE,:],v[:,:,::STRIDE,:]
    else: ks,vs=k,v
    self._kb.append((si,ks)); self._vb.append((si,vs))
    return self.proj_drop(self.proj(o))
for blk in var.blocks: blk.attn.forward=types.MethodType(pf,blk.attn)

def budget():
    early=sum(p*p for i,p in enumerate(PN) if i<=EARLY_MAX)
    late_kept=sum(len(range(0,p*p,STRIDE)) for i,p in enumerate(PN) if i>EARLY_MAX)
    return (early*16 + late_kept*LATE_BITS)/(sum(p*p for p in PN)*16)

CLASSES=[207,388,291,340,817,779,963,437,980,985]; N=16; OUT='./results/quant_evict_combo'
from PIL import Image
bg=budget()
for ci in CLASSES:
    d=osp.join(OUT,'smart_combo','%04d'%ci); os.makedirs(d,exist_ok=True)
    lab=torch.full((N,),ci,dtype=torch.long,device='cuda')
    with torch.inference_mode():
        with torch.autocast('cuda',dtype=torch.float16):
            imgs=var.autoregressive_infer_cfg(B=N,label_B=lab,cfg=1.5,top_k=900,top_p=0.96,g_seed=0)
    arr=imgs.permute(0,2,3,1).mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
    for j in range(N): Image.fromarray(arr[j]).save(osp.join(d,'%03d.png'%j))
print('generated smart_combo budget=%.3f'%bg,flush=True)
import lpips
fn=lpips.LPIPS(net='alex').cuda().eval()
def t(p):
    x=np.asarray(Image.open(p).convert('RGB')); return torch.from_numpy(x).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
prev=json.load(open(osp.join(OUT,'result.json')))
per=[]
for ci in CLASSES:
    for j in range(N):
        with torch.no_grad(): per.append(float(fn(t(osp.join(OUT,'fp16','%04d'%ci,'%03d.png'%j)),t(osp.join(OUT,'smart_combo','%04d'%ci,'%03d.png'%j))).item()))
per=np.array(per); n=len(per)
prev['budgets']['smart_combo']=round(bg,3)
prev['smart_combo']={'lpips':round(float(per.mean()),4),'se':round(float(per.std(ddof=1)/np.sqrt(n)),4)}
json.dump(prev,open(osp.join(OUT,'result.json'),'w'),indent=2)
print('\n=== quant x discard, all configs (LPIPS to FP16, good classes, lower=better) ===')
for name in ['int2_kivi','int3_evict','smart_combo','evict_only']:
    print('%-12s budget=%.3f  LPIPS=%.4f +- %.4f'%(name,prev['budgets'][name],prev[name]['lpips'],prev[name]['se']))
print('SMART_DONE')
