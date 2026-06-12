"""Decisive test of cross-scale PROPAGATION for the key-per-channel benefit.
We store keys per-scale (fp16) and quantize them on read, routing the EARLY-scale keys
by the QUERY scale: an early query sees early keys in direction EARLY_DIR; a late query
sees early keys in direction EARLY_DIR_LATEQ. Late-scale keys and all values are per-token.

Configs (INT3, value per-token):
  uniform : early keys per-token for everyone                -> baseline
  early_pc: early keys per-channel for everyone              -> recovers the gain (= fig7 'early')
  lateblind: early keys per-channel ONLY for early queries,  -> if the gain DISAPPEARS, the
             per-token for late queries                          benefit needs late->early
                                                                 attention = PROPAGATION proven
Compare: gain(early_pc-uniform) split into propagation(early_pc-lateblind)+self(lateblind-uniform).
Run in VAR venv (.venv)."""
import os, os.path as osp, sys, json, types, glob
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
BITS=3; EARLY_DIR='token'; EARLY_DIR_LATEQ='token'; EARLY_MAX=4  # scales 0..4 are "early"

def _q(k, direction):
    if BITS>=16: return k
    rd = -2 if direction=='channel' else -1   # slow path [B,H,L,c]: channel=reduce tokens(-2), token=reduce chans(-1)
    return fake_quant(k, BITS, rd, False, False)

def pf(self,x,attn_bias):
    B,L,C=x.shape
    qkv=F.linear(x,self.mat_qkv.weight,torch.cat((self.q_bias,self.zero_k_bias,self.v_bias))).view(B,L,3,self.num_heads,self.head_dim)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)  # [B,H,L,c]
    if self.attn_l2_norm:
        sm=self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q=F.normalize(q,dim=-1).mul(sm); k=F.normalize(k,dim=-1)
    si=L2SI.get(L,None)
    if not hasattr(self,'_kb') or si==0: self._kb=[]; self._vb=[]
    self._kb.append((si,k)); self._vb.append(v)   # store fp16 per-scale
    # assemble keys with routing by current query scale si
    kparts=[]
    for (bsi,bk) in self._kb:
        if bsi is not None and bsi<=EARLY_MAX:
            d = EARLY_DIR if (si is not None and si<=EARLY_MAX) else EARLY_DIR_LATEQ
        else:
            d = 'token'
        kparts.append(_q(bk,d))
    kk=torch.cat(kparts,dim=2)
    vv=_q(torch.cat(self._vb,dim=2),'token') if BITS<16 else torch.cat(self._vb,dim=2)
    o=slow_attn(query=q,key=kk,value=vv,scale=self.scale,attn_mask=attn_bias,dropout_p=0).transpose(1,2).reshape(B,L,C)
    return self.proj_drop(self.proj(o))
for blk in var.blocks: blk.attn.forward=types.MethodType(pf,blk.attn)

CLASSES=[207,388,291,340,817,779,963,437,980,985]; N=16; OUT='./results/propagation'
CONFIGS={'fp16':(16,'token','token'),'uniform':(3,'token','token'),'early_pc':(3,'channel','channel'),'lateblind':(3,'channel','token')}
from PIL import Image
for name,(bits,ed,edl) in CONFIGS.items():
    globals()['BITS']=bits; globals()['EARLY_DIR']=ed; globals()['EARLY_DIR_LATEQ']=edl
    for ci in CLASSES:
        d=osp.join(OUT,name,'%04d'%ci); os.makedirs(d,exist_ok=True)
        lab=torch.full((N,),ci,dtype=torch.long,device='cuda')
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.float16):
                imgs=var.autoregressive_infer_cfg(B=N,label_B=lab,cfg=1.5,top_k=900,top_p=0.96,g_seed=0)
        arr=imgs.permute(0,2,3,1).mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
        for j in range(N): Image.fromarray(arr[j]).save(osp.join(d,'%03d.png'%j))
    print('generated',name,flush=True)
import lpips
fn=lpips.LPIPS(net='alex').cuda().eval()
def t(p):
    x=np.asarray(Image.open(p).convert('RGB')); return torch.from_numpy(x).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
per={}
for name in ['uniform','early_pc','lateblind']:
    vals=[]
    for ci in CLASSES:
        for j in range(N):
            with torch.no_grad(): vals.append(float(fn(t(osp.join(OUT,'fp16','%04d'%ci,'%03d.png'%j)),t(osp.join(OUT,name,'%04d'%ci,'%03d.png'%j))).item()))
    per[name]=np.array(vals)
res={k:round(float(v.mean()),4) for k,v in per.items()}
res['n']=int(len(per['uniform']))
res['gain_total']=round(float((per['uniform']-per['early_pc']).mean()),4)
res['gain_self']=round(float((per['uniform']-per['lateblind']).mean()),4)
res['gain_propagation']=round(float((per['lateblind']-per['early_pc']).mean()),4)
res['se_propagation']=round(float((per['lateblind']-per['early_pc']).std(ddof=1)/np.sqrt(res['n'])),4)
res['se_self']=round(float((per['uniform']-per['lateblind']).std(ddof=1)/np.sqrt(res['n'])),4)
print('\n=== propagation isolation (INT3, LPIPS to FP16) ===')
print('uniform=%.4f  early_pc=%.4f  lateblind=%.4f  (n=%d)'%(res['uniform'],res['early_pc'],res['lateblind'],res['n']))
print('total gain (uniform-early_pc)        = %.4f'%res['gain_total'])
print('  of which SELF (uniform-lateblind)  = %.4f +- %.4f'%(res['gain_self'],res['se_self']))
print('  of which PROPAGATION (lateblind-early_pc) = %.4f +- %.4f'%(res['gain_propagation'],res['se_propagation']))
json.dump(res,open(osp.join(OUT,'result.json'),'w'),indent=2)
print('PROP_DONE')
