"""Decisive test: is the EARLY-SCALE KV-quantization damage (failure-2, INT2) due to
PROPAGATION (late scales reading quantized early KV) or to the early scales' OWN
generation being corrupted? Parallels the key-direction test (propagation_exp), but here
we route the PRECISION (fp16 vs INT2) of early K AND V by the QUERY scale.

KEY to a clean split: early tokens are generated IDENTICALLY in early_fp16 and prop_blocked
(both feed EARLY queries fp16 early-KV), so the ONLY difference between them is the LATE-query
view of early KV. Late-scale KV is INT2 in all three quantized configs, so we isolate the
early-scale contribution. gain_total = gain_self + gain_prop EXACTLY (telescoping).

Configs (base INT2 uniform per-token):
  base_int2   : all KV INT2                                -> failure baseline
  early_fp16  : early KV fp16 for everyone (late INT2)     -> restores all early-scale importance
  prop_blocked: early KV fp16 for EARLY queries, INT2 for  -> early-self restored, propagation blocked
                LATE queries (late INT2)
If gain_prop ~ 0 -> early-KV damage is the early scales' OWN generation, NOT propagation
(consistent with the key-direction finding -> resolves the 6.2/7 framing). Run in VAR .venv."""
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
BITS=2; EARLY_SELF=True; EARLY_PROP=True; EARLY_MAX=4  # early = scales 0..4

def _qt(x):  # per-token INT (reduce channels, dim -1)
    if BITS>=16: return x
    return fake_quant(x, BITS, -1, False, False)

def pf(self,x,attn_bias):
    B,L,C=x.shape
    qkv=F.linear(x,self.mat_qkv.weight,torch.cat((self.q_bias,self.zero_k_bias,self.v_bias))).view(B,L,3,self.num_heads,self.head_dim)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)  # [B,H,L,c]
    if self.attn_l2_norm:
        sm=self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q=F.normalize(q,dim=-1).mul(sm); k=F.normalize(k,dim=-1)
    si=L2SI.get(L,None)
    if not hasattr(self,'_kb') or si==0: self._kb=[]; self._vb=[]
    self._kb.append((si,k)); self._vb.append((si,v))   # store fp16 per-scale
    late_query = (si is None or si>EARLY_MAX)
    kparts=[]; vparts=[]
    for (bsi,bk),(_,bv) in zip(self._kb,self._vb):
        if bsi is not None and bsi<=EARLY_MAX:
            keep_fp16 = (EARLY_PROP if late_query else EARLY_SELF)
            if keep_fp16: kparts.append(bk); vparts.append(bv)
            else: kparts.append(_qt(bk)); vparts.append(_qt(bv))
        else:
            kparts.append(_qt(bk)); vparts.append(_qt(bv))   # late KV always INT2
    kk=torch.cat(kparts,dim=2); vv=torch.cat(vparts,dim=2)
    o=slow_attn(query=q,key=kk,value=vv,scale=self.scale,attn_mask=attn_bias,dropout_p=0).transpose(1,2).reshape(B,L,C)
    return self.proj_drop(self.proj(o))
for blk in var.blocks: blk.attn.forward=types.MethodType(pf,blk.attn)

CLASSES=[207,388,291,340,817,779,963,437,980,985]; N=16; OUT='./results/failure2_decomp'
CONFIGS={'fp16':(16,True,True),'base_int2':(2,False,False),'early_fp16':(2,True,True),'prop_blocked':(2,True,False)}  # (bits, early_self_fp16, early_prop_fp16)
from PIL import Image
for name,(bits,es,ep) in CONFIGS.items():
    globals()['BITS']=bits; globals()['EARLY_SELF']=es; globals()['EARLY_PROP']=ep
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
for name in ['base_int2','early_fp16','prop_blocked']:
    vals=[]
    for ci in CLASSES:
        for j in range(N):
            with torch.no_grad(): vals.append(float(fn(t(osp.join(OUT,'fp16','%04d'%ci,'%03d.png'%j)),t(osp.join(OUT,name,'%04d'%ci,'%03d.png'%j))).item()))
    per[name]=np.array(vals)
res={k:round(float(v.mean()),4) for k,v in per.items()}
res['n']=int(len(per['base_int2']))
res['gain_total']=round(float((per['base_int2']-per['early_fp16']).mean()),4)
res['gain_self']=round(float((per['base_int2']-per['prop_blocked']).mean()),4)
res['gain_propagation']=round(float((per['prop_blocked']-per['early_fp16']).mean()),4)
res['se_self']=round(float((per['base_int2']-per['prop_blocked']).std(ddof=1)/np.sqrt(res['n'])),4)
res['se_propagation']=round(float((per['prop_blocked']-per['early_fp16']).std(ddof=1)/np.sqrt(res['n'])),4)
print('\n=== failure-2 decomposition (INT2, LPIPS to FP16) ===')
print('base_int2=%.4f  early_fp16=%.4f  prop_blocked=%.4f  (n=%d)'%(res['base_int2'],res['early_fp16'],res['prop_blocked'],res['n']))
print('total early-scale gain (base-early_fp16) = %.4f'%res['gain_total'])
print('  of which SELF (base-prop_blocked)            = %.4f +- %.4f'%(res['gain_self'],res['se_self']))
print('  of which PROPAGATION (prop_blocked-early_fp16)= %.4f +- %.4f'%(res['gain_propagation'],res['se_propagation']))
json.dump(res,open(osp.join(OUT,'result.json'),'w'),indent=2)
print('FAIL2_DONE')
