"""Test whether KIVI quantization and token-discard (eviction) COMPOSE (the report frames
them as complementary; here we verify they can be combined). The patched forward (a) retains
only a strided subset of late-scale tokens (eviction) and (b) fake-quantizes the assembled K
(KIVI per-channel) and V (per-token) on read. Memory budget = retained-token fraction x bits/16.

Configs (good classes, LPIPS to the full FP16 output = no quant + no evict, lower=better):
  fp16       : no quant, no evict                       -> reference
  int2_kivi  : KIVI INT2, no evict          ~12.5% mem  -> pure aggressive quant (expect collapse)
  evict_only : fp16, late stride 24         ~12.4% mem  -> pure aggressive discard
  int3_evict : KIVI INT3 + late stride 2    ~10.1% mem  -> COMBINATION (even tighter budget)
If int3_evict <= evict_only << int2_kivi at an equal-or-tighter budget, the two savings combine:
keeping MORE tokens at near-lossless INT3 beats keeping FEW tokens at fp16. Run in VAR .venv."""
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
BITS=16; KIVI=True; STRIDE=1; LATE_START=5

def _keep(x, si):
    if STRIDE>1 and si is not None and si>=LATE_START: return x[:,:,::STRIDE,:]
    return x

def mem_frac():
    tot=sum(p*p for p in PN); kept=0
    for si,p in enumerate(PN):
        n=p*p; kept += len(range(0,n,STRIDE)) if si>=LATE_START else n
    return kept/tot

def pf(self,x,attn_bias):
    B,L,C=x.shape
    qkv=F.linear(x,self.mat_qkv.weight,torch.cat((self.q_bias,self.zero_k_bias,self.v_bias))).view(B,L,3,self.num_heads,self.head_dim)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
    if self.attn_l2_norm:
        sm=self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q=F.normalize(q,dim=-1).mul(sm); k=F.normalize(k,dim=-1)
    if self.caching:
        si=L2SI.get(L,None)
        if self.cached_k is None:
            kf,vf=k,v; self.cached_k=_keep(k,si); self.cached_v=_keep(v,si)
        else:
            kf=torch.cat((self.cached_k,k),dim=2); vf=torch.cat((self.cached_v,v),dim=2)
            self.cached_k=torch.cat((self.cached_k,_keep(k,si)),dim=2); self.cached_v=torch.cat((self.cached_v,_keep(v,si)),dim=2)
        k,v=kf,vf
    if BITS<16:
        k=fake_quant(k,BITS,-2 if KIVI else -1,False,False)   # key: per-channel(KIVI) reduces tokens(-2)
        v=fake_quant(v,BITS,-1,False,False)                   # value: per-token reduces channels(-1)
    o=slow_attn(query=q,key=k,value=v,scale=self.scale,attn_mask=attn_bias,dropout_p=0).transpose(1,2).reshape(B,L,C)
    return self.proj_drop(self.proj(o))
for blk in var.blocks: blk.attn.forward=types.MethodType(pf,blk.attn)

CLASSES=[207,388,291,340,817,779,963,437,980,985]; N=16; OUT='./results/quant_evict_combo'
CONFIGS={'fp16':(16,True,1),'int2_kivi':(2,True,1),'evict_only':(16,True,24),'int3_evict':(3,True,2)}  # (bits, kivi, stride)
from PIL import Image
budgets={}
for name,(bits,kv,st) in CONFIGS.items():
    globals()['BITS']=bits; globals()['KIVI']=kv; globals()['STRIDE']=st
    budgets[name]=round(mem_frac()*(bits/16.0),3)
    for ci in CLASSES:
        d=osp.join(OUT,name,'%04d'%ci); os.makedirs(d,exist_ok=True)
        lab=torch.full((N,),ci,dtype=torch.long,device='cuda')
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.float16):
                imgs=var.autoregressive_infer_cfg(B=N,label_B=lab,cfg=1.5,top_k=900,top_p=0.96,g_seed=0)
        arr=imgs.permute(0,2,3,1).mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
        for j in range(N): Image.fromarray(arr[j]).save(osp.join(d,'%03d.png'%j))
    print('generated',name,'budget=%.3f'%budgets[name],flush=True)
import lpips
fn=lpips.LPIPS(net='alex').cuda().eval()
def t(p):
    x=np.asarray(Image.open(p).convert('RGB')); return torch.from_numpy(x).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
per={}
for name in ['int2_kivi','evict_only','int3_evict']:
    vals=[]
    for ci in CLASSES:
        for j in range(N):
            with torch.no_grad(): vals.append(float(fn(t(osp.join(OUT,'fp16','%04d'%ci,'%03d.png'%j)),t(osp.join(OUT,name,'%04d'%ci,'%03d.png'%j))).item()))
    per[name]=np.array(vals)
res={'budgets':budgets,'n':int(len(per['int2_kivi']))}
for name in per: res[name]={'lpips':round(float(per[name].mean()),4),'se':round(float(per[name].std(ddof=1)/np.sqrt(res['n'])),4)}
print('\n=== quant x discard combination (LPIPS to FP16, good classes) ===')
for name in ['int2_kivi','evict_only','int3_evict']:
    print('%-12s budget=%.3f  LPIPS=%.4f +- %.4f'%(name,budgets[name],res[name]['lpips'],res[name]['se']))
json.dump(res,open(osp.join(OUT,'result.json'),'w'),indent=2)
print('COMBO_DONE')
