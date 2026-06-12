"""Controlled ablation: does the Key-per-channel benefit (the only difference between
uniform and KIVI) come from the EARLY scales (causally upstream -> compounding) or the
LATE scales (92% of tokens)? We apply Key per-channel quantization only at chosen scales
(Value always per-token, INT3) and measure LPIPS to the FP16 output on VAR-strong classes.
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
BITS=3; CHAN_SCALES=set()  # set per config

def pf(self,x,attn_bias):
    B,L,C=x.shape
    qkv=F.linear(x,self.mat_qkv.weight,torch.cat((self.q_bias,self.zero_k_bias,self.v_bias))).view(B,L,3,self.num_heads,self.head_dim)
    using=self.using_flash and attn_bias is None and qkv.dtype!=torch.float32
    if using or self.using_xform: q,k,v=qkv.unbind(2); dc=1
    else: q,k,v=qkv.permute(2,0,3,1,4).unbind(0); dc=2
    if self.attn_l2_norm:
        sm=self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        if using or self.using_xform: sm=sm.transpose(1,2)
        q=F.normalize(q,dim=-1).mul(sm); k=F.normalize(k,dim=-1)
    if BITS<16:
        si=L2SI.get(L,None)
        kdir_chan = si in CHAN_SCALES
        rdk = (dc if kdir_chan else -1)   # per-channel: reduce over tokens(dc); per-token: reduce channels(-1)
        k=fake_quant(k,BITS,rdk,False,False)
        v=fake_quant(v,BITS,-1,False,False)   # value per-token always
    if self.caching:
        if self.cached_k is None: self.cached_k=k; self.cached_v=v
        else: k=self.cached_k=torch.cat((self.cached_k,k),dim=dc); v=self.cached_v=torch.cat((self.cached_v,v),dim=dc)
    if using:
        from models.basic_var import flash_attn_func
        o=flash_attn_func(q.to(v.dtype),k.to(v.dtype),v.to(v.dtype),dropout_p=0,softmax_scale=self.scale).view(B,L,C)
    elif self.using_xform:
        from models.basic_var import memory_efficient_attention
        o=memory_efficient_attention(q.to(v.dtype),k.to(v.dtype),v,attn_bias=None,p=0,scale=self.scale).view(B,L,C)
    else:
        o=slow_attn(query=q,key=k,value=v,scale=self.scale,attn_mask=attn_bias,dropout_p=0).transpose(1,2).reshape(B,L,C)
    return self.proj_drop(self.proj(o))
for blk in var.blocks: blk.attn.forward=types.MethodType(pf,blk.attn)

CLASSES=[207,388,291,340,817,779,963,437,980,985]; N=16; OUT='./results/ablation'
CONFIGS={'fp16':(16,set()),'uniform':(3,set()),'kivi':(3,set(range(10))),'early':(3,set(range(0,5))),'late':(3,set(range(5,10)))}
from PIL import Image
for name,(bits,cs) in CONFIGS.items():
    globals()['BITS']=bits; globals()['CHAN_SCALES']=cs
    for ci in CLASSES:
        d=osp.join(OUT,name,'%04d'%ci); os.makedirs(d,exist_ok=True)
        lab=torch.full((N,),ci,dtype=torch.long,device='cuda')
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.float16):
                imgs=var.autoregressive_infer_cfg(B=N,label_B=lab,cfg=1.5,top_k=900,top_p=0.96,g_seed=0)
        arr=imgs.permute(0,2,3,1).mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
        for j in range(N): Image.fromarray(arr[j]).save(osp.join(d,'%03d.png'%j))
    print('generated',name,flush=True)
# LPIPS to fp16
import lpips
fn=lpips.LPIPS(net='alex').cuda().eval()
def t(p):
    x=np.asarray(Image.open(p).convert('RGB')); return torch.from_numpy(x).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
per={}
for name in ['uniform','kivi','early','late']:
    vals=[]
    for ci in CLASSES:
        for j in range(N):
            rp=osp.join(OUT,'fp16','%04d'%ci,'%03d.png'%j); cp=osp.join(OUT,name,'%04d'%ci,'%03d.png'%j)
            with torch.no_grad(): vals.append(float(fn(t(rp),t(cp)).item()))
    per[name]=np.array(vals)
res={k:round(float(v.mean()),4) for k,v in per.items()}
res['n']=int(len(per['uniform']))
res['se']={k:round(float(v.std(ddof=1)/np.sqrt(len(v))),4) for k,v in per.items()}
res['std']={k:round(float(v.std(ddof=1)),4) for k,v in per.items()}
# paired diffs
res['paired']={
 'uniform_minus_early':round(float((per['uniform']-per['early']).mean()),4),
 'early_minus_kivi':round(float((per['early']-per['kivi']).mean()),4),
 'se_uniform_minus_early':round(float((per['uniform']-per['early']).std(ddof=1)/np.sqrt(res['n'])),4),
 'se_early_minus_kivi':round(float((per['early']-per['kivi']).std(ddof=1)/np.sqrt(res['n'])),4)}
print('\n=== Key-direction scale ablation (INT3, LPIPS to FP16, lower=closer) ===')
for k in ['uniform','late','early','kivi']:
    print('%-8s key-per-channel scales=%-12s LPIPS=%.4f'%(k,{'uniform':'none','late':'5-9','early':'0-4','kivi':'all'}[k],res[k]))
json.dump(res,open(osp.join(OUT,'lpips.json'),'w'),indent=2)
print('ABLATION_DONE')
