"""Why does KIVI help VAR? KIVI vs uniform differ ONLY in the Key direction (per-channel
vs per-token); both quantize Values per-token. We capture q, K, V of every layer at the
final scale and measure: (1) Value quant error per-channel vs per-token; (2) the error the
two Key-quantization directions induce in the attention logits q.K^T (the quantity attention
actually uses). Run in VAR venv (.venv)."""
import os, os.path as osp, sys, json, types
sys.path.insert(0, '.'); sys.path.insert(0, 'VAR')
sys.path.insert(0, 'analysis')
import numpy as np, torch
import torch.nn.functional as F
from kv_compression.quant import fake_quant
from models.basic_var import slow_attn

CKPT = 'checkpoints'
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import build_vae_var
pn = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4, device='cuda',
                         patch_nums=pn, num_classes=1000, depth=16, shared_aln=False)
vae.load_state_dict(torch.load(osp.join(CKPT, 'vae_ch160v4096z32.pth'), map_location='cpu'), strict=True)
var.load_state_dict(torch.load(osp.join(CKPT, 'var_d16.pth'), map_location='cpu'), strict=True)
vae.eval(); var.eval()
for p in var.parameters(): p.requires_grad_(False)

CAP = {}
def patched(self, x, attn_bias):
    B, L, C = x.shape
    qkv = F.linear(x, self.mat_qkv.weight, torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
    using = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
    if using or self.using_xform: q, k, v = qkv.unbind(2); dc = 1
    else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0); dc = 2
    if self.attn_l2_norm:
        sm = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        if using or self.using_xform: sm = sm.transpose(1, 2)
        q = F.normalize(q, dim=-1).mul(sm); k = F.normalize(k, dim=-1)
    if self.caching:
        if self.cached_k is None: self.cached_k = k; self.cached_v = v
        else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dc); v = self.cached_v = torch.cat((self.cached_v, v), dim=dc)
    if k.shape[dc] == 680 and self.block_idx not in CAP:   # final scale, full cache
        CAP[self.block_idx] = (q.detach().float().cpu(), k.detach().float().cpu(), v.detach().float().cpu(), dc)
    if using:
        from models.basic_var import flash_attn_func
        o = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v.to(v.dtype), dropout_p=0, softmax_scale=self.scale).view(B, L, C)
    elif self.using_xform:
        from models.basic_var import memory_efficient_attention
        o = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=None, p=0, scale=self.scale).view(B, L, C)
    else:
        o = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=0).transpose(1, 2).reshape(B, L, C)
    return self.proj_drop(self.proj(o))
for blk in var.blocks:
    blk.attn.forward = types.MethodType(patched, blk.attn)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.float16):
        var.autoregressive_infer_cfg(B=4, label_B=torch.tensor([207, 980, 817, 388], device='cuda'),
                                     cfg=1.5, top_k=900, top_p=0.96, g_seed=0)
print('captured layers:', len(CAP), flush=True)


def std4(t, dc):  # -> (BH, tokens, channels)
    if dc == 1: B, Lq, H, c = t.shape; t = t.permute(0, 2, 1, 3)
    else: B, H, Lq, c = t.shape
    return t.reshape(t.shape[0]*t.shape[1], t.shape[2], t.shape[3])


def qmse(x, bits, rd):
    return (fake_quant(x, bits, rd, False, False).float() - x.float()).pow(2).mean().item()


vpc, vpt, logit_pc, logit_pt = [], [], [], []
for bi, (q, k, v, dc) in CAP.items():
    Q = std4(q, dc); K = std4(k, dc); V = std4(v, dc)      # (BH, tok, c)
    vpc.append(qmse(V, 3, 1)); vpt.append(qmse(V, 3, 2))    # value: per-channel vs per-token
    # attention logits q.K^T (only query tokens of final scale: Q is BH x 256 x c)
    base = torch.matmul(Q, K.transpose(1, 2))               # (BH, 256, 680)
    Kpc = fake_quant(K, 3, 1, False, False)                 # key per-channel
    Kpt = fake_quant(K, 3, 2, False, False)                 # key per-token
    e_pc = (torch.matmul(Q, Kpc.transpose(1, 2)) - base).pow(2).mean() / base.pow(2).mean()
    e_pt = (torch.matmul(Q, Kpt.transpose(1, 2)) - base).pow(2).mean() / base.pow(2).mean()
    logit_pc.append(float(e_pc)); logit_pt.append(float(e_pt))

res = dict(val_err_perchannel=round(float(np.mean(vpc)), 6), val_err_pertoken=round(float(np.mean(vpt)), 6),
           logit_relerr_key_perchannel=round(float(np.mean(logit_pc)), 5),
           logit_relerr_key_pertoken=round(float(np.mean(logit_pt)), 5))
res['value_pertoken_gain'] = round(res['val_err_perchannel']/res['val_err_pertoken'], 2)
res['key_perchannel_logit_gain'] = round(res['logit_relerr_key_pertoken']/res['logit_relerr_key_perchannel'], 2)
print('\n=== KIVI mechanism on VAR (INT3, mean over 16 layers) ===')
print('VALUE quant MSE: per-channel=%.6f  per-token=%.6f  -> per-token %.2fx lower (KIVI uses per-token for V)'
      % (res['val_err_perchannel'], res['val_err_pertoken'], res['value_pertoken_gain']))
print('KEY -> attention-logit rel.error: per-channel=%.5f  per-token=%.5f  -> per-channel %.2fx lower (KIVI uses per-channel for K)'
      % (res['logit_relerr_key_perchannel'], res['logit_relerr_key_pertoken'], res['key_perchannel_logit_gain']))
json.dump(res, open('./results/kv_dist.json', 'w'), indent=2)
print('SAVED'); print('KV_DIST_DONE')
