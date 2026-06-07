"""Scale-aware KV-cache eviction baseline for VAR (ScaleKV/HACK family).
Keeps early scales (causally critical) fully; subsamples late scales by a stride,
so the retained cache (memory) is reduced. The current scale still attends to the
full just-computed tokens; only a strided subset is RETAINED for future scales.
Controlled via env EVICT_STRIDE (stride for late scales) and EVICT_LATE_START.
"""
import os
import types
import torch
import torch.nn.functional as F

DEFAULT_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
_CFG = None
_ORIG = {}


class EvictConfig:
    def __init__(self, stride_late=1, late_start=5, patch_nums=DEFAULT_PATCH_NUMS):
        self.stride = max(1, int(stride_late))
        self.late_start = int(late_start)
        self.l_to_si = {pn * pn: i for i, pn in enumerate(patch_nums)}
        self.patch_nums = patch_nums
        self.enabled = self.stride > 1

    def mem_frac(self):
        tot = sum(p * p for p in self.patch_nums)
        kept = 0
        for si, p in enumerate(self.patch_nums):
            n = p * p
            kept += len(range(0, n, self.stride)) if si >= self.late_start else n
        return round(kept / tot, 3)

    def as_dict(self):
        return dict(stride=self.stride, late_start=self.late_start, mem_frac=self.mem_frac())


def _keep(x, si, cfg):  # x: [B,H,L,c] new tokens of scale si
    if cfg.enabled and si is not None and si >= cfg.late_start:
        return x[:, :, ::cfg.stride, :]
    return x


def _fwd(self, x, attn_bias):
    from models.basic_var import slow_attn
    B, L, C = x.shape
    qkv = F.linear(input=x, weight=self.mat_qkv.weight,
                   bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                   ).view(B, L, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
    if self.attn_l2_norm:
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q = F.normalize(q, dim=-1).mul(scale_mul)
        k = F.normalize(k, dim=-1)
    if self.caching:
        cfg = _CFG
        si = cfg.l_to_si.get(L, None)
        if self.cached_k is None:
            kf, vf = k, v
            self.cached_k = _keep(k, si, cfg); self.cached_v = _keep(v, si, cfg)
        else:
            kf = torch.cat((self.cached_k, k), dim=2); vf = torch.cat((self.cached_v, v), dim=2)
            self.cached_k = torch.cat((self.cached_k, _keep(k, si, cfg)), dim=2)
            self.cached_v = torch.cat((self.cached_v, _keep(v, si, cfg)), dim=2)
        k, v = kf, vf
    oup = slow_attn(query=q, key=k, value=v, scale=self.scale,
                    attn_mask=attn_bias, dropout_p=0.0).transpose(1, 2).reshape(B, L, C)
    return self.proj_drop(self.proj(oup))


def apply_evict(var, cfg):
    global _CFG
    _CFG = cfg
    for blk in var.blocks:
        a = blk.attn
        if id(a) not in _ORIG:
            _ORIG[id(a)] = a.forward
        a.forward = types.MethodType(_fwd, a)
    return var


def remove_evict(var):
    global _CFG
    _CFG = None
    for blk in var.blocks:
        a = blk.attn
        if id(a) in _ORIG:
            a.forward = _ORIG[id(a)]
    return var
