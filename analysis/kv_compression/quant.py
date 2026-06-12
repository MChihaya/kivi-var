"""KV-cache fake-quantization for VAR.
- deterministic per-token/channel (default), KIVI-style axes, per-scale mixed precision
- stochastic (dithered) rounding: env KVQ_STOCHASTIC=1
- error feedback (error diffusion across tokens, per-channel grid): env KVQ_EF=1
"""
import os
import types
import torch
import torch.nn.functional as F
from models.basic_var import slow_attn

DEFAULT_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
_KVQ = None
_ORIG_FORWARDS = {}


def fake_quant(x, bits, reduce_dim, sym, stochastic=False):
    if bits is None or bits >= 16:
        return x
    xf = x.float()
    if sym:
        m = xf.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(1e-8)
        qpos = 2 ** (bits - 1) - 1
        scale = m / qpos
        u = xf / scale
        q = torch.floor(u + torch.rand_like(u)) if stochastic else torch.round(u)
        q = torch.clamp(q, -qpos - 1, qpos)
        out = q * scale
    else:
        mn = xf.amin(dim=reduce_dim, keepdim=True)
        mx = xf.amax(dim=reduce_dim, keepdim=True)
        qmax = 2 ** bits - 1
        scale = (mx - mn).clamp_min(1e-8) / qmax
        u = (xf - mn) / scale
        q = torch.floor(u + torch.rand_like(u)) if stochastic else torch.round(u)
        q = torch.clamp(q, 0, qmax)
        out = q * scale + mn
    return out.to(x.dtype)


def ef_quant_block(x, bits, sym):
    if bits is None or bits >= 16:
        return x
    xf = x.float()
    qmax = 2 ** bits - 1
    mn = xf.amin(dim=-2, keepdim=True)
    mx = xf.amax(dim=-2, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / qmax
    Lb = xf.shape[-2]
    out = torch.empty_like(xf)
    e = torch.zeros_like(xf[:, :, :1, :])
    for t in range(Lb):
        xt = xf[:, :, t:t + 1, :] + e
        q = torch.clamp(torch.round((xt - mn) / scale), 0, qmax)
        deq = q * scale + mn
        e = xt - deq
        out[:, :, t:t + 1, :] = deq
    return out.to(x.dtype)


class KVQConfig:
    def __init__(self, k_bits=16, v_bits=16, per='token', per_k=None, per_v=None, sym=False,
                 layers='all', scales='all', scale_bits=None, stochastic=None, ef=None,
                 patch_nums=DEFAULT_PATCH_NUMS):
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.per_k = per_k if per_k else per
        self.per_v = per_v if per_v else per
        self.sym = bool(sym)
        self.stochastic = bool(int(os.environ.get('KVQ_STOCHASTIC', '0'))) if stochastic is None else bool(stochastic)
        self.ef = bool(int(os.environ.get('KVQ_EF', '0'))) if ef is None else bool(ef)
        self.layers = layers if layers == 'all' else set(int(x) for x in layers)
        self.scales = scales if scales == 'all' else set(int(x) for x in scales)
        self.l_to_si = {pn * pn: i for i, pn in enumerate(patch_nums)}
        self.scale_bits = None
        if scale_bits:
            self.scale_bits = {int(k): (int(v[0]), int(v[1])) for k, v in scale_bits.items()}
        if self.scale_bits is not None:
            self.enabled = any(kb < 16 or vb < 16 for kb, vb in self.scale_bits.values())
        else:
            self.enabled = (self.k_bits < 16) or (self.v_bits < 16)

    def bits_for(self, block_idx, si):
        if si is None:
            return (16, 16)
        if self.layers != 'all' and block_idx not in self.layers:
            return (16, 16)
        if self.scale_bits is not None:
            return self.scale_bits.get(si, (16, 16))
        if self.scales != 'all' and si not in self.scales:
            return (16, 16)
        return (self.k_bits, self.v_bits)

    def avg_bits(self, patch_nums=DEFAULT_PATCH_NUMS):
        tot = sum(pn * pn for pn in patch_nums)
        acc = 0.0
        for si, pn in enumerate(patch_nums):
            kb, vb = self.bits_for(0, si)
            acc += pn * pn * (kb + vb) / 2.0
        return round(acc / tot, 3)

    def as_dict(self):
        sb = None if self.scale_bits is None else {str(k): list(v) for k, v in self.scale_bits.items()}
        return dict(k_bits=self.k_bits, v_bits=self.v_bits, per_k=self.per_k, per_v=self.per_v,
                    sym=self.sym, stochastic=self.stochastic, ef=self.ef,
                    layers='all' if self.layers == 'all' else sorted(self.layers),
                    scales='all' if self.scales == 'all' else sorted(self.scales),
                    scale_bits=sb, avg_bits=self.avg_bits())


def _patched_forward(self, x, attn_bias):
    B, L, C = x.shape
    qkv = F.linear(input=x, weight=self.mat_qkv.weight,
                   bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                   ).view(B, L, 3, self.num_heads, self.head_dim)
    main_type = qkv.dtype
    using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
    if using_flash or self.using_xform:
        q, k, v = qkv.unbind(dim=2); dim_cat = 1; slow = False
    else:
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2; slow = True
    if self.attn_l2_norm:
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        if using_flash or self.using_xform:
            scale_mul = scale_mul.transpose(1, 2)
        q = F.normalize(q, dim=-1).mul(scale_mul)
        k = F.normalize(k, dim=-1)
    if self.caching:
        cfg = _KVQ
        if cfg is not None and cfg.enabled:
            si = cfg.l_to_si.get(L, None)
            kb, vb = cfg.bits_for(self.block_idx, si)
            if cfg.ef and slow:
                if kb < 16:
                    k = ef_quant_block(k, kb, cfg.sym)
                if vb < 16:
                    v = ef_quant_block(v, vb, cfg.sym)
            else:
                if kb < 16:
                    rd = -1 if cfg.per_k == 'token' else (-2 if slow else 1)
                    k = fake_quant(k, kb, rd, cfg.sym, cfg.stochastic)
                if vb < 16:
                    rd = -1 if cfg.per_v == 'token' else (-2 if slow else 1)
                    v = fake_quant(v, vb, rd, cfg.sym, cfg.stochastic)
        if self.cached_k is None:
            self.cached_k = k; self.cached_v = v
        else:
            k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
            v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
    dropout_p = 0.0
    if using_flash:
        from models.basic_var import flash_attn_func
        oup = flash_attn_func(q.to(main_type), k.to(main_type), v.to(main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
    elif self.using_xform:
        from models.basic_var import memory_efficient_attention
        oup = memory_efficient_attention(q.to(main_type), k.to(main_type), v.to(main_type), attn_bias=None, p=dropout_p, scale=self.scale).view(B, L, C)
    else:
        oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
    return self.proj_drop(self.proj(oup))


def apply_kv_quant(var, cfg):
    global _KVQ
    _KVQ = cfg
    for blk in var.blocks:
        attn = blk.attn
        if id(attn) not in _ORIG_FORWARDS:
            _ORIG_FORWARDS[id(attn)] = attn.forward
        attn.forward = types.MethodType(_patched_forward, attn)
    return var


def remove_kv_quant(var):
    global _KVQ
    _KVQ = None
    for blk in var.blocks:
        attn = blk.attn
        orig = _ORIG_FORWARDS.get(id(attn))
        if orig is not None:
            attn.forward = orig
    return var
