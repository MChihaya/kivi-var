"""KV-cache fake-quantization for Infinity (text-to-image VAR-family).
Replaces Infinity SelfAttention.forward to quantize new K/V (after rope, before
caching cat) so even the final scales' quantization affects attention.
fake_quant is inlined (identical to kv_compression.quant.fake_quant) so this module
has no dependency on the VAR repo. Controlled by env vars:
  INFQ_BITS (16=off), INFQ_PERK/INFQ_PERV (token|channel), INFQ_STOCH (0/1).
KIVI = Key per-channel (reduce over seq dim), Value per-token (reduce over channel dim).
"""
import os
import types
import torch
import torch.nn.functional as F


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


_BITS = 16
_PERK = 'token'
_PERV = 'token'
_STOCH = False
_SYM = False
_ORIG = {}


def _cfg_from_env():
    global _BITS, _PERK, _PERV, _STOCH
    _BITS = int(os.environ.get('INFQ_BITS', '16'))
    _PERK = os.environ.get('INFQ_PERK', 'token')
    _PERV = os.environ.get('INFQ_PERV', 'token')
    _STOCH = bool(int(os.environ.get('INFQ_STOCH', '0')))


def _patched_forward(self, x, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None,
                     rope2d_freqs_grid=None, scale_ind=0):
    import infinity.models.basic as _b
    B, L, C = x.shape
    qkv = F.linear(input=x, weight=self.mat_qkv.weight,
                   bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                   ).view(B, L, 3, self.num_heads, self.head_dim)
    if self.using_flash:
        q, k, v = qkv.unbind(dim=2); L_dim = 1
    else:
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2
    if self.cos_attn:
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
        v = v.contiguous()
    else:
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    if rope2d_freqs_grid is not None:
        q, k = _b.apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid,
                                   self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)
    # ---- quantize NEW k,v before caching (so current and future scales see quantized cache) ----
    if _BITS < 16:
        rdk = -1 if _PERK == 'token' else L_dim
        rdv = -1 if _PERV == 'token' else L_dim
        k = fake_quant(k, _BITS, rdk, _SYM, _STOCH)
        v = fake_quant(v, _BITS, rdv, _SYM, _STOCH)
    if self.caching:
        if self.cached_k is None:
            self.cached_k = k; self.cached_v = v
        else:
            k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim)
            v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
    if self.using_flash:
        if attn_bias_or_two_vector is not None:
            kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
        else:
            kw = dict()
        oup = _b.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
    else:
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            oup = _b.slow_attn(query=q, key=k, value=v, scale=self.scale,
                               attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
    return self.proj_drop(self.proj(oup))


def apply_inf_quant(model):
    _cfg_from_env()
    for m in model.modules():
        if type(m).__name__ == 'SelfAttention' and hasattr(m, 'kv_caching'):
            if id(m) not in _ORIG:
                _ORIG[id(m)] = m.forward
            m.forward = types.MethodType(_patched_forward, m)
    return dict(bits=_BITS, per_k=_PERK, per_v=_PERV, stoch=_STOCH)


def remove_inf_quant(model):
    for m in model.modules():
        if id(m) in _ORIG:
            m.forward = _ORIG[id(m)]
    return model
