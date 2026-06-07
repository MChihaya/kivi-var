"""REAL low-bit KV cache for VAR (not fake-quant): the cache is actually stored as
packed low-bit integers (int8 / int4 / int2) plus KIVI scales, and is dequantized
only transiently, per layer, at attention time. This genuinely reduces the persistent
KV-cache memory (and the measured peak at large batch), unlike fake quantization.

KIVI scheme: Key quantized per-channel (scale shared over tokens), Value per-token
(scale shared over channels) — identical numerics to kv_compression.quant.fake_quant,
so generated images match the fake-quant pipeline.

Forces the slow (SDPA) attention path because the cache is materialized on read.
Controlled by RealKVConfig set via apply_real_quant(var, cfg).
"""
import math
import types
import torch
import torch.nn.functional as F

try:
    from models.basic_var import slow_attn
except Exception:
    slow_attn = None

_CFG = None
_ORIG = {}


class RealKVConfig:
    def __init__(self, k_bits=16, v_bits=16, per_k='channel', per_v='token'):
        self.k_bits = k_bits; self.v_bits = v_bits
        self.per_k = per_k; self.per_v = per_v


# ---------- quantize / pack ----------
def _quant(x, bits, reduce_dim):
    """Asymmetric quantize x along reduce_dim. Returns (q_uint8_levels, mn, scale)."""
    xf = x.float()
    mn = xf.amin(dim=reduce_dim, keepdim=True)
    mx = xf.amax(dim=reduce_dim, keepdim=True)
    qmax = (1 << bits) - 1
    scale = (mx - mn).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round((xf - mn) / scale), 0, qmax).to(torch.uint8)
    return q, mn.to(torch.float16), scale.to(torch.float16)


def _dequant(q, mn, scale):
    return (q.to(torch.float16) * scale + mn)


def _pack(q, bits):
    """Pack uint8 levels in [0,2^bits-1] into a compact uint8 buffer, bit-packed at exactly
    `bits` bits/value with no waste even when bits does not divide 8 (e.g. INT3 packs 8 values
    into 3 bytes). Backward-compatible with bits in {2,4,8}."""
    if bits == 8:
        return q.contiguous(), tuple(q.shape)
    g = math.gcd(bits, 8)
    vals_per_group = 8 // g       # values that fill a whole number of bytes
    bytes_per_group = bits // g   # bytes they occupy (vals_per_group*bits == bytes_per_group*8)
    flat = q.reshape(-1).to(torch.int32)
    n = flat.numel()
    pad = (-n) % vals_per_group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    grp = flat.view(-1, vals_per_group)
    vshift = torch.arange(vals_per_group, device=q.device, dtype=torch.int32) * bits
    code = (grp << vshift).sum(dim=1)                       # [G] integer holding the group's bits
    bshift = torch.arange(bytes_per_group, device=q.device, dtype=torch.int32) * 8
    packed = ((code.unsqueeze(1) >> bshift) & 0xFF).to(torch.uint8).reshape(-1)
    return packed, tuple(q.shape)


def _unpack(packed, bits, shape):
    if bits == 8:
        return packed.view(shape)
    g = math.gcd(bits, 8)
    vals_per_group = 8 // g
    bytes_per_group = bits // g
    mask = (1 << bits) - 1
    p = packed.to(torch.int32).view(-1, bytes_per_group)
    bshift = torch.arange(bytes_per_group, device=packed.device, dtype=torch.int32) * 8
    code = (p << bshift).sum(dim=1)                          # reconstruct the group integer
    vshift = torch.arange(vals_per_group, device=packed.device, dtype=torch.int32) * bits
    out = ((code.unsqueeze(1) >> vshift) & mask).reshape(-1)
    n = 1
    for s in shape:
        n *= s
    return out[:n].view(shape).to(torch.uint8)


class _LayerCache:
    """Per-layer KV cache stored as a list of per-scale packed blocks + scales."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.k_blocks = []  # each: (packed, shape, mn, scale)
        self.v_blocks = []

    def _add(self, blocks, x, bits, reduce_dim):
        if bits >= 16:
            blocks.append(('raw', x.to(torch.float16)))
        else:
            q, mn, scale = _quant(x, bits, reduce_dim)
            packed, shape = _pack(q, bits)
            blocks.append((packed, shape, mn, scale))

    def append(self, k, v):
        # slow path shapes: [B, H, L, c]; channel -> reduce over tokens(-2), token -> reduce over channels(-1).
        # KIVI = (per_k=channel, per_v=token); uniform = (per_k=token, per_v=token).
        kd = -2 if self.cfg.per_k == "channel" else -1
        vd = -2 if self.cfg.per_v == "channel" else -1
        self._add(self.k_blocks, k, self.cfg.k_bits, kd)
        self._add(self.v_blocks, v, self.cfg.v_bits, vd)

    def _read(self, blocks, bits):
        outs = []
        for b in blocks:
            if b[0] == 'raw':
                outs.append(b[1])
            else:
                packed, shape, mn, scale = b
                outs.append(_dequant(_unpack(packed, bits, shape), mn, scale))
        return torch.cat(outs, dim=-2)

    def read(self):
        return self._read(self.k_blocks, self.cfg.k_bits), self._read(self.v_blocks, self.cfg.v_bits)

    def cache_bytes(self):
        tot = 0
        for blocks in (self.k_blocks, self.v_blocks):
            for b in blocks:
                if b[0] == 'raw':
                    tot += b[1].numel() * b[1].element_size()
                else:
                    packed, shape, mn, scale = b
                    tot += packed.numel() * packed.element_size()
                    tot += mn.numel() * mn.element_size() + scale.numel() * scale.element_size()
        return tot


def _patched_forward(self, x, attn_bias):
    from models.basic_var import slow_attn
    B, L, C = x.shape
    qkv = F.linear(input=x, weight=self.mat_qkv.weight,
                   bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                   ).view(B, L, 3, self.num_heads, self.head_dim)
    main_type = qkv.dtype
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # [B,H,L,c] (slow path)
    if self.attn_l2_norm:
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q = F.normalize(q, dim=-1).mul(scale_mul)
        k = F.normalize(k, dim=-1)
    if self.caching:
        if getattr(self, '_qcache', None) is None:
            self._qcache = _LayerCache(_CFG)
        self._qcache.append(k, v)
        k, v = self._qcache.read()
    oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=0.0
                    ).transpose(1, 2).reshape(B, L, C)
    return self.proj_drop(self.proj(oup))


def apply_real_quant(var, cfg):
    global _CFG
    _CFG = cfg
    for blk in var.blocks:
        attn = blk.attn
        if id(attn) not in _ORIG:
            _ORIG[id(attn)] = attn.forward
        attn._qcache = None
        attn.forward = types.MethodType(_patched_forward, attn)
    return var


def reset_caches(var):
    for blk in var.blocks:
        blk.attn._qcache = None


def total_cache_bytes(var):
    tot = 0
    for blk in var.blocks:
        qc = getattr(blk.attn, '_qcache', None)
        if qc is not None:
            tot += qc.cache_bytes()
    return tot


def remove_real_quant(var):
    global _CFG
    _CFG = None
    for blk in var.blocks:
        attn = blk.attn
        if id(attn) in _ORIG:
            attn.forward = _ORIG[id(attn)]
        if hasattr(attn, '_qcache'):
            del attn._qcache
    return var
