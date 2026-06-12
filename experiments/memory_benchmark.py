"""Core result 2 -- the low-bit KV cache reduces both the stored cache and the peak GPU memory.

Stores the KV cache as packed low-bit integers (int8/int4/int3/int2) plus KIVI scales
and dequantizes only one layer at a time at attention time. Prints, per (precision x batch):
the stored cache size (MB) and the measured peak GPU memory (GB).

    uv run experiments/memory_benchmark.py

Expected (report Tables 6-7): int4 cache ~3.6x smaller than fp16; at batch 128 peak ~46.8 -> 27.2 GB;
batch 256 runs out of memory in fp16 but fits in int4/int3/int2 (try --batches 256 on an 80 GB GPU).
"""
import argparse
import gc
import time

ap = argparse.ArgumentParser()
ap.add_argument("--var-repo", default="VAR")
ap.add_argument("--ckpt-dir", default="checkpoints")
ap.add_argument("--bits", default="16,8,4,3,2")
ap.add_argument("--batches", default="16,64,128")
a = ap.parse_args()

import torch
from kivivar import real_quant as RQ
from kivivar.varload import load_var

vae, var, PN = load_var(a.var_repo, a.ckpt_dir)
bitlist = [int(b) for b in a.bits.split(",")]
batches = [int(b) for b in a.batches.split(",")]

# All configs (including the fp16 baseline, stored raw) go through the same dequant-on-read
# attention path, so the only difference between rows is how the KV cache is stored -- a fair
# memory comparison. total_cache_bytes() reports the actually stored cache for every config.
print("%-6s %6s  %14s  %8s  %6s" % ("prec", "batch", "cacheMB(stored)", "peakGB", "sec"))
print("-" * 50)
for bits in bitlist:
    name = "fp16" if bits >= 16 else "int%d" % bits
    RQ.apply_real_quant(var, RQ.RealKVConfig(k_bits=bits, v_bits=bits, per_k="channel", per_v="token"))
    for B in batches:
        RQ.reset_caches(var)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        lab = torch.randint(0, 1000, (B,), device="cuda")
        t0 = time.time()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                var.autoregressive_infer_cfg(B=B, label_B=lab, cfg=1.5, top_k=900, top_p=0.96, g_seed=0)
            sec = time.time() - t0
            cache_mb = RQ.total_cache_bytes(var) / 1e6
            peak = torch.cuda.max_memory_allocated() / 1e9
            print("%-6s %6d  %14.1f  %8.2f  %6.1f" % (name, B, cache_mb, peak, sec))
        except torch.cuda.OutOfMemoryError:
            print("%-6s %6d  %14s  %8s  %6s" % (name, B, "-", "OOM", "-"))
            gc.collect()
            torch.cuda.empty_cache()
    RQ.remove_real_quant(var)
    RQ.reset_caches(var)
