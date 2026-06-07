"""Core result 1 -- KIVI beats uniform quantization on quality.

KIVI (Key per-channel, Value per-token) keeps the generated *distribution* closer to the FP16
(no-quantization) output than uniform (per-token) quantization at the same bit width. The cache
is the REAL packed-int KV cache (kivivar.real_quant). The metric is FID between each config's
images and the FP16 images -- a distribution-level metric, robust to the fact that quantization
changes individual stochastic sampling trajectories (so a per-image metric like LPIPS is noisy).

    uv run experiments/kivi_vs_uniform.py

Expected: FID(KIVI, fp16) < FID(uniform, fp16) at both INT3 and INT2; at INT2 uniform collapses
while KIVI stays far closer.
"""
import argparse
import os
import os.path as osp

ap = argparse.ArgumentParser()
ap.add_argument("--var-repo", default="VAR")
ap.add_argument("--ckpt-dir", default="checkpoints")
ap.add_argument("--n-classes", type=int, default=50, help="number of ImageNet classes")
ap.add_argument("--n", type=int, default=16, help="images per class")
ap.add_argument("--bits", default="3,2")
ap.add_argument("--out", default="out/kivi_vs_uniform")
a = ap.parse_args()

import torch
from kivivar import real_quant as RQ
from kivivar.varload import load_var
from PIL import Image

vae, var, PN = load_var(a.var_repo, a.ckpt_dir)
classes = list(range(0, 1000, max(1, 1000 // a.n_classes)))[: a.n_classes]
bitlist = [int(b) for b in a.bits.split(",")]


def generate(name, k_bits, per_k, per_v):
    if k_bits < 16:
        RQ.apply_real_quant(var, RQ.RealKVConfig(k_bits=k_bits, v_bits=k_bits, per_k=per_k, per_v=per_v))
    d = osp.join(a.out, name)
    os.makedirs(d, exist_ok=True)
    for ci in classes:
        lab = torch.full((a.n,), ci, dtype=torch.long, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            img = var.autoregressive_infer_cfg(B=a.n, label_B=lab, cfg=1.5, top_k=900, top_p=0.96, g_seed=ci)
        arr = img.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        for j in range(a.n):
            Image.fromarray(arr[j]).save(osp.join(d, "%04d_%02d.png" % (ci, j)))
    if k_bits < 16:
        RQ.remove_real_quant(var)
        RQ.reset_caches(var)
    print("generated", name, "(%d images)" % (len(classes) * a.n), flush=True)


generate("fp16", 16, "channel", "token")
for b in bitlist:
    generate("uniform_int%d" % b, b, "token", "token")
    generate("kivi_int%d" % b, b, "channel", "token")

from torch_fidelity import calculate_metrics

print("\n%-7s %-9s  %s" % ("bits", "method", "FID to FP16 (lower = distribution closer to no quantization)"))
print("-" * 70)
for b in bitlist:
    fids = {}
    for name in ("uniform", "kivi"):
        m = calculate_metrics(input1=osp.join(a.out, "%s_int%d" % (name, b)), input2=osp.join(a.out, "fp16"),
                              fid=True, verbose=False)
        fids[name] = m["frechet_inception_distance"]
        print("INT%-4d %-9s  %.2f" % (b, name, fids[name]))
    win = "KIVI" if fids["kivi"] < fids["uniform"] else "uniform"
    print("        -> distribution closer to FP16: %s\n" % win)
