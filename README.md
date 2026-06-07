# KIVI-VAR: real low-bit KV-cache quantization for VAR

Source code to reproduce the **core experiments** of the report *"VAR における KV キャッシュ量子化の
失敗事例分析と省メモリ化の検討"*, on **VAR-d16**:

1. **KIVI** asymmetric quantization (Key per-channel, Value per-token) keeps generation quality
   closer to the FP16 output than uniform (per-token) quantization at the same bit width.
2. The **real packed-integer KV cache** actually reduces the stored cache size and the peak GPU
   memory (unlike a quantize-then-dequantize scheme, which would leave the cache in full precision).

The cache is stored as actually packed low-bit integers (int8/int4/int3/int2) plus small KIVI
scales, and is dequantized only one layer at a time at attention time.

## Setup (uv recommended)

Requires a CUDA GPU. Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
uv sync                  # create the environment and install deps (PyTorch cu121, lpips, ...)
bash scripts/prepare.sh  # clone the VAR code into ./VAR and download checkpoints into ./checkpoints
```

`scripts/prepare.sh` fetches `models.build_vae_var` from the official VAR repo and the two
checkpoints (`var_d16.pth`, `vae_ch160v4096z32.pth`) from the official Hugging Face repo.

## Run the core experiments

```bash
# 1. KIVI vs uniform quantization quality (LPIPS to the FP16 output; lower = closer to no quantization)
uv run experiments/kivi_vs_uniform.py

# 2. Real KV-cache size and peak GPU memory, per precision x batch
uv run experiments/memory_benchmark.py
```

### Expected results (from the report)

`kivi_vs_uniform.py` — KIVI keeps the generated *distribution* closer to the FP16 output than
uniform at the same bit width; the gap widens at INT2, where uniform collapses (50 classes,
16 images each; FID between each config's images and the FP16 images):

| bits | uniform FID→FP16↓ | KIVI FID→FP16↓ |
|------|-------------------|-----------------|
| INT3 | ~294              | **~291**        |
| INT2 | ~323              | **~303**        |

What reproduces is the **ordering** (KIVI < uniform, widening at INT2). FID is a distribution-level
metric, robust to the fact that quantization changes individual stochastic sampling trajectories
(so a per-image metric like LPIPS would be dominated by that noise). The absolute values are high
because this is a small demo sample comparing two *generated* sets — raise `--n-classes` for a
tighter estimate. The report's headline FID-vs-ImageNet (KIVI 7.12 < uniform 10.32 at INT3) is the
large-scale version of the same result. The script also writes the images to `out/`, so you can
see directly that INT2 uniform breaks the subject while INT2 KIVI keeps it.

`memory_benchmark.py` — the packed cache shrinks and the peak drops (measured on one 80 GB GPU):

| batch | fp16 cache | int4 cache | fp16 peak | int4 peak |
|-------|-----------|-----------|-----------|-----------|
| 64    | 5704 MB   | 1599 MB   | 24.3 GB   | 14.5 GB   |
| 128   | 11409 MB  | 3198 MB   | 46.8 GB   | 27.2 GB   |
| 256   | (OOM)     | 6396 MB   | OOM       | fits      |

The int4 cache is ~3.6× smaller than fp16; batch 256 runs out of memory in fp16 but fits when the
cache is packed (try `uv run experiments/memory_benchmark.py --batches 256` on an 80 GB GPU).

## Layout

- `kivivar/real_quant.py` — the real packed-int KV cache (core implementation; KIVI = Key
  per-channel, Value per-token; a generalized bit-packer handles INT3 with no waste).
- `kivivar/evict.py` — a scale-aware token-discard baseline (the report compares quantization
  against it under a matched memory budget).
- `kivivar/varload.py` — load VAR-d16 from a VAR checkout and its checkpoints.
- `experiments/` — the runnable core experiments.
- `scripts/prepare.sh` — fetch the VAR code and checkpoints.

The full study (failure-case analysis, scale ablation, token-discard comparison, and the
generalization to the Infinity-2B model) is described in the report.
