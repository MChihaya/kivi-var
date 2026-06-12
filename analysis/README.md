# Reproducing the report, item by item

This folder contains the analysis scripts behind **every table and figure** of the report.
They are the scripts that actually produced the reported numbers, ported from the working
tree as-is — only machine-local paths were replaced (ImageNet location now comes from
environment variables); the logic is untouched.

## Setup

```bash
uv sync --extra analysis     # core deps + easyocr / insightface / lpips / matplotlib
bash scripts/prepare.sh      # VAR code into ./VAR, checkpoints into ./checkpoints

export IMAGENET_VAL=/path/to/imagenet/val      # analyses that read real images
export IMAGENET_TRAIN=/path/to/imagenet/train  # per-class real references

uv run python analysis/eval/prep_ref.py        # build data/imagenet_val_256 for global FID
mkdir -p figures_out
```

Run everything from the repo root, on one A100-80GB-class GPU. The global-FID sweeps
(10k images per config) take tens of minutes per config; the other analyses take minutes.

## Shared pipeline

| script | role |
|---|---|
| `analysis/generate.py` | class-conditional generation with KV-quantization flags (`--kv_k_bits/--kv_v_bits`, axes `--kv_per_k/--kv_per_v`, per-scale mixed precision `--kv_scale_bits "0-4:8,5-9:2"`, scale/layer masks) and token discard via `EVICT_STRIDE` |
| `analysis/kv_compression/quant.py` | KV quantization hooks (asymmetric round-to-nearest; KIVI axes; per-scale bits; also the stochastic / error-feedback variants that the report's limitations section mentions as tried and rejected) |
| `analysis/kv_compression/evict.py` | scale-aware token discard (keep early scales, stride-subsample late scales) |
| `analysis/eval/compute_metrics.py` | torch-fidelity FID wrapper (caches the real statistics under `data/fid_cache`) |

## Item → command

| report item | how to reproduce |
|---|---|
| §3 paper-protocol FID (ADM evaluator, 50k) | `uv run python analysis/generate.py --classes all --n_per_class 50 --out_dir results/base50k` → `uv run python analysis/eval/pack_npz.py results/base50k samples.npz` → clone `openai/guided-diffusion` into the repo root, download `VIRTUAL_imagenet256_labeled.npz` into `data/fid_ref/`, then `bash analysis/eval/run_adm_eval.sh samples.npz` (needs a TensorFlow env) |
| Table 2 (OCR, real vs generated) and the face robustness claim | `uv run python analysis/eval/prep_perclass_ref.py` → generate the text/face classes with `analysis/generate.py --classes 919,921,530,916,922,917,664,796,281,259,488,489,595,980,437,562 --layout perclass` → `uv run python analysis/failure_analysis/run_failure1.py --gpu --gen_root <gen dir> --out <json>` |
| Fig 2 (high-frequency scatter) | `uv run python analysis/perclass_fid.py` → `uv run python analysis/hf_analysis.py` → `uv run python analysis/build_hf_fig.py` |
| Fig 3 (phase/amplitude swap) | `uv run python analysis/phase_structure_exp.py` → `uv run python analysis/build_phase_fig.py` |
| Table 3 (uniform and scale-wise bit sweep, global FID / 10k) | `bash analysis/kv_compression/run_sweep_bits.sh` → `bash analysis/kv_compression/run_sweep_scales.sh 3` → `bash analysis/kv_compression/run_sweep_scales.sh 2` |
| Fig 4 (collapse grid) | `uv run python analysis/build_figs.py` (montages from the sweep generations; also rebuilds panels used in earlier drafts) |
| §4.2 blocking experiment (early-scale damage comes from its own generation, not propagation) | `uv run python analysis/failure2_decomp.py` |
| §4.2 class-wise collapse (faces/text at INT2) | `bash analysis/kv_compression/run_failure2_phaseB.sh` → `uv run python analysis/failure_analysis/summarize_failure2.py` |
| Table 4 (quantization-direction errors) | `uv run python analysis/kv_dist.py` |
| §5 scale ablation (late-only 0.583 / early-only 0.556 / KIVI 0.552) and the gain-side blocking | `uv run python analysis/kivi_scale_ablation.py` → `uv run python analysis/propagation_exp.py` |
| Table 5 (matched-budget: uniform / KIVI / mixed precision / token discard) | `bash analysis/kv_compression/run_improve.sh` → `bash analysis/kv_compression/run_final.sh` |
| §6.1 quantization+discard combination | `uv run python analysis/quant_evict_combo.py` → `uv run python analysis/smart_combo.py` |
| Tables 6–7 (stored cache size, peak GPU memory) | `uv run experiments/memory_benchmark.py` (see the repo README) |
| Table 8 VAR rows; Fig 5 left panel | `bash analysis/run_good_classes.sh` → `uv run python analysis/var_lpips_good.py` → `uv run python analysis/build_good_fig.py` |
| Table 8 Infinity rows; Fig 5 right panel | see **Infinity-2B** below |
| Fig 5 (combined) | `uv run python analysis/combine_quality_fig.py` |
| Fig 1 | Figure 4 of the original VAR paper, reproduced with attribution — no script |

## Infinity-2B

The Infinity experiments need their own environment (the official
[FoundationVision/Infinity](https://github.com/FoundationVision/Infinity) requirements:
torch 2.5.x, flash-attn, ...) — they do not run in this repo's uv environment:

```bash
git clone https://github.com/FoundationVision/Infinity.git   # into the repo root
# per the Infinity README, download into ./checkpoints:
#   checkpoints/infinity/infinity_2b_reg.pth
#   checkpoints/infinity/infinity_vae_d32reg.pth
#   checkpoints/flan-t5-xl
TORCHDYNAMO_DISABLE=1 <infinity-venv>/bin/python analysis/run_infinity_all.py
uv run python analysis/inf_metrics.py results/infinity   # LPIPS / PSNR (+ OCR) vs fp16
uv run python analysis/inf_lpips_std.py                  # per-image LPIPS mean +- std
uv run python analysis/build_inf_fig.py
```

`analysis/kv_compression/inf_quant.py` is the Infinity-side KV quantization hook
(env-controlled: `INFQ_BITS`, `INFQ_PERK`, `INFQ_PERV`).

## Notes

- Analyses that read real images require ImageNet-1k locally (val; train for the per-class
  references). The dataset cannot be redistributed here.
- `analysis/kv_compression/quant.py` quantizes K/V on write before caching; its quantizer is
  numerically the same as `kivivar/real_quant.py`. The packed storage in `kivivar/` is what
  realizes the memory savings of Tables 6–7.
- The sweep shell scripts also print Inception Score; the report uses FID only.
