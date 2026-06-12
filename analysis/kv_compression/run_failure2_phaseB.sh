#!/bin/bash
OUT=results/failure2; PC=$OUT/perclass; mkdir -p $PC
FC=919,921,530,916,922,917,664,796,281,259,488,489,595,980,437,562

# ---- A) scale-wise GLOBAL FID/IS: early(0-4) vs late(5-9) at INT3 and INT2 ----
glob() {
  name=$1; shift
  echo "===== GLOBAL $name ($@) ====="
  uv run python analysis/generate.py --classes all --n_per_class 10 --batch_size 50 --seed 0 \
      --layout flat --out_dir $OUT/gen_$name "$@" 2>&1 | grep -E "KV-quant|peak_gpu|GEN_DONE"
  uv run python analysis/eval/compute_metrics.py --gen $OUT/gen_$name --real data/imagenet_val_256 \
      --fid --isc --cache_real_name imagenet_val256_50k --out $OUT/metrics_$name.json 2>&1 | grep -E "Frechet|Inception Score"
  rm -rf $OUT/gen_$name
  echo "--- $name done ---"
}
glob early_b3 --kv_k_bits 3 --kv_v_bits 3 --kv_scales 0-4
glob late_b3  --kv_k_bits 3 --kv_v_bits 3 --kv_scales 5-9
glob early_b2 --kv_k_bits 2 --kv_v_bits 2 --kv_scales 0-4
glob late_b2  --kv_k_bits 2 --kv_v_bits 2 --kv_scales 5-9

# ---- B) per-class failure sensitivity (OCR/face/per-class-FID) ----
perc() {
  name=$1; shift
  echo "===== PERCLASS $name ($@) ====="
  uv run python analysis/generate.py --classes $FC --n_per_class 100 --batch_size 50 --seed 0 \
      --layout perclass --out_dir $PC/gen_$name "$@" 2>&1 | grep -E "KV-quant|GEN_DONE"
  uv run python analysis/failure_analysis/run_failure1.py --gpu --gen_root $PC/gen_$name --n 100 \
      --out $PC/table_$name.json 2>&1 | grep -E "SAVED|Traceback"
  rm -rf $PC/gen_$name
  echo "--- $name done ---"
}
perc fp16
perc int2     --kv_k_bits 2 --kv_v_bits 2
perc early_b3 --kv_k_bits 3 --kv_v_bits 3 --kv_scales 0-4
perc late_b3  --kv_k_bits 3 --kv_v_bits 3 --kv_scales 5-9
echo "PHASEB_DONE"
