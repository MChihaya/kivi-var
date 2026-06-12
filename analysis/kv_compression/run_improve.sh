#!/bin/bash
# Phase 5 improvement: KIVI-asymmetric + scale-aware (protect coarse) vs naive uniform.
OUT=results/improve; PC=$OUT/perclass; mkdir -p $PC
FC=919,921,530,916,922,917,664,796,281,259,488,489,595,980,437,562

glob() {
  name=$1; shift
  echo "===== GLOBAL $name ($@) ====="
  uv run python analysis/generate.py --classes all --n_per_class 10 --batch_size 50 --seed 0 \
      --layout flat --out_dir $OUT/gen_$name "$@" 2>&1 | grep -E "KV-quant|GEN_DONE"
  uv run python analysis/eval/compute_metrics.py --gen $OUT/gen_$name --real data/imagenet_val_256 \
      --fid --isc --cache_real_name imagenet_val256_50k --out $OUT/metrics_$name.json 2>&1 | grep -E "Frechet|Inception Score"
  rm -rf $OUT/gen_$name
  echo "--- $name done ---"
}
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

# matched-budget global comparisons (vs uniform int3=10.32, int2=114.6 from failure2)
glob kivi_int3     --kv_k_bits 3 --kv_v_bits 3 --kv_per_k channel --kv_per_v token
glob kivi_int2     --kv_k_bits 2 --kv_v_bits 2 --kv_per_k channel --kv_per_v token
glob sa_int2       --kv_scale_bits 0-4:8,5-9:2
glob sa_kivi_int2  --kv_scale_bits 0-4:8,5-9:2 --kv_per_k channel --kv_per_v token
# per-class failure recovery (vs uniform int2: faces 0.03, chars 0.22)
perc kivi_int2     --kv_k_bits 2 --kv_v_bits 2 --kv_per_k channel --kv_per_v token
perc sa_kivi_int2  --kv_scale_bits 0-4:8,5-9:2 --kv_per_k channel --kv_per_v token
echo IMPROVE_DONE
