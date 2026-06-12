#!/bin/bash
# Scale-wise KV-quant sweep at a fixed bit-width. Usage: bash run_sweep_scales.sh <BIT>
BIT=${1:-3}
OUT=results/failure2
mkdir -p $OUT
NPC=10
run() {
  name=$1; shift
  echo "===== CONFIG $name ($@) ====="
  uv run python analysis/generate.py --classes all --n_per_class $NPC --batch_size 50 --seed 0 \
      --layout flat --out_dir $OUT/gen_$name "$@" 2>&1 | grep -E "KV-quant|peak_gpu|imgs_per_sec|GEN_DONE"
  uv run python analysis/eval/compute_metrics.py --gen $OUT/gen_$name --real data/imagenet_val_256 \
      --fid --isc --cache_real_name imagenet_val256_50k --out $OUT/metrics_$name.json 2>&1 | grep -E "Frechet|Inception Score"
  echo "--- $name done ---"
}
# early scales 0-4 (coarse, ~55 tokens) vs late scales 5-9 (fine, ~625 tokens) at fixed BIT
run early_b${BIT} --kv_k_bits $BIT --kv_v_bits $BIT --kv_scales 0-4
run late_b${BIT}  --kv_k_bits $BIT --kv_v_bits $BIT --kv_scales 5-9
echo "SWEEP_SCALES_DONE"
