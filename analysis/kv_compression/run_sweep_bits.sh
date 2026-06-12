#!/bin/bash
OUT=results/failure2
mkdir -p $OUT
NPC=10              # images per class (x1000 classes = 10k)
run() {
  name=$1; shift
  echo "===== CONFIG $name ($@) ====="
  uv run python analysis/generate.py --classes all --n_per_class $NPC --batch_size 50 --seed 0 \
      --layout flat --out_dir $OUT/gen_$name "$@" 2>&1 | grep -E "KV-quant|peak_gpu|imgs_per_sec|GEN_DONE"
  uv run python analysis/eval/compute_metrics.py --gen $OUT/gen_$name --real data/imagenet_val_256 \
      --fid --isc --cache_real_name imagenet_val256_50k --out $OUT/metrics_$name.json 2>&1 | grep -E "Frechet|Inception Score"
  echo "--- $name done ---"
}
run fp16
run int8 --kv_k_bits 8 --kv_v_bits 8
run int4 --kv_k_bits 4 --kv_v_bits 4
run int3 --kv_k_bits 3 --kv_v_bits 3
run int2 --kv_k_bits 2 --kv_v_bits 2
echo "SWEEP_BITS_DONE"
