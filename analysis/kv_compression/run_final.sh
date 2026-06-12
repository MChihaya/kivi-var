#!/bin/bash
OUT=results/final; PC=$OUT/perclass; mkdir -p $PC
FC=919,921,530,916,922,917,664,796,281,259,488,489,595,980,437,562
# --- global eviction (stride8 ~INT3 mem, stride24 ~INT2 mem) ---
for st in 8 24; do
  echo "===== GLOBAL evict$st ====="
  EVICT_STRIDE=$st uv run python analysis/generate.py --classes all --n_per_class 10 --batch_size 50 --seed 0 \
      --layout flat --out_dir $OUT/gen_evict$st 2>&1 | grep -E "Evict|GEN_DONE"
  uv run python analysis/eval/compute_metrics.py --gen $OUT/gen_evict$st --real data/imagenet_val_256 \
      --fid --isc --cache_real_name imagenet_val256_50k --out $OUT/metrics_evict$st.json 2>&1 | grep -E "Frechet|Inception Score"
  rm -rf $OUT/gen_evict$st
  echo "--- global evict$st done ---"
done
# --- per-class (text/face) for the gaps + eviction ---
percq(){ name=$1; shift
  uv run python analysis/generate.py --classes $FC --n_per_class 50 --batch_size 50 --seed 0 --layout perclass --out_dir $PC/gen_$name "$@" 2>&1 | grep GEN_DONE
  uv run python analysis/failure_analysis/run_failure1.py --gpu --gen_root $PC/gen_$name --n 50 --out $PC/table_$name.json 2>&1 | grep SAVED
  rm -rf $PC/gen_$name; echo "--- perclass $name done ---"
}
perce(){ name=$1; st=$2
  EVICT_STRIDE=$st uv run python analysis/generate.py --classes $FC --n_per_class 50 --batch_size 50 --seed 0 --layout perclass --out_dir $PC/gen_$name 2>&1 | grep GEN_DONE
  uv run python analysis/failure_analysis/run_failure1.py --gpu --gen_root $PC/gen_$name --n 50 --out $PC/table_$name.json 2>&1 | grep SAVED
  rm -rf $PC/gen_$name; echo "--- perclass $name done ---"
}
percq uniform_int3 --kv_k_bits 3 --kv_v_bits 3
percq kivi_int3 --kv_k_bits 3 --kv_v_bits 3 --kv_per_k channel --kv_per_v token
perce evict8 8
perce evict24 24
echo FINAL_DONE
