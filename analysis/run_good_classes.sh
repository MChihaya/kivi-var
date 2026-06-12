#!/bin/bash
PY=${PY:-"uv run python"}
CL="207,388,291,340,817,779,963,437,980,985"
N=16
COMMON="--classes $CL --n_per_class $N --batch_size $N --seed 0 --layout perclass --cfg 1.5"
echo "=== fp16 ==="
$PY analysis/generate.py $COMMON --out_dir results/good_classes/fp16
echo "=== uni_int3 ==="
$PY analysis/generate.py $COMMON --kv_k_bits 3 --kv_v_bits 3 --kv_per_k token --kv_per_v token --out_dir results/good_classes/uni_int3
echo "=== kivi_int3 ==="
$PY analysis/generate.py $COMMON --kv_k_bits 3 --kv_v_bits 3 --kv_per_k channel --kv_per_v token --out_dir results/good_classes/kivi_int3
echo "=== uni_int2 ==="
$PY analysis/generate.py $COMMON --kv_k_bits 2 --kv_v_bits 2 --kv_per_k token --kv_per_v token --out_dir results/good_classes/uni_int2
echo "=== kivi_int2 ==="
$PY analysis/generate.py $COMMON --kv_k_bits 2 --kv_v_bits 2 --kv_per_k channel --kv_per_v token --out_dir results/good_classes/kivi_int2
echo "GOOD_CLASSES_DONE"
