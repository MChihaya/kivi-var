#!/bin/bash
# Paper-protocol eval: FID/sFID/IS/Precision/Recall vs official VIRTUAL reference.
# Usage: bash run_adm_eval.sh <samples.npz> [label]
REF=data/fid_ref/VIRTUAL_imagenet256_labeled.npz
SAMPLES=$1
LABEL=${2:-eval}
echo "=== ADM eval: $LABEL ($SAMPLES) ==="
python guided-diffusion/evaluations/evaluator.py $REF $SAMPLES 2>&1 \
  | grep -E "Inception Score|^FID|sFID|Precision|Recall|Error"
echo "ADM_EVAL_DONE $LABEL"
