#!/usr/bin/env bash
# Offline errors for all 8 datasets, then quadratic T=100 sampling.
# Train missing checkpoints first:
#   ./script/train_raw_multi.sh
#   ./test_exp/script/run_17_raw_multi.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"

WEIGHTS=(equal dyn dyn_norm)
IFS=',' read -r -a NAMES_ARR <<< "$NAMES"
missing=0
for weight in "${WEIGHTS[@]}"; do
  for name in "${NAMES_ARR[@]}"; do
    ckpt="experiments/raw_multi/${weight}/${name}/best.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "missing $ckpt"
      missing=1
    fi
  done
done
if [[ "$missing" -eq 1 ]]; then
  echo "train first: ./script/train_raw_multi.sh"
  exit 1
fi

echo "exp=17_raw_multi names=$NAMES device=$DEVICE"
"$PY" -u test_exp/raw_multi_eval.py \
  --names "$NAMES" \
  --device "$DEVICE" \
  --out test_exp/results/17_raw_multi_sampling
"$PY" -u test_exp/sample_raw_multi.py \
  --names "$NAMES" \
  --device "$DEVICE" \
  --n_sample "$N_SAMPLE" \
  --sample_batch "${SAMPLE_BATCH:-4096}"
echo "done -> test_exp/results/17_raw_multi_sampling/"
