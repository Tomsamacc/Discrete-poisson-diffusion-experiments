#!/usr/bin/env bash
# Eval and sample the consistency stage.
# Needs the 12 new checkpoints plus the existing equal and dyn_norm checkpoints.
# equal and dyn_norm WD/TV are read from experiment 17. They are not resampled.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./test_exp/script/run_18_ratio_cons.sh
set -euo pipefail

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
source "$(dirname "$0")/_common.sh"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)
MODES=(soft anchored hard)
missing=0
for mode in "${MODES[@]}"; do
  for name in "${DATASETS[@]}"; do
    f="experiments/raw_cons/${mode}/${name}/best.pt"
    if [[ ! -f "$f" ]]; then
      echo "missing $f"
      missing=1
    fi
  done
done
for weight in equal dyn_norm; do
  for name in "${DATASETS[@]}"; do
    f="experiments/raw_multi/${weight}/${name}/best.pt"
    if [[ ! -f "$f" ]]; then
      echo "missing $f"
      missing=1
    fi
  done
done
if [[ "$missing" -ne 0 ]]; then
  echo "finish ./script/train_raw_cons.sh before this eval"
  exit 1
fi

"$PY" -u test_exp/cons_eval.py --names "$NAMES" --device "$DEVICE"
"$PY" -u test_exp/sample_raw_cons.py \
  --names "$NAMES" \
  --kernels "$KERNELS" \
  --device "$DEVICE" \
  --n-sample "$N_SAMPLE"
echo "done -> test_exp/results/18_ratio_consistency/"
