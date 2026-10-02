#!/usr/bin/env bash
# Eval and sample dyn_norm consistency.
# Samples only soft and anchored under experiments/raw_cons_dyn_norm/.
# dyn_norm, dyn_norm_proj, hard, eq_soft, and eq_anchored are copied.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./test_exp/script/run_19_ratio_cons_dyn_norm.sh
set -euo pipefail

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
source "$(dirname "$0")/_common.sh"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)
MODES=(soft anchored)
missing=0
for mode in "${MODES[@]}"; do
  for name in "${DATASETS[@]}"; do
    f="experiments/raw_cons_dyn_norm/${mode}/${name}/best.pt"
    if [[ ! -f "$f" ]]; then
      echo "missing $f"
      missing=1
    fi
  done
done
if [[ "$missing" -ne 0 ]]; then
  echo "finish ./script/train_raw_cons_dyn_norm.sh before this eval"
  exit 1
fi

"$PY" -u test_exp/suite19.py \
  --names "$NAMES" \
  --kernels "$KERNELS" \
  --device "$DEVICE" \
  --n-sample "$N_SAMPLE"
echo "done -> test_exp/results/19_ratio_cons_dyn_norm/"
