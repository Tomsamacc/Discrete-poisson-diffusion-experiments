#!/usr/bin/env bash
# Single-head raw ratio with the balanced loss
#   L = (z+1)/gamma * D(R, exp(a))
#   R = gamma X / (z+1)
#   a is the network output, S = exp(a)
# Compare with experiments/direct_prl/direct/k1, whose loss is D(X, m).
# 4 datasets. Skip a run when best.pt exists.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_ratio_bal.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

for name in "${DATASETS[@]}"; do
  out="experiments/ratio_bal/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${out} (best.pt exists) ========"
    continue
  fi
  echo "======== train ${out} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --k_max 1 \
    --moment_k 1 \
    --moment_param raw \
    --moment_loss raw_bal
done

echo "train done -> experiments/ratio_bal/"
echo "then: ./test_exp/script/run_24_ratio_bal.sh"
