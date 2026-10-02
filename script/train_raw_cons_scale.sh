#!/usr/bin/env bash
# Soft consistency on dyn_norm data loss. Anchored is not trained.
#   L_data = w1*L1 + w2*L2 + w3*L3
#   w = (h, h^2/2, h^3/6) / (h + h^2/2 + h^3/6)
#
# Group 1, cons_weight=equal:
#   L = L_data + lambda * (delta2^2 + delta3^2)
#   lambda 0 is experiments/raw_multi/dyn_norm
#   lambda 1 is experiments/raw_cons_dyn_norm/soft
#   new lambdas: 0.1 and 1/3
#
# Group 2, cons_weight=dyn_norm:
#   L = L_data + lambda * (w2*delta2^2 + w3*delta3^2)
#   new lambdas: 1 and 0.3
#
# 4 datasets x 4 new runs = 16. Skip a run when best.pt exists.
# Val logs r = ||trunk grad L_cons|| / ||trunk grad L_data|| by gamma bin.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_raw_cons_scale.sh
# If a run is stopped before epoch 200, delete that directory before launching again.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

run_one() {
  local out="$1"
  local lam="$2"
  local cons_weight="$3"
  local name="$4"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${out} (best.pt exists) ========"
    return
  fi
  echo "======== train ${out} lambda=${lam} cons_weight=${cons_weight} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --k_max 3 \
    --moment_loss raw_multi \
    --raw_weight dyn_norm \
    --weight_steps 100 \
    --consistency soft \
    --lambda_cons "$lam" \
    --cons_weight "$cons_weight"
}

for name in "${DATASETS[@]}"; do
  run_one "experiments/raw_cons_scale/l0.1/${name}" 0.1 equal "$name"
  run_one "experiments/raw_cons_scale/l1o3/${name}" 0.3333333333333333 equal "$name"
done

for name in "${DATASETS[@]}"; do
  run_one "experiments/raw_cons_w/l1/${name}" 1 dyn_norm "$name"
  run_one "experiments/raw_cons_w/l0.3/${name}" 0.3 dyn_norm "$name"
done

echo "train done"
echo "reused lambda 0: experiments/raw_multi/dyn_norm/"
echo "reused lambda 1 unweighted: experiments/raw_cons_dyn_norm/soft/"
echo "new unweighted: experiments/raw_cons_scale/{l0.1,l1o3}/"
echo "new weighted consistency: experiments/raw_cons_w/{l1,l0.3}/"
