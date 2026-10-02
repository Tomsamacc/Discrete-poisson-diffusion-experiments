#!/usr/bin/env bash
# Single-head moment PRL. Same config as scale / A.
#   k=1,2,3 × {direct softplus, ratio D_k e^b}
#   plus k=1 raw: (z+1) e^f / γ
# No sampling. Skip if best.pt exists.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_direct_prl.sh
#   ./test_exp/script/run_14_direct.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

train_one() {
  local param="$1" k="$2" name="$3"
  local out="experiments/direct_prl/${param}/k${k}/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${param} k=${k} ${name} (best.pt exists) ========"
    return
  fi
  echo "======== train ${param} k=${k} ${name} -> ${out} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --moment_k "$k" \
    --moment_param "$param"
}

for param in direct ratio; do
  for k in 1 2 3; do
    for name in "${DATASETS[@]}"; do
      train_one "$param" "$k" "$name"
    done
  done
done
for name in "${DATASETS[@]}"; do
  train_one raw 1 "$name"
done
echo "train done -> experiments/direct_prl/"
echo "then: ./test_exp/script/run_14_direct.sh"
