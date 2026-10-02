#!/usr/bin/env bash
# Two single-head links. Both use L = D(X, m). Four datasets.
#   soft_ratio: S = softplus(f),  m = (z+1)/gamma * S
#   offset:     m = exp(b)
# Same config as scale. Skip a run when best.pt exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

run_one() {
  local kind="$1"
  local name="$2"
  local out="experiments/link/${kind}/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${out} (best.pt exists) ========"
    return
  fi
  echo "======== train ${out} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --k_max 1 \
    --moment_loss "$kind"
}

for name in "${DATASETS[@]}"; do
  run_one soft_ratio "$name"
done
for name in "${DATASETS[@]}"; do
  run_one offset "$name"
done

echo "train done -> experiments/link/"
echo "then: ./test_exp/script/run_25_link.sh"
