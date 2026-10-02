#!/usr/bin/env bash
# Raw multi-output, three losses, same net / seed / data / optimizer.
#   equal:     L1 + L2 + L3
#   dyn:       h L1 + (h^2/2) L2 + (h^3/6) L3
#   dyn_norm:  that sum divided by W = h + h^2/2 + h^3/6
# h is the quadratic T=100 reverse step at the sampled gamma.
# Skip if best.pt exists. No sampling.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_raw_multi.sh
#   ./test_exp/script/run_17_raw_multi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3 pois20 poissmix zip yule_simon)
WEIGHTS=(equal dyn dyn_norm)

for weight in "${WEIGHTS[@]}"; do
  for name in "${DATASETS[@]}"; do
    out="experiments/raw_multi/${weight}/${name}"
    mkdir -p "$out"
    if [[ -f "${out}/best.pt" ]]; then
      echo "======== skip raw multi ${weight} ${name} (best.pt exists) ========"
      continue
    fi
    echo "======== train raw multi ${weight} ${name} -> ${out} ========"
    "$PY" -u train.py -c config_scale.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true \
      --k_max 3 \
      --moment_loss raw_multi \
      --raw_weight "$weight" \
      --weight_steps 100
  done
done
echo "train done -> experiments/raw_multi/{equal,dyn,dyn_norm}/"
echo "then: ./test_exp/script/run_17_raw_multi.sh"
