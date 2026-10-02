#!/usr/bin/env bash
# I. Single-head raw PMF ratio, full Bregman D(R_k, S_hat).
# S_hat = exp(f), R_k = (gamma X)^k / (z+1)^up k. k=1,2,3 trained separately.
# Same hparams as scale: lambda=100, z_rescale, 200 epoch, Adam 1e-3, bs 256.
# No sampling. Skip if best.pt exists.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_raw_ratio.sh
#   ./test_exp/script/run_15_raw.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

for k in 1 2 3; do
  for name in "${DATASETS[@]}"; do
    out="experiments/raw_ratio/k${k}/${name}"
    mkdir -p "$out"
    if [[ -f "${out}/best.pt" ]]; then
      echo "======== skip raw k=${k} ${name} (best.pt exists) ========"
      continue
    fi
    echo "======== train raw k=${k} ${name} -> ${out} ========"
    "$PY" -u train.py -c config_scale.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true \
      --moment_k "$k" \
      --moment_loss raw_ratio
  done
done
echo "train done -> experiments/raw_ratio/"
echo "then: ./test_exp/script/run_15_raw.sh"
