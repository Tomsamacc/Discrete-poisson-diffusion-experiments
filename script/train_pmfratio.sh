#!/usr/bin/env bash
# PMF-ratio k=1,2,3. 不覆盖 experiments/scale。
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_pmfratio.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

for name in "${DATASETS[@]}"; do
  out="experiments/pmfratio/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} (best.pt exists) ========"
    continue
  fi
  echo "======== train ${name} pmfratio k=1,2,3 -> ${out} ========"
  "$PY" -u train.py -c config_pmfratio.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --ratio true \
    --k_max 3
done
echo "train done -> experiments/pmfratio/"
