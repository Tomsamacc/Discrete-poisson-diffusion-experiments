#!/usr/bin/env bash
# PMF-ratio k_max=1 then 2. Same config as config_pmfratio.yml.
# k_max=3 already lives in experiments/pmfratio/{name}/ — skip if best.pt exists.
# Does not touch experiments/scale.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_pmfratio_kmax.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

train_one() {
  local k="$1"
  local name="$2"
  local out
  if [[ "$k" == "3" ]]; then
    out="experiments/pmfratio/${name}"
  else
    out="experiments/pmfratio/kmax${k}/${name}"
  fi
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} k_max=${k} (best.pt exists) ========"
    return
  fi
  echo "======== train ${name} pmfratio k_max=${k} -> ${out} ========"
  "$PY" -u train.py -c config_pmfratio.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --ratio true \
    --k_max "$k"
}

for k in 1 2 3; do
  for name in "${DATASETS[@]}"; do
    train_one "$k" "$name"
  done
done
echo "train done -> experiments/pmfratio/kmax{1,2}/ and experiments/pmfratio/{name}/"
echo "then: ./test_exp/script/run_12_kmax.sh"
