#!/usr/bin/env bash
# 前台跑：
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./train_h05.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p experiments/h05

DATASETS=(gamma_ltj nb poismix_mod pois20 poissmix poissmix3 zip yule_simon)
NAMES="gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon"

for name in "${DATASETS[@]}"; do
  out="experiments/h05/${name}"
  mkdir -p "$out"

  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} (best.pt exists) ========"
  else
    echo "======== train ${name} -> ${out} ========"
    "$PY" -u train.py -c config_h05.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true
  fi

  if [[ -f "${out}/samples_poisson.npy" && -f "${out}/samples_nb.npy" && -f "${out}/samples_repoisson.npy" && -f "${out}/samples_twopois.npy" ]]; then
    echo "======== skip sample ${name} ========"
  else
    echo "======== sample ${name} 200 steps h=0.5 ========"
    "$PY" -u sample.py -c config_h05.yml \
      --ckpt "${out}/best.pt" \
      --out_dir "$out" \
      --lbd 100 \
      --snr_min 0 \
      --snr_max 100 \
      --sample_steps 200 \
      --kernels poisson,nb,repoisson,twopois
  fi
done

echo "======== plot h05 ========"
"$PY" -u test.py --exp_root experiments/h05 --kernels poisson,nb,repoisson,twopois --names "$NAMES"
echo "done"
