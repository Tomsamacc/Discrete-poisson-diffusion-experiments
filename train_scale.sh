#!/usr/bin/env bash
# 前台跑：
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./train_scale.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p experiments/scale

DATASETS=(gamma_ltj nb poismix_mod)

for name in "${DATASETS[@]}"; do
  out="experiments/scale/${name}"
  mkdir -p "$out"

  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} (best.pt exists) ========"
  else
    echo "======== train ${name} scale=true -> ${out} ========"
    "$PY" -u train.py -c config_scale.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true
  fi

  if [[ -f "${out}/samples_poisson.npy" && -f "${out}/samples_nb.npy" && -f "${out}/samples_repoisson.npy" ]]; then
    echo "======== skip sample ${name} ========"
  else
    echo "======== sample ${name} ========"
    "$PY" -u sample.py -c config_scale.yml \
      --ckpt "${out}/best.pt" \
      --out_dir "$out" \
      --lbd 100 \
      --snr_min 0 \
      --snr_max 100 \
      --sample_steps 100 \
      --kernels poisson,nb,repoisson
  fi
done

echo "======== plot scale ========"
"$PY" -u test.py --exp_root experiments/scale
echo "done"
