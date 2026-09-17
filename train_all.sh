#!/usr/bin/env bash
# 前台跑，不要 nohup：
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./train_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p experiments

DATASETS=(gamma_ltj nb poismix_mod)

for name in "${DATASETS[@]}"; do
  out="experiments/${name}"
  mkdir -p "$out"

  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} (best.pt exists) ========"
  else
    echo "======== train ${name} -> ${out} ========"
    "$PY" -u train.py -c config.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100
  fi

  if [[ -f "${out}/samples_poisson.npy" && -f "${out}/samples_nb.npy" && -f "${out}/samples_repoisson.npy" ]]; then
    echo "======== skip sample ${name} ========"
  else
    echo "======== sample ${name} poisson/nb/repoisson ========"
    "$PY" -u sample.py -c config.yml \
      --ckpt "${out}/best.pt" \
      --out_dir "$out" \
      --lbd 100 \
      --snr_min 0 \
      --snr_max 100 \
      --sample_steps 100 \
      --kernels poisson,nb,repoisson
  fi
done

echo "======== plot all ========"
"$PY" -u test.py --exp_root experiments
echo "done"
