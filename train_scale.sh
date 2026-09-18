#!/usr/bin/env bash
# 100 steps, h=1, scale=True. 不覆盖 experiments/{gamma_ltj,nb,...} 那次没 scale 的结果。
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

DATASETS=(gamma_ltj nb poismix_mod pois20 poissmix poissmix3 zip yule_simon)
NAMES="gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon"

for name in "${DATASETS[@]}"; do
  out="experiments/scale/${name}"
  mkdir -p "$out"

  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip train ${name} (best.pt exists) ========"
  else
    echo "======== train ${name} scale=true 100 steps -> ${out} ========"
    "$PY" -u train.py -c config_scale.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true
  fi

  if [[ -f "${out}/samples_poisson.npy" && -f "${out}/samples_nb.npy" && -f "${out}/samples_repoisson.npy" && -f "${out}/samples_twopois.npy" ]]; then
    echo "======== skip sample ${name} ========"
  else
    echo "======== sample ${name} 100 steps h=1 ========"
    "$PY" -u sample.py -c config_scale.yml \
      --ckpt "${out}/best.pt" \
      --out_dir "$out" \
      --lbd 100 \
      --snr_min 0 \
      --snr_max 100 \
      --sample_steps 100 \
      --kernels poisson,nb,repoisson,twopois
  fi
done

echo "======== plot scale (100 steps) ========"
"$PY" -u test.py --exp_root experiments/scale --kernels poisson,nb,repoisson,twopois --names "$NAMES"
echo "done"
