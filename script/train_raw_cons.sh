#!/usr/bin/env bash
# Ratio consistency, first stage. Data loss stays L1+L2+L3.
#   soft:      L1+L2+L3 + 1 * (delta_2^2 + delta_3^2)
#   anchored:  L1+L2+L3 + 1 * stopgrad residual on a2, a3
#   hard:      L1 only. S2 and S3 are products of S1 at inference.
# 4 datasets x 3 modes = 12 runs. Skip a run when best.pt exists.
# Datasets: gamma_ltj, nb, poismix_mod (two Poisson), poissmix3 (three Poisson).
# Does not train consistency together with dyn or dyn_norm.
# Does not sweep lambda_cons.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_raw_cons.sh
#   ./test_exp/script/run_18_ratio_cons.sh
# If a run is stopped before epoch 200, delete that directory before launching again.
# best.pt is written whenever val improves, so a partial directory would be skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)
MODES=(soft anchored hard)

for mode in "${MODES[@]}"; do
  for name in "${DATASETS[@]}"; do
    out="experiments/raw_cons/${mode}/${name}"
    mkdir -p "$out"
    if [[ -f "${out}/best.pt" ]]; then
      echo "======== skip raw cons ${mode} ${name} (best.pt exists) ========"
      continue
    fi
    echo "======== train raw cons ${mode} ${name} -> ${out} ========"
    if [[ "$mode" == "hard" ]]; then
      "$PY" -u train.py -c config_scale.yml \
        --data "data/${name}" \
        --out_dir "$out" \
        --lbd 100 \
        --scale true \
        --consistency hard \
        --lambda_cons 1 \
        --moment_loss raw_ratio \
        --moment_k 1 \
        --moment_param raw \
        --k_max 1
    else
      "$PY" -u train.py -c config_scale.yml \
        --data "data/${name}" \
        --out_dir "$out" \
        --lbd 100 \
        --scale true \
        --k_max 3 \
        --moment_loss raw_multi \
        --raw_weight equal \
        --weight_steps 100 \
        --consistency "$mode" \
        --lambda_cons 1
    fi
  done
done
echo "train done -> experiments/raw_cons/{soft,anchored,hard}/"
echo "then: ./test_exp/script/run_18_ratio_cons.sh"
