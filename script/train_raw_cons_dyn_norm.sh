#!/usr/bin/env bash
# Same ratio-consistency runs, with dyn_norm head weights.
#   w = (h, h^2/2, h^3/6) / (h + h^2/2 + h^3/6)
#   soft:      w1*L1 + w2*L2 + w3*L3 + 1 * (delta_2^2 + delta_3^2)
#   anchored:  w1*L1 + w2*L2 + w3*L3 + 1 * stopgrad residual on a2, a3
# hard is L1 only, so these weights do not change it. The equal hard
# checkpoints in experiments/raw_cons/hard/ stay as they are.
# 4 datasets x 2 modes = 8 runs. Skip a run when best.pt exists.
# Does not touch experiments/raw_cons/ or experiments/raw_multi/.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_raw_cons_dyn_norm.sh
# If a run is stopped before epoch 200, delete that directory before launching again.
# best.pt is written whenever val improves, so a partial directory would be skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)
MODES=(soft anchored)

for mode in "${MODES[@]}"; do
  for name in "${DATASETS[@]}"; do
    out="experiments/raw_cons_dyn_norm/${mode}/${name}"
    mkdir -p "$out"
    if [[ -f "${out}/best.pt" ]]; then
      echo "======== skip raw cons dyn_norm ${mode} ${name} (best.pt exists) ========"
      continue
    fi
    echo "======== train raw cons dyn_norm ${mode} ${name} -> ${out} ========"
    "$PY" -u train.py -c config_scale.yml \
      --data "data/${name}" \
      --out_dir "$out" \
      --lbd 100 \
      --scale true \
      --k_max 3 \
      --moment_loss raw_multi \
      --raw_weight dyn_norm \
      --weight_steps 100 \
      --consistency "$mode" \
      --lambda_cons 1
  done
done
echo "train done -> experiments/raw_cons_dyn_norm/{soft,anchored}/"
echo "then: ./test_exp/script/run_19_ratio_cons_dyn_norm.sh"
