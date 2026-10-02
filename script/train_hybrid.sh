#!/usr/bin/env bash
# Hybrid m1-protected training. Run after ./script/train_raw_cons_scale.sh.
#
# Shared trunk, three raw log-S heads. Inference is one forward pass.
#   m_hat_k = (z+1)...(z+k) / gamma^k * exp(a_k)
#
# L1 is the direct posterior-mean Bregman D(X, m_hat_1).
# L2 and L3 stay raw-ratio Bregman D(R_k, exp(a_k)).
#
# hybrid_eq:  L = L1 + L2 + L3
# hybrid_dyn: L = L1 + (h/2) L2 + (h^2/6) L3
#   h is the quadratic T=100 reverse step at the training gamma.
#
# 4 datasets x 2 losses = 8 runs. Skip a run when best.pt exists.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_hybrid.sh
# If a run is stopped before epoch 200, delete that directory before launching again.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

run_one() {
  local out="$1"
  local weight="$2"
  local name="$3"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${out} (best.pt exists) ========"
    return
  fi
  echo "======== train ${out} hybrid_weight=${weight} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --k_max 3 \
    --moment_loss hybrid \
    --hybrid_weight "$weight" \
    --weight_steps 100
}

for name in "${DATASETS[@]}"; do
  run_one "experiments/hybrid/hybrid_eq/${name}" equal "$name"
  run_one "experiments/hybrid/hybrid_dyn/${name}" dyn "$name"
done

echo "train done -> experiments/hybrid/{hybrid_eq,hybrid_dyn}/"
echo "then: ./test_exp/script/run_21_hybrid.sh"
