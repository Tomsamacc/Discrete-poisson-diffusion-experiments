#!/usr/bin/env bash
# Higher heads, same loss. m1 is softplus(f1) and L1 = D(X, m1) in every run.
# k=2,3 are turned into S_k and trained with D(R_k, S_k).
#   raw_log:       S = exp(a),        m = c S
#   direct_ratio:  S = softplus(u),   m = c S
#   log_moment:    m = exp(b),        S = m / c
#   root_moment:   m = softplus(r)^k, S = m / c
#   direct_moment: m = softplus(v),   S = m / c
#   c_k = (z+1)...(z+k) / gamma^k
# equal: L1 + L2 + L3
# dyn:   L1 + (h/2) L2 + (h^2/6) L3
# 5 links x 2 weights x 4 datasets. Skip a run when best.pt exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)
PARAMS=(raw_log direct_ratio log_moment root_moment direct_moment)
WEIGHTS=(equal dyn)
if [[ $# -ge 1 ]]; then
  WEIGHTS=("$1")
fi

run_one() {
  local weight="$1"
  local param="$2"
  local name="$3"
  local out="experiments/hm/${weight}/${param}/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${out} (best.pt exists) ========"
    return
  fi
  echo "======== train ${out} ========"
  "$PY" -u train.py -c config_scale.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --k_max 3 \
    --moment_loss hm \
    --hm_param "$param" \
    --hybrid_weight "$weight" \
    --weight_steps 100
}

for weight in "${WEIGHTS[@]}"; do
  for param in "${PARAMS[@]}"; do
    for name in "${DATASETS[@]}"; do
      run_one "$weight" "$param" "$name"
    done
  done
done

echo "train done -> experiments/hm/"
echo "then: ./test_exp/script/run_26_hm.sh"
