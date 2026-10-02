#!/usr/bin/env bash
# PDF next matrix: C balanced k=1, D raw k=1, E balanced k=2, F balanced k=3.
# Same config as config_pmfratio.yml. Does not touch scale / existing kmax{1,2,3}.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/train_pmfratio_next.sh
#   ./test_exp/script/run_13_next.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASETS=(gamma_ltj nb poismix_mod poissmix3)

# tag  out_dir_suffix          k_max  ratio_loss
JOBS=(
  "C  balanced_k1  1  balanced"
  "D  raw_k1       1  raw"
  "E  balanced_k2  2  balanced"
  "F  balanced_k3  3  balanced"
)

train_one() {
  local tag="$1" sub="$2" k="$3" rloss="$4" name="$5"
  local out="experiments/pmfratio/${sub}/${name}"
  mkdir -p "$out"
  if [[ -f "${out}/best.pt" ]]; then
    echo "======== skip ${tag} ${name} k_max=${k} ${rloss} (best.pt exists) ========"
    return
  fi
  echo "======== train ${tag} ${name} k_max=${k} ratio_loss=${rloss} -> ${out} ========"
  "$PY" -u train.py -c config_pmfratio.yml \
    --data "data/${name}" \
    --out_dir "$out" \
    --lbd 100 \
    --scale true \
    --ratio true \
    --k_max "$k" \
    --ratio_loss "$rloss"
}

for job in "${JOBS[@]}"; do
  read -r tag sub k rloss <<<"$job"
  for name in "${DATASETS[@]}"; do
    train_one "$tag" "$sub" "$k" "$rloss" "$name"
  done
done
echo "train done -> experiments/pmfratio/{balanced_k1,raw_k1,balanced_k2,balanced_k3}/"
echo "then: ./test_exp/script/run_13_next.sh"
