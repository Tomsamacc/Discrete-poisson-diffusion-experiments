#!/usr/bin/env bash
# (12) k_max=1/2/3 vs learned-mean: Poisson T=100 + oracle |log m| by γ bin.
# Train first: ./script/train_pmfratio_kmax.sh
#   ./test_exp/script/run_12_kmax.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
echo "exp=kmax_diag names=$NAMES"
"$PY" -u test_exp/kmax_diag.py --names "$NAMES" --device "$DEVICE" --n_sample "$N_SAMPLE"
echo "done -> test_exp/results/12_kmax/"
