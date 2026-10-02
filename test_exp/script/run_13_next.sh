#!/usr/bin/env bash
# (13) PDF next: A–F Poisson quadratic + G oracle sampler + signed tails + γ-grid.
# Train first: ./script/train_pmfratio_next.sh
#   ./test_exp/script/run_13_next.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
echo "exp=13_next names=$NAMES"
"$PY" -u test_exp/next_ratio.py --names "$NAMES" --device "$DEVICE" --n_sample "$N_SAMPLE"
echo "done -> test_exp/results/13_next/"
