#!/usr/bin/env bash
# Collect single-head moment errors and grad-norm quantiles. No sampling.
#   ./test_exp/script/run_14_direct.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
echo "exp=14_direct names=$NAMES"
"$PY" -u test_exp/direct_prl_eval.py --names "$NAMES" --device "$DEVICE"
echo "done -> test_exp/results/14_direct/"
