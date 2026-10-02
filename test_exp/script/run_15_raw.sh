#!/usr/bin/env bash
# Collect raw-ratio moment / ratio errors and grad norms. No sampling.
#   ./test_exp/script/run_15_raw.sh
set -euo pipefail
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
source "$(dirname "$0")/_common.sh"
echo "exp=15_raw names=$NAMES device=$DEVICE"
"$PY" -u test_exp/raw_ratio_eval.py --names "$NAMES" --device "$DEVICE"
echo "done -> test_exp/results/15_raw/"
