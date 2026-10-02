#!/usr/bin/env bash
# Collect raw multi-output errors and trunk gradient stats. No sampling.
#   ./test_exp/script/run_16_raw_multi.sh
set -euo pipefail
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
source "$(dirname "$0")/_common.sh"
echo "exp=16_raw_multi names=$NAMES device=$DEVICE"
"$PY" -u test_exp/raw_multi_eval.py --names "$NAMES" --device "$DEVICE"
echo "done -> test_exp/results/16_raw_multi/"
