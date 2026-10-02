#!/usr/bin/env bash
# J + raw-target scale. No training, no sampling.
#   ./test_exp/script/run_15_diag.sh
set -euo pipefail
NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
source "$(dirname "$0")/_common.sh"
echo "exp=15_diag names=$NAMES device=$DEVICE"
"$PY" -u test_exp/dynamics_diag.py --names "$NAMES" --device "$DEVICE" --n "${N_DIAG:-4096}"
echo "done -> test_exp/results/15_diag/"
