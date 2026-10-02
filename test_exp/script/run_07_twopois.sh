#!/usr/bin/env bash
# (7) Two-Poisson stability: v,c3,g,w,x1,x2,hx, fallbacks. Quadratic T=100.
#   ./test_exp/script/run_07_twopois.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=twopois ckpt=$CKPT names=$NAMES n_diag=$N_DIAG"
"$PY" -u test_exp/next_exps.py --exp twopois \
  --ckpt_root "$CKPT" --names "$NAMES" \
  --n_diag "$N_DIAG" --device "$DEVICE"
echo "done -> test_exp/results/07_twopois/"
