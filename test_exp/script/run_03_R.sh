#!/usr/bin/env bash
# (3) missing posterior variance R_t = h v / m. Quadratic T=100.
#   ./test_exp/script/run_03_R.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=R ckpt=$CKPT names=$NAMES n_diag=$N_DIAG"
"$PY" -u test_exp/next_exps.py --exp R \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels "$KERNELS" \
  --n_diag "$N_DIAG" --device "$DEVICE"
echo "done -> test_exp/results/03_R/"
