#!/usr/bin/env bash
# (6) first-step: true E[X], m_θ(z=0,γ≈0 / 0.01), exact gap PMF.
#   ./test_exp/script/run_06_first_step.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=first_step ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp first_step \
  --ckpt_root "$CKPT" --names "$NAMES" --device "$DEVICE"
echo "done -> test_exp/results/06_first_step/"
