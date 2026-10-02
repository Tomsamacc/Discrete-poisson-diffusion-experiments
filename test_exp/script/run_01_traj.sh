#!/usr/bin/env bash
# (1) trajectory-marginal z_γ vs true Pois(γ X). Quadratic T=100.
#   ./test_exp/script/run_01_traj.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=traj ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp traj \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels "$KERNELS" \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/01_traj_marginal/"
