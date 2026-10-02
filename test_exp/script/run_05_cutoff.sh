#!/usr/bin/env bash
# (5) exact mixed-Poisson reverse for γ < γ_cut, then mean-Poisson.
#     cuts 0.01, 0.05, 0.1, 0.2, 0.5, 1.0. Quadratic T=100.
#   ./test_exp/script/run_05_cutoff.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=cutoff ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp cutoff \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels poisson \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/05_cutoff/"
