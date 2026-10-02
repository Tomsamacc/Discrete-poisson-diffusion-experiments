#!/usr/bin/env bash
# (11) θ-trapezoidal mean-Poisson (Heun, θ=1/2) on quadratic γ.
#     euler T=100 (100 NFE, reused), trap T=50 (100 NFE), trap T=100 (200 NFE).
#   ./test_exp/script/run_11_trap.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=trap ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp trap \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels poisson \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/11_trap/"
