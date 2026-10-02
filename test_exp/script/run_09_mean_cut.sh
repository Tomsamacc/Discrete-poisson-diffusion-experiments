#!/usr/bin/env bash
# (9) quadratic T=100: oracle posterior mean for γ<γ_cut, then learned mean + Poisson.
#     cuts 0.1 and 1.0. cut=0 / cut=inf reused from 02 p=2 and 04 quad oracle.
#   ./test_exp/script/run_09_mean_cut.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=mean_cut ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp mean_cut \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels poisson \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/09_mean_cut/"
