#!/usr/bin/env bash
# (4) analytic posterior mean + Poisson vs learned mean + Poisson.
#     uniform T=100 and quadratic T=100.
#   ./test_exp/script/run_04_oracle_mean.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=oracle_mean ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp oracle_mean \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels poisson \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/04_oracle_mean/"
