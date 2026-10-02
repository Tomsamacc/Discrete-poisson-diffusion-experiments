#!/usr/bin/env bash
# (12) NB from oracle (m1, m2) vs learned shifted moments.
#     uniform T=100 and quadratic T=100. Learned reused from 02_exponent p1 and p2.
#   ./test_exp/script/run_12_oracle_nb.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=oracle_nb ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp oracle_nb \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels nb \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/12_oracle_nb/"
