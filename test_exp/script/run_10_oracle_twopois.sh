#!/usr/bin/env bash
# (10) Two-Poisson from oracle (m1,m2,m3)=E[X^k|z] vs learned +k moments.
#     uniform T=100 and quadratic T=100. Learned reused from 08 / 02 p=2.
#   ./test_exp/script/run_10_oracle_twopois.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=oracle_twopois ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp oracle_twopois \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels twopois \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/10_oracle_twopois/"
