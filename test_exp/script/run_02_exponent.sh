#!/usr/bin/env bash
# (2) schedule exponent sweep: γ_t = 100 (t/100)^p, p=1,1.5,2,3,4. T=100.
#   ./test_exp/script/run_02_exponent.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=exponent ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp exponent \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels "$KERNELS" \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/02_exponent/"
