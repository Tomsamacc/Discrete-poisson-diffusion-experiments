#!/usr/bin/env bash
# (8) uniform T=100,200,500,1000, same ckpt. γ_max=100.
#   ./test_exp/script/run_08_sweep.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
echo "exp=sweep ckpt=$CKPT names=$NAMES"
"$PY" -u test_exp/next_exps.py --exp sweep \
  --ckpt_root "$CKPT" --names "$NAMES" --kernels "$KERNELS" \
  --n_sample "$N_SAMPLE" --device "$DEVICE"
echo "done -> test_exp/results/08_sweep/"
