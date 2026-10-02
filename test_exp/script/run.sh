#!/usr/bin/env bash
# 旧实验：sweep / quad / oracle first-step。不重训。
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./test_exp/script/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon}"
CKPT="${CKPT_ROOT:-experiments/scale}"

echo "ckpt_root=$CKPT names=$NAMES"
"$PY" -u test_exp/sample_tests.py \
  --ckpt_root "$CKPT" \
  --out_root test_exp \
  --names "$NAMES" \
  --stage all
"$PY" -u test_exp/plot_tests.py --out_root test_exp --names "$NAMES"
echo "done -> test_exp/sweep_log.png test_exp/quad_vs_uniform.png test_exp/oracle.png"
