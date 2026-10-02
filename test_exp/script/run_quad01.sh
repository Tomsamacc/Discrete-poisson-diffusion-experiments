#!/usr/bin/env bash
# 100-step quadratic, γ: 0.1→λ, z0 ~ Pois(0.1 X). No retrain.
#   ./test_exp/script/run_quad01.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon}"
CKPT="${CKPT_ROOT:-experiments/scale}"

echo "ckpt_root=$CKPT names=$NAMES stage=quad01 γ0=0.1 z0~Pois(0.1 X)"
"$PY" -u test_exp/sample_tests.py \
  --ckpt_root "$CKPT" \
  --out_root test_exp \
  --names "$NAMES" \
  --stage quad01
"$PY" -u test_exp/plot_tests.py --out_root test_exp --names "$NAMES" --stage quad01
echo "done -> test_exp/quad_g01.png test_exp/metrics_quad01.txt"
