#!/usr/bin/env bash
# Sample PMF-ratio ckpts: uniform T=100 h=1 and quadratic γ_t=100(t/100)^2.
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./script/sample_pmfratio.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,poissmix3}"
echo "sample pmfratio names=$NAMES"
"$PY" -u sample_pmfratio.py --names "$NAMES"
echo "done -> experiments/pmfratio/{name}/{unif_T100,quad_T100}/ and metrics.txt"
