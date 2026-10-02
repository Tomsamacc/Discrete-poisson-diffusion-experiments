#!/usr/bin/env bash
# Offline m1/m2/m3 errors, then quadratic T=100 sampling.
# Poisson, NB, and TwoPois for hybrid_eq and hybrid_dyn.
# Direct learned-mean and dyn_norm rows are copied, not resampled.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

"$PY" -u test_exp/hybrid_eval.py
"$PY" -u test_exp/sample_hybrid.py
