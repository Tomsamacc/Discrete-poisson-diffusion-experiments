#!/usr/bin/env bash
# Sample the lambda sweep and the weighted-consistency models.
# dyn_norm (lambda 0) and unweighted lambda 1 are copied.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

"$PY" -u test_exp/sample_cons_scale.py
