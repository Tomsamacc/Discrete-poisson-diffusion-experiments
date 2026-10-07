#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$HOME/venv/itdpdm/bin/python" ]]; then
    PY="$HOME/venv/itdpdm/bin/python"
  elif [[ -x /home/zhaog30/miniconda3/envs/itdpdm/bin/python ]]; then
    PY=/home/zhaog30/miniconda3/envs/itdpdm/bin/python
  else
    PY=python
  fi
fi
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "$ROOT"
exec "$PY" -u test_exp/nb_trunk_cases.py "$@"
