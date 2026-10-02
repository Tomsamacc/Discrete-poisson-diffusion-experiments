# sourced by run_0*.sh
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/zhaog30/miniconda3/envs/itdpdm/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

NAMES="${NAMES:-gamma_ltj,nb,poismix_mod,pois20,poissmix,poissmix3,zip,yule_simon}"
KERNELS="${KERNELS:-poisson,nb,twopois}"
CKPT="${CKPT_ROOT:-experiments/scale}"
N_SAMPLE="${N_SAMPLE:-50000}"
N_DIAG="${N_DIAG:-8192}"
DEVICE="${DEVICE:-cuda}"
