#!/usr/bin/env bash
# Round after quadratic diagnostics: mean-cut, oracle Two-Poisson, θ-trapezoidal.
# 不重训。冻 experiments/scale/*/best.pt。输出 test_exp/results/09..11。
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./test_exp/script/run_round2.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "======== 9 mean_cut (oracle mean for γ<cut) ========"
"$HERE/run_09_mean_cut.sh"
echo "======== 10 oracle Two-Poisson moments ========"
"$HERE/run_10_oracle_twopois.sh"
echo "======== 11 θ-trapezoidal mean-Poisson ========"
"$HERE/run_11_trap.sh"
echo "round2 done -> test_exp/results/{09_mean_cut,10_oracle_twopois,11_trap}/"
