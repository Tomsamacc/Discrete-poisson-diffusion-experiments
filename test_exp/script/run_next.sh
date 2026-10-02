#!/usr/bin/env bash
# 二次 schedule 之后的 8 个诊断，按建议优先级。
# 不重训。冻 experiments/scale/*/best.pt。输出 test_exp/results/。
#   conda activate itdpdm
#   cd /home/zhaog30/LTJ_experiments
#   ./test_exp/script/run_next.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "======== 1 traj ========"
"$HERE/run_01_traj.sh"
echo "======== 2 exponent ========"
"$HERE/run_02_exponent.sh"
echo "======== 4 oracle mean ========"
"$HERE/run_04_oracle_mean.sh"
echo "======== 5 cutoff ========"
"$HERE/run_05_cutoff.sh"
echo "======== 3 R ========"
"$HERE/run_03_R.sh"
echo "======== 7 twopois ========"
"$HERE/run_07_twopois.sh"
echo "======== 6 first step ========"
"$HERE/run_06_first_step.sh"
echo "======== 8 sweep ========"
"$HERE/run_08_sweep.sh"
echo "all next exps done -> test_exp/results/"
