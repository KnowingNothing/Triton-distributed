#/bin/bash

set -e

function run_tutorials_testcases() {
  # tutorials
  # 09
  bash ./scripts/launch_amd.sh ./tutorials/09-AMD-overlapping-allgather-gemm.py
  # 10
  bash ./scripts/launch_amd.sh ./tutorials/10-AMD-overlapping-gemm-reduce-scatter.py
}

run_tutorials_testcases
