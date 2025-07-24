#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_unittest_testcases() {
  # distributed ops
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_distributed-notify-wait.py
  # ag gemm
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 11008 4096
  # gemm rs
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_gemm_rs_intra_node.py 8192 4096 12288
}

run_unittest_testcases
