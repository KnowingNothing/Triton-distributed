#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_unittest_testcases() {
  # rocshmem api tests
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_rocshmem_api.py
  # distributed ops
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_distributed-notify-wait.py
  # ag gemm
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 11008 4096 --stress
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 11008 4096 --stress --autotune
  # for MI300X
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 11008 4096 --use_copy_kernel --stress --autotune
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 11008 4096 --use_copy_kernel --use_fused_kernel --stress --autotune
  # gemm rs
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_gemm_rs_intra_node.py 8192 4096 12288
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_gemm_rs_intra_node.py 8192 4096 12288 --transpose_weight # transpose weight performance is much worse
  # # gemm ar
  bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_gemm_ar_intra_node.py 8192 4096 12288 --stress
}

run_unittest_testcases
