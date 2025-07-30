#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_nvshmem_team_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_team_split.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_pp.py
}

run_nvshmem_team_testcases
