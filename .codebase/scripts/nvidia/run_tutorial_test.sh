#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_tutorial_testcases() {
  # tutorials
  # 01
  bash scripts/launch.sh ./tutorials/01-distributed-notify-wait.py
  # 02
  bash scripts/launch.sh ./tutorials/02-intra-node-allgather.py
  # 03
  bash scripts/launch.sh ./tutorials/03-inter-node-allgather.py
  # 04
  bash scripts/launch.sh ./tutorials/04-deepseek-infer-all2all.py
  # 05
  bash scripts/launch.sh ./tutorials/05-intra-node-reduce-scatter.py
  # 06
  bash scripts/launch.sh ./tutorials/06-inter-node-reduce-scatter.py
  # 07
  bash scripts/launch.sh ./tutorials/07-overlapping-allgather-gemm.py
  # 08
  bash scripts/launch.sh ./tutorials/08-overlapping-gemm-reduce-scatter.py
}

run_tutorial_testcases
