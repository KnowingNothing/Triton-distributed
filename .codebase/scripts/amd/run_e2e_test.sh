#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

DENSE_MODEL=".codebase/hf_configs/Qwen/Qwen3-0.6B"

function run_amd_testcases() {
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_mlp.py --M 4096 --model ${DENSE_MODEL}
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_mlp.py --M 4096 --model ${DENSE_MODEL} --per_op
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_attn.py --bsz 32 --seq_len 128 --model ${DENSE_MODEL} --mode prefill
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_attn.py --bsz 4096 --seq_len 128 --model ${DENSE_MODEL} --mode decode
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 16 --seq_len 256 --model ${DENSE_MODEL} --check
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 32 --seq_len 128 --model ${DENSE_MODEL} --mode prefill
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 4096 --seq_len 128 --model ${DENSE_MODEL} --mode decode
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_e2e_inference.py --bsz 4096 --gen_len 128 --model ${DENSE_MODEL} --max_length 150
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_e2e_inference.py --bsz 4096 --gen_len 128 --model ${DENSE_MODEL} --max_length 150 --triton_dist
}

export RANDOM_PARAMS=1
run_amd_testcases
