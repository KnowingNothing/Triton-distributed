#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_amd_testcases() {
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --per_op
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 16 --seq_len 256 --model Qwen/Qwen3-32B --check
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150
  CUDA_GRAPH=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --triton_dist
}

run_amd_testcases
