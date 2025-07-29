#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_e2e_testcases_dense() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model Qwen/Qwen3-32B --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --mode decode --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --check --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --mode decode --use_allreduce --allreduce_method two_shot_multimem
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --triton_dist
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --triton_dist_AR
}

function run_e2e_testcases_moe() {
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_moe.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-30B-A3B
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-30B-A3B --mode prefill
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-30B-A3B --mode decode
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-30B-A3B --check
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-30B-A3B --mode prefill
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-30B-A3B --mode decode
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model Qwen/Qwen3-30B-A3B --no_graph
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --triton_dist --model Qwen/Qwen3-30B-A3B
}

run_e2e_testcases_dense
run_e2e_testcases_moe