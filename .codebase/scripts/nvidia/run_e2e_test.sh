#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

DENSE_MODEL=".codebase/hf_configs/Qwen/Qwen3-0.6B"
MOE_MODEL=".codebase/hf_configs/Qwen/Qwen3-30B-A3B"

function run_e2e_testcases_dense() {
  # tp mlp
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model ${DENSE_MODEL} --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model ${DENSE_MODEL} --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 2048 --model ${DENSE_MODEL} --mode gemm_ar
  
  # tp attn prefill
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 8 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode gemm_ar
  
  # tp attn decode
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode gemm_ar
  
  # tp e2e check
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model ${DENSE_MODEL} --check --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model ${DENSE_MODEL} --check --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model ${DENSE_MODEL} --check --mode gemm_ar

  # tp e2e prefill
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model ${DENSE_MODEL} --run_type prefill --mode gemm_ar

  # tp e2e decode
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode ag_rs
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode allreduce
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model ${DENSE_MODEL} --run_type decode --mode gemm_ar

  # e2e inference
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model ${DENSE_MODEL} --backend torch
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model ${DENSE_MODEL} --backend triton_dist
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --model ${DENSE_MODEL} --backend triton_dist_AR
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --model ${DENSE_MODEL} --backend triton_dist_gemm_ar
}

function run_e2e_testcases_moe() {
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_moe.py --bsz 32 --seq_len 128 --model ${MOE_MODEL}
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model ${MOE_MODEL} --run_type prefill --mode ag_rs
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model ${MOE_MODEL} --run_type decode --mode ag_rs
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model ${MOE_MODEL} --check  --mode ag_rs
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model ${MOE_MODEL} --run_type prefill --mode ag_rs
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model ${MOE_MODEL} --run_type decode --mode ag_rs
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model ${MOE_MODEL} --no_graph --backend torch
  bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model ${MOE_MODEL} --backend triton_dist
}

export RANDOM_PARAMS=1
run_e2e_testcases_dense
run_e2e_testcases_moe