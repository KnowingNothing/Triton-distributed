#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$(realpath python)

function run_nvshmem_team_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_team_split.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_pp.py
  # pre attn a2a
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --local-copy --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 2 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --local-copy --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --local-copy --no-apply_pack --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --local-copy --dp --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --local-copy --no-apply_pack --dp --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_pre_attn_all2all_intra_node.py 1 65536 112 128 --num_comm_sm 16 --comm_op QKVPackA2A --sp_size 8 --gqa 12 --dp --verify --iters 30
  # gemm a2a
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_gemm_all2all_intra_node.py 1 65536 6144 128 14336 --iters 30 --num_comm_sm 16 --sm_margin 4 --dtype=bfloat16 --comm_op QKVPackA2A --gqa 12 --sp_size 8 --verify
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_gemm_all2all_intra_node.py 1 65536 6144 128 14336 --iters 30 --num_comm_sm 16 --sm_margin 4 --dtype=bfloat16 --comm_op QKVPackA2A --gqa 12 --sp_size 8 --dp --verify
  # a2a gemm
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_all2all_gemm_intra_node.py 1 96 65536 128 6144 --iters 30 --dtype=bfloat16 --num_comm_sm 16 --sm_margin 4 --fuse_sync --sp_size 8 --verify
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_all2all_gemm_intra_node.py 1 96 65536 128 6144 --iters 30 --dtype=bfloat16 --num_comm_sm 16 --sm_margin 4 --fuse_sync --sp_size 8 --dp --verify
  # post attn a2a
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py 1 65536 96 128 --num_comm_sm 16 --a2a_only --sp_size 8  --fuse_sync --local-copy --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py 1 65536 96 128 --num_comm_sm 2 --a2a_only --sp_size 8  --fuse_sync --local-copy --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py 1 65536 96 128 --num_comm_sm 16 --a2a_only --sp_size 8  --fuse_sync --local-copy --dp --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py 1 65536 96 128 --num_comm_sm 16 --a2a_only --sp_size 8  --fuse_sync --verify --iters 30
  CUDA_DEVICE_MAX_CONNECTIONS=8 bash scripts/launch.sh python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py 1 65536 96 128 --num_comm_sm 16 --a2a_only --sp_size 8  --fuse_sync --dp --verify --iters 30
}

run_nvshmem_team_testcases
