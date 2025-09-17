#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../../..)
pushd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:`realpath python`

function run_unittest_testcases() {
  python3 python/triton_dist/test/nvidia/test_language_extra.py
}

function run_simt_testcases() {
  #############
  # ad-hoc test
  #############
  # simt
  python3 python/triton_dist/test/nvidia/test_simt.py
  # basic
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma_multi_barrier
}

function run_aot_testcases() {
  #############
  # ad-hoc test
  #############
  # aot compilation
  USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_compile_aot.py
}

function run_inductor_patch_testcases() {
  #############
  # ad-hoc test
  #############
  # incompibility of triton 3.4.0 with `torch.compile` (2.7.0 ... 2.8.0)
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_patch_torch_compile.py
}

function run_ag_gemm_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness
if [ -z "$L20_NO_RUN" ]; then
  bash scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness --local_world_size 2
fi
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_autotune
if [ -z "$L20_NO_RUN" ]; then
  bash scripts/launch.sh --nproc_per_node 4 python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_autotune --local_world_size 2
fi
}

function run_gemm_rs_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568 --check
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py 4096 4096 12288 --fuse_scatter --check
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py 4096 4096 12288 --fuse_scatter --no-persistent --check
}

function run_allgather_testcases() {
if [ -z "$L20_NO_RUN" ]; then
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_small_msg.py
fi
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_gather.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_fast_allgather.py --iters 10 --warmup_iters 20 --mode push_2d_ll --minbytes 4096 --maxbytes 8192
}

function run_ep_all2all_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_moe_inference.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_single_2d.py
}

function run_nvshmem_api_testcases() {
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ring_put.py
}

function run_flash_decoding_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent
  USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent_aot
  USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_aot
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case perf
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
  USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
}

function run_ag_moe_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20 --autotune
}

function run_moe_reduce_rs_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
}

function run_moe_reduce_ar_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_ar.py  8192 2048 1536 32 2
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_ar.py  8192 2048 1536 32 2 --autotune
}

function run_ep_a2a_testcases() {
  NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check
  NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8
  NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 4096 -N 6144 --topk 6  --drop_ratio 0.3  --check --with-scatter-indices  --has_weight
}

function run_sp_ag_attention_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_ag_attention_intra_node.py --batch_size 1 --q_head 32 --kv_head 32 --max_seqlen_q 8192 --max_seqlen_k 8192 --head_dim 128 --seqlens_q 8192 --seqlens_k 8192
}

function run_allreduce_testcases() {
  NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method double_tree --stress --iters 2 --verify_hang 50
  NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2 --verify_hang 50
  NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method two_shot --stress --iters 2 --verify_hang 50
  NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot_tma --stress --iters 2 --verify_hang 50
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot_multimem --stress --iters 2 --verify_hang 50
  NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method two_shot_multimem --stress --iters 2 --verify_hang 50
}

function run_gemm_ar_testcases() {
  # Skip GEMM AR tests for GPUs with compute capability lower than 9.0
  sm_version=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)

  if echo "$sm_version" | awk '$1 >= 9.0 {exit 0} {exit 1}'; then
      NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 32 5120 25600 --no-copy-to-local --low-latency
      NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 32 5120 25600 --check  --low-latency
      NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 28000 7168 4096 --no-copy-to-local --num_comm_sms 4
      NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 28000 7168 4096 --check --num_comm_sms 4
  else
      echo "Skipping GEMM AR tests for GPU with compute capability lower than 9.0"
  fi
}

function run_gdn_testcases() {
  USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_gdn.py --num_heads 12
}

function run_a2a_single_gemm_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_single_gemm.py --M 7168 --N 9216 --K 3072 --dtype int8 --check --iters 10
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_single_gemm.py --M 7168 --N 9216 --K 3072 --dtype float8_e4m3fn --check --iters 10
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_to_all_single_gemm.py --M 7168 --N 9216 --K 3072 --dtype float8_e5m2 --check --iters 10
}

function run_utils_testcases() {
  bash scripts/launch.sh python/triton_dist/test/nvidia/test_utils.py --case max_occupancy
}

# run all cases
run_unittest_testcases
run_simt_testcases
run_aot_testcases
run_ag_gemm_testcases
run_gemm_rs_testcases
run_allgather_testcases
run_ep_all2all_testcases
run_nvshmem_api_testcases
run_flash_decoding_testcases
run_ag_moe_testcases
run_moe_reduce_rs_testcases
run_ep_a2a_testcases
run_sp_ag_attention_testcases
run_allreduce_testcases
run_gemm_ar_testcases
run_utils_testcases
run_inductor_patch_testcases
run_moe_reduce_ar_testcases
run_gdn_testcases
run_a2a_single_gemm_testcases
